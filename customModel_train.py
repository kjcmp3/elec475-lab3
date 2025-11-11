#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinySegNet on VOC2012 (21 classes): training + evaluation template with mIoU logging.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

# --------------------------
# Constants
# --------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES = 21
IGNORE_INDEX = 255

def resolve_voc_root(data_root: Path) -> Path:
    """Return a path whose tree contains VOCdevkit/VOC2012."""
    if (data_root / "VOCdevkit" / "VOC2012").is_dir():
        return data_root
    if (data_root / "VOC2012").is_dir():
        return data_root.parent if data_root.name == "VOC2012" else data_root
    raise FileNotFoundError(f"VOCdevkit/VOC2012 not found under: {data_root}")

# --------------------------
# Transforms (paired)
# --------------------------
def normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

def train_transform(img, mask, crop=320):
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0 / short
    new_size = (int(round(h * scale)), int(round(w * scale)))

    img = TF.resize(img, new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

    # random crop
    i = torch.randint(0, new_size[0]-crop+1, ()).item()
    j = torch.randint(0, new_size[1]-crop+1, ()).item()
    img = TF.crop(img, i, j, crop, crop)
    mask = TF.crop(mask, i, j, crop, crop)

    if torch.rand(()) < 0.5:
        img = TF.hflip(img); mask = TF.hflip(mask)

    img_t = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t

def val_transform(img, mask, crop=320):
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0 / short
    new_size = (int(round(h * scale)), int(round(w * scale)))

    img = TF.resize(img, new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

    img = TF.center_crop(img, [crop, crop])
    mask = TF.center_crop(mask, [crop, crop])

    img_t = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t

# --------------------------
# Dataset wrapper
# --------------------------
class VOCSegPair(Dataset):
    def __init__(self, root: Path, image_set: str, crop_size: int, is_train: bool):
        self.ds = VOCSegmentation(root=str(root), year="2012", image_set=image_set, download=False)
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        if self.is_train:
            img_t, mask_t = train_transform(img, mask, crop=self.crop_size)
        else:
            img_t, mask_t = val_transform(img, mask, crop=self.crop_size)
        return img_t, mask_t

# --------------------------
# Model (same as your TinySegNet)
# --------------------------
class SE(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DWConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, use_se=False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.se = SE(cout, r=4) if use_se else nn.Identity()

    def forward(self, x):
        x = F.silu(self.dw_bn(self.dw(x)))
        x = F.silu(self.pw_bn(self.pw(x)))
        x = self.se(x)
        return x

class ASPPLite(nn.Module):
    def __init__(self, cin, rates=(1, 6, 12, 18), branch_ch=64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, branch_ch, 3 if r > 1 else 1, padding=r if r > 1 else 0,
                          dilation=r if r > 1 else 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU()
            ) for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class TinySegNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        chs = [24, 40, 96, 192]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )
        self.s1 = DWConvBlock(16,  chs[0], stride=1, use_se=False)   # 1/2
        self.s2 = DWConvBlock(chs[0], chs[1], stride=2, use_se=False) # 1/4
        self.s3 = DWConvBlock(chs[1], chs[2], stride=2, use_se=True)  # 1/8
        self.s4 = DWConvBlock(chs[2], chs[3], stride=2, use_se=True)  # 1/16

        self.aspp = ASPPLite(chs[3], rates=(1,6,12,18), branch_ch=64)  # -> 256

        self.mid_proj = nn.Conv2d(chs[1], 64, 1)
        self.low_proj = nn.Conv2d(chs[0], 32, 1)

        self.dec_mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.dec_low = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        self.dropout_head = nn.Dropout(0.5)
        self.cls = nn.Conv2d(64, num_classes, 1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        n, _, H, W = x.shape
        x = self.stem(x)
        low  = self.s1(x)
        mid  = self.s2(low)
        h8   = self.s3(mid)
        high = self.s4(h8)

        ctx = self.aspp(high)
        up_mid = F.interpolate(ctx, scale_factor=2, mode="bilinear", align_corners=False)
        fuse_mid = self.dec_mid(up_mid + self.mid_proj(mid))

        up_low = F.interpolate(fuse_mid, scale_factor=2, mode="bilinear", align_corners=False)
        fuse_low = self.dec_low(up_low + self.low_proj(low))

        logits_4x = self.cls(self.dropout_head(fuse_low))
        logits = F.interpolate(logits_4x, size=(H, W), mode="bilinear", align_corners=False)

        return {"out": logits, "taps": {"low": low, "mid": mid, "high": high}}

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --------------------------
# Metrics
# --------------------------
@torch.no_grad()
def eval_miou(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pred = model(imgs)["out"].argmax(1)
        valid = masks != IGNORE_INDEX
        for c in range(NUM_CLASSES):
            pc = (pred == c) & valid
            tc = (masks == c) & valid
            inter[c] += (pc & tc).sum()
            union[c] += (pc | tc).sum()
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return float(iou.mean().item()), iou.detach().cpu().numpy()

# --------------------------
# Train
# --------------------------
def poly_lr_lambda(it: int, total: int, power: float = 0.9):
    return (1.0 - it / float(total)) ** power

def train_one_epoch(model, loader, optim, device, epoch_it, total_its):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    running = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        loss = ce(model(imgs)["out"], masks)
        loss.backward()
        optim.step()
        running += float(loss.item()) * imgs.size(0)
        epoch_it[0] += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
    return running / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="Folder that contains VOCdevkit (so VOCdevkit/VOC2012 exists).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", dest="workers", type=int, default=2)
    ap.add_argument("--num-workers", dest="workers", type=int, help="Alias for --workers")
    ap.add_argument("--crop-size", dest="crop_size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    try:
        voc_root = resolve_voc_root(Path(args.data_root))
    except FileNotFoundError as e:
        print(str(e))
        print("Hint: in Colab, use /content/data prepared by your KaggleHub cell.")
        raise
    print(f"[i] Using VOC root: {voc_root}")

    # Device handling
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[i] Device: {device}")

    # Datasets / Loaders
    train_set = VOCSegPair(voc_root, image_set="train", crop_size=args.crop_size, is_train=True)
    val_set   = VOCSegPair(voc_root, image_set="val",   crop_size=args.crop_size, is_train=False)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_mem, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=max(1, args.batch_size//2), shuffle=False,
                              num_workers=args.workers, pin_memory=pin_mem)

    # Model
    model = TinySegNet(num_classes=NUM_CLASSES)
    print(f"[i] TinySegNet params: {count_params(model)/1e6:.3f}M")
    model.to(device)

    # Optimizer + poly LR
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_iters = max(1, args.epochs * len(train_loader))
    scheduler = LambdaLR(optim, lr_lambda=lambda it: poly_lr_lambda(it, total_iters, power=0.9))

    best_miou = 0.0
    global_it = [0]

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, device, global_it, total_iters)
        scheduler.step()
        miou, _ = eval_miou(model, val_loader, device)
        print(f"Epoch {epoch:03d}/{args.epochs} | Train CE: {train_loss:.4f} | Val mIoU: {miou:.4f} | LR: {optim.param_groups[0]['lr']:.3e}")
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": best_miou}, "tinyseg_best.pt")
            print(f"[i] New best mIoU {best_miou:.4f}. Saved -> tinyseg_best.pt")

    print(f"[✓] Training done. Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
