#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate TinySegNet on VOC2012 val: mIoU + optional colorized predictions.
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image

# --------------------------
# Constants (match training)
# --------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 21
IGNORE_INDEX  = 255

def resolve_voc_root(data_root: Path) -> Path:
    if (data_root / "VOCdevkit" / "VOC2012").is_dir():
        return data_root
    if (data_root / "VOC2012").is_dir():
        return data_root.parent if data_root.name == "VOC2012" else data_root
    raise FileNotFoundError(f"VOCdevkit/VOC2012 not found under: {data_root}")

def normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

def val_transform(img, mask, crop=320):
    # Resize shorter side to 360, center-crop to 320 (same as your training val_transform)
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0 / short
    new_size = (int(round(h * scale)), int(round(w * scale)))
    img = TF.resize(img, new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)
    img = TF.center_crop(img, [crop, crop])
    mask = TF.center_crop(mask, [crop, crop])
    img_t  = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t

class VOCSegVal(torch.utils.data.Dataset):
    def __init__(self, root: Path, crop_size: int = 320):
        self.ds = VOCSegmentation(root=str(root), year="2012", image_set="val", download=False)
        self.crop = crop_size
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, mask = self.ds[i]
        return val_transform(img, mask, crop=self.crop)

# --------------------------
# Model (import from train)
# --------------------------
def load_model(num_classes=NUM_CLASSES):
    try:
        # If your training file is in the same folder, reuse the class definition:
        from customModel_train import TinySegNet   # type: ignore
        return TinySegNet(num_classes=num_classes)
    except Exception:
        # Fallback: minimal inline copy if import path differs.
        import torch.nn as nn, torch.nn.functional as F
        class SE(nn.Module):
            def __init__(self, c: int, r: int = 4):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(c, c // r, 1)
                self.fc2 = nn.Conv2d(c // r, c, 1)
            def forward(self, x):
                s = self.pool(x); s = F.silu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
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
                return self.se(x)
        class ASPPLite(nn.Module):
            def __init__(self, cin, rates=(1,6,12,18), branch_ch=64):
                super().__init__()
                self.branches = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(cin, branch_ch, 3 if r>1 else 1, padding=r if r>1 else 0,
                                  dilation=r if r>1 else 1, bias=False),
                        nn.BatchNorm2d(branch_ch), nn.SiLU()
                    ) for r in rates
                ])
                self.proj = nn.Sequential(nn.Conv2d(branch_ch*len(rates), 256, 1, bias=False),
                                          nn.BatchNorm2d(256), nn.SiLU(), nn.Dropout(0.5))
            def forward(self, x):
                feats = [b(x) for b in self.branches]
                return self.proj(torch.cat(feats, dim=1))
        class TinySegNet(nn.Module):
            def __init__(self, num_classes=NUM_CLASSES):
                super().__init__()
                chs = [24,40,96,192]
                self.stem = nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1,bias=False),
                                          nn.BatchNorm2d(16), nn.SiLU())
                self.s1 = DWConvBlock(16,chs[0],stride=1,use_se=False)
                self.s2 = DWConvBlock(chs[0],chs[1],stride=2,use_se=False)
                self.s3 = DWConvBlock(chs[1],chs[2],stride=2,use_se=True)
                self.s4 = DWConvBlock(chs[2],chs[3],stride=2,use_se=True)
                self.aspp = ASPPLite(chs[3], rates=(1,6,12,18), branch_ch=64)
                self.mid_proj = nn.Conv2d(chs[1], 64, 1)
                self.low_proj = nn.Conv2d(chs[0], 32, 1)
                self.dec_mid = nn.Sequential(nn.Conv2d(256,256,3,padding=1,bias=False),
                                             nn.BatchNorm2d(256), nn.SiLU())
                self.dec_low = nn.Sequential(nn.Conv2d(64,64,3,padding=1,bias=False),
                                             nn.BatchNorm2d(64), nn.SiLU())
                self.dropout_head = nn.Dropout(0.5)
                self.cls = nn.Conv2d(64, num_classes, 1)
            def forward(self, x):
                _,_,H,W = x.shape
                import torch.nn.functional as F
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
                logits = F.interpolate(logits_4x, size=(H,W), mode="bilinear", align_corners=False)
                return {"out": logits}
        return TinySegNet(num_classes=num_classes)

# --------------------------
# Metrics
# --------------------------
@torch.no_grad()
def eval_miou(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
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
# Color map + saving
# --------------------------
def voc_color_map():
    # Standard PASCAL VOC palette (first few; rest by bit-twiddling)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r=g=b=0; c=i
        for j in range(8):
            r |= ((c & 1) << (7-j)); g |= (((c>>1) & 1) << (7-j)); b |= (((c>>2) & 1) << (7-j))
            c >>= 3
        cmap[i] = [r,g,b]
    return cmap

def save_color_mask(mask_np: np.ndarray, path: Path, cmap: np.ndarray):
    pal = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    pal.putpalette(cmap.flatten().tolist())
    pal.save(path)

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="tinyseg_best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--save-limit", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    voc_root = resolve_voc_root(Path(args.data_root))
    val_set = VOCSegVal(voc_root, crop_size=args.crop_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model + weights
    model = load_model(NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Evaluate
    miou, per_cls = eval_miou(model, val_loader, device)
    print("\n==== Evaluation ====")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU (len=21):")
    print([float(f"{x:.4f}") for x in per_cls])

    # Optional prediction dumps
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmap = voc_color_map()
        saved = 0
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(val_loader):
                logits = model(imgs.to(device))["out"]
                preds = logits.argmax(1).cpu().numpy()
                for b in range(preds.shape[0]):
                    save_color_mask(preds[b], out_dir / f"val_{i:04d}_{b}.png", cmap)
                    saved += 1
                    if saved >= args.save_limit:
                        break
                if saved >= args.save_limit:
                    break
        print(f"[i] Saved {saved} colorized predictions to: {out_dir}")

if __name__ == "__main__":
    main()