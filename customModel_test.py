#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval TinySegNet on VOC2012 val: mIoU + optional PNG dumps.

Usage:
  python customModel_test.py \
    --data-root /content/data \
    --ckpt tinyseg_best.pt \
    --batch-size 8 \
    --crop-size 320 \
    --device cuda \
    --save-dir preds_val
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image

# --------------------------
# Constants / palette
# --------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 21
IGNORE_INDEX  = 255

def voc_palette():
    palette = [0]*(256*3)
    colors = [
        (0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),
        (128,0,128),(0,128,128),(128,128,128),(64,0,0),(192,0,0),
        (64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),
        (192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),
        (0,64,128)
    ]
    for i,(r,g,b) in enumerate(colors):
        palette[i*3+0]=r; palette[i*3+1]=g; palette[i*3+2]=b
    return palette

# --------------------------
# TinySegNet (MATCHES TRAIN)
# --------------------------
class SE(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x*s

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
    def __init__(self, cin, rates=(1,6,12,18), branch_ch=64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, branch_ch, 3 if r>1 else 1, padding=r if r>1 else 0,
                          dilation=r if r>1 else 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU()
            ) for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch*len(rates), 256, 1, bias=False),
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
        self.s1 = DWConvBlock(16,   chs[0], stride=1, use_se=False)  # 1/2
        self.s2 = DWConvBlock(chs[0], chs[1], stride=2, use_se=False) # 1/4
        self.s3 = DWConvBlock(chs[1], chs[2], stride=2, use_se=True)  # 1/8
        self.s4 = DWConvBlock(chs[2], chs[3], stride=2, use_se=True)  # 1/16

        self.aspp = ASPPLite(chs[3], rates=(1,6,12,18), branch_ch=64)  # -> 256

        # Projections (MATCH TRAIN)
        self.ctx_to_64 = nn.Conv2d(256, 64, 1, bias=False)
        self.mid_proj  = nn.Conv2d(chs[1], 64, 1, bias=False)
        self.mid_to_32 = nn.Conv2d(64, 32, 1, bias=False)
        self.low_proj  = nn.Conv2d(chs[0], 32, 1, bias=False)

        self.dec_mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.dec_low = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        self.dropout_head = nn.Dropout(0.5)
        self.cls = nn.Conv2d(64, num_classes, 1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        _, _, H, W = x.shape
        x = self.stem(x)
        low  = self.s1(x)        # 1/2
        mid  = self.s2(low)      # 1/4
        h8   = self.s3(mid)      # 1/8
        high = self.s4(h8)       # 1/16

        ctx   = self.aspp(high)
        up_m  = F.interpolate(ctx, size=mid.shape[-2:], mode="bilinear", align_corners=False)  # to 1/4
        up_m  = self.ctx_to_64(up_m)                                                           # 256->64
        fuse_m = self.dec_mid(up_m + self.mid_proj(mid))                                       # 64@1/4

        up_l  = F.interpolate(fuse_m, size=low.shape[-2:], mode="bilinear", align_corners=False)  # to 1/2
        up_l  = self.mid_to_32(up_l)                                                               # 64->32
        fuse_l = self.dec_low(up_l + self.low_proj(low))                                           # 64@1/2

        logits_4x = self.cls(self.dropout_head(fuse_l))                                        # [N,21,1/2,1/2]
        logits = F.interpolate(logits_4x, size=(H, W), mode="bilinear", align_corners=False)
        return {"out": logits, "taps": {"low": low, "mid": mid, "high": high}}

# --------------------------
# Transforms (eval) — sizes multiple of 16
# --------------------------
def normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

def _nearest_multiple_of_16(n: int) -> int:
    return max(16, (n // 16) * 16)

def val_transform(img, mask, crop=320):
    crop = _nearest_multiple_of_16(crop)
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0/short
    new_size = (int(round(h*scale)), int(round(w*scale)))
    img  = TF.resize(img,  new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)
    img  = TF.center_crop(img,  [crop, crop])
    mask = TF.center_crop(mask, [crop, crop])
    img_t  = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t

class VOCSegPair(Dataset):
    def __init__(self, root: Path, crop_size: int = 320):
        self.ds = VOCSegmentation(root=str(root), year="2012", image_set="val", download=False)
        self.crop = crop_size
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        return val_transform(img, mask, crop=self.crop)

# --------------------------
# Utils
# --------------------------
def resolve_voc_root(data_root: Path) -> Path:
    if (data_root / "VOCdevkit" / "VOC2012").is_dir():
        return data_root
    if (data_root / "VOC2012").is_dir():
        return data_root.parent if data_root.name == "VOC2012" else data_root
    raise FileNotFoundError(f"VOCdevkit/VOC2012 not found under: {data_root}")

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
    iou = torch.where(union > 0, inter/union, torch.zeros_like(union))
    return float(iou.mean().item()), iou.detach().cpu().numpy()

def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    m = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    m.putpalette(voc_palette())
    return m

def safe_load_ckpt(path: str, map_location):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint not found or empty: {p}")
    try:
        return torch.load(p, map_location=map_location)  # PyTorch>=2.6: weights_only=True by default
    except Exception as e1:
        print(f"[!] Retry torch.load(weights_only=False) due to: {e1}")
        return torch.load(p, map_location=map_location, weights_only=False)

def load_state_forgiving(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model_sd = model.state_dict()
    compatible = {}
    skipped_mismatch = []
    skipped_missing = []
    for k, v in state.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                compatible[k] = v
            else:
                skipped_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            skipped_missing.append(k)

    print(f"[i] Loading {len(compatible)} / {len(model_sd)} tensors from checkpoint")
    if skipped_mismatch:
        print(f"[!] Skipped {len(skipped_mismatch)} keys with shape mismatch (e.g., {skipped_mismatch[0]})")
    if skipped_missing:
        print(f"[!] Skipped {len(skipped_missing)} unexpected keys (e.g., {skipped_missing[0]})")

    model.load_state_dict(compatible, strict=False)

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="Folder that contains VOCdevkit (so VOCdevkit/VOC2012 exists).")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to tinyseg_best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-dir", type=str, default="", help="If set, saves colorized PNGs here.")
    ap.add_argument("--save-max", type=int, default=50, help="Max samples to save if save-dir set.")
    args = ap.parse_args()

    device = torch.device(args.device)
    voc_root = resolve_voc_root(Path(args.data_root))
    print(f"[i] Using VOC root: {voc_root}")
    print(f"[i] Device: {device}")

    # dataset / loader
    val_set = VOCSegPair(voc_root, crop_size=args.crop_size)
    pin = device.type == "cuda"
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # model + checkpoint
    model = TinySegNet(num_classes=NUM_CLASSES).to(device)
    ckpt = safe_load_ckpt(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)  # supports {'model': state_dict} or raw state_dict
    load_state_forgiving(model, state)

    # evaluate
    miou, per_cls = eval_miou(model, val_loader, device)
    print(f"\n==== EVAL ====")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU (len=21):", [round(float(x), 4) for x in per_cls])

    # optional PNG dumps
    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        model.eval()
        saved = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                pred = model(imgs)["out"].argmax(1).detach().cpu().numpy()
                for i in range(pred.shape[0]):
                    colorize_mask(pred[i]).save(outdir / f"val_{saved:05d}.png")
                    saved += 1
                    if saved >= args.save_max:
                        break
                if saved >= args.save_max:
                    break
        print(f"[i] Saved {saved} PNGs to {outdir}")

if __name__ == "__main__":
    main()
