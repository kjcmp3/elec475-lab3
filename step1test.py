#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test: load torchvision's pretrained FCN-ResNet50 (21 classes),
report mIoU on VOC 2012 val, and optionally save prediction images.

Examples:
  pip install torch torchvision kagglehub tqdm
  python step1test.py --data-root /content/data --batch-size 2 --max-images 20 \
      --save-dir preds_fcn --save-max 30
"""

import argparse
import os
import sys
import platform
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from PIL import Image


# -------------------- VOC helpers --------------------
def voc_palette() -> List[int]:
    """Standard 21-class VOC palette as a flat list of length 256*3."""
    palette = [0] * (256 * 3)
    colors = [
        (0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128),
        (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0),
        (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
        (192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0),
        (0,64,128)
    ]
    for i, (r, g, b) in enumerate(colors):
        palette[i * 3 + 0] = r
        palette[i * 3 + 1] = g
        palette[i * 3 + 2] = b
    return palette

def colorize_mask(mask_hw: torch.Tensor) -> Image.Image:
    """mask_hw: [H,W] uint8/long -> palettized PIL (VOC colors)."""
    m = Image.fromarray(mask_hw.to(torch.uint8).cpu().numpy(), mode="P")
    m.putpalette(voc_palette())
    return m

def try_download_voc_with_kagglehub() -> Optional[Path]:
    """Try to get Kaggle dataset via kagglehub; return path that contains VOCdevkit/VOC2012."""
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None
    try:
        root = Path(kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset"))
        for c in [root, *root.rglob("*")]:
            if (c / "VOCdevkit" / "VOC2012").is_dir():
                return c / "VOCdevkit"
        return None
    except Exception:
        return None


# -------------------- Transforms / Loader --------------------
class ResizePair:
    """Resize image and segmentation mask together to (size, size)."""
    def __init__(self, size: int):
        self.size = size
        self.img_resize = transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.mask_resize = transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __call__(self, img, mask):
        return self.img_resize(img), self.mask_resize(mask)

def build_dataloader(voc_root: str | Path, batch_size: int,
                     max_images: Optional[int],
                     device: torch.device):
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    meta = getattr(weights, "meta", {}) or {}
    mean = meta.get("mean", (0.485, 0.456, 0.406))
    std  = meta.get("std",  (0.229, 0.224, 0.225))

    resize_pair = ResizePair(520)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=mean, std=std)

    def transform(img, target):
        img, target = resize_pair(img, target)
        img = norm(to_tensor(img))
        target = pil_to_tensor(target).squeeze(0).long()  # [H,W], {0..20,255}
        return img, target

    dataset = VOCSegmentation(
        root=str(voc_root),
        year="2012",
        image_set="val",
        download=False,
        transforms=transform,
    )

    if max_images is not None and max_images >= 0 and max_images < len(dataset):
        class _Slice(torch.utils.data.Dataset):
            def __init__(self, base, n): self.base, self.n = base, n
            def __len__(self): return self.n
            def __getitem__(self, idx): return self.base[idx]
        dataset = _Slice(dataset, max_images)

    workers = 0 if platform.system().lower().startswith("win") else 2
    pin = (device.type == "cuda")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin
    )
    return loader, mean, std


# -------------------- Eval / Saving --------------------
def denorm_to_pil(img_chw: torch.Tensor, mean, std) -> Image.Image:
    """img_chw: [3,H,W] tensor -> PIL.Image (clipped to [0,1])."""
    mean_t = torch.tensor(mean, dtype=img_chw.dtype, device=img_chw.device).view(3,1,1)
    std_t  = torch.tensor(std,  dtype=img_chw.dtype, device=img_chw.device).view(3,1,1)
    x = img_chw * std_t + mean_t
    x = x.clamp(0, 1).mul(255).round().byte().cpu()
    return Image.fromarray(x.permute(1,2,0).numpy(), mode="RGB")

def stack_horiz(imgs: List[Image.Image]) -> Image.Image:
    """Simple horizontal concatenation."""
    h = max(im.height for im in imgs)
    resized = [im if im.height == h else im.resize((int(im.width * h / im.height), h), Image.BILINEAR) for im in imgs]
    w_total = sum(im.width for im in resized)
    out = Image.new("RGB", (w_total, h))
    x = 0
    for im in resized:
        out.paste(im.convert("RGB"), (x, 0))
        x += im.width
    return out

@torch.no_grad()
def evaluate_miou_and_save(model,
                           loader,
                           device,
                           mean, std,
                           save_dir: Optional[Path],
                           save_max: int) -> Tuple[float, List[float]]:
    """Compute mIoU and optionally save up to save_max visualizations."""
    num_classes = 21
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)

    to_save = 0
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for images, targets in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # [B,H,W]

        logits = model(images)["out"]      # [B,21,H,W]
        preds = logits.argmax(1)           # [B,H,W]

        valid = (targets != 255)
        for c in range(num_classes):
            pc = (preds == c) & valid
            tc = (targets == c) & valid
            inter[c] += pc.logical_and(tc).sum()
            union[c] += pc.logical_or(tc).sum()

        # --- Save visuals ---
        if save_dir and to_save < save_max:
            b = images.size(0)
            for i in range(b):
                if to_save >= save_max:
                    break
                img_pil = denorm_to_pil(images[i].cpu(), mean, std)
                pred_pal = colorize_mask(preds[i])
                targ_pal = colorize_mask(targets[i].clamp_min(0))  # ignore 255 colors out as black
                # Overlay: blend original with colored prediction
                pred_rgb = pred_pal.convert("RGB")
                overlay = Image.blend(img_pil, pred_rgb, alpha=0.5)
                grid = stack_horiz([img_pil, pred_pal, targ_pal, overlay])
                grid.save(save_dir / f"val_{to_save:05d}.png")
                to_save += 1

    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    miou = float(iou.mean().item())
    return miou, [float(x.item()) for x in iou]


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="",
                        help="Folder that contains VOCdevkit (so VOCdevkit/VOC2012 exists).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=50,
                        help="Limit val samples (set <0 for all).")
    parser.add_argument("--save-dir", type=str, default="",
                        help="If set, saves predictions (input | pred | target | overlay).")
    parser.add_argument("--save-max", type=int, default=50,
                        help="Max images to save to --save-dir.")
    args = parser.parse_args()

    # Locate or download VOC
    if args.data_root:
        dr = Path(args.data_root)
        if (dr / "VOCdevkit" / "VOC2012").is_dir():
            voc_root = dr
        elif (dr / "VOC2012").is_dir():
            voc_root = dr.parent if dr.name == "VOC2012" else dr
        else:
            print(f"[!] --data-root provided but VOCdevkit/VOC2012 not found under: {dr}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[i] Trying KaggleHub download of gopalbhattrai/pascal-voc-2012-dataset ...")
        voc_root = try_download_voc_with_kagglehub()
        if voc_root is None:
            print("[!] Could not auto-download. Please download/extract manually and pass --data-root /path/to/VOCdevkit",
                  file=sys.stderr)
            sys.exit(1)

    print(f"[i] Using VOC root: {voc_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    # Pretrained FCN-ResNet50 (21 classes, VOC label mapping)
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights).to(device)

    # Data
    max_images = None if (args.max_images is not None and args.max_images < 0) else args.max_images
    loader, mean, std = build_dataloader(voc_root, args.batch_size, max_images, device)

    # Evaluate + save
    save_dir = Path(args.save_dir) if args.save_dir else None
    miou, per_class_iou = evaluate_miou_and_save(
        model, loader, device, mean, std, save_dir, args.save_max
    )

    print("\n==== Results ====")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU (len={}):".format(len(per_class_iou)))
    print([round(x, 4) for x in per_class_iou])
    if save_dir:
        print(f"[i] Saved up to {args.save_max} visuals to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
