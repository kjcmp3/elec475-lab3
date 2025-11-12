#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate KD-trained TinyVOCSeg on VOC2012 val set.

- Uses the TinyVOCSeg student model from customModel.py
- Computes mIoU over 21 classes
- Saves colorized masks + overlay images + triptychs

Example (Colab):
!python kd_test.py \
  --data-root /content/data \
  --ckpt tinyseg_kd_best.pt \
  --batch-size 8 \
  --crop-size 320 \
  --device cuda \
  --save-dir preds_val_kd \
  --save-max 30
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image

# import your KD-trained student model
from customModel import TinyVOCSeg   # <-- make sure this matches your file/class name

# -----------------------
# Constants
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 21
IGNORE_INDEX  = 255


def voc_palette():
    """Pascal VOC color palette (21 classes)."""
    palette = [0] * (256 * 3)
    colors = [
        (0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),
        (128,0,128),(0,128,128),(128,128,128),(64,0,0),(192,0,0),
        (64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),
        (192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),
        (0,64,128)
    ]
    for i, (r, g, b) in enumerate(colors):
        palette[i*3+0] = r
        palette[i*3+1] = g
        palette[i*3+2] = b
    return palette


# -----------------------
# Dataset + transforms
# -----------------------
def normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

def val_transform(img: Image.Image, mask: Image.Image, crop: int = 320):
    """
    Resize shorter side to ~360, then center-crop `crop` (multiple of 16 recommended).
    Returns tensor image, tensor mask, and PIL image (for overlays).
    """
    h, w = img.height, img.width
    scale = 360.0 / min(h, w)
    new_size = (int(round(h * scale)), int(round(w * scale)))
    img  = TF.resize(img,  new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

    img  = TF.center_crop(img,  [crop, crop])
    mask = TF.center_crop(mask, [crop, crop])

    img_t  = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t, img  # keep PIL for overlay

class VOCSegPair(Dataset):
    def __init__(self, root: Path, crop_size: int = 320):
        self.ds = VOCSegmentation(root=str(root), year="2012",
                                  image_set="val", download=False)
        self.crop = crop_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        x, y, pil = val_transform(img, mask, crop=self.crop)
        return {"img": x, "mask": y, "pil": pil}


# -----------------------
# Helpers
# -----------------------
def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    m = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    m.putpalette(voc_palette())
    return m

def overlay_pred_on_image(pil_img: Image.Image,
                          pred_np: np.ndarray,
                          alpha: float = 0.5) -> Image.Image:
    color = colorize_mask(pred_np).convert("RGBA")
    base  = pil_img.convert("RGBA")
    return Image.blend(base, color, alpha).convert("RGB")

def concat_triptych(img: Image.Image,
                    mask_np: np.ndarray,
                    overlay: Image.Image) -> Image.Image:
    """Return a single wide image: [original | mask | overlay]."""
    mask_img = colorize_mask(mask_np).convert("RGB")
    w, h = img.size
    out = Image.new("RGB", (w * 3, h))
    out.paste(img, (0, 0))
    out.paste(mask_img, (w, 0))
    out.paste(overlay, (2 * w, 0))
    return out

def load_state_forgiving(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """
    Load only keys whose shapes match; report mismatches/unexpected.
    Helpful if you tweak the model slightly between runs.
    """
    model_sd = model.state_dict()
    ok = {}
    skipped_mismatch = []
    skipped_unexp = []

    for k, v in state.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                ok[k] = v
            else:
                skipped_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            skipped_unexp.append(k)

    print(f"[i] Loading {len(ok)} / {len(model_sd)} tensors from checkpoint")
    if skipped_mismatch:
        ex = skipped_mismatch[0]
        print(f"[!] Skipped {len(skipped_mismatch)} keys with shape mismatch "
              f"(e.g., {ex[0]}: ckpt {ex[1]} vs model {ex[2]})")
    if skipped_unexp:
        print(f"[!] Skipped {len(skipped_unexp)} unexpected keys (e.g., {skipped_unexp[0]})")

    model.load_state_dict(ok, strict=False)

@torch.no_grad()
def eval_miou(model: nn.Module,
              loader: DataLoader,
              device: torch.device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    for batch in loader:
        imgs  = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        pred  = model(imgs)["out"].argmax(1)
        valid = masks != IGNORE_INDEX

        for c in range(NUM_CLASSES):
            pc = (pred == c) & valid
            tc = (masks == c) & valid
            inter[c] += (pc & tc).sum()
            union[c] += (pc | tc).sum()

    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return float(iou.mean().item()), iou.detach().cpu().numpy()


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True,
                    help="KD checkpoint, e.g., tinyseg_kd_best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-dir", type=str, default="preds_val_kd")
    ap.add_argument("--save-max", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Overlay alpha for blending mask + image")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    pin = (device.type == "cuda")

    val_set = VOCSegPair(Path(args.data_root), crop_size=args.crop_size)

    def collate_keep_pil(batch: List[Dict]):
        """Stack tensors; keep PILs as list to avoid default_collate errors."""
        imgs  = torch.stack([b["img"] for b in batch])
        masks = torch.stack([b["mask"] for b in batch])
        pils  = [b["pil"] for b in batch]
        return {"img": imgs, "mask": masks, "pil": pils}

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_keep_pil,
    )

    # Instantiate student model and load KD checkpoint
    model = TinyVOCSeg(n_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    load_state_forgiving(model, state)

    print(f"[i] Using VOC root: {args.data_root}")
    print(f"[i] Device: {device}")

    # quick sanity: class histogram on first batch
    with torch.no_grad():
        first = next(iter(val_loader))
        logits = model(first["img"].to(device))["out"]
        preds0 = logits.argmax(1).cpu()
        hist = torch.bincount(preds0.flatten(), minlength=NUM_CLASSES).numpy()
        frac_bg = float(hist[0] / max(1, hist.sum()))
        print(f"[i] Pred class histogram (batch 0): {hist.tolist()}")
        print(f"[i] Fraction background (class 0): {frac_bg:.3f}")

    # mIoU eval
    miou, per_cls = eval_miou(model, val_loader, device)
    print("\n==== EVAL ====")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU (len=21):", [round(float(x), 4) for x in per_cls])

    # save predictions
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["img"].to(device, non_blocking=True)
            pil_imgs = batch["pil"]
            pred = model(imgs)["out"].argmax(1).cpu().numpy()

            for i in range(pred.shape[0]):
                mask_img = colorize_mask(pred[i])
                overlay  = overlay_pred_on_image(pil_imgs[i], pred[i], alpha=args.alpha)
                trip     = concat_triptych(pil_imgs[i], pred[i], overlay)

                mask_img.save(outdir / f"mask_{saved:04d}.png")
                overlay.save(outdir  / f"overlay_{saved:04d}.jpg",   quality=95)
                trip.save(outdir     / f"triptych_{saved:04d}.jpg",  quality=95)

                saved += 1
                if saved >= args.save_max:
                    break
            if saved >= args.save_max:
                break

    print(f"[i] Saved {saved} results -> {outdir} (mask_*, overlay_*, triptych_*)")


if __name__ == "__main__":
    main()
