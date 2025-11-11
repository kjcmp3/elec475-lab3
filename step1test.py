#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test: load torchvision's pretrained FCN-ResNet50 (21 classes) and
report mIoU on VOC 2012 val. Tries to auto-download via kagglehub if no
--data-root is provided.

Run examples:
  pip install torch torchvision kagglehub tqdm
  py step1test.py --data-root "C:\\Users\\kjake\\Downloads\\VOCDataset" --batch-size 1 --max-images 20
"""

import argparse
import os
import sys
from pathlib import Path
import platform

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def try_download_voc_with_kagglehub() -> Path | None:
    """
    Try to fetch the Kaggle dataset via kagglehub, return a Path that contains
    VOCdevkit/VOC2012 if successful. Returns None on failure.
    """
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    try:
        root = Path(kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset"))
        candidates = [root] + list(root.rglob("*"))
        for c in candidates:
            if (c / "VOCdevkit" / "VOC2012").is_dir():
                return c / "VOCdevkit"
        return None
    except Exception:
        return None


class ResizePair:
    """Resize image and segmentation mask together."""
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


def build_dataloader(voc_root: str | Path, batch_size: int, max_images: int | None, device: torch.device):
    # Use robust mean/std (fallback to ImageNet defaults if not in meta)
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    meta = getattr(weights, "meta", {}) or {}
    mean = meta.get("mean", (0.485, 0.456, 0.406))
    std = meta.get("std", (0.229, 0.224, 0.225))

    resize_pair = ResizePair(520)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=mean, std=std)

    def transform(img, target):
        img, target = resize_pair(img, target)
        img = norm(to_tensor(img))
        target = pil_to_tensor(target).squeeze(0).long()  # [H,W], 0..20 or 255
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

    # CPU/Windows-friendly defaults
    workers = 0 if platform.system().lower().startswith("win") else 2
    pin = (device.type == "cuda")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin
    )
    return loader


@torch.no_grad()
def evaluate_miou(model, loader, device):
    """Compute mean IoU over 21 VOC classes (ignore label 255)."""
    num_classes = 21
    inter = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    model.eval()
    for images, targets in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device)
        targets = targets.to(device)  # [B,H,W], values in {0..20,255}

        logits = model(images)["out"]        # [B,21,H,W]
        preds = logits.argmax(1)             # [B,H,W]

        valid = (targets != 255)
        for c in range(num_classes):
            pred_c = (preds == c) & valid
            targ_c = (targets == c) & valid
            inter[c] += (pred_c & targ_c).sum().item()
            union[c] += (pred_c | targ_c).sum().item()

    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    miou = iou.mean().item()
    return miou, iou.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="",
                        help="Folder that contains VOCdevkit (so VOCdevkit/VOC2012 exists).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=50,
                        help="Limit val samples (set <0 for all).")
    args = parser.parse_args()

    # Locate or download VOC
    if args.data_root:
        dr = Path(args.data_root)
        # Accept either <root>/VOCdevkit/VOC2012 or <root> already == VOCdevkit
        if (dr / "VOCdevkit" / "VOC2012").is_dir():
            voc_root = dr
        elif (dr / "VOC2012").is_dir():
            # If you pass .../VOCdevkit, that's fine too.
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
    loader = build_dataloader(voc_root, args.batch_size, max_images, device)

    # Evaluate
    miou, per_class_iou = evaluate_miou(model, loader, device)
    print("\n==== Results ====")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU (len={}):".format(len(per_class_iou)))
    print([round(x, 4) for x in per_class_iou])


if __name__ == "__main__":
    main()
