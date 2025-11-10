#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_eval_fcn_resnet50.py

Evaluate torchvision FCN-ResNet50 (21-class VOC) on PASCAL VOC 2012 val split.

Run examples:

# Local (PyCharm terminal):
# python src/step1_eval_fcn_resnet50.py --data-root ./data/VOC --batch-size 8 --num-workers 4

# Colab (with optional torchmetrics install):
# !pip install torchmetrics
# !python src/step1_eval_fcn_resnet50.py --data-root /content/data/VOC --batch-size 16 --num-workers 2
"""

import argparse
import time
import random
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor

# Try torchmetrics for robust mIoU; fall back to custom IoU if unavailable.
_HAS_TORCHMETRICS = True
try:
    from torchmetrics.classification import JaccardIndex
except Exception:
    _HAS_TORCHMETRICS = False

# 21 PASCAL VOC classes (including background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IGNORE_INDEX = 255
NUM_CLASSES = 21


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VOCEvalWrapper(torch.utils.data.Dataset):
    """
    Wrap VOCSegmentation to:
      - Apply official preprocessing to images (weights.transforms()).
      - Convert masks to torch.long and resize with nearest to match image size.
      - Preserve IGNORE_INDEX=255 exactly (no normalization/augment on masks).
    """
    def __init__(self, root: str, weights: FCN_ResNet50_Weights, year: str = "2012", image_set: str = "val", download: bool = True):
        self.voc = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.preprocess = weights.transforms()  # official preprocessing for FCN-ResNet50 (COCO_WITH_VOC_LABELS)
        self.ignore_index = IGNORE_INDEX

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.voc[idx]  # PIL.Image RGB, PIL.Image 'P' (palette) with labels [0..20] and 255 ignore
        img_t = self.preprocess(img)  # CxHxW float tensor

        # Convert mask to tensor (no normalization); ensure dtype long and values preserved.
        # pil_to_tensor returns uint8 tensor [1,H,W]; squeeze channel.
        mask = pil_to_tensor(target).squeeze(0).to(torch.int64)  # [H,W], values in {0..20,255}

        # Resize mask to image tensor spatial size using nearest neighbor to keep labels/255 exact.
        h, w = img_t.shape[-2:]
        if mask.shape[-2:] != (h, w):
            # add N,C,H,W -> (1,1,H,W) for interpolate
            mask = mask.unsqueeze(0).unsqueeze(0).to(torch.float32)
            mask = F.interpolate(mask, size=(h, w), mode="nearest").squeeze(0).squeeze(0).to(torch.int64)

        return img_t, mask


def build_dataloader(data_root: str, batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, FCN_ResNet50_Weights]:
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1  # 21-class head aligned to VOC labels
    dataset = VOCEvalWrapper(root=data_root, weights=weights, year="2012", image_set="val", download=True)

    def _collate(batch):
        # Default collate works (tensors same size), but keep custom to be explicit.
        imgs = torch.stack([b[0] for b in batch], dim=0)
        masks = torch.stack([b[1] for b in batch], dim=0)
        return imgs, masks

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )
    return loader, weights


def build_model(weights: FCN_ResNet50_Weights, device: torch.device) -> torch.nn.Module:
    model = fcn_resnet50(weights=weights, progress=True)
    model.eval()
    model.to(device)
    return model


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    pred: [N,H,W] int64 in [0..C-1]
    target: [N,H,W] int64 in [0..C-1] or ignore_index
    returns: [C,C] where rows are GT, cols are Pred.
    """
    with torch.no_grad():
        pred = pred.view(-1).to(torch.int64)
        target = target.view(-1).to(torch.int64)
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
        k = (target * num_classes + pred)
        hist = torch.bincount(k, minlength=num_classes * num_classes)
        hist = hist.reshape(num_classes, num_classes)
    return hist


def iou_from_confmat(confmat: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    confmat: [C,C], rows GT, cols Pred
    returns:
      per_class_iou: [C] with NaN where undefined
      miou: mean over classes where denominator > 0
    """
    with torch.no_grad():
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        denom = tp + fp + fn
        per_class_iou = tp.to(torch.float64) / denom.clamp_min(1).to(torch.float64)
        # mark classes with denom==0 as NaN so they don't count toward mean
        per_class_iou = torch.where(denom == 0, torch.tensor(float('nan'), dtype=per_class_iou.dtype, device=per_class_iou.device), per_class_iou)
        valid = ~torch.isnan(per_class_iou)
        miou = per_class_iou[valid].mean().item() if valid.any() else float('nan')
    return per_class_iou, miou


def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device,
             amp: bool = False,
             use_torchmetrics: bool = _HAS_TORCHMETRICS,
             want_per_class: bool = True):
    model.eval()
    total_images = 0
    start_time = time.perf_counter()
    infer_time_sum = 0.0

    if use_torchmetrics:
        # macro-average mIoU ignoring 255
        miou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average="macro")
        miou_metric = miou_metric.to(device)
        # per-class IoU by setting average=None (returns [C]), still honoring ignore_index
        perclass_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average=None).to(device) if want_per_class else None
    else:
        confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    # Accumulate which classes appear (in GT) for summary
    class_presence = torch.zeros(NUM_CLASSES, dtype=torch.int64, device=device)

    autocast_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            total_images += imgs.size(0)

            tic = time.perf_counter()
            with autocast_ctx():
                out = model(imgs)["out"]  # [N,C,H,W] logits
            infer_time_sum += (time.perf_counter() - tic)

            # Predictions
            preds = out.argmax(1)  # [N,H,W] int64 later

            # Update class presence from GT (ignore 255)
            gt_valid = masks != IGNORE_INDEX
            if gt_valid.any():
                present = torch.bincount(masks[gt_valid].view(-1), minlength=NUM_CLASSES)
                class_presence += present.to(class_presence.dtype)

            if use_torchmetrics:
                # torchmetrics expects same device and dtype long for preds/target
                miou_metric.update(preds, masks)
                if want_per_class:
                    perclass_metric.update(preds, masks)
            else:
                confmat += compute_confusion_matrix(preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    elapsed = time.perf_counter() - start_time
    throughput = total_images / infer_time_sum if infer_time_sum > 0 else float('nan')

    if use_torchmetrics:
        miou = miou_metric.compute().item()
        per_class = perclass_metric.compute().detach().cpu().tolist() if want_per_class else None
    else:
        per_class_t, miou = iou_from_confmat(confmat)
        per_class = per_class_t.detach().cpu().tolist() if want_per_class else None

    # Effective classes (present in GT at least once)
    effective_classes = int((class_presence > 0).sum().item())

    stats = {
        "miou": miou,
        "per_class": per_class,
        "throughput": throughput,
        "total_images": total_images,
        "effective_classes": effective_classes,
        "elapsed_sec": elapsed
    }
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FCN-ResNet50 (VOC 2012 val) with torchvision weights.")
    p.add_argument("--data-root", type=str, required=True, help="Root dir for VOC dataset (VOCdevkit will be created here).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", help="Enable pinned memory for DataLoader.")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable pinned memory.")
    p.set_defaults(pin_memory=True)
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-per-class", action="store_true", help="Do not print per-class IoU table.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data & model
    loader, weights = build_dataloader(args.data_root, args.batch_size, args.num_workers, args.pin_memory)
    model = build_model(weights, device)

    # Summary
    print("=" * 80)
    print("FCN-ResNet50 VOC 2012 Evaluation")
    print(f"Device: {device.type.upper()} (AMP={'ON' if args.amp and device.type=='cuda' else 'OFF'})")
    print(f"Dataset (val) size: {len(loader.dataset)} images")
    print(f"Batch size: {args.batch_size} | Num workers: {args.num_workers} | Pin memory: {args.pin_memory}")
    print(f"Classes: {NUM_CLASSES} (ignore_index={IGNORE_INDEX} preserved)")
    print("=" * 80)

    # Evaluate
    stats = evaluate(model, loader, device, amp=args.amp, use_torchmetrics=_HAS_TORCHMETRICS, want_per_class=(not args.no_per_class))

    # Pretty print results
    miou = stats["miou"]
    per_class = stats["per_class"]
    throughput = stats["throughput"]
    total_images = stats["total_images"]
    effective_classes = stats["effective_classes"]

    print("\nFinal Results")
    print("-" * 80)
    print(f"Images evaluated: {total_images}")
    print(f"Effective classes present in GT: {effective_classes}/{NUM_CLASSES}")
    print(f"mIoU: {miou:.4f}")
    if per_class is not None:
        print("Per-class IoU:")
        row = []
        for cname, val in zip(VOC_CLASSES, per_class):
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                row.append(f"{cname}: n/a")
            else:
                row.append(f"{cname}: {val:.4f}")
        # Print as a single comma-separated line, as requested
        print(", ".join(row))
    print(f"Throughput: {throughput:.2f} images/sec")
    print("-" * 80)
    print("\nmIoU: {:.2f}\nPer-class IoU:{}".format(miou, ("\n" + ", ".join([f"{n}: {0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v:.2f}" for n, v in zip(VOC_CLASSES, per_class)])) if per_class is not None else " (skipped)"))
    print(f"Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    main()
