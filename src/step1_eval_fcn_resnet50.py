#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/step1_eval_fcn_resnet50.py

Evaluate torchvision FCN-ResNet50 (21-class VOC) on PASCAL VOC 2012 val split.

Run examples:

# Local (PyCharm terminal):
# python src/step1_eval_fcn_resnet50.py --data-root ./data/VOC --batch-size 8 --num-workers 4

# Colab (with optional torchmetrics + Kaggle auto-download):
# !pip install torchmetrics kaggle
# # Add Kaggle API credentials: upload kaggle.json or use:
# # import os, json; os.makedirs('/root/.kaggle', exist_ok=True); json.dump({"username":"<user>","key":"<key>"}, open('/root/.kaggle/kaggle.json','w')); os.chmod('/root/.kaggle/kaggle.json', 0o600)
# !python src/step1_eval_fcn_resnet50.py --data-root /content/data/VOC --auto-download-kaggle --batch-size 16 --num-workers 2
"""

import argparse
import time
import random
import math
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple

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

KAGGLE_DATASET = "gopalbhattrai/pascal-voc-2012-dataset"  # per user request


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _voc_expected_paths(root: Path) -> dict:
    vocdevkit = root / "VOCdevkit" / "VOC2012"
    return {
        "root": root,
        "voc2012": vocdevkit,
        "jpeg": vocdevkit / "JPEGImages",
        "segcls": vocdevkit / "SegmentationClass",
        "imgsets": vocdevkit / "ImageSets" / "Segmentation" / "val.txt",
    }


def has_voc2012_structure(root: Path) -> bool:
    p = _voc_expected_paths(root)
    return p["jpeg"].is_dir() and p["segcls"].is_dir() and p["imgsets"].is_file()


def _extract_archive(archive_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix in [".zip"]:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(out_dir)
    elif archive_path.suffix in [".gz", ".tgz"] or archive_path.name.endswith(".tar"):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(out_dir)
    else:
        print(f"[WARN] Unknown archive format: {archive_path.name} (skipping)")


def try_download_kaggle(root: Path) -> bool:
    """
    Attempt to download and unpack the Kaggle dataset into root so that
    torchvision.datasets.VOCSegmentation can find VOCdevkit/VOC2012.
    Requires `kaggle` CLI with credentials.
    """
    try:
        # ensure kaggle CLI exists
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print("[ERROR] Kaggle CLI not found. Install with `pip install kaggle` and place kaggle.json credentials in ~/.kaggle/ .")
        return False

    dl_dir = root / "_kaggle_tmp"
    dl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading Kaggle dataset `{KAGGLE_DATASET}` to {dl_dir} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dl_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] kaggle download failed: {e}")
        return False

    # Extract all archives we find
    for p in dl_dir.iterdir():
        if p.is_file() and any(p.name.endswith(ext) for ext in [".zip", ".tar", ".tar.gz", ".tgz"]):
            print(f"[INFO] Extracting {p.name} ...")
            _extract_archive(p, dl_dir)

    # Heuristics: find a folder that contains VOCdevkit/VOC2012 or can be moved to that structure
    # Many Kaggle mirrors ship 'VOCdevkit/VOC2012/...'. If we find such, move into root.
    candidate = None
    for sub in dl_dir.rglob("VOC2012"):
        if (sub / "JPEGImages").is_dir() and (sub / "SegmentationClass").is_dir():
            candidate = sub
            break

    if candidate is None:
        # As a fallback, try known top-level names
        for guess in ["VOCdevkit/VOC2012", "VOC2012"]:
            sub = dl_dir / guess
            if (sub / "JPEGImages").is_dir() and (sub / "SegmentationClass").is_dir():
                candidate = sub
                break

    if candidate is None:
        print("[ERROR] Could not find VOC2012 structure after extraction. Please inspect contents and place under <data-root>/VOCdevkit/VOC2012/.")
        return False

    # Ensure destination structure
    dest = root / "VOCdevkit" / "VOC2012"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Move/merge candidate into destination
    if dest.exists():
        print(f"[INFO] Destination {dest} exists. Merging contents...")
    for item in candidate.iterdir():
        target = dest / item.name
        if target.exists():
            # merge directories if needed
            if item.is_dir():
                for inner in item.iterdir():
                    shutil.move(str(inner), str(target))
            else:
                # overwrite file
                shutil.move(str(item), str(target))
        else:
            shutil.move(str(item), str(target))

    print(f"[INFO] VOC2012 is prepared under {dest}.")
    return has_voc2012_structure(root)


class VOCEvalWrapper(torch.utils.data.Dataset):
    """
    Wrap VOCSegmentation to:
      - Apply official preprocessing to images (weights.transforms()).
      - Convert masks to torch.long and resize with nearest to match image size.
      - Preserve IGNORE_INDEX=255 exactly (no normalization/augment on masks).
    """
    def __init__(self, root: str, weights: FCN_ResNet50_Weights, year: str = "2012", image_set: str = "val"):
        self.voc = VOCSegmentation(root=root, year=year, image_set=image_set, download=False)
        self.preprocess = weights.transforms()  # official preprocessing for FCN-ResNet50 (COCO_WITH_VOC_LABELS)
        self.ignore_index = IGNORE_INDEX

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.voc[idx]  # PIL.Image RGB, PIL.Image 'P' with labels [0..20] and 255 ignore
        img_t = self.preprocess(img)  # CxHxW float tensor

        # Convert mask to tensor (no normalization); ensure dtype long and values preserved.
        mask = pil_to_tensor(target).squeeze(0).to(torch.int64)  # [H,W], values in {0..20,255}

        # Resize mask to image tensor spatial size using nearest neighbor to keep labels/255 exact.
        h, w = img_t.shape[-2:]
        if mask.shape[-2:] != (h, w):
            mask = mask.unsqueeze(0).unsqueeze(0).to(torch.float32)
            mask = F.interpolate(mask, size=(h, w), mode="nearest").squeeze(0).squeeze(0).to(torch.int64)

        return img_t, mask


def build_dataloader(data_root: str, batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, FCN_ResNet50_Weights]:
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1  # 21-class head aligned to VOC labels

    dataset = VOCEvalWrapper(root=data_root, weights=weights, year="2012", image_set="val")

    def _collate(batch):
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
        miou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average="macro").to(device)
        perclass_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average=None).to(device) if want_per_class else None
    else:
        confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

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

            preds = out.argmax(1)

            gt_valid = masks != IGNORE_INDEX
            if gt_valid.any():
                present = torch.bincount(masks[gt_valid].view(-1), minlength=NUM_CLASSES)
                class_presence += present.to(class_presence.dtype)

            if use_torchmetrics:
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
    p.add_argument("--auto-download-kaggle", action="store_true",
                   help=f"Try to download {KAGGLE_DATASET} with Kaggle CLI if VOC not found under data-root.")
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

    # Ensure VOC data exists (Kaggle auto-download if requested)
    data_root = Path(args.data_root).expanduser().resolve()
    if not has_voc2012_structure(data_root):
        if args.auto-download-kaggle:
            ok = try_download_kaggle(data_root)
            if not ok:
                raise SystemExit("[FATAL] VOC2012 not found and Kaggle auto-download failed. "
                                 "Please place VOC under <data-root>/VOCdevkit/VOC2012/")
        else:
            raise SystemExit("[FATAL] VOC2012 structure not found. Either prepare "
                             "<data-root>/VOCdevkit/VOC2012/ or pass --auto-download-kaggle.")

    # Build data & model
    loader, weights = build_dataloader(str(data_root), args.batch_size, args.num_workers, args.pin_memory)
    model = build_model(weights, device)

    # Summary
    print("=" * 80)
    print("FCN-ResNet50 VOC 2012 Evaluation")
    print(f"Device: {device.type.upper()} (AMP={'ON' if args.amp and device.type=='cuda' else 'OFF'})")
    print(f"Dataset (val) size: {len(loader.dataset)} images")
    print(f"Batch size: {args.batch_size} | Num workers: {args.num_workers} | Pin memory: {args.pin_memory}")
    print(f"Classes: {NUM_CLASSES} (ignore_index={IGNORE_INDEX} preserved)")
    print(f"Using weights: {FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.name}")
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
        print(", ".join(row))
    print(f"Throughput: {throughput:.2f} images/sec")
    print("-" * 80)
    print("\nmIoU: {:.2f}\nPer-class IoU:{}".format(
        miou, ("\n" + ", ".join([f"{n}: {0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v:.2f}" for n, v in zip(VOC_CLASSES, per_class)]))
        if per_class is not None else " (skipped)")
    )
    print(f"Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    main()
