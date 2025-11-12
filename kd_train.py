#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation training script:

Teacher: pretrained FCN-ResNet50 (21 classes, VOC label mapping)
Student: custom TinyVOCSeg from customModel.py

Loss:
  L_total = alpha * CE(student, y) +
            beta  * response_KD(student_logits, teacher_logits, T) +
            gamma * feature_KD(student_feat, teacher_feat)

Feature KD uses cosine loss on a high-level feature map.

Run example (Colab):
  !python kd_train.py \
      --data-root /content/data \
      --epochs 40 \
      --batch-size 8 \
      --alpha 1.0 \
      --beta  1.0 \
      --gamma 0.1 \
      --temperature 4.0 \
      --device cuda
"""

import argparse
from pathlib import Path
from typing import Tuple

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
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from tqdm import tqdm

from distillation_losses import (
    response_distillation_loss,
    feature_distillation_cosine_loss,
    IGNORE_INDEX,
)

# ---- import your student model (TinyVOCSeg) ----
from customModel import TinyVOCSeg  # <--- matches your tiny_segnet.py

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 21


# --------------------------
# Dataset / transforms
# --------------------------
def resolve_voc_root(data_root: Path) -> Path:
    if (data_root / "VOCdevkit" / "VOC2012").is_dir():
        return data_root
    if (data_root / "VOC2012").is_dir():
        return data_root.parent if data_root.name == "VOC2012" else data_root
    raise FileNotFoundError(f"VOCdevkit/VOC2012 not found under: {data_root}")


def normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def train_transform(img, mask, crop=320):
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0 / short
    new_size = (int(round(h * scale)), int(round(w * scale)))

    img  = TF.resize(img, new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

    i = torch.randint(0, new_size[0] - crop + 1, ()).item()
    j = torch.randint(0, new_size[1] - crop + 1, ()).item()
    img  = TF.crop(img, i, j, crop, crop)
    mask = TF.crop(mask, i, j, crop, crop)

    if torch.rand(()) < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    img_t  = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t


def val_transform(img, mask, crop=320):
    h, w = img.height, img.width
    short = min(h, w)
    scale = 360.0 / short
    new_size = (int(round(h * scale)), int(round(w * scale)))

    img  = TF.resize(img, new_size, InterpolationMode.BILINEAR)
    mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

    img  = TF.center_crop(img, [crop, crop])
    mask = TF.center_crop(mask, [crop, crop])

    img_t  = normalize(TF.to_tensor(img))
    mask_t = TF.pil_to_tensor(mask).squeeze(0).long()
    return img_t, mask_t


class VOCSegPair(Dataset):
    def __init__(self, root: Path, image_set: str, transform_kind: str = "train", crop_size: int = 320):
        self.ds = VOCSegmentation(root=str(root), year="2012", image_set=image_set, download=False)
        self.kind = transform_kind
        self.crop = crop_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        if self.kind == "train":
            return train_transform(img, mask, crop=self.crop)
        else:
            return val_transform(img, mask, crop=self.crop)


# --------------------------
# Teacher wrapper for feature taps
# --------------------------
class FCNTeacherWithFeat(nn.Module):
    """
    Wrap torchvision fcn_resnet50 to get both output logits and an intermediate feature map.

    We hook backbone.layer3 (ResNet stage 3) as teacher feature.
    """
    def __init__(self):
        super().__init__()
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = fcn_resnet50(weights=weights)
        self._feat = None

        # hook on layer3
        self.model.backbone.layer3.register_forward_hook(self._hook_layer3)

        # freeze teacher parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def _hook_layer3(self, module, inp, out):
        self._feat = out  # [B, C_t, H_t, W_t]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)["out"]   # [B,21,H,W]
        feat = self._feat            # [B,1024,H_t,W_t] for ResNet-50
        return out, feat


# --------------------------
# Feature alignment module
# --------------------------
class FeatureAlign(nn.Module):
    """
    Projects student + teacher feature maps to shared [B, C_common, H_common, W_common]
    using bilinear resize and 1x1 convs. Trainable, learned with the student.
    """
    def __init__(self, c_student: int, c_teacher: int, c_common: int = 256):
        super().__init__()
        self.proj_s = nn.Conv2d(c_student, c_common, 1, bias=False)
        self.proj_t = nn.Conv2d(c_teacher, c_common, 1, bias=False)
        self.bn_s = nn.BatchNorm2d(c_common)
        self.bn_t = nn.BatchNorm2d(c_common)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor):
        # resize student to teacher's spatial size
        H_t, W_t = feat_t.shape[-2:]
        feat_s_resized = F.interpolate(
            feat_s, size=(H_t, W_t), mode="bilinear", align_corners=False
        )

        s = self.bn_s(self.proj_s(feat_s_resized))
        t = self.bn_t(self.proj_t(feat_t))
        return s, t


# --------------------------
# mIoU metric (for logging)
# --------------------------
@torch.no_grad()
def eval_miou(student: nn.Module, loader: DataLoader, device: torch.device) -> float:
    student.eval()
    num_classes = NUM_CLASSES
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        pred = student(imgs)["out"].argmax(1)
        valid = masks != IGNORE_INDEX
        for c in range(num_classes):
            pc = (pred == c) & valid
            tc = (masks == c) & valid
            inter[c] += (pc & tc).sum()
            union[c] += (pc | tc).sum()

    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return float(iou.mean().item())


# --------------------------
# Train loop
# --------------------------
def poly_lr_lambda(e: int, total_epochs: int, power: float = 0.9):
    return (1.0 - e / float(total_epochs)) ** power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # KD hyperparams
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for CE w.r.t. ground truth")
    ap.add_argument("--beta",  type=float, default=1.0, help="weight for response-based KD")
    ap.add_argument("--gamma", type=float, default=0.1, help="weight for feature-based KD")
    ap.add_argument("--temperature", type=float, default=4.0, help="temperature for KD")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    voc_root = resolve_voc_root(Path(args.data_root))
    print(f"[i] Using VOC root: {voc_root}")

    device = torch.device(args.device)
    pin = (device.type == "cuda")

    train_set = VOCSegPair(voc_root, image_set="train", transform_kind="train", crop_size=args.crop_size)
    val_set   = VOCSegPair(voc_root, image_set="val",   transform_kind="val",   crop_size=args.crop_size)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True
    )
    val_loader   = DataLoader(
        val_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin
    )

    # ---- Student model (TinyVOCSeg) ----
    student = TinyVOCSeg(n_classes=NUM_CLASSES).to(device)
    print(f"[i] Student params: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e6:.3f}M")

    # ---- Teacher model (frozen) ----
    teacher = FCNTeacherWithFeat().to(device)
    teacher.eval()

    # ---- Feature align (student 'high' tap: 192 ch, teacher layer3: 1024 ch) ----
    feat_align = FeatureAlign(c_student=192, c_teacher=1024, c_common=256).to(device)

    params = list(student.parameters()) + list(feat_align.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: poly_lr_lambda(e, args.epochs, power=0.9))

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        feat_align.train()
        teacher.eval()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 1. Student forward
            s_out = student(imgs)
            s_logits = s_out["out"]          # [B,21,H,W]
            s_feat   = s_out["taps"]["high"] # [B,192,h_s,w_s]

            # 2. Teacher forward (no grad)
            with torch.no_grad():
                t_logits, t_feat = teacher(imgs)  # t_logits: [B,21,H,W], t_feat: [B,1024,h_t,w_t]

            # 3a. Supervised CE loss
            loss_ce = ce_loss_fn(s_logits, masks)

            # 3b. Response-based KD loss
            loss_kd_resp = response_distillation_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                targets=masks,
                T=args.temperature,
                ignore_index=IGNORE_INDEX,
            )

            # 3c. Feature-based KD loss (cosine on aligned features)
            s_feat_aligned, t_feat_aligned = feat_align(s_feat, t_feat)
            loss_kd_feat = feature_distillation_cosine_loss(s_feat_aligned, t_feat_aligned)

            # 3d. Total loss
            loss = args.alpha * loss_ce + args.beta * loss_kd_resp + args.gamma * loss_kd_feat
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)
            pbar.set_postfix({
                "CE": f"{loss_ce.item():.3f}",
                "KD_resp": f"{loss_kd_resp.item():.3f}",
                "KD_feat": f"{loss_kd_feat.item():.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validate student mIoU
        miou = eval_miou(student, val_loader, device)
        print(f"Epoch {epoch:03d}/{args.epochs} | Train total: {train_loss:.4f} | Val mIoU: {miou:.4f} | LR: {optimizer.param_groups[0]['lr']:.3e}")

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {"model": student.state_dict(), "epoch": epoch, "miou": best_miou},
                "tinyseg_kd_best.pt",
            )
            print(f"[i] New best KD mIoU {best_miou:.4f}. Saved -> tinyseg_kd_best.pt")

    print(f"[✓] KD training done. Best student mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
