#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval TinySegNet on VOC2012 val: mIoU + optional PNG dumps.
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
# TinySegNet (same as training)
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
        self.aspp = ASPPLite(chs[3], rates=(1,6,12,18), branch_ch=64) # -> 256

        self.mid_proj = nn.Conv2d(chs[1], 64, 1)
        self.low_proj = nn.Conv2d(chs[0], 32, 1)
        self.dec_mid  = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(256), nn.SiLU())
        self.dec_low  = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.SiLU())
        self.dropout_head = nn.Dropout(0.5)
        self.cls = nn.Conv2d(64, num_classes, 1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        _, _, H, W = x.shape
        x = self.stem(x)
        low  = self.s1(x)        # 1/2
        mid  = self.s2(low)      # 1/4
        h8   = self.s3(mid)      # 1/8
        high = self.s4(h8)       # 1/16
        ctx = self.aspp(high)    # [N,256,H/16,W/16]

        up_mid   = F.interpolate(ctx, scale_factor=2, mode="bilinear", align_corners=False)
        fuse_mid = self.dec_mid(up_mid + self.mid_proj(mid))         # [N,256,H/8,W/8]
        up_low   = F.interpolate(fuse_mid, scale_factor=2, mode="bilinear", align_corners=False)
        fuse_low = self.dec_low(up_low + self.low_proj(low))         # [N,64,H/4,W/4]
        logits_4x = self.cls(self.dropout_head(fuse_low))            # [N,21,H/4,]()_
