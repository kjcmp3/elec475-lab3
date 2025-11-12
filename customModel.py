# tiny_segnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# -------------------------
# Building blocks
# -------------------------
class SqueezeExcite(nn.Module):
    """Channel attention: GAP → FC↓ → SiLU → FC↑ → Sigmoid → scale."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(1, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.silu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DepthwiseSeparable(nn.Module):
    """
    3x3 depthwise (stride, padding=1) + BN + SiLU,
    then 1x1 pointwise + BN + (optional SE) + SiLU.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        use_se: bool = False,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)

        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.SiLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(out_ch, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.pw_bn(self.pw(x))
        if self.use_se:
            x = self.se(x)
        x = self.act(x)
        return x


class ASPPLite(nn.Module):
    """
    Four 3x3 atrous branches with rates {1,6,12,18}, each → 64 ch.
    Concatenate to 256 ch → BN → SiLU → Dropout(0.5).
    """
    def __init__(
        self,
        in_ch: int,
        branch_ch: int = 64,
        rates=(1, 6, 12, 18),
        p_drop: float = 0.5,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, branch_ch, 3,
                        padding=r, dilation=r, bias=False
                    ),
                    nn.BatchNorm2d(branch_ch),
                    nn.SiLU(inplace=True),
                )
            )

        self.out_bn = nn.BatchNorm2d(branch_ch * len(rates))
        self.out_act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        y = torch.cat(ys, dim=1)  # B, 256, H, W
        y = self.out_act(self.out_bn(y))
        y = self.drop(y)
        return y


# -------------------------
# Model
# -------------------------
class TinyVOCSeg(nn.Module):
    """
    Lightweight TinySeg for VOC (≈0.5M params, depending on PyTorch version).

    Encoder:
      Stage1: 3 -> 24, stride 2  (tap: low, ~1/2)
      Stage2: 24 -> 40, stride 2 (tap: mid, ~1/4)
      Stage3: 40 -> 96, stride 2 (SE on, ~1/8)
      Stage4: 96 -> 192, stride 2 (SE on, ~1/16, → ASPP-lite)

    Decoder:
      ASPP(high, 192) -> 256 ch
        → 1×1 ctx_to_64 → 64 ch
        → up to 1/4, fuse (+) with mid (40→64) → dec_mid (3×3, 64→64)
        → up to 1/2, align 64→32, fuse (+) with low(24→32) → dec_low (3×3, 32→64)
        → 1×1 classifier (64→21), upsample to input size.

    Returns:
      dict(out=logits, taps={'low':..., 'mid':..., 'high':...})
    """
    def __init__(self, n_classes: int = 21):
        super().__init__()

        # -------- Encoder --------
        self.st1 = DepthwiseSeparable(3,   24, stride=2, use_se=False)  # 1/2
        self.st2 = DepthwiseSeparable(24,  40, stride=2, use_se=False)  # 1/4
        self.st3 = DepthwiseSeparable(40,  96, stride=2, use_se=True)   # 1/8
        self.st4 = DepthwiseSeparable(96, 192, stride=2, use_se=True)   # 1/16

        # -------- Context (ASPP) --------
        self.aspp = ASPPLite(192, branch_ch=64, rates=(1, 6, 12, 18), p_drop=0.5)
        # ASPP output is 256 channels
        self.ctx_to_64 = nn.Conv2d(256, 64, 1, bias=False)

        # -------- Decoder: mid (1/4) fusion --------
        self.mid_proj = nn.Conv2d(40, 64, 1, bias=False)
        self.mid_bn   = nn.BatchNorm2d(64)
        self.dec_mid  = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # -------- Decoder: low (1/2) fusion --------
        self.low_proj  = nn.Conv2d(24, 32, 1, bias=False)
        self.low_bn    = nn.BatchNorm2d(32)
        self.low_align = nn.Conv2d(64, 32, 1, bias=False)  # align mid path → 32 ch

        self.dec_low = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # -------- Head --------
        self.dropout_head = nn.Dropout2d(0.5)
        self.cls = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        # Encoder
        x1 = self.st1(x)   # low:  B, 24, H/2,  W/2
        x2 = self.st2(x1)  # mid:  B, 40, H/4,  W/4
        x3 = self.st3(x2)  #      B, 96, H/8,  W/8
        x4 = self.st4(x3)  # high: B,192, H/16, W/16

        # Context
        ctx = self.aspp(x4)                     # B,256,H/16,W/16
        ctx64 = self.ctx_to_64(ctx)             # B, 64,H/16,W/16

        # ---- Fuse with mid (1/4) ----
        up_mid = F.interpolate(
            ctx64, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )                                       # B,64,H/4,W/4
        mid_feat = self.mid_bn(self.mid_proj(x2))  # B,64,H/4,W/4
        fuse_mid = self.dec_mid(up_mid + mid_feat)  # B,64,H/4,W/4

        # ---- Fuse with low (1/2) ----
        up_low = F.interpolate(
            fuse_mid, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )                                       # B,64,H/2,W/2
        up_low32 = self.low_align(up_low)       # B,32,H/2,W/2
        low_feat = self.low_bn(self.low_proj(x1))  # B,32,H/2,W/2
        fuse_low = self.dec_low(up_low32 + low_feat)  # B,64,H/2,W/2

        # Head: to full resolution
        logits_half = self.cls(self.dropout_head(fuse_low))        # B,21,H/2,W/2
        logits = F.interpolate(
            logits_half, size=(H, W), mode="bilinear", align_corners=False
        )                                                          # B,21,H,W

        taps = {"low": x1, "mid": x2, "high": x4}
        return {"out": logits, "taps": taps}


# -------------------------
# Utilities
# -------------------------
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyVOCSeg(n_classes=21)
    x = torch.randn(1, 3, 320, 320)  # crop size per spec
    out = model(x)

    print("Logits:", tuple(out["out"].shape))          # (1, 21, 320, 320)
    for k, v in out["taps"].items():
        print(f"tap[{k}]:", tuple(v.shape))
    print("Trainable params:", f"{count_trainable_params(model)/1e6:.3f}M")
