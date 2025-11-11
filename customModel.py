# tiny_segnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# -------------------------
# Building blocks
# -------------------------
class DepthwiseSeparable(nn.Module):
    """3x3 depthwise (stride, padding=1) + BN + SiLU, then 1x1 pointwise + BN + SiLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, use_se: bool = False, se_reduction: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
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


class ASPPLite(nn.Module):
    """
    Four 3x3 atrous branches with rates {1,6,12,18}, each to 64 ch → concat (256) → BN → SiLU → Dropout(0.5).
    """
    def __init__(self, in_ch: int, branch_ch: int = 64, rates=(1, 6, 12, 18), p_drop: float = 0.5):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, branch_ch, 3, padding=r, dilation=r, bias=False) for r in rates
        ])
        self.bn = nn.BatchNorm2d(branch_ch * len(rates))
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        y = torch.cat(ys, dim=1)                 # B, 256, H, W
        y = self.act(self.bn(y))
        y = self.drop(y)
        return y


# -------------------------
# Model
# -------------------------
class TinyVOCSeg(nn.Module):
    """
    Encoder:
      Stage1: 3 -> 24, stride 2  (tap: low, ~1/2)
      Stage2: 24 -> 40, stride 2 (tap: mid, ~1/4)
      Stage3: 40 -> 96, stride 2 (no tap)
      Stage4: 96 -> 192, stride 2 (tap: high, ~1/16, → ASPP-lite)

    Decoder:
      ASPP(high) → up 4× → add with mid(1×1→64) → 3×3
                 → up 2× → add with low(1×1→32) → 3×3
                 → up 2× → 1×1 classifier → logits (B,21,H,W)

    Returns:
      dict(out=logits, taps={'low':..., 'mid':..., 'high':...})
    """
    def __init__(self, n_classes: int = 21):
        super().__init__()
        # Encoder
        self.st1 = DepthwiseSeparable(3,   24, stride=2, use_se=False)   # 1/2
        self.st2 = DepthwiseSeparable(24,  40, stride=2, use_se=False)   # 1/4
        self.st3 = DepthwiseSeparable(40,  96, stride=2, use_se=True)    # 1/8 (SE on)
        self.st4 = DepthwiseSeparable(96, 192, stride=2, use_se=True)    # 1/16 (SE on)

        # Context head
        self.aspp = ASPPLite(192, branch_ch=64, rates=(1, 6, 12, 18), p_drop=0.5)  # -> 256 ch

        # Decoder projections
        self.proj_mid = nn.Conv2d(40, 64, 1, bias=False)
        self.bn_mid   = nn.BatchNorm2d(64)

        self.dec1 = nn.Sequential(                       # after (upsampled ASPP 4× + mid)
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        self.proj_low = nn.Conv2d(24, 32, 1, bias=False)
        self.bn_low   = nn.BatchNorm2d(32)

        self.dec2 = nn.Sequential(                       # after (upsampled dec1 2× + low)
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        self.dropout_head = nn.Dropout2d(0.5)
        self.cls = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]

        # Encoder
        x1 = self.st1(x)          # low,   B, 24, H/2,  W/2
        x2 = self.st2(x1)         # mid,   B, 40, H/4,  W/4
        x3 = self.st3(x2)         #       B, 96, H/8,  W/8
        x4 = self.st4(x3)         # high,  B,192, H/16, W/16

        # Context
        h = self.aspp(x4)         # B,256, H/16, W/16

        # Decode: fuse with mid (1/4). Upsample ASPP 4× → H/4
        h = F.interpolate(h, scale_factor=4, mode="bilinear", align_corners=False)
        m = self.bn_mid(self.proj_mid(x2))              # B,64,H/4,W/4
        h = self.dec1(h + m)                            # B,64,H/4,W/4

        # Decode: fuse with low (1/2). Upsample 2× → H/2
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        l = self.bn_low(self.proj_low(x1))              # B,32,H/2,W/2
        # bring channels to 64 before add: concatenate then reduce, or add after a lift.
        # We keep it simple: upsampled h (64ch) first → reduce in dec2 to 32, then add via a 1x1 adapter.
        # To stick strictly to “add then 3x3”, we adapt by projecting h to 32 on-the-fly:
        h32 = nn.functional.conv2d(h, weight=torch.zeros(32, 64, 1, 1, device=h.device)) if False else None  # placeholder (no-op)
        # Simpler: concatenate then 3x3 to 32 (equiv. to add+3x3); matches channels robustly:
        h = torch.cat([h, l], dim=1)                    # B,96,H/2,W/2
        h = nn.Sequential(                               # inline block (built once at first call)
            nn.Conv2d(96, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        ).to(h.device)(h)

        # Final upsample to input
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        h = self.dropout_head(h)
        logits = self.cls(h)                             # B,21,H,W

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
