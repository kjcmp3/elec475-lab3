# distillation_losses.py
import torch
import torch.nn.functional as F

# VOC ignore label
IGNORE_INDEX = 255


def response_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float = 4.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Response-based KD loss (Hinton-style) over valid pixels only.

    Args:
        student_logits: [B, C, H, W] raw logits from student (z_s).
        teacher_logits: [B, C, H, W] raw logits from teacher (z_t).
        targets:        [B, H, W] int labels, used only for the ignore mask.
        T:              temperature.
        ignore_index:   label to ignore in the mask.

    Returns:
        Scalar KD loss (T^2 * KL(p_t^T || p_s^T)).
    """
    device = student_logits.device

    # Mask out ignore pixels
    # valid: [B, H, W]
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        # no valid pixels → return 0 with grad
        return torch.zeros((), device=device, requires_grad=True)

    # [B, C, H, W] -> [B, H, W, C] -> [N, C] for valid pixels only
    s = student_logits.permute(0, 2, 3, 1)[valid]  # [N, C]
    t = teacher_logits.permute(0, 2, 3, 1)[valid]  # [N, C]

    # Softmax with temperature
    s_T = s / T
    t_T = t / T

    log_p_s = F.log_softmax(s_T, dim=-1)
    p_t     = F.softmax(t_T, dim=-1)

    # KL(p_t || p_s) with "batchmean" over N
    kd = F.kl_div(log_p_s, p_t, reduction="batchmean")

    # Hinton-style scaling
    kd = kd * (T * T)
    return kd


def feature_distillation_cosine_loss(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Feature-based KD loss using cosine distance between aligned features.

    Both feats are expected to have shape [B, C, H, W] (already aligned).

    We compute cosine similarity per spatial location and then average:
      L_feat = 1 - mean_cos_sim
    """
    # [B, C, H, W] -> [B, C, H*W]
    B, C, H, W = feat_s.shape
    feat_s_flat = feat_s.view(B, C, -1)   # [B, C, N]
    feat_t_flat = feat_t.view(B, C, -1)   # [B, C, N]

    # Normalize along channel dim
    feat_s_norm = feat_s_flat / (feat_s_flat.norm(dim=1, keepdim=True) + eps)
    feat_t_norm = feat_t_flat / (feat_t_flat.norm(dim=1, keepdim=True) + eps)

    # Cosine sim per spatial location: [B, N]
    cos_sim = (feat_s_norm * feat_t_norm).sum(dim=1)

    # Average over batch and spatial locations
    mean_cos = cos_sim.mean()
    loss = 1.0 - mean_cos
    return loss
