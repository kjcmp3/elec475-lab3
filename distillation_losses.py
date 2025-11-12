# distillation_losses.py
import torch
import torch.nn.functional as F

IGNORE_INDEX = 255

def softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    logits: [B, C, H, W]
    returns: [B, C, H, W] probabilities with temperature T
    """
    return F.softmax(logits / T, dim=1)

def response_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float = 4.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Hinton-style KL / cross-entropy between softened teacher & student outputs.

    student_logits, teacher_logits: [B, C, H, W]
    targets: [B, H, W] (used only to mask ignore_index)
    """
    B, C, H, W = student_logits.shape

    # mask out void pixels
    valid = (targets != ignore_index).view(B, 1, H, W)   # [B,1,H,W]
    valid = valid.expand(-1, C, -1, -1)                 # [B,C,H,W]

    # flatten all valid spatial locations
    s = student_logits[valid].view(-1, C)  # [N_valid, C]
    t = teacher_logits[valid].view(-1, C)  # [N_valid, C]

    # softened distributions
    log_p_s = F.log_softmax(s / T, dim=1)   # [N_valid, C]
    p_t     = F.softmax(t / T, dim=1)       # [N_valid, C]

    # cross-entropy: -sum p_t * log p_s
    kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return kd_loss

def feature_distillation_cosine_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-based feature distillation.

    student_feat: [B, C_s, H_s, W_s]
    teacher_feat: [B, C_t, H_t, W_t]
    We up/downsample and linearly project both to a common shape if needed
    before applying cosine similarity. This function assumes they've
    already been spatially+channel aligned to same shape [B, C, H, W].
    """
    assert student_feat.shape == teacher_feat.shape, (
        f"Shapes must match for cosine feat loss, got {student_feat.shape} vs {teacher_feat.shape}"
    )

    B = student_feat.shape[0]
    # flatten to [B, C*H*W]
    s_flat = student_feat.view(B, -1)
    t_flat = teacher_feat.view(B, -1)

    # cosine similarity per sample
    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B]
    loss = (1.0 - cos_sim).mean()
    return loss
