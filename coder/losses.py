"""
Loss functions for the Arthritis MIL model.

Contains:
  - focal_loss:        multi-class focal loss (Lin et al., 2017) with ignore_index
  - compute_loss:      multi-objective training loss combining per-joint, patient,
                       anatomy, and entropy regularization terms
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Handles class imbalance (RA >> PsA >> OA in clinical prevalence) by
    down-weighting well-classified examples and focusing on hard ones.

    Args:
        logits:       (..., C) — raw scores per class
        targets:      (...,)   — ground-truth class indices
        gamma:        focusing parameter (≥0). γ=0 → standard CE
        alpha:        (C,) optional per-class weighting factor
        ignore_index: targets equal to this are masked out

    Returns:
        scalar loss averaged over non-ignored positions

    bf16/fp16 safety: cross_entropy internally casts logits to float32
    when computing the softmax, so we accept any input dtype.
    """
    # Standard cross-entropy (ignored positions get 0 loss)
    ce_loss = F.cross_entropy(
        logits, targets, reduction="none", ignore_index=ignore_index
    )                                                    # (...,) — 0 at ignored positions

    # Convert to probability of correct class
    pt = torch.exp(-ce_loss)                             # (...,) — p_t in [0, 1]

    # Focal modulation: (1 - p_t)^γ
    focal_weight = (1.0 - pt) ** gamma                   # (...,)

    # Apply focal weight
    focal = focal_weight * ce_loss                       # (...,)

    # Apply per-class alpha weighting
    if alpha is not None:
        # alpha: (C,) — gather per-target-class
        alpha_t = alpha.gather(0, targets.clamp(min=0))  # (...,)
        alpha_t = alpha_t * (targets != ignore_index).float()  # zero out ignored
        focal = alpha_t * focal                          # (...,)

    # Mask out ignored positions
    mask = (targets != ignore_index).float()             # (...,)
    focal = focal * mask                                 # (...,)

    return focal.sum() / mask.sum().clamp(min=1.0)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Objective Training Loss
# ═══════════════════════════════════════════════════════════════════════

def compute_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """Compute the multi-objective training loss.

    L_total = w_j * L_joint + w_p * L_patient + w_a * L_anatomy + w_e * L_entropy

    Args:
        output: dict from model forward with keys:
            'per_joint_logits'   (B, N, C) or None
            'patient_logits'     (B, C)
            'attention_weights'  (B, N)
            'explanation_loss'   scalar
        batch: data batch dict with keys:
            'joint_labels'       (B, N) — per-joint class, -100 for padding
            'patient_label'      (B,)   — patient-level class
        config: ModelConfig

    Returns:
        dict with 'loss', 'loss_joint', 'loss_patient',
             'loss_anatomy', 'loss_entropy'
    """
    device = output["patient_logits"].device
    losses = {}

    # ── 1. Per-joint classification loss ──
    if output["per_joint_logits"] is not None and "joint_labels" in batch:
        logits = output["per_joint_logits"]                         # (B, N, C)
        labels = batch["joint_labels"]                               # (B, N)

        if config.loss_type == "focal":
            # Flatten to (B*N, C) for focal loss
            losses["loss_joint"] = focal_loss(
                logits.reshape(-1, config.n_classes),
                labels.reshape(-1),
                gamma=config.focal_gamma,
                alpha=(
                    torch.tensor(config.focal_alpha, device=device)
                    if config.focal_alpha is not None else None
                ),
                ignore_index=-100,
            )
        else:
            losses["loss_joint"] = F.cross_entropy(
                logits.reshape(-1, config.n_classes),
                labels.reshape(-1),
                ignore_index=-100,
            )
    else:
        losses["loss_joint"] = torch.tensor(0.0, device=device)

    # ── 2. Patient-level classification loss ──
    if "patient_label" in batch:
        logits = output["patient_logits"]                            # (B, C)
        labels = batch["patient_label"]                              # (B,)

        if config.loss_type == "focal":
            losses["loss_patient"] = focal_loss(
                logits, labels,
                gamma=config.focal_gamma,
                alpha=(
                    torch.tensor(config.focal_alpha, device=device)
                    if config.focal_alpha is not None else None
                ),
            )
        else:
            losses["loss_patient"] = F.cross_entropy(logits, labels)
    else:
        losses["loss_patient"] = torch.tensor(0.0, device=device)

    # ── 3. Anatomy prior explanation loss ──
    losses["loss_anatomy"] = output.get(
        "explanation_loss", torch.tensor(0.0, device=device)
    )

    # ── 4. Entropy regularization on attention weights ──
    attn = output["attention_weights"]                               # (B, N)

    # bf16/fp16 safety: compute entropy in float32
    attn_float = attn.float()
    attn_entropy = -(attn_float * torch.log(attn_float + 1e-8)).sum(dim=-1).mean()
    losses["loss_entropy"] = config.entropy_reg_weight * attn_entropy.to(attn.dtype)

    # ── 5. Total loss ──
    losses["loss"] = (
        config.per_joint_loss_weight * losses["loss_joint"]
        + config.patient_loss_weight * losses["loss_patient"]
        + losses["loss_anatomy"]
        + losses["loss_entropy"]
    )

    return losses
