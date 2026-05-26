"""
AnatomyExplanationModule: clinically interpretable explanations for arthritis predictions.

Provides:
  1. Per-joint attention weights α_i — relative importance for the patient-level decision
  2. Per-joint logits — spatial map of disease activity across joints
  3. Anatomy-guided explanation loss — Dice alignment between attention and disease-specific joint priors

Shape conventions:
    attention_weights:  (B, N)
    per_joint_logits:   (B, N, C)
    joint_group_labels: (B, N)   — 0..n_groups-1, -1 for unknown
    target_disease:     (B,)     — ground-truth disease class
"""

import torch
import torch.nn as nn
from typing import Optional

from config import ModelConfig


class AnatomyExplanationModule(nn.Module):
    """Produces clinically interpretable explanations and an optional anatomy-guided loss.

    When use_anatomy_prior_loss is enabled, computes a Dice loss between
    attention weights and disease-specific anatomical priors
    (e.g., RA→MCP/PIP, OA→DIP, PsA→PIP/DIP).

    Inductive bias: Anatomical priors encode domain knowledge that different
    arthritis types preferentially affect specific joint groups. Aligning
    attention with these priors improves clinical plausibility of explanations
    without requiring pixel-level ROI annotations for every image.

    The priors are a soft target — attention should be higher on joints
    typically affected by the diagnosed disease.

    Anatomy priors (based on clinical literature):
        RA:   MCP 2-5, PIP 2-5, wrist    (spares DIP)
        PsA:  DIP 2-5, PIP 2-5           (may involve all fingers)
        OA:   DIP 2-5, CMC/thumb base    (spares MCP)
        Normal: uniform (all joints equally plausible)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.explanation_method = config.explanation_method
        self.use_prior_loss = config.use_anatomy_prior_loss
        self.prior_weight = config.anatomy_prior_loss_weight

        # Register anatomy priors as a persistent buffer (saved with checkpoint)
        priors = self._build_anatomy_priors(config.n_classes)
        self.register_buffer("anatomy_priors", priors, persistent=True)

    # ------------------------------------------------------------------
    # Anatomy Prior Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anatomy_priors(n_classes: int) -> torch.Tensor:
        """Build disease-to-joint-group prior masks.

        Joint group indices for a hand X-ray:
            0-3:   DIP 2-5        (4 groups)
            4-8:   PIP 2-5        (4 groups)
            8-11:  MCP 2-5        (4 groups)
            12:    wrist
            13:    CMC/thumb
            14-17: carpals        (4 groups)
            Total: 18 joint groups

        Returns:
            (C, n_groups) float tensor — prior relevance mask per disease class
        """
        n_groups = 18
        priors = torch.zeros(n_classes, n_groups)

        # Class order: 0=RA, 1=PsA, 2=OA, 3=normal
        # RA:   MCP (8-11), PIP (4-7), wrist (12)
        priors[0, 4:12] = 1.0    # PIP + MCP
        priors[0, 12] = 1.0      # wrist

        # PsA:  DIP (0-3), PIP (4-7)
        priors[1, 0:8] = 1.0     # DIP + PIP

        # OA:   DIP (0-3), CMC (13)
        priors[2, 0:4] = 1.0     # DIP
        priors[2, 13] = 1.0      # CMC/thumb

        # Normal: uniform over all groups
        if n_classes > 3:
            priors[3, :] = 1.0 / n_groups

        return priors  # (C, n_groups)

    # ------------------------------------------------------------------
    # Explanation Loss (fully vectorized)
    # ------------------------------------------------------------------

    def compute_explanation_loss(
        self,
        attention_weights: torch.Tensor,
        joint_group_labels: torch.Tensor,
        target_disease: torch.Tensor,
    ) -> torch.Tensor:
        """Dice loss between attention weights and disease-specific anatomical prior.

        Fully vectorized — no Python loops — compatible with torch.compile.

        Args:
            attention_weights:  (B, N) — MIL attention α_i over joints
            joint_group_labels: (B, N) — anatomical group ID (0..n_groups-1) or -1
            target_disease:     (B,)   — ground-truth disease class index

        Returns:
            Scalar loss value (0 if use_prior_loss=False)
        """
        if not self.use_prior_loss:
            return torch.tensor(0.0, device=attention_weights.device)

        B, N = attention_weights.shape                       # batch, joints

        # ── Gather prior row for each sample's disease ──
        # self.anatomy_priors: (C, n_groups)
        disease_priors = self.anatomy_priors[target_disease]  # (B, n_groups)

        # ── Map joint group labels → prior values ──
        # Handle -1 (unknown groups) by clamping to 0 and zeroing them out later
        has_label = (joint_group_labels >= 0).float()        # (B, N)
        group_labels_clamped = joint_group_labels.clamp(0, self.anatomy_priors.size(1) - 1)  # (B, N)

        # Batch-indexed gather: for each (b, j), get disease_priors[b, group_labels[b,j]]
        batch_idx = torch.arange(B, device=attention_weights.device)[:, None].expand(-1, N)  # (B, N)
        prior_mask = disease_priors[batch_idx, group_labels_clamped]  # (B, N)

        # Zero out positions where group label is unknown
        prior_mask = prior_mask * has_label                   # (B, N)

        # ── Smoothed Dice coefficient ──
        intersection = (attention_weights * prior_mask).sum(dim=-1)   # (B,)
        union = attention_weights.sum(dim=-1) + prior_mask.sum(dim=-1)  # (B,)
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)   # (B,) — per-sample Dice

        dice_loss = 1.0 - dice.mean()                         # scalar

        return self.prior_weight * dice_loss

    # ------------------------------------------------------------------
    # Explanation Output Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_top_k_joints(
        self,
        attention_weights: torch.Tensor,
        joint_names: Optional[list] = None,
        k: int = 5,
    ) -> list:
        """Return top-k joints by attention weight for interpretability.

        Args:
            attention_weights: (B, N) — attention weights
            joint_names: list of N joint name strings (optional)
            k: number of top joints to return

        Returns:
            List of (joint_index, weight, name) tuples per batch item
        """
        B, N = attention_weights.shape
        results = []

        for b in range(B):
            weights = attention_weights[b]                     # (N,)
            top_indices = weights.argsort(descending=True)[:k]  # (k,)
            top_weights = weights[top_indices]                  # (k,)

            batch_result = []
            for rank in range(k):
                idx = top_indices[rank].item()
                name = joint_names[idx] if joint_names else f"joint_{idx}"
                batch_result.append((idx, top_weights[rank].item(), name))
            results.append(batch_result)

        return results

    def forward(
        self,
        attention_weights: torch.Tensor,
        joint_group_labels: torch.Tensor,
        target_disease: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for compute_explanation_loss.

        Args:
            attention_weights:  (B, N)
            joint_group_labels: (B, N)
            target_disease:     (B,)

        Returns:
            scalar loss
        """
        return self.compute_explanation_loss(
            attention_weights, joint_group_labels, target_disease
        )
