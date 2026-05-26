"""
ArthritisClassificationHead: dual-path classification for per-joint and patient-level predictions.

Shape conventions:
    Joint features:  (B, N, d_model)
    Bag rep:         (B, d_model)
    Per-joint logits: (B, N, C)   — C = n_classes (4: RA, PsA, OA, normal)
    Patient logits:   (B, C)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import ModelConfig


class ArthritisClassificationHead(nn.Module):
    """Produces predictions at two granularities:

    (A) Per-joint head: classifies each joint independently.
        Used for fine-grained localization of disease activity.
        Trained with per-joint labels when available.

    (B) Patient-level head: classifies the entire patient from the
        attention-weighted bag representation. Provides the final diagnosis.

    Both heads share the bag representation but have separate parameters.

    Inductive bias: Separating per-joint and patient-level classifiers lets
    the model learn that a single affected joint may indicate disease
    (per-joint head) while the pattern across all joints is more specific
    (patient-level head). This matches clinical practice where one erosive
    joint is enough for diagnosis but the distribution confirms type.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_classes = config.n_classes
        self.disease_specific = config.disease_specific_heads
        d = config.d_model

        # ── (A) Per-joint classifier ──
        if config.per_joint_classifier:
            self.joint_head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, config.per_joint_hidden),
                nn.GELU(),
                nn.Dropout(config.per_joint_dropout),
                nn.Linear(config.per_joint_hidden, config.n_classes),
            )
        else:
            self.joint_head = None

        # ── (B) Patient-level classifier ──
        if self.disease_specific:
            self.patient_heads = nn.ModuleDict({
                name: nn.Linear(d, 1)
                for name in ["RA", "PsA", "OA", "normal"][:config.n_classes]
            })
        else:
            self.patient_head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(config.mil_dropout),
                nn.Linear(d // 2, config.n_classes),
            )

    def forward(
        self,
        joint_features: torch.Tensor,
        bag_rep: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward pass through both classification heads.

        Args:
            joint_features: (B, N, d) — per-joint feature vectors
            bag_rep:        (B, d)    — patient bag representation

        Returns:
            per_joint_logits: (B, N, C) or None
            patient_logits:   (B, C)
        """
        # ── Per-joint logits ──
        per_joint_logits = None
        if self.joint_head is not None:
            per_joint_logits = self.joint_head(joint_features)   # (B, N, C)

        # ── Patient logits ──
        if self.disease_specific:
            logits_list = [
                head(bag_rep) for head in self.patient_heads.values()
            ]                                                    # list of (B, 1)
            patient_logits = torch.cat(logits_list, dim=-1)      # (B, C)
        else:
            patient_logits = self.patient_head(bag_rep)          # (B, C)

        return per_joint_logits, patient_logits
