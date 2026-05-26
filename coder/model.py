"""
ArthritisMILModel: Anatomy-Aware Per-Joint MIL for Three-Way Arthritis Discrimination.

   ┌──────────────────────────────────────────────────────────────────┐
   │              PATIENT-LEVEL OUTPUT (RA / PsA / OA / normal)       │
   └──────────────────────────────────────────────────────────────────┘
                                    ▲
                    ┌───────────────┴────────────────┐
                    │    Patient-Level Aggregation    │
                    │  (attention-weighted pool of   │
                    │   per-joint or bag logits)     │
                    └───────────────┬────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
 ┌────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
 │ Per-Joint Head │   │  Patient-Level Head  │   │  XAI Module      │
 │ (B, N, 4)      │   │  (B, 4)               │   │  α_i importance  │
 └────────┬───────┘   └───────────┬──────────┘   └──────────────────┘
          │                       ▲
          └───────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           │ Gated Attn MIL Pool │
           │ z = Σ α_i · h_i     │
           └──────────┬──────────┘
                      │
           ┌──────────┴──────────┐
           │  Multi-View Fusion  │
           └──────────┬──────────┘
                      │
     ┌────────────────┼────────────────┐
     │ PA View        │ Oblique View   │  ...
     ▼                ▼                ▼
 ┌──────────┐   ┌──────────┐     ┌──────────┐
 │Detection │   │Detection │     │Detection │
 │YOLOv7    │   │YOLOv7    │     │YOLOv7    │
 └────┬─────┘   └────┬─────┘     └────┬─────┘
      ▼               ▼                ▼
 ┌──────────┐   ┌──────────┐     ┌──────────┐
 │ROI Extr  │   │ROI Extr  │     │ROI Extr  │
 │(N₁ joints)│  │(N₂ joints)│    │(N₃ joints)│
 └────┬─────┘   └────┬─────┘     └────┬─────┘
      ▼               ▼                ▼
 ┌──────────┐   ┌──────────┐     ┌──────────┐
 │Foundation│   │Foundation│     │Foundation│
 │Backbone  │   │Backbone  │     │Backbone  │
 │DINOv2    │   │DINOv2    │     │DINOv2    │
 └──────────┘   └──────────┘     └──────────┘

Pipeline:
  1. Joint detection (YOLOv7 or pre-computed boxes)
  2. Per-joint ROI extraction → (B, N, 1, 224, 224)
  3. Foundation model backbone → (B, N, d)
  4. Multi-view fusion → (B, N_total, d)
  5. Gated attention MIL → bag_rep (B, d) + α_i (B, N)
  6. Dual-path classification → per-joint (B, N, C) + patient (B, C)
  7. XAI outputs → attention weights, per-joint logits, explanation loss

Loss (total):
    L = w_j * CE(per_joint_logits, y_joint)
      + w_p * CE(patient_logits, y_patient)
      + w_a * L_anatomy(α_i, joint_groups, y_patient)
      + w_e * H(α_i)

where w_j, w_p, w_a, w_e are loss weights from ModelConfig.

Domain: Computer Vision (primary) + Scientific ML (secondary)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from config import ModelConfig
from backbone import FoundationBackbone
from layers import (
    JointDetectionModule,
    ROIFeatureExtractor,
    MultiViewFusion,
    GatedAttentionMIL,
)
from heads import ArthritisClassificationHead
from explanation import AnatomyExplanationModule


class ArthritisMILModel(nn.Module):
    """Complete anatomy-aware per-joint MIL architecture for arthritis discrimination.

    Composes all sub-modules into an end-to-end pipeline.
    Supports gradient checkpointing on the backbone (the compute bottleneck).

    Usage:
        cfg = ModelConfig(backbone="dinov2_vitl14", detection_model="none")
        model = ArthritisMILModel(cfg)
        output = model(x, boxes={"PA": [torch.randn(15, 4)]})
        patient_pred = output["patient_logits"].argmax(dim=-1)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Stage 1: Joint detection (bypassed if boxes provided)
        self.detector = JointDetectionModule(config) \
            if config.detection_model != "none" else None

        # Stage 2: ROI extraction (differentiable crop + resize)
        self.roi_extractor = ROIFeatureExtractor(config)

        # Stage 3: Foundation model backbone (per-joint feature extractor)
        self.backbone = FoundationBackbone(config)

        # Stage 4: Multi-view fusion
        self.view_fusion = MultiViewFusion(config) \
            if len(config.input_views) > 1 else nn.Identity()

        # Stage 5: MIL aggregator (gated attention)
        self.mil_aggregator = GatedAttentionMIL(config)

        # Stage 6: Dual-path classification heads
        self.classifier = ArthritisClassificationHead(config)

        # Stage 7: Anatomy-guided explanation module
        self.explanation = AnatomyExplanationModule(config)

        # Parameter count tracking
        self._log_param_count()

    def _log_param_count(self) -> None:
        """Log total and trainable parameter counts on init."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        # Print is acceptable on __init__ for model construction diagnostics
        print(
            f"[ArthritisMILModel] Total: {total:,}  "
            f"Trainable: {trainable:,}  Frozen: {frozen:,}"
        )

    # ------------------------------------------------------------------
    # Gradient Checkpointing Hook
    # ------------------------------------------------------------------

    def forward_with_checkpointing(
        self, rois: torch.Tensor
    ) -> torch.Tensor:
        """Run the backbone with gradient checkpointing to save memory.

        The backbone (especially DINOv2 ViT-L on 30 ROIs) is the memory
        bottleneck.  Checkpointing trades compute for memory by not storing
        intermediate activations.

        Args:
            rois: (B, N, 1, H, W) — joint ROI crops

        Returns:
            (B, N, d_model) — per-joint features
        """
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.backbone, rois, use_reentrant=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
        boxes: Optional[Dict[str, List[torch.Tensor]]] = None,
        joint_group_labels: Optional[torch.Tensor] = None,
        return_explanations: bool = True,
        use_checkpoint: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass.

        Args:
            x:        (B, 1, H_full, W_full) — primary view full X-ray (optional
                      if `views` is provided and contains all needed views)
            views:    {view_name: (B, 1, H_v, W_v)} — additional views (optional).
                      If None and x is given, treated as single-view ("PA").
                      If provided, replaces the default view set.
            boxes:    {view_name: list of (N_i, 4)} — pre-computed joint boxes
                      If None, runs YOLOv7 detection on each view.
            joint_group_labels: (B, N_total) — anatomical group IDs (for XAI loss)
            return_explanations: if True, compute explanation outputs
            use_checkpoint: if True, use gradient checkpointing on backbone

        Returns:
            dict with keys:
                per_joint_logits   (B, N, C) or None
                patient_logits     (B, C)
                attention_weights  (B, N)
                bag_representation (B, d)
                joint_features     (B, N, d)
                explanation_loss   scalar
                view_membership    (B, N) — view index per joint (for multi-view attribution)
        """
        # ── Determine views ──
        if views is None:
            if x is None:
                raise ValueError("Either x or views must be provided.")
            views = {"PA": x}
        view_names = list(views.keys())
        n_views = len(view_names)

        # ── Process each view ──
        view_features_dict: Dict[str, torch.Tensor] = {}
        all_masks: List[torch.Tensor] = []

        for v_name in view_names:
            img = views[v_name]                                    # (B, 1, H_v, W_v)
            B = img.shape[0]

            # 1. Joint detection
            if boxes is not None and v_name in boxes:
                v_boxes = boxes[v_name]                            # list of (N_i, 4)
            elif self.detector is not None:
                v_boxes = self.detector(img)                       # list of (N_i, 4)
            else:
                raise ValueError(
                    f"No boxes provided for view '{v_name}' "
                    f"and no detector configured. Either pass boxes or set "
                    f"detection_model≠'none'."
                )

            # 2. Extract ROI crops
            rois, mask = self.roi_extractor(img, v_boxes)
            # rois: (B, N_v, 1, H_roi, W_roi)
            # mask: (B, N_v) — 1 for valid joints, 0 for padding

            # 3. Backbone features (with optional gradient checkpointing)
            if use_checkpoint and self.training:
                features = self.forward_with_checkpointing(rois)   # (B, N_v, d)
            else:
                features = self.backbone(rois)                     # (B, N_v, d)

            view_features_dict[v_name] = features
            all_masks.append(mask)

        # 4. Multi-view fusion
        if n_views > 1:
            fused, view_membership = self.view_fusion(
                view_features_dict, view_names
            )
        else:
            fused = view_features_dict[view_names[0]]              # (B, N, d)
            view_membership = torch.zeros(
                fused.size(0), fused.size(1),
                device=fused.device, dtype=torch.long
            )

        # Combined validity mask across all views
        full_mask = torch.cat(all_masks, dim=1) if n_views > 1 else all_masks[0]
        # full_mask: (B, N_total)

        # 5. MIL aggregation
        bag_rep, attention_weights = self.mil_aggregator(fused, mask=full_mask)
        # bag_rep: (B, d)
        # attention_weights: (B, N_total)

        # 6. Dual-path classification
        per_joint_logits, patient_logits = self.classifier(fused, bag_rep)
        # per_joint_logits: (B, N_total, C) or None
        # patient_logits: (B, C)

        # 7. Explanation loss (anatomy-guided Dice loss)
        explanation_loss = torch.tensor(0.0, device=fused.device)
        if self.explanation.use_prior_loss and joint_group_labels is not None:
            # Use predicted class (detached to avoid extra gradient path)
            # as the target disease for the anatomical prior lookup.
            target_pred = patient_logits.argmax(dim=-1).detach()  # (B,)
            explanation_loss = self.explanation(
                attention_weights, joint_group_labels, target_pred
            )

        # ── Assemble output dict ──
        output = {
            "per_joint_logits": per_joint_logits,
            "patient_logits": patient_logits,
            "attention_weights": attention_weights,
            "bag_representation": bag_rep,
            "joint_features": fused,
            "explanation_loss": explanation_loss,
            "view_membership": view_membership,
        }

        return output

# ═══════════════════════════════════════════════════════════════════════
# Parameter Count Helper
# ═══════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> None:
    """Print total and trainable parameter counts for any nn.Module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
