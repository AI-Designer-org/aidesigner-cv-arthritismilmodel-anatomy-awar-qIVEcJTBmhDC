"""
Supporting layers for the Arthritis MIL architecture.

Contains:
  - _normalize_xray:      per-image normalization for X-ray contrast
  - JointDetectionModule:  YOLOv7 wrapper for localizing joints in full X-ray
  - ROIFeatureExtractor:   differentiable crop + resize via torchvision.ops.roi_align
  - MultiViewFusion:       concatenative or cross-attention fusion across X-ray views
  - GatedAttentionMIL:     gated attention mechanism for MIL pooling (Ilse et al., 2018)

Shape conventions:
    Image:         (B, C, H, W)          — batch, channels, height, width
    Joint crops:   (B, N, C, H_roi, W_roi)
    Features:      (B, N, d_model)
    Boxes:         List[Tensor(N_i, 4)]  — per-image (x1, y1, x2, y2)
    Attention:     (B, N)
    Bag:           (B, d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════
# Helper: X-ray Normalization
# ═══════════════════════════════════════════════════════════════════════

def _normalize_xray(x: torch.Tensor) -> torch.Tensor:
    """Per-image normalization: zero-mean, unit-variance.

    X-rays have inconsistent brightness/contrast across acquisitions.
    Per-image normalization removes acquisition-level variance while
    preserving anatomical contrast.

    Args:
        x: (B, C, H, W) — raw X-ray

    Returns:
        (B, C, H, W) — normalized
    """
    B, C, H, W = x.shape                                  # batch, channels, height, width
    x_flat = x.view(B, C, -1)                              # (B, C, H*W)
    mean = x_flat.mean(dim=-1, keepdim=True)               # (B, C, 1)
    std = x_flat.std(dim=-1, keepdim=True).clamp(min=1e-6) # (B, C, 1)
    return (x - mean.view(B, C, 1, 1)) / std.view(B, C, 1, 1)  # (B, C, H, W)


# ═══════════════════════════════════════════════════════════════════════
# Block 1 — Joint Detection Module
# ═══════════════════════════════════════════════════════════════════════

class JointDetectionModule(nn.Module):
    """Localizes individual joints in a full X-ray image using YOLOv7.

    This module is designed for OFFLINE use (not in the hot training loop
    since DINOv2 is the bottleneck).  YOLOv7 runs at inference time or
    as a pre-processing step to cache detection boxes.

    If joint coordinates are already available (structured annotations),
    this module is bypassed by setting detection_model="none" in config
    and passing pre-computed boxes via the model forward method.

    Input:  (B, 1, H_full, W_full)          — full grayscale X-ray
    Output: List[Tensor(N_i, 4)] per batch  — (x1, y1, x2, y2)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conf_threshold = config.detection_confidence
        self.iou_threshold = config.detection_iou_threshold
        # Lazy-load YOLOv7 — only when first called
        self._detector = None

    def _load_detector(self):
        """Load YOLOv7 via torch hub on first call."""
        if self._detector is None:
            self._detector = torch.hub.load(
                "WongKinYiu/yolov7", "yolov7", pretrained=True
            )
            self._detector.eval()
            for param in self._detector.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Detect joints in full X-ray images.

        Args:
            x: (B, 1, H, W) — grayscale X-ray images

        Returns:
            List of (N_i, 4) tensors, one per batch item.
            Each row is (x1, y1, x2, y2) in pixel coordinates.
        """
        self._load_detector()

        B = x.shape[0]                                       # batch

        # Normalize and repeat to 3 channels (YOLOv7 expects RGB)
        x_norm = _normalize_xray(x)                          # (B, 1, H, W)
        x_rgb = x_norm.expand(-1, 3, -1, -1)                 # (B, 3, H, W)

        # YOLOv7 inference
        with torch.no_grad():
            detections = self._detector(x_rgb)                # list per-image of (N_det, 6)

        boxes = []
        for det in detections:
            # Filter by confidence
            mask = det[:, 4] > self.conf_threshold            # (N_det,)
            det = det[mask]                                   # (N_filtered, 6)

            # Apply NMS
            keep = torch.ops.torchvision.nms(
                det[:, :4], det[:, 4], self.iou_threshold
            )
            boxes.append(det[keep, :4])                       # (N_keep, 4)

        return boxes


# ═══════════════════════════════════════════════════════════════════════
# Block 2 — ROI Feature Extractor (differentiable crop + resize)
# ═══════════════════════════════════════════════════════════════════════

class ROIFeatureExtractor(nn.Module):
    """Crops joint ROIs from full X-ray using detected/annotated boxes.

    Uses torchvision.ops.roi_align for differentiable, batched cropping
    with gradient flow through box coordinates.

    Pads to max_joints_per_view across the batch with a validity mask
    so downstream MIL can handle variable joint counts.

    Input:  img (B, 1, H, W), boxes List[Tensor(N_i, 4)]
    Output: rois (B, N_max, 1, H_roi, W_roi), mask (B, N_max)

    Inductive bias: Fixed-size crops standardize feature extraction across
    differently-sized joints (MCP vs. wrist vs. DIP), though discarding
    relative spatial context within the hand.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.output_size = (config.img_size, config.img_size)   # (H_roi, W_roi)
        self.max_joints = config.max_joints_per_view

    def forward(
        self, img: torch.Tensor, boxes: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and pad joint ROI crops.

        Args:
            img:   (B, 1, H_full, W_full) — full X-ray images
            boxes: list of (N_i, 4) tensors — detected/annotated joint boxes

        Returns:
            rois:  (B, N_max, 1, H_roi, W_roi) — padded joint crops
            mask:  (B, N_max) — 1 for real joints, 0 for padding
        """
        B = img.shape[0]                                       # batch
        device = img.device
        H_full, W_full = img.shape[-2:]                        # height, width

        # Build RoI-Align input: (batch_index, x1, y1, x2, y2)
        all_rois = []
        per_image_counts = []

        for b in range(B):
            n = boxes[b].size(0)                               # N_i joints in this image
            per_image_counts.append(n)

            if n > 0:
                # Clamp boxes to image boundaries
                box_clamped = boxes[b].clone()
                box_clamped[:, 0] = box_clamped[:, 0].clamp(0, W_full - 1)  # x1
                box_clamped[:, 1] = box_clamped[:, 1].clamp(0, H_full - 1)  # y1
                box_clamped[:, 2] = box_clamped[:, 2].clamp(0, W_full - 1)  # x2
                box_clamped[:, 3] = box_clamped[:, 3].clamp(0, H_full - 1)  # y2

                batch_idx = torch.full((n, 1), b, device=device, dtype=torch.float)
                roi_batch = torch.cat([batch_idx, box_clamped], dim=1)   # (n, 5)
                all_rois.append(roi_batch)

        # Batched RoI-Align
        if all_rois:
            rois_flat = torch.cat(all_rois, dim=0)             # (total_joints, 5)
            crops = torchvision_roi_align(
                img, rois_flat, self.output_size, spatial_scale=1.0, aligned=True
            )                                                  # (total_joints, 1, H_roi, W_roi)
        else:
            crops = torch.zeros(0, 1, *self.output_size, device=device)

        # Split back per image and pad to max_joints
        padded = torch.zeros(
            B, self.max_joints, 1, *self.output_size, device=device
        )                                                      # (B, N_max, 1, H_roi, W_roi)
        mask = torch.zeros(B, self.max_joints, device=device)  # (B, N_max)

        start = 0
        for b in range(B):
            n = per_image_counts[b]
            if n > 0:
                k = min(n, self.max_joints)
                padded[b, :k] = crops[start : start + k]
                mask[b, :k] = 1.0
                start += n

        return padded, mask


def torchvision_roi_align(
    img: torch.Tensor,
    boxes: torch.Tensor,
    output_size: Tuple[int, int],
    spatial_scale: float = 1.0,
    aligned: bool = True,
) -> torch.Tensor:
    """Wrapper around torchvision.ops.roi_align with lazy import."""
    import torchvision.ops as ops
    return ops.roi_align(img, boxes, output_size, spatial_scale, aligned=aligned)


# ═══════════════════════════════════════════════════════════════════════
# Block 3 — Multi-View Fusion
# ═══════════════════════════════════════════════════════════════════════

class MultiViewFusion(nn.Module):
    """Fuses joint-level features from multiple X-ray views into a unified bag.

    Strategy:
      1. Add view-specific learned embedding to each joint feature
      2. Concatenate all joints across views into a single bag
      3. Optionally apply cross-attention to model inter-view joint correlations

    Input:  view_features Dict[view_name, (B, N_v, d_model)]
            view_names    List[str] — ordered keys
    Output: fused         (B, N_total, d_model)

    Inductive bias: View embeddings let the model distinguish joints from
    different perspectives. Cross-attention allows associating the same
    anatomical joint across views.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fusion_type = config.multi_view_fusion
        d = config.d_model

        # Learned view embeddings
        if config.use_view_embedding:
            self.view_embeddings = nn.ParameterDict({
                view: nn.Parameter(torch.randn(1, 1, d) * 0.02)
                for view in config.input_views
            })

        # Optional cross-attention fusion
        if self.fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                d, config.fusion_n_heads,
                dropout=config.fusion_dropout, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(d)

    def forward(
        self,
        view_features: Dict[str, torch.Tensor],
        view_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse features from multiple views.

        Args:
            view_features: {view_name: (B, N_v, d)} per-view features
            view_names: ordered list of view keys

        Returns:
            fused: (B, N_total, d) — concatenated features with optional cross-attention
            view_membership: (B, N_total) — view index for each joint (for attribution)
        """
        fused_list = []
        membership_list = []

        for v_idx, name in enumerate(view_names):
            feats = view_features[name]                        # (B, N_v, d)

            # Add view embedding
            if hasattr(self, "view_embeddings") and name in self.view_embeddings:
                feats = feats + self.view_embeddings[name]     # (B, N_v, d)

            # Track which view each joint belongs to
            N_v = feats.size(1)                                # joints in this view
            membership = torch.full(
                (feats.size(0), N_v), v_idx,
                device=feats.device, dtype=torch.long
            )                                                  # (B, N_v)

            fused_list.append(feats)
            membership_list.append(membership)

        x = torch.cat(fused_list, dim=1)                       # (B, N_total, d)
        view_membership = torch.cat(membership_list, dim=1)    # (B, N_total)

        # Optional cross-attention refinement
        if self.fusion_type == "cross_attention":
            x_res = x                                          # (B, N_total, d)
            x, _ = self.cross_attn(x, x, x)                    # (B, N_total, d)
            x = self.layer_norm(x + x_res)                     # (B, N_total, d)

        return x, view_membership


# ═══════════════════════════════════════════════════════════════════════
# Block 4 — Gated Attention MIL Aggregator
# ═══════════════════════════════════════════════════════════════════════

class GatedAttentionMIL(nn.Module):
    """Gated attention mechanism for multi-instance pooling (Ilse et al., 2018).

    Computes a content-based attention weight α_i for each joint:
        a_i = tanh(V · h_i)           — "what is this joint?"
        b_i = sigmoid(U · h_i)        — "how important is it?" (gate)
        α_i = softmax(wᵀ · (a_i ⊙ b_i))

    Bag representation: z = Σ α_i · h_i

    The attention weights α_i are directly interpretable as the relative
    importance of each joint for the patient-level decision.

    Input:  (B, N, d_model)    — joint features
            mask (B, N)        — 1 for real joints, 0 for padding
    Output: bag_rep (B, d)     — attention-weighted bag
            alpha   (B, N)     — attention weights

    Inductive bias: The gating mechanism (sigmoid) allows the model to
    learn non-linear "importance queries" that the tanh projection alone
    cannot express — critical for arthritis because disease-relevant
    patterns (erosion, JSN) are non-linear features of the joint appearance.

    Pre-norm: LayerNorm before gated attention stabilizes training and
    rescues vanishing gradients in the saturating sigmoid/tanh gates.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        L = config.mil_hidden_dim                     # hidden dim
        d = config.d_model                             # feature dim

        self.layer_norm = nn.LayerNorm(d)
        self.V = nn.Linear(d, L, bias=False)           # tanh projection
        self.U = nn.Linear(d, L, bias=False)           # gate projection
        self.w = nn.Linear(L, 1, bias=False)           # attention vector
        self.dropout = nn.Dropout(config.mil_dropout)
        self.gated = config.mil_gated

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through gated attention MIL.

        Args:
            x:    (B, N, d) — joint features
            mask: (B, N)    — 1 for valid joints, 0 for padding

        Returns:
            bag_rep: (B, d) — attention-weighted bag representation
            alpha:   (B, N) — attention weights (sum to 1 over valid joints)
        """
        # Pre-norm for training stability
        x = self.layer_norm(x)                         # (B, N, d)

        # Gated attention scores
        a = torch.tanh(self.V(x))                      # (B, N, L) — content
        if self.gated:
            b = torch.sigmoid(self.U(x))               # (B, N, L) — gate
            gated = a * b                              # (B, N, L)
        else:
            gated = a                                  # (B, N, L)

        # Score each joint
        scores = self.w(gated).squeeze(-1)             # (B, N)
        scores = self.dropout(scores)                  # (B, N)

        # Masked softmax (fp16-safe: compute in float32)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # bf16/fp16 safety: cast to float32 before softmax
        alpha = F.softmax(scores.float(), dim=-1).to(x.dtype)   # (B, N)

        # Weighted aggregation via batched matmul
        bag_rep = torch.bmm(alpha.unsqueeze(1), x)     # (B, 1, d)
        bag_rep = bag_rep.squeeze(1)                   # (B, d)

        return bag_rep, alpha
