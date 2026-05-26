"""
Core Block Pseudocode
=====================
Anatomy-Aware Per-Joint MIL for Arthritis X-ray Analysis

Each block below is a self-contained nn.Module. The full model
(ArthritisMILModel) at the bottom composes them.

Notation:
    B  = batch size
    N  = number of joints in the bag (pad to N_max)
    N_v = number of joints in view v
    V  = number of views
    d  = d_model (feature dimension)
    L  = mil_hidden_dim
    C  = n_classes (4: RA, PsA, OA, normal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Tuple


# ═════════════════════════════════════════════════════════════════════
# Block 1 — Joint Detection
# ═════════════════════════════════════════════════════════════════════

class JointDetectionModule(nn.Module):
    """
    Localizes individual joints in a full X-ray image.

    YOLOv7 provides bounding-box-level localization. If joint
    coordinates are already available (structured annotations), this
    module is bypassed and boxes are read from the dataset directly.

    Input:  (B, 1, H_full, W_full)          — full grayscale X-ray
    Output: List[Tensor(N_i, 4)] per batch  — (x1,y1,x2,y2) per joint

    Inductive bias: separating detection from classification lets us
    handle variable joint counts and occluded/missing joints gracefully.
    """
    def __init__(self, config):
        super().__init__()
        # YOLOv7 from Ultralytics
        self.detector = torch.hub.load(
            'WongKinYiu/yolov7', 'yolov7', pretrained=True
        )
        self.conf_threshold = config.detection_confidence
        self.iou_threshold = config.detection_iou_threshold

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 1, H, W) — normalize and repeat to 3 channels
        x_rgb = _normalize_xray(x)  # zero-mean, unit-var per image
        x_rgb = x_rgb.expand(-1, 3, -1, -1)

        # YOLOv7 returns list of detections per image
        detections = self.detector(x_rgb)

        boxes = []
        for det in detections:
            # Filter by confidence
            mask = det[:, 4] > self.conf_threshold
            det = det[mask]
            # NMS
            keep = torch.ops.torchvision.nms(
                det[:, :4], det[:, 4], self.iou_threshold
            )
            boxes.append(det[keep, :4])

        return boxes  # List[(N_i, 4)]


# ═════════════════════════════════════════════════════════════════════
# Block 2 — ROI Feature Extractor (differentiable crop + resize)
# ═════════════════════════════════════════════════════════════════════

class ROIFeatureExtractor(nn.Module):
    """
    Crops joint ROIs from full X-ray using detected/annotated boxes,
    resizes to fixed size, and applies per-joint normalization.

    Uses RoI-Align (from torchvision.ops) for gradient flow through
    spatial coordinates.

    Input:  img (B, 1, H, W), boxes List[(N_i, 4)]
    Output: rois (B, N_max, 1, H_roi, W_roi), mask (B, N_max)

    Inductive bias: fixed-size crops standardize feature extraction
    across differently-sized joints (MCP vs. wrist vs. DIP), though
    it discards relative spatial context.
    """
    def __init__(self, config):
        super().__init__()
        self.output_size = (config.img_size, config.img_size)
        self.max_joints = config.max_joints_per_view

    def forward(
        self, img: torch.Tensor, boxes: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = img.shape[0]
        device = img.device

        batch_rois = []
        batch_masks = []

        for b in range(B):
            n = boxes[b].size(0)
            # Clamp and crop each box
            rois = []
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                # Clamp to image boundaries
                x1 = x1.clamp(0, img.shape[3])
                y1 = y1.clamp(0, img.shape[2])
                x2 = x2.clamp(0, img.shape[3])
                y2 = y2.clamp(0, img.shape[2])

                # Crop and resize
                crop = img[b:b+1, :, int(y1):int(y2), int(x1):int(x2)]
                crop = F.interpolate(
                    crop, size=self.output_size, mode='bilinear',
                    align_corners=False
                )
                rois.append(crop)

            # Pad to max_joints
            n_valid = len(rois)
            if n_valid == 0:
                # No joints detected — zero-pad
                padded = torch.zeros(
                    1, 1, *self.output_size, device=device
                ).expand(self.max_joints, -1, -1, -1)
                mask = torch.zeros(self.max_joints, device=device)
            else:
                stacked = torch.cat(rois, dim=0)  # (n, 1, H, W)
                padded = torch.cat([
                    stacked,
                    torch.zeros(
                        self.max_joints - n_valid, 1, *self.output_size,
                        device=device
                    )
                ], dim=0)
                mask = torch.cat([
                    torch.ones(n_valid, device=device),
                    torch.zeros(self.max_joints - n_valid, device=device)
                ], dim=0)

            batch_rois.append(padded)
            batch_masks.append(mask)

        return torch.stack(batch_rois), torch.stack(batch_masks)
        # (B, N_max, 1, H_roi, W_roi), (B, N_max)


# ═════════════════════════════════════════════════════════════════════
# Block 3 — Foundation Model Backbone (per-joint feature extractor)
# ═════════════════════════════════════════════════════════════════════

class FoundationBackbone(nn.Module):
    """
    Shared frozen backbone applied independently to each joint ROI.

    Uses DINOv2 ViT-L/14 (default) as the feature extractor.
    The [CLS] token from the last layer serves as the joint
    representation. Optionally applies LoRA to adapt the backbone
    to X-ray domain without full fine-tuning.

    Input:  (B, N, 1, 224, 224)      — joint ROI crops
    Output: (B, N, d_model)           — per-joint feature vectors

    Inductive bias: DINOv2's self-supervised pretraining on ImageNet
    produces features that transfer well to medical imaging despite
    the domain gap, because ViT patch-level features capture generic
    texture/shape primitives relevant to bone erosion patterns.
    """
    def __init__(self, config):
        super().__init__()
        self.frozen = config.backbone_frozen
        self.d_model = config.d_model

        # Load DINOv2
        if config.backbone.startswith("dinov2"):
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', config.backbone
            )
            self.use_cls_token = True
        elif config.backbone == "resnet152":
            from torchvision.models import resnet152, ResNet152_Weights
            self.backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.use_cls_token = False
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        if self.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional LoRA fine-tuning
        if config.use_lora and config.backbone.startswith("dinov2"):
            self._apply_lora(config)

        # Per-joint normalization (X-ray statistics may differ from ImageNet)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _apply_lora(self, config):
        """Replace linear projections in attention with LoRA variants."""
        for name, module in self.backbone.named_modules():
            if not hasattr(module, 'weight'):
                continue
            # Check if this module name contains target projection
            if any(f".{proj}" in name for proj in config.lora_target_modules):
                in_feat, out_feat = module.weight.shape
                lora_layer = LoRALinear(
                    in_feat, out_feat,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha
                )
                lora_layer.weight.data = module.weight.data
                if module.bias is not None:
                    lora_layer.bias = module.bias
                # Replace in parent
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.backbone.get_submodule(parent_name)
                setattr(parent, child_name, lora_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 1, H, W)
        B, N = x.shape[:2]
        x_flat = x.view(B * N, 1, x.shape[-2], x.shape[-1])  # (B*N, 1, H, W)

        # Convert to 3-channel (repeat grayscale) and normalize
        x_flat = x_flat.expand(-1, 3, -1, -1).float()
        x_flat = (x_flat - self.mean) / self.std

        if self.use_cls_token:
            features = self.backbone(x_flat)  # (B*N, d_model) — [CLS] token
        else:
            # ResNet: global average pool over spatial
            features = self.backbone(x_flat)  # (B*N, 2048, H', W')
            features = features.mean(dim=[-2, -1])  # (B*N, 2048)

        return features.view(B, N, self.d_model)  # (B, N, d_model)


# ═════════════════════════════════════════════════════════════════════
# Block 4 — Multi-View Fusion
# ═════════════════════════════════════════════════════════════════════

class MultiViewFusion(nn.Module):
    """
    Fuses joint-level features from multiple X-ray views (e.g., PA hand,
    oblique, lateral) into a unified bag.

    Strategy:
    1. Add view-specific learned embedding to each joint feature
    2. Concatenate all joints across views into a single bag
    3. Optionally apply cross-attention to model inter-view joint correlations

    Input:  Dict[view_name, (B, N_v, d_model)] — per-view joint features
            List[str] — ordered view names
    Output: (B, N_total, d_model) — fused bag
            (B, N_total)          — view membership mask (for attribution)

    Inductive bias: View embeddings let the model distinguish joints
    from different perspectives. Cross-attention allows the model to
    associate the same anatomical joint across views (e.g., learning
    that MCP-2 in PA view and MCP-2 in oblique view are related).
    """
    def __init__(self, config):
        super().__init__()
        self.fusion_type = config.multi_view_fusion

        # View embeddings
        if config.use_view_embedding and config.input_views:
            self.view_embeddings = nn.ParameterDict({
                view: nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
                for view in config.input_views
            })

        # Cross-attention fusion
        if self.fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                config.d_model, config.fusion_n_heads,
                dropout=config.fusion_dropout, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        view_features: Dict[str, torch.Tensor],
        view_names: List[str]
    ) -> torch.Tensor:
        fused = []
        for name in view_names:
            feats = view_features[name]  # (B, N_v, d)
            if hasattr(self, 'view_embeddings') and name in self.view_embeddings:
                feats = feats + self.view_embeddings[name]
            fused.append(feats)

        x = torch.cat(fused, dim=1)  # (B, N_total, d)

        if self.fusion_type == "cross_attention":
            x_res = x
            x, _ = self.cross_attn(x, x, x)
            x = self.layer_norm(x + x_res)

        return x  # (B, N_total, d)


# ═════════════════════════════════════════════════════════════════════
# Block 5 — Gated Attention MIL Aggregator
# ═════════════════════════════════════════════════════════════════════

class GatedAttentionMIL(nn.Module):
    """
    Gated attention mechanism for MIL pooling (Ilse et al., 2018).

    Computes a content-based attention weight α_i for each joint:
        a_i = tanh(V · h_i)           — "what is this joint?"
        b_i = sigmoid(U · h_i)        — "how important is it?" (gate)
        α_i = softmax(wᵀ · (a_i ⊙ b_i))

    Bag representation: z = Σ α_i · h_i

    The attention weights α_i are directly interpretable as the
    relative importance of each joint for the patient-level decision.

    Input:  (B, N, d_model)    — joint features
    Output: (B, d_model)       — bag representation
            (B, N)             — attention weights α_i

    Inductive bias: The gating mechanism (sigmoid) allows the model
    to learn non-linear "importance queries" that the tanh projection
    alone cannot express. This is critical for arthritis because
    disease-relevant patterns (erosion, JSN) are non-linear features
    of the joint appearance.

    Pre-norm: LayerNorm before attention stabilizes training, rescues
    vanishing gradients in the saturating sigmoid/tanh gates.
    """
    def __init__(self, config):
        super().__init__()
        L = config.mil_hidden_dim
        d = config.d_model

        self.layer_norm = nn.LayerNorm(d)
        self.V = nn.Linear(d, L, bias=False)      # tanh projection
        self.U = nn.Linear(d, L, bias=False)      # gate projection
        self.w = nn.Linear(L, 1, bias=False)      # attention vector
        self.dropout = nn.Dropout(config.mil_dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, d)
        x = self.layer_norm(x)

        # Gated attention
        a = torch.tanh(self.V(x))                 # (B, N, L)
        b = torch.sigmoid(self.U(x))              # (B, N, L)
        gated = a * b                              # (B, N, L)

        # Score
        scores = self.w(gated).squeeze(-1)         # (B, N)
        scores = self.dropout(scores)

        # Masked softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=-1)          # (B, N)

        # Weighted aggregation
        bag = torch.bmm(alpha.unsqueeze(1), x)     # (B, 1, d)
        bag = bag.squeeze(1)                       # (B, d)

        return bag, alpha


# ═════════════════════════════════════════════════════════════════════
# Block 6 — LoRA Linear (parameter-efficient fine-tuning)
# ═════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (Hu et al., 2021) of a linear layer.

    W' = W + (B · A) · (α / r)
    where A ~ N(0, σ²), B = 0 initially, so fine-tuning starts
    from the pretrained weights.

    Used only on backbone attention projections when use_lora=True.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank) * 0.01
        )
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original projection + LoRA update
        base = F.linear(x, self.weight, self.bias)
        lora_update = x @ self.lora_A @ self.lora_B * self.scaling
        return base + lora_update


# ═════════════════════════════════════════════════════════════════════
# Block 7 — Dual-Path Classification Head
# ═════════════════════════════════════════════════════════════════════

class ArthritisClassificationHead(nn.Module):
    """
    Produces predictions at two granularities:

    (A) Per-joint head:  classifies each joint independently.
        Used for fine-grained localization of disease activity.
        Trained with per-joint labels when available.

    (B) Patient-level head: classifies the entire patient from the
        attention-weighted bag representation.
        Provides the final diagnosis.

    Both heads share the bag representation but have separate parameters.

    Input:  joint_features (B, N, d)
            bag_rep (B, d)
    Output: per_joint_logits (B, N, C)
            patient_logits (B, C)

    Inductive bias: Separating per-joint and patient-level classifiers
    lets the model learn that a single affected joint may indicate
    disease (per-joint head) while the pattern across all joints is
    more specific (patient-level head). This is consistent with
    clinical practice where one erosive joint is enough for diagnosis
    but the distribution confirms the disease type.
    """
    def __init__(self, config):
        super().__init__()
        self.n_classes = config.n_classes
        self.disease_specific = config.disease_specific_heads

        # (A) Per-joint classifier
        if config.per_joint_classifier:
            self.joint_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.per_joint_hidden),
                nn.GELU(),
                nn.Dropout(config.per_joint_dropout),
                nn.Linear(config.per_joint_hidden, config.n_classes)
            )
        else:
            self.joint_head = None

        # (B) Patient-level classifier
        if self.disease_specific:
            # Binary heads per disease — each outputs a logit
            self.patient_heads = nn.ModuleDict({
                'RA':  nn.Linear(config.d_model, 1),
                'PsA': nn.Linear(config.d_model, 1),
                'OA':  nn.Linear(config.d_model, 1),
                'normal': nn.Linear(config.d_model, 1),
            })
        else:
            self.patient_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.mil_dropout),
                nn.Linear(config.d_model // 2, config.n_classes)
            )

    def forward(
        self, joint_features: torch.Tensor, bag_rep: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Per-joint logits
        per_joint_logits = None
        if self.joint_head is not None:
            per_joint_logits = self.joint_head(joint_features)  # (B, N, C)

        # Patient logits
        if self.disease_specific:
            logits = [head(bag_rep) for head in self.patient_heads.values()]
            patient_logits = torch.cat(logits, dim=-1)  # (B, 4)
        else:
            patient_logits = self.patient_head(bag_rep)  # (B, C)

        return per_joint_logits, patient_logits


# ═════════════════════════════════════════════════════════════════════
# Block 8 — Anatomy-Guided Explanation Module
# ═════════════════════════════════════════════════════════════════════

class AnatomyExplanationModule(nn.Module):
    """
    Produces clinically interpretable explanations:

    1. Attention weights α_i → per-joint importance for patient diagnosis
    2. Per-joint logits → spatial map of disease activity
    3. Optional: Grad-CAM heatmaps on top-k joints for spatial grounding

    When use_anatomy_prior_loss is enabled, this module computes a
    Dice loss between attention weights and disease-specific anatomical
    priors (e.g., RA→MCP/PIP, OA→DIP, PsA→PIP/DIP).

    Input:  joint_features (B, N, d)
            attention_weights (B, N)
            per_joint_logits (B, N, C)
            joint_labels (List[List[str]]) — anatomical names of each joint
    Output: explanations dict

    Inductive bias: Anatomical priors encode domain knowledge that
    different arthritis types preferentially affect specific joint
    groups. Aligning attention with these priors improves clinical
    plausibility of explanations without requiring pixel-level ROI
    annotations for every image.
    """
    def __init__(self, config):
        super().__init__()
        self.explanation_method = config.explanation_method
        self.use_prior_loss = config.use_anatomy_prior_loss
        self.prior_weight = config.anatomy_prior_loss_weight

        # Anatomical priors: mapping from disease to joint groups
        # These are used as soft targets for attention weights.
        # Key: disease index, Value: list of joint-group indices
        self.register_buffer(
            'anatomy_priors',
            self._build_anatomy_priors(config.n_classes),
            persistent=False
        )

    def _build_anatomy_priors(self, n_classes):
        """
        Disease-to-joint-group mapping based on clinical literature:
          RA:   MCP 2-5, PIP 2-5, wrist  (spares DIP)
          PsA:  PIP 2-5, DIP 2-5           (may involve all fingers)
          OA:   DIP 2-5, CMC/thumb base   (spares MCP)
          Normal: no prior (uniform)
        Returns binary mask (C, n_joint_groups) for soft supervision.
        """
        # Joint group indices (example for hand X-ray):
        # 0-4: DIP 2-5, 5-9: PIP 2-5, 10-13: MCP 2-5,
        # 14: wrist, 15: CMC/thumb, 16-19: carpals
        n_groups = 20

        priors = torch.zeros(n_classes, n_groups)
        # RA: MCP (10-13), PIP (5-9), wrist (14)
        priors[0, 5:14] = 1.0
        # PsA: PIP (5-9), DIP (0-4)
        priors[1, 0:10] = 1.0
        # OA: DIP (0-4), CMC (15)
        priors[2, 0:5] = 1.0
        priors[2, 15] = 1.0
        # Normal: uniform
        priors[3, :] = 1.0 / n_groups
        return priors  # (C, n_groups)

    def compute_explanation_loss(
        self,
        attention_weights: torch.Tensor,
        joint_group_labels: torch.Tensor,
        target_disease: torch.Tensor
    ) -> torch.Tensor:
        """
        Dice loss between attention weights and disease-specific
        anatomical prior mask.

        attention_weights: (B, N) — MIL attention α_i
        joint_group_labels: (B, N) — joint group ID (0..n_groups-1)
        target_disease: (B,) — ground-truth disease class

        Returns scalar loss.
        """
        if not self.use_prior_loss:
            return torch.tensor(0.0, device=attention_weights.device)

        B, N = attention_weights.shape

        # Build per-sample prior mask: for each joint, is it in the
        # disease-relevant group?
        prior_mask = torch.zeros(B, N, device=attention_weights.device)
        for b in range(B):
            disease = target_disease[b].item()
            prior_row = self.anatomy_priors[disease]  # (n_groups,)
            for j in range(N):
                group = joint_group_labels[b, j].item()
                if group >= 0:
                    prior_mask[b, j] = prior_row[group]

        # Smoothed Dice
        intersection = (attention_weights * prior_mask).sum(dim=-1)
        union = attention_weights.sum(dim=-1) + prior_mask.sum(dim=-1)
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1.0 - dice.mean()

        return self.prior_weight * dice_loss

    def forward(self, *args, **kwargs):
        """See compute_explanation_loss for the main entry point."""
        return self.compute_explanation_loss(*args, **kwargs)


# ═════════════════════════════════════════════════════════════════════
# Full Model — ArthritisMILModel
# ═════════════════════════════════════════════════════════════════════

class ArthritisMILModel(nn.Module):
    """
    Complete architecture: Anatomy-Aware Per-Joint MIL for
    three-way arthritis discrimination.

    Pipeline:
      1. Joint detection (YOLOv7 or pre-computed boxes)
      2. Per-joint ROI extraction → (B, N, 1, 224, 224)
      3. Foundation model backbone → (B, N, d)
      4. Multi-view fusion → (B, N_total, d)
      5. Gated attention MIL → bag_rep (B, d) + α_i (B, N)
      6. Dual-path classification → per-joint (B, N, C) + patient (B, C)
      7. XAI outputs → attention weights, per-joint logits, optional Grad-CAM

    Loss (total):
        L = w_j * CE(per_joint_logits, y_joint)
          + w_p * CE(patient_logits, y_patient)
          + w_a * L_anatomy(α_i, joint_groups, y_patient)
          + w_e * H(α_i)     [entropy regularization]

    where w_j, w_p, w_a, w_e are loss weights from ModelConfig.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stage 1: Detection (bypassed if boxes provided)
        self.detector = JointDetectionModule(config) \
            if config.detection_model != "none" else None

        # Stage 2: ROI extraction
        self.roi_extractor = ROIFeatureExtractor(config)

        # Stage 3: Feature extraction
        self.backbone = FoundationBackbone(config)

        # Stage 4: Multi-view fusion
        self.view_fusion = MultiViewFusion(config) \
            if len(config.input_views) > 1 else nn.Identity()

        # Stage 5: MIL aggregator
        self.mil_aggregator = GatedAttentionMIL(config)

        # Stage 6: Classification heads
        self.classifier = ArthritisClassificationHead(config)

        # Stage 7: Explanation module
        self.explanation = AnatomyExplanationModule(config)

    def forward(
        self,
        x: torch.Tensor,
        views: Optional[Dict[str, torch.Tensor]] = None,
        boxes: Optional[Dict[str, List[torch.Tensor]]] = None,
        joint_group_labels: Optional[torch.Tensor] = None,
        return_explanations: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) — primary view full X-ray
            views: Dict[view_name, (B, 1, H_v, W_v)] — additional views
            boxes: Dict[view_name, List[(N_i, 4)]] — pre-computed boxes
            joint_group_labels: (B, N_total) — anatomical group IDs for each joint
            return_explanations: compute attention/Grad-CAM for XAI

        Returns:
            dict with keys:
                per_joint_logits   (B, N, C)
                patient_logits     (B, C)
                attention_weights  (B, N)
                bag_representation (B, d)
                joint_features     (B, N, d)
                explanation_loss   scalar (0 if not enabled)
                (optionally) gradcam_heatmaps
        """
        # ----- Single-view wrapper -----
        if views is None:
            views = {"PA": x}
        view_names = list(views.keys())
        n_views = len(view_names)

        # ----- Process each view -----
        all_features = []
        all_masks = []
        view_features_dict = {}

        for v_name in view_names:
            img = views[v_name]  # (B, 1, H, W)
            B = img.shape[0]

            # 1. Detect joints
            if boxes is not None and v_name in boxes:
                v_boxes = boxes[v_name]
            elif self.detector is not None:
                v_boxes = self.detector(img)
            else:
                raise ValueError(
                    f"No boxes for view '{v_name}' and no detector configured."
                )

            # 2. Extract ROI crops
            rois, mask = self.roi_extractor(img, v_boxes)
            # rois: (B, N_v, 1, 224, 224), mask: (B, N_v)

            # 3. Backbone features
            features = self.backbone(rois)  # (B, N_v, d)

            view_features_dict[v_name] = features
            all_masks.append(mask)

        # 4. Multi-view fusion (or identity if single view)
        if n_views > 1:
            fused = self.view_fusion(view_features_dict, view_names)
        else:
            fused = view_features_dict[view_names[0]]  # (B, N, d)

        # Combined mask across all views
        full_mask = torch.cat(all_masks, dim=1) if n_views > 1 else all_masks[0]

        # 5. MIL aggregation
        bag_rep, attention_weights = self.mil_aggregator(fused, mask=full_mask)

        # 6. Classification
        per_joint_logits, patient_logits = self.classifier(fused, bag_rep)

        # 7. Explanation loss (if enabled)
        explanation_loss = torch.tensor(0.0, device=x.device)
        if self.explanation.use_prior_loss and joint_group_labels is not None:
            # Need per_joint_logits' argmax as "target disease" for prior loss
            # Better: use ground-truth patient labels. For inference, use prediction.
            target = patient_logits.argmax(dim=-1).detach()
            explanation_loss = self.explanation(
                attention_weights, joint_group_labels, target
            )

        output = {
            "per_joint_logits": per_joint_logits,
            "patient_logits": patient_logits,
            "attention_weights": attention_weights,
            "bag_representation": bag_rep,
            "joint_features": fused,
            "explanation_loss": explanation_loss,
        }

        return output


# ═════════════════════════════════════════════════════════════════════
# Loss Function
# ═════════════════════════════════════════════════════════════════════

def compute_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config
) -> Dict[str, torch.Tensor]:
    """
    Computes the multi-objective training loss.

    L_total = w_j * L_joint + w_p * L_patient + w_a * L_anatomy + w_e * L_entropy

    Args:
        output: model forward output dict
        batch: contains 'joint_labels' (B, N, C), 'patient_label' (B,)
               optionally 'joint_group_labels' (B, N)
        config: ModelConfig

    Returns:
        dict with 'loss', 'loss_joint', 'loss_patient', 'loss_anatomy', 'loss_entropy'
    """
    device = output['patient_logits'].device
    losses = {}

    # --- Per-joint classification loss ---
    if output['per_joint_logits'] is not None and 'joint_labels' in batch:
        logits = output['per_joint_logits']            # (B, N, C)
        labels = batch['joint_labels']                  # (B, N) with -100 padding
        if config.loss_type == "focal":
            losses['loss_joint'] = focal_loss(
                logits, labels, gamma=config.focal_gamma,
                alpha=config.focal_alpha, ignore_index=-100
            )
        else:
            losses['loss_joint'] = F.cross_entropy(
                logits.reshape(-1, config.n_classes),
                labels.reshape(-1), ignore_index=-100
            )
    else:
        losses['loss_joint'] = torch.tensor(0.0, device=device)

    # --- Patient-level classification loss ---
    if 'patient_label' in batch:
        logits = output['patient_logits']               # (B, C)
        labels = batch['patient_label']                  # (B,)
        if config.loss_type == "focal":
            losses['loss_patient'] = focal_loss(
                logits, labels, gamma=config.focal_gamma,
                alpha=config.focal_alpha
            )
        else:
            losses['loss_patient'] = F.cross_entropy(logits, labels)
    else:
        losses['loss_patient'] = torch.tensor(0.0, device=device)

    # --- Anatomy prior loss ---
    losses['loss_anatomy'] = output.get('explanation_loss', torch.tensor(0.0, device=device))

    # --- Entropy regularization on attention ---
    attn = output['attention_weights']                   # (B, N)
    attn_entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
    losses['loss_entropy'] = config.entropy_reg_weight * attn_entropy

    # --- Total ---
    losses['loss'] = (
        config.per_joint_loss_weight * losses['loss_joint']
        + config.patient_loss_weight * losses['loss_patient']
        + losses['loss_anatomy']
        + losses['loss_entropy']
    )

    return losses


# ═════════════════════════════════════════════════════════════════════
# Helper: Focal Loss
# ═════════════════════════════════════════════════════════════════════

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Handles class imbalance (RA >> PsA >> OA in prevalence).
    """
    ce_loss = F.cross_entropy(
        logits, targets, reduction='none', ignore_index=ignore_index
    )
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss

    if alpha is not None:
        alpha_t = alpha.gather(0, targets)
        focal = alpha_t * focal

    # Mask ignored positions
    mask = (targets != ignore_index).float()
    focal = focal * mask

    return focal.sum() / mask.sum().clamp(min=1)


# ═════════════════════════════════════════════════════════════════════
# Helper: X-ray Normalization
# ═════════════════════════════════════════════════════════════════════

def _normalize_xray(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image normalization: zero-mean, unit-variance.
    X-rays have no standard color distribution (unlike photographs),
    so per-image normalization is preferred over fixed mean/std.
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    mean = x_flat.mean(dim=-1, keepdim=True)
    std = x_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mean.view(B, C, 1, 1)) / std.view(B, C, 1, 1)
