"""
ModelConfig: Anatomy-Aware Per-Joint MIL for Arthritis Discrimination
Domain: Computer Vision (primary) + Scientific ML (secondary)

All hyperparameters in one place — no magic numbers in implementation code.
Covers the full pipeline: detection → ROI extraction → backbone → fusion → MIL → classification → XAI.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # ═══════════════════════════════════════════════════════════════════
    # INPUT SPECIFICATION
    # ═══════════════════════════════════════════════════════════════════

    in_channels: int = 1
    """X-ray input: single-channel grayscale."""

    img_size: int = 224
    """Resize joint ROI crops to this spatial size (H == W)."""

    input_views: Tuple[str, ...] = ("PA",)
    """Expected view names, e.g. ("PA", "oblique", "lateral").
    Controls the view-embedding vocabulary for multi-view fusion."""

    # ═══════════════════════════════════════════════════════════════════
    # JOINT DETECTION MODULE
    # ═══════════════════════════════════════════════════════════════════

    detection_model: str = "yolov7"
    """Detection backbone: "yolov7" | "none" (if coordinates provided)."""

    detection_input_size: int = 640
    """YOLOv7 inference resolution for full X-ray."""

    detection_confidence: float = 0.5
    """Minimum confidence threshold for detected joints."""

    detection_iou_threshold: float = 0.45
    """NMS IoU threshold."""

    max_joints_per_view: int = 30
    """Maximum joint ROIs per view (padding length)."""

    # ═══════════════════════════════════════════════════════════════════
    # FOUNDATION MODEL BACKBONE (per-joint feature extractor)
    # ═══════════════════════════════════════════════════════════════════

    backbone: str = "dinov2_vitl14"
    """Foundation model:
       "dinov2_vitl14" | "dinov2_vitb14" | "dinov2_vitg14"
       | "resnet152" | "tiny_debug" (for smoke testing)."""

    d_model: int = 1024
    """Feature dimension. DINOv2 ViT-L/14 → 1024; ViT-B/14 → 768; ViT-G/14 → 1536."""

    backbone_frozen: bool = True
    """Freeze backbone weights during MIL training. Saves compute, prevents overfitting."""

    use_lora: bool = False
    """Parameter-efficient fine-tuning via LoRA on backbone attention projections."""

    lora_rank: int = 8
    """LoRA rank hyperparameter."""

    lora_alpha: int = 16
    """LoRA alpha scaling."""

    lora_target_modules: Tuple[str, ...] = ("q", "k", "v", "o")
    """Which attention projections get LoRA adapters."""

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-VIEW FUSION
    # ═══════════════════════════════════════════════════════════════════

    multi_view_fusion: str = "concat"
    """Fusion strategy: "concat" | "cross_attention" | "none"."""

    fusion_n_heads: int = 4
    """Heads in cross-attention fusion layer."""

    fusion_dropout: float = 0.1

    use_view_embedding: bool = True
    """Add learned view-identity embedding to each joint feature before fusion.
    Inductive bias: allows model to distinguish joints from different anatomical
    perspectives even after feature extraction."""

    # ═══════════════════════════════════════════════════════════════════
    # GATED ATTENTION MIL AGGREGATOR
    # ═══════════════════════════════════════════════════════════════════

    mil_hidden_dim: int = 512
    """Hidden dimension of the gated attention network (V and U projections)."""

    mil_gated: bool = True
    """Use gated attention (Ilse et al. 2018). False = simple softmax attention."""

    mil_dropout: float = 0.2
    """Dropout applied before attention scoring."""

    # ═══════════════════════════════════════════════════════════════════
    # PER-JOINT CLASSIFIER
    # ═══════════════════════════════════════════════════════════════════

    per_joint_classifier: bool = True
    """Enable per-joint classification head."""

    per_joint_hidden: int = 256
    """Hidden dim of per-joint MLP."""

    per_joint_dropout: float = 0.2

    # ═══════════════════════════════════════════════════════════════════
    # CLASSIFICATION HEAD
    # ═══════════════════════════════════════════════════════════════════

    n_classes: int = 4
    """Output classes: RA, PsA, OA, normal."""

    disease_specific_heads: bool = False
    """False = unified softmax head over 4 classes.
    True = 4 binary heads (one per disease). Start with unified."""

    patient_pooling: str = "attention"
    """How to aggregate per-joint logits → patient level:
       "attention" | "mean" | "max"."""

    # ═══════════════════════════════════════════════════════════════════
    # ANATOMY-GUIDED EXPLANATION (XAI)
    # ═══════════════════════════════════════════════════════════════════

    use_anatomy_prior_loss: bool = False
    """Supervise attention weights with disease-joint anatomical priors.
    Adds L_anatomy = Dice(attention_weights, prior_mask) to loss."""

    anatomy_prior_loss_weight: float = 0.1
    """Coefficient for L_anatomy in total loss."""

    explanation_method: str = "attention"
    """Primary explanation: "attention" | "gradcam" | "both"."""

    # ═══════════════════════════════════════════════════════════════════
    # LOSS FUNCTION
    # ═══════════════════════════════════════════════════════════════════

    loss_type: str = "focal"
    """"ce" → cross-entropy | "focal" → focal loss (γ=2.0)."""

    focal_gamma: float = 2.0
    """Focusing parameter for focal loss."""

    focal_alpha: Optional[Tuple[float, ...]] = None
    """Per-class weighting. None = uniform."""

    per_joint_loss_weight: float = 1.0
    """Weight for per-joint classification loss."""

    patient_loss_weight: float = 1.0
    """Weight for patient-level classification loss."""

    entropy_reg_weight: float = 0.01
    """Entropy regularization on attention weights.
    Encourages sharper (low weight) or more uniform (high weight) attention."""

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    warmup_steps: int = 500
    batch_size: int = 16
    n_epochs: int = 100
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 15
    mixed_precision: str = "fp16"

    # ═══════════════════════════════════════════════════════════════════
    # DATA AUGMENTATION
    # ═══════════════════════════════════════════════════════════════════

    use_augmentation: bool = True
    augmentation_strength: float = 1.0
    random_rotation_deg: int = 10
    random_brightness_contrast: float = 0.1
    random_gaussian_noise_std: float = 0.02

    # ═══════════════════════════════════════════════════════════════════
    # MODEL IO
    # ═══════════════════════════════════════════════════════════════════

    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "arthritis_mil"
    seed: int = 42

    def __post_init__(self):
        """Validate config and set d_model based on backbone choice."""
        assert self.n_classes in (3, 4), "Must be 3 (RA/PsA/OA) or 4 (+normal)"
        assert self.loss_type in ("ce", "focal"), f"Unknown loss: {self.loss_type}"
        assert self.multi_view_fusion in ("concat", "cross_attention", "none"), \
            f"Unknown fusion: {self.multi_view_fusion}"
        assert self.patient_pooling in ("attention", "mean", "max"), \
            f"Unknown pooling: {self.patient_pooling}"

        if self.use_lora and not self.backbone_frozen:
            raise ValueError("LoRA requires backbone_frozen=True (LoRA adapters are "
                             "trained while original weights stay frozen).")

        # Auto-set d_model based on backbone
        if self.backbone == "dinov2_vitb14":
            self.d_model = 768
        elif self.backbone == "dinov2_vitl14":
            self.d_model = 1024
        elif self.backbone == "dinov2_vitg14":
            self.d_model = 1536
        elif self.backbone == "resnet152":
            self.d_model = 2048
        elif self.backbone == "tiny_debug":
            self.d_model = 64
