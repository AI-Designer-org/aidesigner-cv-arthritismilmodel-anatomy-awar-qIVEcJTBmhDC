"""
ModelConfig: Anatomy-Aware Per-Joint MIL for Arthritis Discrimination
Domain: Computer Vision (primary) + Scientific ML (secondary)
=====================================================================

This configuration governs the full architecture described in this design.
Every hyperparameter appears here — no magic numbers in implementation code.

Usage:
    from dataclasses import dataclass, field
    from typing import Optional, List, Tuple

    cfg = ModelConfig()
    model = ArthritisMILModel(cfg)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


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
    """Foundation model: "dinov2_vitl14" | "dinov2_vitb14" | "resnet152" | "orthofoundation"."""

    d_model: int = 1024
    """Feature dimension. DINOv2 ViT-L/14 → 1024; ViT-B/14 → 768."""

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

    backbone_normalization: str = "imagenet"
    """Normalization stats for backbone input: "imagenet" | "xray_standard"."""

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

    mil_use_residual: bool = True
    """Residual connection: bag_rep = f(features) + bag_rep. Stabilizes deep aggregator."""

    # ═══════════════════════════════════════════════════════════════════
    # PER-JOINT CLASSIFIER
    # ═══════════════════════════════════════════════════════════════════

    per_joint_classifier: bool = True
    """Enable per-joint classification head. Provides fine-grained joint-level predictions."""

    per_joint_hidden: int = 256
    """Hidden dimension of the per-joint MLP."""

    per_joint_dropout: float = 0.2

    # ═══════════════════════════════════════════════════════════════════
    # CLASSIFICATION HEAD
    # ═══════════════════════════════════════════════════════════════════

    n_classes: int = 4
    """Output classes: RA, PsA, OA, normal."""

    disease_specific_heads: bool = False
    """False = unified softmax head over 4 classes.
    True = 3 binary disease heads + 1 normal head.
    Rationale: binary heads may handle class imbalance better but lose
    competitive normal-vs-disease calibration. Start with unified head."""

    patient_pooling: str = "attention"
    """How to aggregate per-joint logits to patient level:
    "attention" → attention-weighted mean of per-joint logits
    "mean"     → uniform average
    "max"      → max over joints
    Used only when per_joint_classifier=True and we need a
    per-joint-derived patient prediction."""

    # ═══════════════════════════════════════════════════════════════════
    # ANATOMY-GUIDED EXPLANATION (XAI)
    # ═══════════════════════════════════════════════════════════════════

    use_anatomy_prior_loss: bool = False
    """Supervise attention weights with disease-joint anatomical priors.
    When enabled, adds L_anatomy = Dice(attention_weights, prior_mask) to loss.
    WARNING: requires structured anatomical priors (joint-group -> disease mapping)."""

    anatomy_prior_loss_weight: float = 0.1
    """Coefficient for L_anatomy in total loss."""

    anatomy_prior_type: str = "attention_dice"
    """"attention_dice" → Dice between attention α_i and disease-joint prior mask.
    "gradcam_dice"  → Dice between Grad-CAM heatmap and prior mask (not implemented)."""

    explanation_method: str = "attention"
    """Primary explanation method exposed to clinician:
    "attention" → per-joint attention weights (lightweight, inherent)
    "gradcam"   → Grad-CAM on top-k joints (more precise spatially)
    "both"      → provide both for comparison."""

    # ═══════════════════════════════════════════════════════════════════
    # LOSS FUNCTION
    # ═══════════════════════════════════════════════════════════════════

    loss_type: str = "focal"
    """"ce"    → standard cross-entropy
    "focal" → focal loss (γ=2.0 handles class imbalance)
    "asymmetric" → asymmetric focal loss (different γ per class)."""

    focal_gamma: float = 2.0
    """Focusing parameter for focal loss. Higher = more focus on hard examples."""

    focal_alpha: Optional[Tuple[float, ...]] = None
    """Per-class weighting for focal loss. E.g., (0.25, 0.25, 0.25, 0.25).
    If None, uniform weighting is used."""

    per_joint_loss_weight: float = 1.0
    """Weight for per-joint classification loss in total loss."""

    patient_loss_weight: float = 1.0
    """Weight for patient-level classification loss."""

    entropy_reg_weight: float = 0.01
    """Entropy regularization on attention weights. Encourages sharper or
    more uniform attention (lower → sharper peaks, higher → more uniform)."""

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING CONFIG
    # ═══════════════════════════════════════════════════════════════════

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    """"cosine"  → cosine annealing with warm restarts
    "plateau" → ReduceLROnPlateau
    "linear"  → linear decay."""

    warmup_steps: int = 500
    batch_size: int = 16
    n_epochs: int = 100
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 15
    mixed_precision: str = "fp16"
    """Mixed precision: "fp16" | "bf16" | "no"."""

    # ═══════════════════════════════════════════════════════════════════
    # DATA AUGMENTATION
    # ═══════════════════════════════════════════════════════════════════

    use_augmentation: bool = True
    augmentation_strength: float = 1.0
    """Scale factor: 0.0 = no augmentation, 1.0 = full, >1.0 = aggressive."""

    random_rotation_deg: int = 10
    random_brightness_contrast: float = 0.1
    random_gaussian_noise_std: float = 0.02
    random_elastic_transform: bool = False
    """Elastic transforms may distort anatomical structure. Use with caution."""

    random_crop_scale: Tuple[float, float] = (0.9, 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════════════

    eval_per_joint: bool = True
    """Compute per-joint macro-F1, weighted-F1, confusion matrix."""

    eval_patient_level: bool = True
    """Compute patient-level accuracy, AUROC (one-vs-rest), AUPRC."""

    eval_explanation: bool = True
    """Compute explanation Dice between attention weights and anatomical priors,
    plus Deletion/Insertion scores for faithfulness."""

    explanation_dice_threshold: float = 0.5
    """Threshold for binarizing attention weights before Dice computation."""

    eval_calibration: bool = True
    """Expected Calibration Error (ECE) per disease class."""

    eval_robustness: bool = True
    """Stratified performance by severity, joint site, view."""

    # ═══════════════════════════════════════════════════════════════════
    # MODEL IO
    # ═══════════════════════════════════════════════════════════════════

    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "arthritis_mil"
    seed: int = 42

    def __post_init__(self):
        """Validation and automatic adjustments."""
        assert self.n_classes in (3, 4), "Must be 3 (RA/PsA/OA) or 4 (+normal)"
        assert self.loss_type in ("ce", "focal", "asymmetric")
        assert self.multi_view_fusion in ("concat", "cross_attention", "none")
        assert self.patient_pooling in ("attention", "mean", "max")

        if self.use_lora and not self.backbone_frozen:
            raise ValueError("LoRA requires backbone_frozen=True (LoRA adapters are "
                             "trained while original weights stay frozen).")

        if self.backbone == "dinov2_vitb14":
            self.d_model = 768
        elif self.backbone == "dinov2_vitl14":
            self.d_model = 1024
        elif self.backbone == "dinov2_vitg14":
            self.d_model = 1536
        elif self.backbone == "resnet152":
            self.d_model = 2048
        # "orthofoundation" d_model depends on release — default to 1024
