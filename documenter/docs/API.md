# API Reference

## config.py

### `class ModelConfig`
Central dataclass containing every hyperparameter for the full detection→backbone→fusion→MIL→classification→XAI pipeline. No magic numbers in implementation code.

**Fields:**

| Field | Type | Default | Rationale |
|---|---|---|---|
| `in_channels` | `int` | `1` | X-ray input: single-channel grayscale |
| `img_size` | `int` | `224` | Joint ROI crop size (H == W) |
| `input_views` | `Tuple[str, ...]` | `("PA",)` | View names for view-embedding vocabulary |
| `detection_model` | `str` | `"yolov7"` | Detection mode: `"yolov7"` or `"none"` (pre-computed boxes) |
| `detection_input_size` | `int` | `640` | YOLOv7 inference resolution |
| `detection_confidence` | `float` | `0.5` | Min confidence for detected joints |
| `detection_iou_threshold` | `float` | `0.45` | NMS IoU threshold |
| `max_joints_per_view` | `int` | `30` | Max joint ROIs per view (padding length) |
| `backbone` | `str` | `"dinov2_vitl14"` | Foundation model: `"dinov2_vitl14"`, `"dinov2_vitb14"`, `"dinov2_vitg14"`, `"resnet152"`, `"tiny_debug"` |
| `d_model` | `int` | `1024` | Feature dimension; auto-set from backbone choice in `__post_init__` |
| `backbone_frozen` | `bool` | `True` | Freeze backbone weights during MIL training |
| `use_lora` | `bool` | `False` | LoRA on backbone attention projections |
| `lora_rank` | `int` | `8` | LoRA rank |
| `lora_alpha` | `int` | `16` | LoRA alpha scaling |
| `lora_target_modules` | `Tuple[str, ...]` | `("q","k","v","o")` | Which attention projections get LoRA adapters |
| `multi_view_fusion` | `str` | `"concat"` | Fusion: `"concat"`, `"cross_attention"`, `"none"` |
| `fusion_n_heads` | `int` | `4` | Heads in cross-attention fusion |
| `fusion_dropout` | `float` | `0.1` | Dropout in fusion |
| `use_view_embedding` | `bool` | `True` | Learned view-identity embedding per joint |
| `mil_hidden_dim` | `int` | `512` | Hidden dim of gated attention (V and U projections) |
| `mil_gated` | `bool` | `True` | Gated vs. simple softmax attention |
| `mil_dropout` | `float` | `0.2` | Dropout before attention scoring |
| `per_joint_classifier` | `bool` | `True` | Enable per-joint classification head |
| `per_joint_hidden` | `int` | `256` | Hidden dim of per-joint MLP |
| `per_joint_dropout` | `float` | `0.2` | Dropout in per-joint MLP |
| `n_classes` | `int` | `4` | Output classes: RA, PsA, OA, normal |
| `disease_specific_heads` | `bool` | `False` | `False`=unified softmax; `True`=4 binary heads |
| `patient_pooling` | `str` | `"attention"` | Per-joint→patient aggregation: `"attention"`, `"mean"`, `"max"` |
| `use_anatomy_prior_loss` | `bool` | `False` | Supervise attention with disease-joint priors |
| `anatomy_prior_loss_weight` | `float` | `0.1` | Coefficient for L_anatomy in total loss |
| `explanation_method` | `str` | `"attention"` | Primary explanation: `"attention"`, `"gradcam"`, `"both"` |
| `loss_type` | `str` | `"focal"` | `"ce"` or `"focal"` (γ=2.0) |
| `focal_gamma` | `float` | `2.0` | Focal loss focusing parameter |
| `focal_alpha` | `Optional[Tuple]` | `None` | Per-class weighting (None = uniform) |
| `per_joint_loss_weight` | `float` | `1.0` | Weight for per-joint CE loss |
| `patient_loss_weight` | `float` | `1.0` | Weight for patient-level CE loss |
| `entropy_reg_weight` | `float` | `0.01` | Entropy regularization on attention weights |
| `optimizer` | `str` | `"adamw"` | Optimizer type |
| `learning_rate` | `float` | `1e-4` | Peak learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `lr_scheduler` | `str` | `"cosine"` | `"cosine"`, `"plateau"`, or `"linear"` |
| `warmup_steps` | `int` | `500` | Linear warmup steps |
| `batch_size` | `int` | `16` | Batch size |
| `n_epochs` | `int` | `100` | Max training epochs |
| `gradient_clip_val` | `float` | `1.0` | Global norm clipping |
| `mixed_precision` | `str` | `"fp16"` | `"fp16"`, `"bf16"`, or `"no"` |
| `use_augmentation` | `bool` | `True` | Enable data augmentation |
| `augmentation_strength` | `float` | `1.0` | Scale factor (0.0=none, >1.0=aggressive) |
| `random_rotation_deg` | `int` | `10` | Max rotation degrees |
| `random_brightness_contrast` | `float` | `0.1` | Brightness/contrast jitter range |
| `random_gaussian_noise_std` | `float` | `0.02` | Gaussian noise std |
| `seed` | `int` | `42` | Random seed |
| `checkpoint_dir` | `str` | `"./checkpoints"` | Model save directory |
| `experiment_name` | `str` | `"arthritis_mil"` | Experiment name for logging |

**Constructor:** `ModelConfig(**kwargs)`

**Methods:**
- `__post_init__()` — validates n_classes, loss_type, multi_view_fusion, patient_pooling; raises `ValueError` if LoRA requested without `backbone_frozen=True`; auto-sets `d_model` from backbone choice.

---

## model.py

### `class ArthritisMILModel(nn.Module)`
Complete anatomy-aware per-joint MIL architecture for arthritis discrimination. Composes seven sub-modules into an end-to-end pipeline: detection → ROI extraction → backbone → multi-view fusion → MIL aggregation → dual-path classification → XAI.

**Constructor:** `ArthritisMILModel(config: ModelConfig)`

**Methods:**

- `forward(x=None, views=None, boxes=None, joint_group_labels=None, return_explanations=True, use_checkpoint=False) -> Dict[str, Tensor]`

  End-to-end forward pass.

  Args:
  - `x`: (B, 1, H_full, W_full) — primary view full X-ray (optional if `views` covers all)
  - `views`: `{view_name: (B, 1, H_v, W_v)}` — multi-view dict (optional)
  - `boxes`: `{view_name: list of (N_i, 4)}` — pre-computed joint boxes (optional; runs YOLOv7 if None)
  - `joint_group_labels`: (B, N_total) — anatomical group IDs for XAI loss
  - `return_explanations`: if True, compute explanation outputs
  - `use_checkpoint`: gradient checkpointing on backbone

  Returns dict with keys:
  - `per_joint_logits`: (B, N, C) or None
  - `patient_logits`: (B, C)
  - `attention_weights`: (B, N) — α_i, sum to 1 over valid joints
  - `bag_representation`: (B, d) — attention-weighted bag
  - `joint_features`: (B, N, d) — fused per-joint features
  - `explanation_loss`: scalar — Dice-based anatomy loss (0 if disabled)
  - `view_membership`: (B, N) — view index per joint

  Shape invariants:
  - Batch B ≥ 1; N ≤ sum of max_joints_per_view across views
  - dtype in {float32, bfloat16}; float16 supported via GradScaler

- `forward_with_checkpointing(rois: Tensor) -> Tensor`

  Run backbone with gradient checkpointing. Trades compute for memory.

  Args:
  - `rois`: (B, N, 1, H, W) — joint ROI crops

  Returns:
  - (B, N, d_model) — per-joint features

  Side effect: backbone intermediate activations are not stored.

### `count_params(model: nn.Module) -> None`
Print total and trainable parameter counts for any nn.Module.

---

## backbone.py

### `class BaseOperator(ABC, nn.Module)`
Abstract base class for the core feature-extraction operator. Subclasses must implement `forward(x) -> Tensor`.

### `class FoundationBackbone(BaseOperator)`
Shared (optionally frozen) backbone applied independently to each joint ROI. Supports DINOv2 ViT variants (B/L/G), ResNet-152, and a tiny_debug backbone for smoke testing. Grayscale input is replicated to 3 channels and normalized with ImageNet statistics.

**Constructor:** `FoundationBackbone(config: ModelConfig)`

**Methods:**
- `forward(x: Tensor) -> Tensor`
  Args: `x`: (B, N, 1, H, W) — joint ROI crops, H == W == img_size
  Returns: (B, N, d_model) — feature vectors
  Shape: flattens B×N, runs backbone, restores dims.

### `class LoRALinear(nn.Module)`
Low-Rank Adaptation (Hu et al., 2021) of a linear layer: W' = W + (B·A)·(α/r). Applied to backbone attention projections when `use_lora=True`.

**Constructor:** `LoRALinear(in_features, out_features, rank=8, alpha=16)`

**Methods:**
- `forward(x: Tensor) -> Tensor`
  Args: `x`: (..., in_features)
  Returns: (..., out_features) — base linear + LoRA update

---

## layers.py

### `_normalize_xray(x: Tensor) -> Tensor`
Per-image normalization: zero-mean, unit-variance. Removes acquisition-level variance in X-ray brightness/contrast while preserving anatomical contrast.

Args: `x`: (B, C, H, W) — raw X-ray
Returns: (B, C, H, W) — normalized

### `class JointDetectionModule(nn.Module)`
Localizes individual joints in a full X-ray image using YOLOv7 (loaded via torch.hub). Designed for offline use — runs at inference or as a pre-processing step. Bypassed when `detection_model="none"`.

**Constructor:** `JointDetectionModule(config: ModelConfig)`

**Methods:**
- `forward(x: Tensor) -> List[Tensor]`
  Args: `x`: (B, 1, H_full, W_full) — full X-ray
  Returns: List of (N_i, 4) tensors, one per batch item — (x1, y1, x2, y2) in pixel coordinates

### `class ROIFeatureExtractor(nn.Module)`
Crops joint ROIs from full X-ray using detected/annotated boxes. Uses `torchvision.ops.roi_align` for differentiable, batched cropping. Pads to `max_joints_per_view` with a validity mask.

**Constructor:** `ROIFeatureExtractor(config: ModelConfig)`

**Methods:**
- `forward(img: Tensor, boxes: List[Tensor]) -> Tuple[Tensor, Tensor]`
  Args: `img`: (B, 1, H_full, W_full), `boxes`: list of (N_i, 4)
  Returns: `rois`: (B, N_max, 1, H_roi, W_roi), `mask`: (B, N_max) — 1 for real joints, 0 for padding

### `class MultiViewFusion(nn.Module)`
Fuses joint-level features from multiple X-ray views. Adds view-specific learned embeddings, concatenates across views, and optionally applies cross-attention.

**Constructor:** `MultiViewFusion(config: ModelConfig)`

**Methods:**
- `forward(view_features: Dict[str, Tensor], view_names: List[str]) -> Tuple[Tensor, Tensor]`
  Args: `view_features`: `{view_name: (B, N_v, d)}`, `view_names`: ordered key list
  Returns: `fused`: (B, N_total, d), `view_membership`: (B, N_total) — view index per joint

### `class GatedAttentionMIL(nn.Module)`
Gated attention mechanism for multi-instance pooling (Ilse et al., 2018). Computes content-based attention weights α_i for each joint via tanh×sigmoid gating, then aggregates to a bag representation via α-weighted sum.

**Constructor:** `GatedAttentionMIL(config: ModelConfig)`

**Methods:**
- `forward(x: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]`
  Args: `x`: (B, N, d) — joint features, `mask`: (B, N) — 1 for valid, 0 for padding
  Returns: `bag_rep`: (B, d), `alpha`: (B, N) — attention weights, sum to 1 over valid joints

  bf16/fp16 safety: softmax computed in float32 before casting to input dtype.

---

## heads.py

### `class ArthritisClassificationHead(nn.Module)`
Dual-path classification head. Produces per-joint logits (classify each joint independently) and patient-level logits (from the bag representation). Supports both unified softmax and disease-specific binary heads.

**Constructor:** `ArthritisClassificationHead(config: ModelConfig)`

**Methods:**
- `forward(joint_features: Tensor, bag_rep: Tensor) -> Tuple[Optional[Tensor], Tensor]`
  Args: `joint_features`: (B, N, d), `bag_rep`: (B, d)
  Returns: `per_joint_logits`: (B, N, C) or None, `patient_logits`: (B, C)

  Shape invariants:
  - If `per_joint_classifier=False`, per_joint_logits is None
  - If `disease_specific_heads=True`, patient_logits output is same shape (B, C) but from 4 independent binary classifiers

---

## explanation.py

### `class AnatomyExplanationModule(nn.Module)`
Produces clinically interpretable explanations and an optional anatomy-guided Dice loss.

Anatomy priors (from clinical literature):
- RA: MCP 2-5, PIP 2-5, wrist (18 joint groups: DIP[0-3], PIP[4-7], MCP[8-11], wrist[12], CMC[13], carpals[14-17])
- PsA: DIP 2-5, PIP 2-5
- OA: DIP 2-5, CMC/thumb
- Normal: uniform over all groups

**Constructor:** `AnatomyExplanationModule(config: ModelConfig)`

**Methods:**
- `compute_explanation_loss(attention_weights, joint_group_labels, target_disease) -> Tensor`
  Args:
  - `attention_weights`: (B, N) — MIL α_i
  - `joint_group_labels`: (B, N) — anatomical group ID (0..17) or -1 for unknown
  - `target_disease`: (B,) — ground-truth disease class index

  Returns: scalar Dice loss (0 if `use_prior_loss=False`)

  Fully vectorized (no Python loops), compatible with torch.compile.

- `get_top_k_joints(attention_weights, joint_names=None, k=5) -> list`
  Return top-k joints by attention weight for interpretability.
  Args: `attention_weights`: (B, N), `joint_names`: optional list of N strings
  Returns: List of `(joint_index, weight, name)` tuples per batch item

  Decorated with `@torch.no_grad()`.

- `forward(attention_weights, joint_group_labels, target_disease) -> Tensor`
  Alias for `compute_explanation_loss`.

---

## losses.py

### `focal_loss(logits, targets, gamma=2.0, alpha=None, ignore_index=-100) -> Tensor`
Multi-class focal loss (Lin et al., 2017): FL(p_t) = -α_t · (1-p_t)^γ · log(p_t).

Args:
- `logits`: (..., C) — raw scores
- `targets`: (...,) — ground-truth class indices
- `gamma`: focusing parameter ≥ 0 (γ=0 → standard CE)
- `alpha`: (C,) — optional per-class weighting
- `ignore_index`: targets equal to this are masked out (loss = 0)

Returns: scalar, averaged over non-ignored positions

bf16/fp16 safety: accepts any input dtype; cross_entropy internally casts to float32 for softmax.

### `compute_loss(output, batch, config) -> Dict[str, Tensor]`
Compute the multi-objective training loss.

L_total = w_j·L_joint + w_p·L_patient + w_a·L_anatomy + w_e·H(α)

Args:
- `output`: dict from model forward (`per_joint_logits`, `patient_logits`, `attention_weights`, `explanation_loss`)
- `batch`: dict (`joint_labels`: (B, N), `patient_label`: (B,))
- `config`: ModelConfig

Returns: dict with keys `loss`, `loss_joint`, `loss_patient`, `loss_anatomy`, `loss_entropy`. Each is a scalar tensor.
