# Architecture

## 1. Motivation

X-ray imaging remains the primary modality for assessing structural joint damage in rheumatic diseases. Current deep learning systems are designed as *single-disease specialists* — they quantify Sharp/van der Heijde scores for rheumatoid arthritis (Hügle et al., EULAR 2025; autoscoRA, *Arthritis Rheumatol* 2026), grade Kellgren-Lawrence severity for osteoarthritis (EfficientNet, ConvNeXt, ViT), or score erosion patterns in psoriatic arthritis (PsA ViT, MLMI 2024). A systematic review of the field (Semin Arthritis Rheum, 2025) reports that 88% of studies focus on RA, 22% on OA, and only 9% on PsA — and **none jointly discriminate all three at per-joint granularity**.

This matters clinically because RA, PsA, and OA have overlapping radiographic presentations but fundamentally different treatment pathways. A unified model that produces per-joint probability distributions over {RA, PsA, OA, healthy} could support differential diagnosis, particularly in early disease where features are subtle.

The architecture tested here makes three specific hypotheses:

1. **Multi-instance learning (MIL) naturally maps to per-joint annotations** — each joint is an "instance" in a "bag" (the patient), and the model learns which joints are most informative for each disease.
2. **A frozen foundation model backbone (DINOv2) provides sufficient feature quality** despite the ImageNet-to-radiology domain gap, avoiding overfitting on limited clinical data.
3. **Gated attention weights double as clinically interpretable explanations** — when aligned with disease-specific anatomical priors (RA→MCP/PIP, OA→DIP, PsA→PIP/DIP) via a Dice loss, attention should highlight joints a rheumatologist would consider relevant.

## 2. At a glance

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        PATIENT-LEVEL OUTPUT (RA / PsA / OA / normal)                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
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
   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
   │  Per-Joint Head      │   │  Patient-Level Head  │   │  XAI Module          │
   │  (B, N, 4)           │   │  (B, 4)               │   │  • α_i importance    │
   │  Classify each       │   │  Unified softmax or   │   │  • Per-joint logits  │
   │  joint independently │   │  4 disease-specific   │   │  • Anatomical Dice   │
   │                      │   │  binary heads         │   │  • Grad-CAM (opt.)   │
   └──────────┬───────────┘   └───────────┬──────────┘   └──────────────────────┘
              │                           ▲
              └───────────────┬───────────┘
                              │
               ┌──────────────┴──────────────┐
               │  Gated Attention MIL Pool   │
               │  z = Σ α_i · h_i            │
               │  α_i = softmax(w · gate)    │
               │  (B, N) attention weights   │
               └──────────────┬──────────────┘
                              │
               ┌──────────────┴──────────────┐
               │  Multi-View Fusion          │
               │  • View embeddings added    │
               │  • Concatenate across views │
               │  • Optional cross-attention │
               └──────────────┬──────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │ PA View            │ Oblique View       │ Lateral View ...
         ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Detection   │      │ Detection   │      │ Detection   │
   │ YOLOv7      │      │ YOLOv7      │      │ YOLOv7      │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ ROI Extract │      │ ROI Extract │      │ ROI Extract │
   │ (N₁ joints) │      │ (N₂ joints) │      │ (N₃ joints) │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Foundation  │      │ Foundation  │      │ Foundation  │
   │ Backbone    │      │ Backbone    │      │ Backbone    │
   │ DINOv2      │      │ DINOv2      │      │ DINOv2      │
   │ (frozen)    │      │ (frozen)    │      │ (frozen)    │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          ▼                    ▼                    ▼
   (B, N₁, d)          (B, N₂, d)           (B, N₃, d)
          │                    │                    │
          └────────────────────┴────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   INPUT IMAGE     │
                    │  (B, 1, H, W)    │
                    │  Multi-view X-ray │
                    └───────────────────┘
```

| Property | Value |
|---|---|
| Parameter count (default config, frozen backbone) | ~1–2M trainable (DINOv2 ViT-L/14 frozen) |
| Parameter count (with LoRA) | ~3–4M trainable |
| Time complexity (inference) | O(N_joints · d²) for backbone forward + O(N_joints · L) for MIL, where L = mil_hidden_dim |
| Space complexity | O(B · N_max · d) for joint features + O(B · N_max) for attention weights |
| Hardware requirements (recommended) | 1× NVIDIA RTX 4090 (24 GB) or A100; bf16 mixed precision |
| Custom kernels required | None — all operations are standard PyTorch ops |

## 3. The core component

### 3.1 Intuition

The architecture treats each patient as a "bag" of joints. A YOLOv7 detector (or pre-computed annotations) locates individual joints in the X-ray. Each joint ROI is cropped to a fixed 224×224 patch and passed through a shared, frozen DINOv2 ViT-L/14 backbone to produce a feature vector. These per-joint features form the bag.

The key question MIL answers is: *which joints in this bag are most informative for the diagnosis?* A gated attention mechanism learns a non-linear importance score α_i for each joint. The weighted sum of joint features (the "bag representation") captures the patient-level signature. Two classifiers operate on these representations: one classifies each joint independently (fine-grained disease localization), and the other classifies the bag representation (patient-level diagnosis).

The attention weights α_i double as explanations — they show which joints the model considered most important. An optional anatomy-guided loss encourages α_i to align with disease-specific joint groups (e.g., RA typically affects MCP and PIP joints but spares DIP).

### 3.2 Equations

**Gated attention (Ilse et al., 2018):**

Let h_i ∈ ℝᵈ be the feature vector for joint i.

```
a_i = tanh(V · h_i)                content embedding
b_i = sigmoid(U · h_i)             gate (learned importance query)
α_i = softmax(wᵀ · (a_i ⊙ b_i))   attention weight over N joints
z = Σ α_i · h_i                    bag representation
```

where V, U ∈ ℝ^{L×d}, w ∈ ℝ^{L} are learned, L = mil_hidden_dim.

**Multi-objective loss:**

```
L_total = w_j · CE_joint(y_j, ŷ_j)
        + w_p · CE_patient(y_p, ŷ_p)
        + w_a · Dice(α, prior_mask)
        + w_e · H(α)
```

The Dice term for anatomy alignment, given attention weights α and a disease-specific prior mask m:

```
Dice = (2 · Σ(α · m) + ε) / (Σα + Σm + ε)
L_anatomy = 1 − Dice
```

**Focal loss (Lin et al., 2017)** replaces standard CE for class imbalance:

```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
```

where γ = 2.0 focuses the model on hard-to-classify examples.

### 3.3 Reference implementation walk-through

The forward pass in `coder/model.py::ArthritisMILModel.forward`:

```python
# For each view: detection → ROI extraction → backbone
for v_name in view_names:
    img = views[v_name]                    # (B, 1, H_v, W_v)
    v_boxes = self.detector(img)           # List[(N_i, 4)]
    rois, mask = self.roi_extractor(img, v_boxes)  # (B, N_v, 1, 224, 224)
    features = self.backbone(rois)         # (B, N_v, d_model)

# Multi-view fusion (concatenate with view embeddings)
fused, membership = self.view_fusion(view_features, view_names)  # (B, N_total, d)

# Gated attention MIL
bag_rep, attention_weights = self.mil_aggregator(fused, mask=full_mask)  # (B, d), (B, N)

# Dual-path classification
per_joint_logits, patient_logits = self.classifier(fused, bag_rep)  # (B, N, C), (B, C)
```

Shape at each step for a single view with default config (d_model=1024, max_joints=30):

| Step | Shape | Description |
|---|---|---|
| Input X-ray | (B, 1, 640, 640) | Full grayscale image (detection resolution) |
| Detection boxes | List[(N_i, 4)] | Per-image, variable N_i (padded to N_max=30) |
| ROI crops | (B, 30, 1, 224, 224) | Fixed-size normalized crops |
| Backbone features | (B, 30, 1024) | DINOv2 ViT-L/14 [CLS] tokens |
| Multi-view fused | (B, 30, 1024) | Or (B, 60, 1024) for 2 views |
| Attention weights | (B, 30) | α_i, sum to 1 over valid (non-padded) joints |
| Bag representation | (B, 1024) | Σ α_i · h_i, attention-weighted |
| Per-joint logits | (B, 30, 4) | Per-joint {RA, PsA, OA, normal} |
| Patient logits | (B, 4) | Patient-level diagnosis |

## 4. Tensor shape evolution

Default config: B=16, N_max=30, single view, DINOv2 ViT-L/14 (d=1024), 4 classes.

| Stage | Shape | Learnable | Notes |
|---|---|---|---|
| Input X-ray | (16, 1, 640, 640) | ✗ | Full-resolution grayscale |
| X-ray normalization | (16, 1, 640, 640) | ✗ | Per-image zero-mean unit-variance |
| YOLOv7 detection | List of (N_i, 4) | ✗ (frozen) | Boxes filtered by conf=0.5 + NMS |
| ROI crop + pad | (16, 30, 1, 224, 224) | ✗ | torchvision.ops.roi_align |
| DINOv2 backbone | (16, 30, 1024) | ✗ (frozen) | [CLS] token per joint; LoRA optional |
| View embedding | (16, 30, 1024) | ✓ (if used) | Added per-joint; no change |
| Multi-view fusion | (16, 30×V, 1024) | ✓ (attn) | V = number of views |
| Gated MIL (pre-norm) | (16, 30, 1024) | ✗ | LayerNorm applied |
| Gated attention scores | (16, 30) | ✓ | logits before softmax |
| Attention weights α_i | (16, 30) | ✗ | softmax, sum=1 over valid joints |
| Bag representation | (16, 1024) | ✗ | α-weighted sum of features |
| Per-joint MLP | (16, 30, 256) → (16, 30, 4) | ✓ | LayerNorm → Linear → GELU → Dropout → Linear |
| Patient MLP | (16, 512) → (16, 4) | ✓ | LayerNorm → Linear(d, d/2) → GELU → Dropout → Linear |
| Explanation loss | scalar | ✗ | Dice(α, prior_mask), 0 if disabled |
| Attention entropy | scalar | ✗ | H(α) = −Σ(α · log α) |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| Detection-based MIL (YOLOv7 → ROI) | Tile-sampling MIL (grid patches) | Per-joint annotations are available; detection preserves anatomical identity and handles variable joint counts | Error propagation from detector; YOLOv7 must be trained or run offline |
| Frozen DINOv2 backbone | Full fine-tuning; scratch-trained CNN | Prevents overfitting on limited clinical data (1–2M vs. 300M trainable params); proven self-supervised features transfer to X-ray | Domain gap (ImageNet → radiology); may miss X-ray-specific textures; cannot adapt features to arthritis-specific patterns |
| Gated attention (tanh × sigmoid) | Simple softmax attention | Non-linear gating captures the non-linear visual signatures of disease patterns (erosions, JSN) | 2× parameters in attention network; marginal compute overhead (~5ms per batch) |
| Pre-norm (LayerNorm before attention) | Post-norm | Prevents vanishing gradients in tanh/sigmoid saturating nonlinearities; stabilizes training with frozen high-d features | LayerNorm adds ~2% compute per forward pass |
| Dual-path classification (per-joint + patient) | Single patient-level head | Matches clinical reasoning: a single affected joint indicates disease, but the pattern across joints confirms the type | Joint-level labels required for per-joint head; multi-objective loss needs tuning |
| View embeddings for multi-view | Raw concatenation; no view identity | Lets model distinguish PA MCP from oblique MCP; learnable view-specific transformations | Extra parameters (V × d); requires view labels at inference |
| Anatomy prior loss (optional Dice) | No explanation supervision; Grad-CAM only | Encodes clinical knowledge (RA→MCP/PIP, OA→DIP, PsA→PIP/DIP) without per-image ROI annotations | Prior loss weight must be tuned; assumes prior accuracy — if priors are wrong, explanations degrade |
| Focal loss (γ=2.0) | Standard cross-entropy; class-weighted CE | Addresses expected class imbalance (RA > OA > PsA) by focusing on hard examples | γ=2.0 is conservative; may under-focus on very rare classes — adjust γ or switch to asymmetric focal |
| Per-image normalization | Dataset-level normalization (mean/std over all images) | X-rays have inconsistent brightness/contrast across acquisitions; per-image norm removes acquisition variance while preserving anatomical contrast | Loses absolute intensity information (though X-ray intensity is not diagnostically absolute) |
| Padding to N_max (30) with mask | Variable-size batches (no padding) | Enables standard batching, avoids dynamic graphs, works with PyTorch DataLoader | 5% attention mass on padded positions would indicate defective masking (verified: zero mass) |

## 6. Domain-specific considerations

### Computer Vision (primary domain)

| Concern | How the architecture addresses it |
|---|---|
| **Spatial inductive bias** | YOLOv7 provides absolute pixel coordinates; ROI cropping discards absolute position but preserves relative anatomy within the joint. Joint-group labels (MCP-2, PIP-3) are preserved as metadata for anatomical attribution. |
| **Scale / resolution** | Detection at 640×640 (YOLOv7 default). ROI crops resized to 224×224 for DINOv2. No explicit multi-scale processing — scale cues (large vs. small erosion) are lost in the fixed-size crop. |
| **Dense vs. global attention** | Hybrid: per-joint features are extracted from local ROIs (dense within each joint) and aggregated globally via permutation-invariant MIL. This matches the clinical observation that disease can be diagnosed from either a single joint or a pattern across joints. |
| **Rotation invariance** | Data augmentation (random rotation ±10°) provides mild invariance. Hand X-rays are well-aligned in clinical protocols; aggressive rotation is not needed. |
| **Permutation invariance** | The MIL aggregator is deliberately permutation-invariant: joints have a fixed anatomical order, but disease patterns are defined by which *group* is affected, not by spatial configuration. |

### Scientific ML / Clinical (secondary domain)

| Concern | How the architecture addresses it |
|---|---|
| **Clinical validation** | Explanation Dice scores provide a quantitative bridge between model behavior and clinical knowledge. The XAI module outputs attention weights, top-k joints, and per-joint logits that a rheumatologist can review. |
| **Stratified evaluation** | Metrics can be stratified by disease severity, joint site (MCP/PIP/DIP/wrist), and image view. Stratification infrastructure is in the ablation runner but requires clinical metadata. |
| **Data provenance** | Architecture assumes anonymized, IRB-approved data. All preprocessing is deterministic and auditable. Splits are patient-level (not image-level) to prevent leakage. |
| **Calibration** | Temperature scaling analysis is planned (ECE per disease class) but not yet implemented. |
| **Explanation faithfulness** | Deletion/Insertion AUC metrics are planned (via Captum) but not yet implemented. Current explanation quality is assessed via Dice alignment with priors. |

### Permutation invariance and joint order

Joints have an anatomical ordering (DIP→PIP→MCP→wrist radially outward), but the MIL aggregator is permutation-invariant by design — it does not use position encoding. This is intentional: disease patterns are defined by which joint *group* is affected (e.g., "RA typically affects MCP and PIP"), not by the exact spatial configuration. Joint-group labels (metadata) preserve anatomical identity without imposing order bias that could confuse the set-based MIL reasoning.

## 7. Known limitations

- **No real-data validation** — All mechanical tests pass (shapes, gradients, numerics, masking), but classification accuracy, explanation fidelity, and baseline comparisons require real clinical X-ray data. Until then, all performance claims are `TODO: unverified`.
- **Detection model not trained** — YOLOv7 module is a wrapper for a pretrained model. Actual training on joint annotations (or structured coordinate extraction) is needed for clinically accurate detection. The architecture supports bypassing detection entirely via pre-computed boxes.
- **Baseline models not fully implemented** — Image-level ViT and single-disease specialist baselines are specified in the evaluation plan but are not built as runnable modules. Ablation configs exist but the actual baseline training pipeline is missing.
- **Explanation faithfulness unvalidated** — The Dice-based anatomy prior loss can align attention with clinical priors, but attention weights are not guaranteed to be faithful explanations (Jain & Wallace, 2019). Deletion/Insertion AUC and Grad-CAM comparison are planned but not implemented.
- **Calibration not implemented** — ECE computation and temperature scaling are specified in the evaluation requirements but not yet built.
- **Tile-sampling MIL incomplete** — Ablation 7 (detection-based vs. tile-sampling) is only partially implemented. No grid-based tile sampler exists; only the config option is defined.
- **No explicit equivariance** — The architecture has no built-in rotation or reflection equivariance. Hand X-rays are standardized in acquisition, so this is acceptable, but it limits generalization to non-standard orientations.
- **DINOv2 domain gap** — DINOv2 was trained on ImageNet (natural images). X-ray grayscale radiographs have fundamentally different texture and frequency characteristics. The frozen-backbone assumption may fail if the domain gap is too large; the fallback (scratch-trained ResNet-152) is available.
