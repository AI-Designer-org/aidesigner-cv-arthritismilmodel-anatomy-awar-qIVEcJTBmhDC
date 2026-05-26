# Architecture Overview: Anatomy-Aware Per-Joint MIL for Arthritis Discrimination

## High-Level Design

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

## Parameter-Activation Flow

```
              Tensor shape                   Learnable?   Module
              ─────────────────────────────────────────────────────
Input:        (B, 1, 640, 640)               ✗           Raw X-ray
  │
  ▼
Detection:    List[(N_i, 4)] per image       ✗ (frozen)  YOLOv7
  │
  ▼
ROI Crop:     (B, N_max, 1, 224, 224)        ✗           GridSample
  │
  ▼
Backbone:     (B, N_max, d=1024)             ✗ (frozen)  DINOv2
  │
  ▼
View Fusion:  (B, N_total, 1024)             ✓ (emb)    Embed + Concat
  │
  ▼
MIL Pool:     (B, 1024)   +   (B, N) α_i     ✓            GatedAttn
  │
  ├─► Joint Head:   (B, N, 4)                ✓            MLP
  ├─► Patient Head: (B, 4)                   ✓            MLP
  └─► XAI:          α_i + Grad-CAM           ✗            Post-hoc
```

Where:
- `B` = batch size
- `N_i` = joints detected in image i (variable)
- `N_max` = max_joints_per_view (padded to 30)
- `N_total` = sum of N_max across views (e.g., 60 for 2 views)
- `d` = d_model (1024 for DINOv2 ViT-L/14)
- Trainable parameters: ~1–2M (attention aggregator + heads) or ~3–4M (+LoRA)

## Novel Component Detail: Gated Attention MIL Aggregator

```
Joint features h_i  (d=1024)
         │
    ┌────┴────┐
    ▼         ▼
tanh(V·h_i)  sigmoid(U·h_i)       V, U: Linear(d → L=512)
    │         │
    └────┬────┘
         ▼
    a_i ⊙ b_i                     Element-wise gate
         │
         ▼
    wᵀ · (a_i ⊙ b_i)             w: Linear(L → 1)
         │
         ▼
    softmax over N joints         α_i = attention weight for joint i
         │
    ┌────┴────┐
    ▼         ▼
  z = Σ α_i·h_i                 Bag representation (B, d)
  α_i ∈ [0,1]                   Interpretable joint importance
```

## Novel Component Detail: Anatomy-Guided Explanation Loss

```
During training (when use_anatomy_prior_loss=True):

  attention_weights α_i       joint_group_labels      target_disease
       (B, N)                     (B, N)                  (B,)
         │                          │                       │
         └──────────────┬───────────┘                       │
                        ▼                                   │
               Lookup prior mask:                           │
               which joint groups are                       │
               relevant for this disease?                   │
                        │                                   │
                        ▼                                   ▼
               Prior mask (B, N)   ←───   anatomy_priors[RA] = MCP/PIP/wrist
                                          anatomy_priors[PsA] = PIP/DIP
                                          anatomy_priors[OA] = DIP/CMC
                                          anatomy_priors[normal] = uniform
                        │
                        ▼
               Dice(α_i, prior_mask)
                        │
                        ▼
               L_anatomy = 1 - Dice
```

## Inference-Time Explanation Output

```
┌─────────────────────────────────────────────────────────────────┐
│  EXPLANATION OUTPUT (per patient)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Patient-level diagnosis:  RA  (p=0.87)                        │
│                                                                 │
│  Top-3 most influential joints:                                 │
│    Joint      α_i    Per-joint prediction    Confidence         │
│    ──────    ─────   ────────────────────   ──────────         │
│    MCP-2     0.31    RA                      0.92              │
│    PIP-3     0.24    RA                      0.88              │
│    MCP-3     0.18    RA                      0.85              │
│                                                                 │
│  Anatomical alignment score (Dice):  0.74                       │
│  (RA-expected joints: MCP 2-5, PIP 2-5, wrist)                 │
│                                                                 │
│  [Grad-CAM overlay available on top-3 joint ROIs]               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Inductive Biases (One Sentence Each)

| # | Design choice | Inductive bias statement |
|---|---|---|
| 1 | **Detection-based MIL** (not tile-sampling) | Separating detection from classification lets us handle variable joint counts, missing joints, and occlusions naturally — the detector finds what's available, and the MIL works with whatever it gets. |
| 2 | **Frozen DINOv2 backbone** | DINOv2's self-supervised ViT features capture generic shape/texture primitives (bone contours, joint space width, erosion boundaries) that transfer to X-ray despite the ImageNet-to-radiology domain gap, so freezing avoids overfitting on limited clinical data. |
| 3 | **Gated attention (tanh × sigmoid)** | The gating mechanism lets the MIL learn non-linear "importance queries" — critical because disease-relevant features (e.g., subtle erosions vs. joint space narrowing) have non-linear visual signatures that linear attention would miss. |
| 4 | **Pre-norm before attention** | LayerNorm before the tanh/sigmoid gates prevents vanishing gradients in the saturating nonlinearities, stabilizing training especially with frozen high-dimensional features. |
| 5 | **Dual-path classification** | Separating per-joint and patient-level classifiers lets the model capture both "any single joint can indicate disease" (per-joint) and "the pattern across joints confirms the type" (patient-level) — matching clinical reasoning. |
| 6 | **View embeddings for multi-view** | Adding a learned view embedding before fusion allows the model to distinguish e.g., PA-view MCP from oblique-view MCP, so it can learn view-specific feature transformations. |
| 7 | **Anatomy prior loss (optional)** | Aligning attention weights with disease-specific joint groups encodes the clinical prior that RA affects MCP/PIP, OA affects DIP, and PsA affects PIP/DIP — improving explanation plausibility without per-image ROI annotations. |
| 8 | **Focal loss** | Focusing on hard-to-classify examples counters the expected class imbalance (RA > OA > PsA prevalent in clinical datasets) and the inherent subtlety of early arthritis changes. |
| 9 | **Per-image normalization** | X-rays have inconsistent brightness/contrast across acquisitions; normalizing each image to zero-mean unit-variance removes acquisition-level variance while preserving anatomical contrast. |
| 10 | **Padding with masking (not variable batch)** | Padding joints to N_max with a validity mask lets us batch efficiently across images with different joint counts, avoiding dynamic computation graphs and enabling standard data-loading. |
