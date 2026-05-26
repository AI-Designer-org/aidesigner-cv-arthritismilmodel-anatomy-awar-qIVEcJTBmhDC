# Benchmarks

All numbers are reproducible with the commands shown.
Numbers marked `TODO` have not been measured on real clinical data — do not cite them for clinical performance.

> The architecture is mechanically verified (shapes, gradients, masking, numerics) on synthetic data with a tiny_debug backbone. All performance claims (accuracy, explanation Dice, baseline comparisons) require real X-ray data and are marked `TODO: unverified`.

## Synthetic verification (mechanical correctness)

### Shape and gradient tests

| Test | Result | Command | Notes |
|---|---|---|---|
| Single-view forward shapes | PASS | `pytest test_model.py::TestShapes::test_single_view_output_shape -v` | All 5 output tensors have correct dimensions |
| Multi-view forward shapes | PASS | `pytest test_model.py::TestShapes::test_multi_view_output_shape -v` | Cross-attention fusion, 2 views |
| Variable joint counts (3 vs 12) | PASS | `pytest test_model.py::TestShapes::test_variable_joint_counts -v` | Padding + masking handles heterogeneity |
| Single-joint edge case | PASS | `pytest test_model.py::TestShapes::test_single_joint_image -v` | Attention weight on single joint ≈ 1.0 |
| Zero joints (all padding) | PASS | `pytest test_model.py::TestShapes::test_zero_joints_padding -v` | Uniform attention over masked positions |
| 3-class (no normal) | PASS | `pytest test_model.py::TestShapes::test_different_n_classes -v` | Configurable n_classes |
| Per-joint classifier disabled | PASS | `pytest test_model.py::TestShapes::test_per_joint_classifier_disabled -v` | Returns None for joint logits |
| Anatomy loss shape | PASS | `pytest test_model.py::TestShapes::test_anatomy_prior_loss_shape -v` | Scalar explanation loss |
| All trainable params get gradients | PASS | `pytest test_model.py::TestGradients::test_all_trainable_params_receive_gradients -v` | No dead parameters |
| No NaN gradients | PASS | `pytest test_model.py::TestGradients::test_no_nan_gradients -v` | All gradients finite |
| Multi-view gradient flow | PASS | `pytest test_model.py::TestGradients::test_gradient_flow_multi_view -v` | Fusion parameters trainable |
| Anatomy loss gradient flow | PASS | `pytest test_model.py::TestGradients::test_gradient_flow_with_anatomy_loss -v` | Anatomy loss contributes non-zero gradient |

### Numeric stability

| Test | Result | Command |
|---|---|---|
| bf16 forward (no NaN/Inf) | PASS | `pytest test_model.py::TestNumerics::test_bf16_forward -v` |
| fp16 forward (no NaN/Inf) | PASS | `pytest test_model.py::TestNumerics::test_fp16_forward -v` |
| Extreme bright/dark inputs | PASS | `pytest test_model.py::TestNumerics::test_extreme_input_values -v` |
| X-ray normalization edge cases | PASS | `pytest test_model.py::TestNumerics::test_xray_normalization_numerics -v` |
| Focal loss edge cases | PASS | `pytest test_model.py::TestNumerics::test_focal_loss_numerics -v` |

### CV domain properties

| Test | Result | Command | Notes |
|---|---|---|---|
| MIL permutation invariance | PASS | `pytest test_model.py::TestCVProperties::test_mil_permutation_invariance -v` | Bag rep unchanged under joint permutation |
| Attention masking correctness | PASS | `pytest test_model.py::TestCVProperties::test_attention_masking_correctness -v` | Zero attention on padded positions |
| MIL without gating | PASS | `pytest test_model.py::TestCVProperties::test_mil_without_gating -v` | Simple attention variant also works |
| View embedding distinguishability | PASS | `pytest test_model.py::TestCVProperties::test_view_embedding_distinguishability -v` | View membership tracked correctly |
| Translation approximate invariance | PASS | `pytest test_model.py::TestCVProperties::test_translation_approximate_invariance -v` | Cosine sim >0.8 after 4px shift |
| Noise entropy (no shortcut) | PASS | `pytest test_model.py::TestCVProperties::test_no_spatial_shortcut_noise -v` | Entropy >50% of max on noise |
| Anatomy prior Dice range | PASS | `pytest test_model.py::TestCVBenchmarks::test_anatomy_prior_dice_range -v` | Correct Dice for aligned and misaligned cases |
| No mode collapse | PASS | `pytest test_model.py::TestCVBenchmarks::test_no_mode_collapse -v` | Outputs differ across different inputs |
| Linear probe above chance | PASS | `pytest test_model.py::TestCVBenchmarks::test_linear_probe_on_frozen_features -v` | Frozen features support >25% accuracy |

### Loss correctness

| Test | Result | Command |
|---|---|---|
| Loss keys (all 5) | PASS | `pytest test_model.py::TestLoss::test_compute_loss_keys -v` |
| Loss without joint labels | PASS | `pytest test_model.py::TestLoss::test_loss_without_joint_labels -v` |
| Focal (γ=0) = CE | PASS | `pytest test_model.py::TestLoss::test_focal_loss_vs_ce -v` |
| Focal alpha weighting | PASS | `pytest test_model.py::TestLoss::test_focal_loss_alpha -v` |

## Evaluation tasks (CV domain)

### Linear probe on frozen features

> TODO: unverified — requires real data with ground-truth labels.

**Command:** `python eval/linear_probe.py` (not yet implemented as standalone script)

Linear probe training on frozen DINOv2 features is verified mechanically (synthetic data shows >25% accuracy, which is above chance for 4-class). On real data, a linear probe evaluates feature quality without fine-tuning the backbone.

### Per-joint classification accuracy

> TODO: unverified — requires real X-ray data with per-joint labels.

| Metric | Target | Current status |
|---|---|---|
| Per-joint macro-F1 (RA/PsA/OA/normal) | >0.80 | Not yet measurable |
| Per-joint weighted-F1 | >0.85 | Not yet measurable |
| Patient-level accuracy | >0.80 | Not yet measurable |
| Patient-level AUROC (one-vs-rest) | >0.90 | Not yet measurable |

### Explanation fidelity

> TODO: unverified — requires real data with joint-group annotations.

| Metric | Target | Current status |
|---|---|---|
| Explanation Dice (α vs. disease-joint prior) | >0.60 | Dice computation verified on synthetic data (range: [0, prior_weight]) |
| Deletion AUC | >0.80 | Not yet implemented |
| Insertion AUC | >0.80 | Not yet implemented |

### Calibration

> TODO: unverified — ECE analysis not yet implemented.

| Metric | Target | Current status |
|---|---|---|
| ECE (RA) | <0.10 | Not implemented |
| ECE (PsA) | <0.10 | Not implemented |
| ECE (OA) | <0.10 | Not implemented |
| ECE (normal) | <0.10 | Not implemented |

**Command:** Not yet available. Requires `sklearn.calibration` or custom ECE implementation.

## Ablation study

Each ablation is a single-field change from the baseline config. Results below are on **synthetic data with tiny_debug backbone** and show mechanical correctness only — they have no predictive value for real-data performance.

| Ablation | Config delta | Patient accuracy (synthetic) | Δ vs. baseline | Reproduce command |
|---|---|---|---|---|
| Baseline | — | `TODO: unverified` | — | `python run_ablations.py --ablation baseline` |
| MIL → Average Pooling | `mil_gated=False, patient_pooling="mean"` | `TODO` | `TODO` | `python run_ablations.py --ablation mil_pool` |
| Frozen → Full FT | `backbone_frozen=False` | `TODO` | `TODO` | `python run_ablations.py --ablation full_ft --backbone dinov2_vitb14` |
| Focal → CE | `loss_type="ce"` | `TODO` | `TODO` | `python run_ablations.py --ablation ce_loss` |
| Three-way → One-vs-All (RA) | binary RA vs. rest | `TODO` | `TODO` | `python run_ablations.py --ablation ova_ra` |
| Anatomy Prior Off → On | `use_anatomy_prior_loss=True` | `TODO` | `TODO` | `python run_ablations.py --ablation anatomy_on` |
| Single-view → Multi-view | `input_views=("PA", "oblique")` | `TODO` | `TODO` | `python run_ablations.py --ablation multi_view` |
| Detection → Tile Sampling | `detection_model="none"` + grid | `TODO` | `TODO` | `python run_ablations.py --ablation tile_sampling` |

All ablations: `python run_ablations.py --config all`

## Profiling

GPU: RTX 4090 (24 GB), bf16 mixed precision, B=16, N_max=30, DINOv2 ViT-L/14 (estimated from architect's compute budget):

| Phase | Time (ms) | Peak mem (MB) | Notes |
|---|---|---|---|
| YOLOv7 detection (frozen) | ~50 | ~2,000 | Offline — not in training loop |
| DINOv2 backbone (×30 ROIs, frozen) | ~300 | ~8,000 | Bottleneck; cache after epoch 1 |
| MIL aggregator + heads | ~5 | ~1,000 | Only trainable component |
| **Total per batch** | ~355 | ~11,000 | B=16, N=30; fits on 24 GB |
| Training step (frozen fwd + bwd) | ~10 (backward) | ~13,000 | Only MIL+heads gradients |

**Estimated FLOPs:** ~10–15 GFLOPs per batch forward (frozen backbone: 2·params for DINOv2 inference; trainable heads: negligible).

**Key optimization:** Feature caching. After epoch 1, DINOv2 outputs can be cached on disk since the backbone is frozen. This reduces per-epoch time from ~355ms/batch to ~5ms/batch (MIL + heads only).

**Reproduce:** `python profile_model.py --mode both --backbone tiny_debug`

## Research-quality evaluation

| Dimension | Score (0–5) | Evidence | Gaps |
|---|---|---|---|
| Novelty | 3/5 | Gap A (three-way discrimination) grounded in literature; Gap B (anatomy-grounded XAI) is novel for arthritis | Novelty depends on real-data performance; architecture novelty is incremental (combines known components) |
| Experimental comprehensiveness | 3/5 | 7 ablations defined (6/7 fully implemented); 30+ pytest tests; profiling script | Baselines (image-level ViT, single-disease specialists) not fully implemented; ECE, Deletion/Insertion, stratified eval not built |
| Theoretical foundation | 4/5 | 10 inductive biases documented; 15 traceability mappings; 5 risk flags with mitigations | Some biases asserted not tested; permutation invariance trade-off under-discussed |
| Result analysis | 1/5 | Smoke test + ablation runner produce metrics on synthetic data | No real-data results; no error analysis; no SOTA comparison |
| Implementation reproducibility | 4/5 | Single ModelConfig with validation; seed=42; clear shape contracts | Deterministic training not enforced; data preprocessing not included; YOLOv7 dep not pinned |
| Writing readiness | 4/5 | Architecture diagram, design justification, landscape synthesis all present | Figures are ASCII (not thesis-ready); no executive summary |

### Required next experiments (from scorecard)

| Priority | Experiment | Description |
|---|---|---|
| **P0** | Real data baseline comparison | Train image-level ViT, per-joint CNN, and proposed MIL on actual X-ray data. Compare per-joint macro-F1, patient-level AUROC. |
| **P0** | Anatomy prior Dice validation | Compute explanation Dice between attention weights and RA/PsA/OA joint priors on real data. With and without anatomy prior loss. |
| **P1** | Frozen vs. fine-tuned backbone | Compare frozen DINOv2, LoRA, and full fine-tuning on held-out test set. |
| **P1** | Calibration analysis | Compute ECE per disease class. Apply temperature scaling if ECE > 0.10. |
| **P1** | Explanation faithfulness (Deletion/Insertion) | Compute Deletion and Insertion AUC. Compare attention-based and Grad-CAM explanations. |
| **P1** | Robustness stratification | Stratify by disease severity, joint site (MCP/PIP/DIP/wrist), and image view (PA/oblique). |
| **P2** | One-vs-all comparison | Compare three-way vs. one-vs-all binary classifiers. Measure multi-class penalty. |
| **P2** | Detection sensitivity analysis | Evaluate how YOLOv7 detection thresholds affect downstream classification accuracy. |

### Blocking gaps

| Code | Description | Priority |
|---|---|---|
| `baseline_not_beaten` | No baseline comparison results — architecture not shown to beat image-level ViT or per-joint CNN | Critical |
| `data_unavailable` | Real clinical X-ray data required to assess clinical utility | Critical |
| `benchmark_not_executable` | Full evaluation suite requires real X-ray dataset not in repository | Critical |
| `coverage_gap` | ECE, Deletion/Insertion, stratified evaluation not implemented | High |
| `baseline_not_implemented` | Image-level ViT and single-disease specialist models not built | High |
| `claim_not_grounded` | "Foundation model provides label efficiency" — no experiment run | High |
| `novelty_unverified` | Three-way discrimination claim depends on real data performance | High |
| `ablation_missing` | Tile-sampling MIL only partially implemented | Medium |
