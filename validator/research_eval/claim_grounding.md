# Claim Grounding Report
## Anatomy-Aware Per-Joint MIL for Arthritis Discrimination

Every architectural, performance, or novelty claim below must point to one of:
- a source file path and function/class/line number,
- a test or benchmark command,
- an ablation result,
- profiler output,
- or `TODO: unverified`.

Claims with **no grounding** are flagged as `TODO: unverified`.

---

## 1. Novelty Claims

### Claim A: "No existing system jointly performs three-way RA-vs-PsA-vs-OA discrimination at per-joint granularity"

**Status:** grounded (literature)
**Source:** `research/01_landscape_synthesis.md` lines 78-86
**Evidence:**
- Systematic review (Semin Arthritis Rheum, 2025): 88% RA, 22% OA, 9% PsA — none jointly on all three
- EULAR 2025 ViT system explicitly lists PsA as future work
- PsA ViT system (MLMI 2024): PsA-only scoring
- Grounding file: `research/02_lifecycle_contract.yaml` lines 12-22 (status: grounded)
**Validation hook:** Per-joint macro-F1 > 0.80 on held-out test with all 4 classes
**Current verification:** ❌ — requires real data

### Claim B: "Attention-based MIL with foundation model backbone can produce anatomically grounded per-joint explanations"

**Status:** hypothesis
**Source:** `architect/04_design_justification.md` line 12
**Evidence:**
- Anatomy-aware MIL for RA scoring (Bo et al., MICCAI AMAI 2025): RA-only, no anatomical validation
- Expert-guided explanation loss (Uddin et al. 2025): applicable but not tested on arthritis
- **Implementation:** `coder/explanation.py::AnatomyExplanationModule` lines 44-146
- **Test:** `test_model.py::TestCVProperties::test_anatomy_prior_dice_range` — Dice loss computation verified
- **Ablation config:** `run_ablations.py::_make_ablations` → `anatomy_on`
**Validation hook:** Explanation Dice > 0.6; Deletion/Insertion AUC > 0.8
**Current verification:** ⚠️ — Dice computation verified on synthetic data; clinical validation TODO

### Claim C: "Foundation models for MSK radiographs can be fine-tuned for arthritis-specific classification"

**Status:** hypothesis
**Source:** `research/02_lifecycle_contract.yaml` lines 36-43
**Evidence:**
- OrthoFoundation (1.2M knee images): not evaluated on RA/PsA/OA
- SKELEX (1.2M MSK radiographs): fracture detection and OA grading only
- **Implementation:** `coder/backbone.py::FoundationBackbone` lines 43-184 — supports DINOv2, ResNet-152, extensible to OrthoFoundation/SKELEX
- **Ablation config:** `run_ablations.py::_make_ablations` → `full_ft` (backbone_frozen=True → False)
**Validation hook:** Frozen backbone outperforms scratch-trained ResNet-50 by >5% F1
**Current verification:** ❌ — requires real data and backbone comparison

---

## 2. Architecture Claims

### Claim D: "Detection-based MIL handles variable joint counts"

**Status:** verified (mechanical)
**Source:** `coder/layers.py::ROIFeatureExtractor` lines 130-213
**Evidence:**
- Padding + masking strategy: `coder/layers.py` lines 198-213
- Masked softmax attention: `coder/layers.py::GatedAttentionMIL.forward` lines 390-397
- **Test:** `test_model.py::TestShapes::test_variable_joint_counts` — 3 vs 12 joints
- **Test:** `test_model.py::TestShapes::test_single_joint_image` — 1 joint edge case
- **Test:** `test_model.py::TestShapes::test_zero_joints_padding` — 0 joints edge case
- **Test:** `test_model.py::TestCVProperties::test_attention_masking_correctness` — zero attention on padded positions
**Command:** `pytest test_model.py::TestShapes -v`
**Verified:** ✅

### Claim E: "Gated attention enables non-linear importance weighting"

**Status:** verified (mechanical)
**Source:** `architect/03_architecture_diagram.md` lines 109-131
**Evidence:**
- Implementation: `coder/layers.py::GatedAttentionMIL` lines 320-400
- Both gated and non-gated variants work: `test_model.py::TestCVProperties::test_mil_without_gating`
- Permutation invariance: `test_model.py::TestCVProperties::test_mil_permutation_invariance`
**Command:** `pytest test_model.py::TestCVProperties::test_mil_permutation_invariance -v`
**Verified:** ✅ — gated and non-gated both produce correct outputs

### Claim F: "Multi-view fusion with view embeddings improves over single-view"

**Status:** hypothesis
**Source:** `architect/04_design_justification.md` line 145
**Evidence:**
- Implementation: `coder/layers.py::MultiViewFusion` lines 232-313
- Concatenative and cross-attention modes: `test_model.py::TestMultiView::test_cross_attention_fusion_shape`
- View embedding distinguishability: `test_model.py::TestCVProperties::test_view_embedding_distinguishability`
- Ablation config: `run_ablations.py` → `multi_view`
**Validation hook:** Multi-view variant outperforms single-view by >3% patient-level AUROC
**Current verification:** ⚠️ — architecture works; performance gain requires real data

### Claim G: "Dual-path classification captures both per-joint and patient-level patterns"

**Status:** verified (mechanical)
**Source:** `coder/heads.py::ArthritisClassificationHead` lines 18-99
**Evidence:**
- Both heads produce correct shapes: `test_model.py::TestShapes::test_single_view_output_shape`
- Per-joint classifier can be disabled: `test_model.py::TestShapes::test_per_joint_classifier_disabled`
- Disease-specific heads work: `test_model.py::TestShapes::test_disease_specific_heads`
- Disease-specific heads are distinguishable: `test_model.py::TestCVBenchmarks::test_disease_specific_heads_separate`
- Dual-path loss computation: `coder/losses.py::compute_loss` lines 80-173
**Command:** `pytest test_model.py::TestShapes::test_disease_specific_heads -v`
**Verified:** ✅

### Claim H: "Frozen backbone prevents overfitting with limited data"

**Status:** hypothesis
**Source:** `architect/04_design_justification.md` line 108
**Evidence:**
- Frozen backbone implementation: `coder/backbone.py::FoundationBackbone.__init__` lines 101-103
- Frozen immutability verified: `test_model.py::TestBackbone::test_backbone_frozen_immutable`
- Ablation config: `run_ablations.py` → `full_ft`
**Validation hook:** Frozen backbone test accuracy within 5% of full fine-tuning
**Current verification:** ❌ — requires real data comparison

---

## 3. Performance Claims

### Claim I: "Per-joint classification accuracy >80% on held-out test"

**Status:** TODO: unverified
**Validation hook:** Per-joint macro-F1 > 0.80 on held-out test set
**Missing:** Requires real X-ray dataset with per-joint labels

### Claim J: "Attention weights show higher mass on disease-relevant joints"

**Status:** TODO: unverified
**Validation hook:** Explanation Dice > 0.6 with disease-joint priors
**Missing:** Requires real data with joint-group annotations and clinical validation

### Claim K: "Patient-level aggregation improves over image-level baseline by >5% F1"

**Status:** TODO: unverified
**Validation hook:** MIL model outperforms image-level ViT by >5% F1
**Missing:** Requires image-level ViT baseline implementation + real data

### Claim L: "Foundation model backbone yields >10% improvement over scratch-trained CNN"

**Status:** TODO: unverified
**Validation hook:** Frozen DINOv2 > scratch-trained ResNet-50 by 10% F1
**Missing:** Requires both backbone implementations + real data

### Claim M: "Training completes in 2-4 hours on RTX 4090"

**Status:** TODO: unverified
**Source:** `architect/04_design_justification.md` line 218
**Estimated (from profile_model.py):** ~355ms per batch (DINOv2 ViT-L/14 forward)
**Gap:** Actual training time depends on dataset size, batch size, and number of epochs. Profile script provides per-step timing but not end-to-end training time.

---

## 4. Implementation Claims

### Claim N: "All hyperparameters in a single config with validation"

**Status:** verified
**Source:** `coder/config.py::ModelConfig` lines 13-226
**Evidence:**
- `__post_init__` validates n_classes, loss_type, multi_view_fusion, patient_pooling, LoRA constraint
- Auto-sets d_model based on backbone choice
- No magic numbers in implementation code

### Claim O: "Gradient checkpointing reduces memory at compute cost"

**Status:** verified
**Source:** `coder/model.py::ArthritisMILModel.forward_with_checkpointing` lines 146-162
**Evidence:**
- Checkpointing forward pass verified: `test_model.py::TestShapes::test_gradient_checkpointing`
- Uses `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`

### Claim P: "Focal loss handles class imbalance"

**Status:** verified (mechanical)
**Source:** `coder/losses.py::focal_loss` lines 21-73
**Evidence:**
- Numeric stability tested: `test_model.py::TestNumerics::test_focal_loss_numerics`
- Equivalence at γ=0: `test_model.py::TestLoss::test_focal_loss_vs_ce`
- Alpha weighting: `test_model.py::TestLoss::test_focal_loss_alpha`
- Ablation: `run_ablations.py` → `ce_loss`
**Command:** `pytest test_model.py::TestLoss -v`

---

## 5. Summary

| Claim | Status | Grounding |
|-------|--------|-----------|
| A: Three-way gap | ✅ Grounded (literature) | `research/01_landscape_synthesis.md` |
| B: Anatomy-grounded XAI | ⚠️ Partially verified | `coder/explanation.py` + synthetic Dice test |
| C: Foundation model adaptation | ❌ Unverified | Requires real data |
| D: Variable joint counts | ✅ Verified | 3 tests pass |
| E: Gated attention | ✅ Verified | Permutation invariance test |
| F: Multi-view fusion | ⚠️ Architecture verified | Shape tests pass; performance gain unverified |
| G: Dual-path classification | ✅ Verified | Shape tests + loss computation verified |
| H: Frozen backbone prevents overfitting | ❌ Unverified | Requires real data |
| I: >80% joint accuracy | ❌ Unverified | Requires real data |
| J: Attention on relevant joints | ❌ Unverified | Requires real data + clinical validation |
| K: >5% improvement over image-level ViT | ❌ Unverified | Requires baseline + real data |
| L: >10% over scratch CNN | ❌ Unverified | Requires real data |
| M: 2-4h training | ❌ Unverified | Profile provides estimate; real timing needed |
| N: Single config | ✅ Verified | `coder/config.py` with `__post_init__` |
| O: Gradient checkpointing | ✅ Verified | `test_model.py::test_gradient_checkpointing` |
| P: Focal loss | ✅ Verified | 4 loss tests pass |
