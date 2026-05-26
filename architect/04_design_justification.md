# Design Justification, Traceability, Risks & Ablations

---

## 1. Research-to-Architecture Traceability

Every non-standard architectural decision traces to either an upstream novelty claim, a known baseline limitation, a domain invariant, an evaluation requirement, or implementation practicality.

| # | Research contract item | Architecture decision | Evidence status | Validation hook |
|---|---|---|---|---|
| 1 | **Novelty claim A**: No existing system jointly discriminates RA/PsA/OA at per-joint granularity | **Dual-path classification** (per-joint head + patient head) with 4-class output {RA, PsA, OA, normal} | **grounded** — literature confirms no prior work | Per-joint macro-F1 > 0.80 with all 4 classes present; confusion matrix shows balanced discrimination |
| 2 | **Novelty claim B**: MIL + foundation model can produce anatomically grounded per-joint explanations | **Gated attention weights α_i** as inherent explanation; **AnatomyPriorLoss** (Dice between α_i and disease-joint prior mask) | **hypothesis** — MIL explanation for arthritis is novel; anatomic grounding is novel | Explanation Dice (α_i vs. RA→MCP/PIP prior) > 0.6; Deletion/Insertion area-under-curve > 0.8 |
| 3 | **Novelty claim C**: Foundation models can be fine-tuned for arthritis classification | **Frozen DINOv2 backbone** with optional **LoRA** adapters; configuration supports OrthoFoundation/SKELEX as drop-in | **hypothesis** — not tested on inflammatory arthritis | Frozen backbone variant outperforms scratch-trained ResNet-50 by >5% F1; LoRA fine-tuning adds >3% over frozen |
| 4 | **Data constraint**: Per-joint structural annotations available | **Detection-based MIL** (YOLOv7 + ROI cropping) rather than tile-sampling MIL | **grounded** — annotations enable supervised per-joint learning | Detection mAP > 0.90 on held-out joint localization test |
| 5 | **Data constraint**: Multiple views/sites per patient | **MultiViewFusion** with view embeddings + optional cross-attention | **grounded** — data property constrains architecture | Multi-view variant outperforms single-view by >3% patient-level AUROC |
| 6 | **Data constraint**: Limited labeled data (clinical setting) | **Frozen backbone** + lightweight trainable head (~1–2M params); data augmentation pipeline | **grounded** — standard practice in medical DL | Frozen backbone test accuracy within 5% of full fine-tuning; augmentation improves generalization >3% |
| 7 | **Data constraint**: Expected class imbalance (RA > OA > PsA) | **Focal loss** γ=2.0; optional per-class α weights; disease-specific binary heads as alternative | **grounded** — class imbalance is clinical reality | Per-class F1 variance < 0.10; minority class (PsA) F1 > 0.70 |
| 8 | **Evaluation requirement**: Per-joint patient-level split | **Patient-level train/test split** enforced in data loader; no image-level leakage | **grounded** — stated in evaluation plan | Implement patient-level GroupKFold cross-validation |
| 9 | **Evaluation requirement**: Explanation faithfulness (Deletion/Insertion) | **Attention weight analysis** pipeline in XAI module; faithfulness metrics computed post-hoc | **grounded** — required by evaluation plan | Deletion/Insertion curves computed for test set |
| 10 | **Evaluation requirement**: Calibration (ECE per disease) | **Patient-level logits** used for temperature scaling analysis | **grounded** — required by evaluation plan | ECE < 0.10 per disease class after calibration |
| 11 | **Baseline requirement**: Image-level ViT as baseline | Separate **ImageLevelViTBaseline** model (not part of main architecture) | **grounded** — must compare against it | MIL model outperforms image-level ViT by >5% F1 |
| 12 | **Baseline requirement**: Per-joint CNN (YOLO + ResNet) | **JointDetectionModule** + independent per-joint ResNet (ablation variant) | **grounded** — must compare against it | MIL aggregator outperforms majority-vote pooling by >3% F1 |
| 13 | **Blocking unknown**: Joint annotations (bbox vs. label) | **DetectionModelConfig** supports both modes: "yolov7" for bbox, "none" for coordinate-based cropping | **TODO: unverified** — need to verify annotation format | If bboxes: detection module trains with supervision. If coordinates: ROI crops from centers |
| 14 | **Blocking unknown**: Foundation model availability | **DINOv2** as default (public, proven); OrthoFoundation/SKELEX as config-drop-in option | **TODO: unverified** — need institutional access verification | Backbone config swap produces valid forward pass |
| 15 | **Blocking unknown**: Minimum dataset size | **Data efficiency** via frozen backbone + strong augmentation + focal loss; mitigation: one-vs-all fallback | **TODO: unverified** — power analysis needed | Monitor validation loss divergence — if unstable, apply one-vs-all fallback |

---

## 2. Domain-Specific Design Considerations

### CV Domain (Primary)

| Concern | How the architecture addresses it |
|---|---|
| **Spatial handling** | YOLOv7 provides absolute pixel coordinates; ROI cropping normalizes to fixed size, discarding absolute position but preserving relative anatomy within the joint. Joint-group labels (MCP-2, PIP-3, etc.) are preserved as metadata for anatomical attribution. |
| **Scale invariance** | No explicit multi-scale processing. YOLOv7 has inherent FPN-based multi-scale detection. ROI crops are resized to a fixed 224×224, which standardizes feature extraction but may lose scale cues (e.g., a large erosion vs. a small one). |
| **Dense vs. global** | Per-joint features are extracted from local ROIs (dense within each joint), then aggregated globally via MIL attention. This is a hybrid: local patches + global bag-level reasoning. |
| **Rotation invariance** | Data augmentation (random rotation ±10°) provides mild rotation invariance. Hand X-rays are typically well-aligned in clinical acquisition protocols, so aggressive rotation is not needed. |

### SciML Domain (Secondary)

| Concern | How the architecture addresses it |
|---|---|
| **Clinical validation** | Explanation Dice scores provide a quantitative bridge between model behavior and clinical knowledge (disease-joint associations). The XAI module is designed to produce outputs a rheumatologist can evaluate. |
| **Domain-specific evaluation** | Metrics are stratified by: (a) disease severity (mild/moderate/severe), (b) joint site (MCP/PIP/DIP/wrist), (c) image view (PA/oblique). Stratification is implemented in the evaluation harness, not the model. |
| **Data provenance** | The architecture assumes anonymized, IRB-approved data. All preprocessing is deterministic and auditable. Data splits are patient-level to prevent leakage between training and test sets. |

### GenAI Domain (Supporting — Data Augmentation)

While not in the main architecture, diffusion-based synthetic X-ray generation is noted as a future augmentation strategy. The architecture's modular design allows synthetic ROI crops to be injected at the detection stage or as additional joint features at the MIL stage without modifying the core blocks.

---

## 3. Implementation Risk Flags

### Risk 1: DINOv2 Domain Gap — X-ray vs. Natural Images
- **What**: DINOv2 was trained on ImageNet (natural images). X-ray grayscale radiographs have fundamentally different texture, contrast, and frequency characteristics. The [CLS] token may encode features irrelevant to bone erosion patterns.
- **Mitigation**: 
  - Use per-image normalization (zero-mean, unit-variance) before DINOv2 input
  - Evaluate frozen vs. LoRA fine-tuned performance empirically
  - Fallback: ResNet-152 with ImageNet weights (proven for medical imaging transfer)
- **Falsification**: If frozen DINOv2 underperforms scratch-trained ResNet-50 with same data budget, the domain gap is too large for frozen transfer.

### Risk 2: Attention Weight Reliability for Explanation
- **What**: MIL attention weights α_i may not correspond to clinical relevance. "Attention is not explanation" (Jain & Wallace, 2019) — attention weights can be rank-correlated with other features without being causal.
- **Mitigation**:
  - Compute Deletion/Insertion faithfulness metrics (required by evaluation plan)
  - Compare against Grad-CAM on per-joint ROIs
  - Anatomy prior loss provides soft supervision to align attention with clinical priors
- **Falsification**: If Deletion AUC < 0.6 (random baseline), attention weights are not faithful explanations. Must switch to Grad-CAM or integrated gradients as primary explanation.

### Risk 3: Variable Joint Counts and Padding Artifacts
- **What**: Different X-rays show different numbers of joints (missing fingers, amputations, field-of-view cuts). Padding to N_max = 30 creates a large padding ratio for images with few joints, and the mask may not perfectly exclude padded positions from attention softmax.
- **Mitigation**:
  - Masked softmax with -1e9 fill (standard, numerically stable)
  - Entropy regularization encourages attention mass on non-padded joints
  - Monitor effective N (non-padded joints) distribution at inference
- **Falsification**: If the model consistently assigns >5% attention mass to padded positions (detectable via mask == 0 positions), the masking is defective.

### Risk 4: Multi-View Joint Correspondence Ambiguity
- **What**: When fusing PA + oblique views, the same anatomical joint appears in both. Without explicit correspondence, the MIL sees 2× the joints and may double-count evidence or attend inconsistently.
- **Mitigation**: View embeddings differentiate the source view. Cross-attention fusion can learn implicit correspondence. The per-joint classifier processes each detection independently so double-counting is naturally handled (two independent observations of the same joint).
- **Falsification**: If multi-view fusion does NOT outperform single-view (ablated), the fusion strategy adds complexity without benefit. Default to single-view (best view) for primary experiments.

### Risk 5: Numerical Instability in Focal Loss with Ignore Index
- **What**: Focal loss exponentiates cross-entropy: `(1 - pt)^γ * ce`. With `ignore_index=-100`, masked positions produce `pt ≈ exp(0) = 1`, so `(1-1)^γ = 0` — correct behavior. But if -100 logits are not properly masked before softmax, NaN gradients can occur.
- **Mitigation**: The `focal_loss` helper applies `ignore_index` to the final loss, not to logits. For safety, also mask logits before CE: `logits = logits.masked_fill(ignore_mask.unsqueeze(-1), -1e9)`.
- **Falsification**: Monitor training loss for NaN. If occurs, switch to standard CE loss as immediate fix.

---

## 4. Suggested Ablations

Each ablation is a single-field ModelConfig change tied to a specific hypothesis.
Ordered by "turn this off first if it doesn't work."

### Ablation 1: MIL → Average Pooling (drop novel aggregator)
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| MIL type | `mil_gated` | True | False (switch to `patient_pooling="mean"`) |
| **Hypothesis tested** | The gated attention mechanism adds value over uniform pooling of joint features. If MIL attention ≠ simple average, the aggregator is learning meaningful importance weighting. |
| **Expected metric movement** | Patient-level F1 drops by ≥3%. Per-joint F1 unchanged (same backbone + joint head). |
| **Failure interpretation** | If F1 does NOT drop (or drops <1%), the attention mechanism is not learning useful joint weighting. The model may be overfitting on attention — increase entropy_reg_weight or reduce mil_hidden_dim. |
| **Owning stage** | `ml-architect` → reconsider attention design. If fundamental, revert to simple pooling and route to `ml-research`. |

### Ablation 2: Frozen Backbone → Full Fine-Tuning
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Backbone training | `backbone_frozen` | True | False |
| **Hypothesis tested** | LoRA/frozen backbone provides sufficient feature quality for arthritis classification. Full fine-tuning may overfit given limited labeled data. |
| **Expected metric movement** | Frozen: X% test F1. Full fine-tuning: X ± 3% (if X-ray domain benefits from adaptation) or X - 5% (if overfitting). |
| **Failure interpretation** | If frozen backbone underperforms full fine-tuning by >3%, the domain gap is larger than expected. Apply LoRA (use_lora=True) as a middle ground. If full fine-tuning underperforms frozen, it confirms overfitting on limited data — keep frozen. |
| **Owning stage** | `ml-coder` (experimental comparison). Route frozen/full-tune decision to `ml-architect` for architecture revision. |

### Ablation 3: Focal Loss → Standard Cross-Entropy
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Loss type | `loss_type` | "focal" | "ce" |
| **Hypothesis tested** | Class imbalance (especially PsA underrepresentation) degrades multi-class CE. Focal loss improves minority-class recall. |
| **Expected metric movement** | PsA F1 drops by ≥5% under CE. RA and OA F1 may improve slightly (dominant classes benefit from no γ modulation). |
| **Failure interpretation** | If focal loss does NOT improve PsA F1, either (a) PsA is not actually underrepresented in this dataset (check label distribution), or (b) γ=2.0 is too aggressive — reduce to γ=1.0 or switch to class-weighted CE. |
| **Owning stage** | `ml-validator` (as hyperparameter selection report). |

### Ablation 4: Three-Way → One-vs-All (reduce task complexity)
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Classes | `n_classes` | 4 | (separate binary: RA vs. rest, PsA vs. rest, OA vs. rest) |
| **Hypothesis tested** | The three-way discrimination task is meaningfully harder than one-vs-all. The per-joint model captures disease-specific patterns that help separate all three simultaneously. |
| **Expected metric movement** | Per-disease AUROC for one-vs-all should be ≥ three-way per-class AUROC. The gap measures the "multi-class penalty." |
| **Failure interpretation** | If one-vs-all matches three-way performance (AUROC gap < 0.02), the model is not leveraging disease interactions. The three-way formulation is still valid as a unified model (practicality) but the novelty claim of "joint discrimination" weakens. |
| **Owning stage** | `ml-research` (novelty claim reassessment). |

### Ablation 5: Anatomy Prior Loss → No Explanation Supervision
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Anatomy loss | `use_anatomy_prior_loss` | True → False | `anatomy_prior_loss_weight` = 0 |
| **Hypothesis tested** | Supervising attention with anatomical priors improves explanation plausibility (Dice) without degrading classification accuracy. |
| **Expected metric movement** | Explanation Dice drops by ≥0.1. Classification F1 unchanged or slightly improved (prior acts as regularizer). |
| **Failure interpretation** | If Dice does NOT drop significantly, the model's attention was already aligned with priors — removing the loss saves compute. If classification F1 drops, the prior loss was an important regularizer (keep it). If Dice drops but F1 improves, the prior was constraining the model — consider lower prior weight. |
| **Owning stage** | `ml-validator` (XAI metrics). Route explanation quality findings to `ml-architect`. |

### Ablation 6: Multi-View Fusion → Single-View
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Views | `input_views` | ("PA", "oblique") | ("PA",) |
| **Hypothesis tested** | Additional X-ray views provide complementary diagnostic information that improves classification beyond the best single view. |
| **Expected metric movement** | Patient-level AUROC drops by ≥2% when using only PA view. Certain joints (e.g., PIP in oblique view) may show higher per-joint sensitivity. |
| **Failure interpretation** | If single-view matches multi-view, the secondary view adds no discriminative information. May reflect: (a) view standardization issues (not all patients have both views), or (b) primary view captures all relevant features. If performance drops, the cost of multi-view acquisition must be weighed against the gain. |
| **Owning stage** | `ml-architect` (view fusion strategy) + `ml-research` (clinical practicality assessment). |

### Ablation 7: Detection-based → Tile-Sampling MIL (skip detection)
| Field | Config field | Baseline value | Ablated value |
|---|---|---|---|
| Detection | `detection_model` | "yolov7" | "none" (use grid-based tile sampling instead) |
| **Hypothesis tested** | The detection stage is necessary for per-joint granularity. Without it, tile-sampling MIL (e.g., 32×32 patches) loses anatomical correspondence between joints. |
| **Expected metric movement** | Per-joint accuracy drops significantly (>15%) because tile-sampling cannot reliably isolate individual joints. Patient-level accuracy may drop 5-10%. |
| **Failure interpretation** | If tile-sampling MIL approaches detection-based performance, the annotation granularity is not critical for the task. This would route back to `ml-research` to reassess the "per-joint" novelty claim. |
| **Owning stage** | `ml-architect` (if detection matters) → `ml-research` (if it doesn't). |

---

## 5. Domain-Specific Considerations in Detail

### LM-inspired concerns for this CV architecture

While this is primarily a CV design, two LM-like concerns apply:

| LM concern | Translation to this architecture |
|---|---|
| **Position / order scheme** | Joints have an anatomical ordering (DIP→PIP→MCP→wrist radially outward). The MIL aggregator is **permutation-invariant by design** — it does not use position encoding. This is intentional: disease patterns are defined by which joint *group* is affected, not by the exact spatial configuration. Joint-group labels (metadata) preserve anatomical identity without imposing order bias. |
| **Causal contract** | No causality is needed (diagnosis is not sequential). The model processes all joints simultaneously. |

### CV-specific: Resolution handling

- **Detection resolution**: YOLOv7 operates at 640×640 (standard). Hand X-rays at clinical resolution (2000×2500 px) are downsampled for detection, which is acceptable for joint localization.
- **ROI resolution**: 224×224 crops preserve sufficient detail for erosion/JSN assessment. If computational budget allows, 384×384 crops (DINOv2 ViT-L supports this) may improve fine-detail classification at the cost of 3× compute.

### SciML-specific: Symmetry considerations

- The architecture has **no explicit equivariance** to rotation or reflection. Hand X-rays have a standard anatomical orientation (PA view with fingers up), so rotation equivariance is not needed.
- Translation equivariance is provided by the detection module (convolutional YOLOv7 backbone) but lost at the MIL stage (permutation-invariant pooling). This is acceptable because joint *identity* matters clinically, not joint *position*.

### Training dynamics

```
Training curves expected:
                    ┌─────────────────────────┐
                    │   Loss convergence       │
                    │                          │
    Loss            │  ┌── patient CE loss     │
                    │  │  ── per-joint CE loss │
                    │  │  .. attention entropy │
                    │  │  -· anatomy prior     │
                    │  └── total loss          │
                    │                          │
                    └─────────────────────────┘
                           Epoch
```

- **Order of convergence**: Patient-level loss converges first (fewer parameters, stronger gradient signal from bag label). Per-joint loss takes longer (sparser per-joint labels, higher variance).
- **Anatomy prior loss**: Should decrease monotonically if attention aligns with priors. If it oscillates, prior_weight is too high relative to CE losses.
- **Attention entropy**: Should initially be high (uniform attention) and decrease as the model learns to focus on discriminative joints. If entropy drops to 0 too quickly (<10 epochs), model may be attending to only 1 joint — increase entropy_reg_weight.

---

## 6. Compute Budget Estimate

| Component | Forward | Backward | Memory (B=16, N=30) |
|---|---|---|---|
| YOLOv7 detection | ~50ms | — (frozen) | ~2 GB |
| DINOv2 ViT-L/14 (×30 ROIs) | ~300ms | — (frozen) | ~8 GB |
| MIL aggregator + heads | ~5ms | ~10ms | ~1 GB |
| **Total per batch** | ~355ms | ~10ms (trainable only) | ~11 GB |

- A single RTX 4090 (24 GB) can handle B=16 with N_max=30.
- Bottleneck is the DINOv2 forward pass on 30 ROIs per image. Can be optimized by:
  (a) Caching features after the first epoch (if backbone is frozen)
  (b) Processing ROIs in smaller chunks when GPU memory is tight
- Expected training time: ~2-4 hours for 100 epochs on a 200-patient dataset (frozen backbone).

---

## 7. Summary of Novel Contributions

| # | Contribution | Where in architecture | Novelty level |
|---|---|---|---|
| 1 | **Three-way per-joint arthritis discrimination** (RA/PsA/OA/normal) | 4-class output head, per-joint + patient dual path | **First known system** to jointly discriminate all three at per-joint granularity |
| 2 | **Anatomy-guided explanation alignment** | AnatomyExplanationModule with Dice loss between α_i and disease-specific joint priors | **Novel application** — XAI alignment for arthritis; extends Uddin et al. (2025) to MSK |
| 3 | **Detection-based MIL for multi-disease arthritis** | YOLOv7 → ROI → Foundation → GatedAttn | **Novel combination** — detection-based MIL (common in pathology) applied to rheumatology X-ray with foundation model features |
| 4 | **Multi-view fusion with view embeddings for hand X-ray** | MultiViewFusion with view embeddings + optional cross-attention | **New application** — XFMamba-style fusion adapted to hand X-ray views for arthritis |
| 5 | **Dual-path loss with explanation regularization** | L = w_j·CE_joint + w_p·CE_patient + w_a·L_anatomy + w_e·H(α) | **Novel loss formulation** — jointly optimizing classification + explanation alignment + attention sharpness |
