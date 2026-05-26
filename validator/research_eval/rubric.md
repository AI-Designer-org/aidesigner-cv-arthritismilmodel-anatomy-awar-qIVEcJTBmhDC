# Research-Quality Evaluation Rubric
## Anatomy-Aware Per-Joint MIL for Arthritis Discrimination (RA / PsA / OA)

**Domain:** Computer Vision (primary) + Scientific ML (secondary)
**Task Level:** level_1 — concrete method direction provided by the thesis proposal
**Upstream contract present:** Yes (research lifecycle contract from `ml-research`)

---

## Scoring Scale (0–5)

| Score | Meaning |
|-------|---------|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## Dimension 1: Novelty

### Questions
- Does the architecture actually address a gap not covered by prior work?
- Is the novelty claim specific enough to be falsifiable?
- Can the novelty be demonstrated with the proposed evaluation?

### Evidence expected
- **Gap A (Three-way discrimination):** Per-joint macro-F1 > 0.80 on held-out test with all 4 classes; confusion matrix shows balanced discrimination (RA vs PsA vs OA vs normal)
- **Gap B (Anatomy-grounded XAI):** Explanation Dice > 0.6 between attention weights and disease-specific joint priors; Deletion/Insertion AUC > 0.8
- **Gap D (Foundation model adaptation):** Frozen backbone outperforms scratch-trained ResNet-50 by >5% F1

### CV-specific questions
- Does the benchmark suite test invariance/equivariance claims, resolution behavior, and feature quality?
- Does it compare against a simple CNN/ViT-style baseline?

---

## Dimension 2: Experimental Comprehensiveness

### Questions
- Are all required baselines implemented and runnable?
- Are all required evaluation metrics computed?
- Are all blocking unknowns identified and mitigated?
- Are the proposed ablations tied to specific architectural claims?

### Required baselines (from research contract)
1. Image-level ViT fine-tuned on same dataset
2. Per-joint CNN baseline (YOLO + ResNet-50 + majority vote)
3. Single-disease specialist models
4. Ablation: frozen backbone vs. full fine-tuning
5. Ablation: MIL attention vs. average pooling
6. Ablation: focal loss vs. cross-entropy
7. Ablation: multi-view vs. single-view
8. Ablation: detection-based vs. tile-sampling MIL
9. Ablation: anatomy prior loss on vs. off

### Required metrics (from research contract)
| Metric | Status |
|--------|--------|
| Per-joint macro-F1, weighted-F1 | Implemented in eval harness |
| Patient-level accuracy, AUROC | Implemented in eval harness |
| Explanation Dice (attention vs. anatomical priors) | Implemented in explanation module |
| Deletion / Insertion metrics | TODO: post-hoc analysis script |
| Expected Calibration Error per class | TODO: calibration analysis |
| Performance stratified by severity, site, view | TODO: stratified evaluation |

---

## Dimension 3: Theoretical Foundation

### Questions
- Are the inductive biases of each architectural decision stated and justified?
- Is there a clear connection between clinical domain knowledge and design choices?
- Are the limits of the architecture acknowledged?

### Expected
- 10 inductive biases documented in architecture diagram (completed)
- Per-justification traceability to research claims (completed)
- Risk flags identified and mitigation strategies described (completed)
- Falsification conditions stated for each claim (completed)

---

## Dimension 4: Result Analysis

### Questions
- Are the evaluation results reproducible from the provided code?
- Is there error analysis beyond aggregate metrics?
- Are failure modes discussed?

### Current state
- TODO: Full evaluation suite requires real X-ray data (not available in synthetic-only validation)
- Synthetic benchmark provides mechanical verification but no clinical insight
- Confusion matrix analysis, per-disease error analysis: TODO

---

## Dimension 5: Implementation Reproducibility

### Questions
- Can someone reproduce the reported results from the code + config?
- Are seeds set deterministically?
- Are training hyperparameters fully specified?

### Current state
- Seed set to 42 in config
- All hyperparameters in single ModelConfig dataclass
- config.py has `__post_init__` validation
- Deterministic training requires `torch.use_deterministic_algorithms(True)` — not enforced in code
- Data preprocessing pipeline: TODO (requires real X-ray data)

---

## Dimension 6: Writing Readiness

### Questions
- Is the architecture documented end-to-end?
- Are there visual diagrams for the pipeline?
- Are the inductive biases and design decisions explained in accessible language?

### Current state
- Architecture diagram in `03_architecture_diagram.md` — complete with ASCII art
- Design justification in `04_design_justification.md` — comprehensive
- Inductive bias table with one-sentence statements for all 10 design choices
- Research justification in `research/01_landscape_synthesis.md`
- TODO: Figures in publication quality (thesis-ready)

---

## Domain-Specific Research Questions for This Architecture

### CV Domain
| Question | Status |
|----------|--------|
| Does it test invariance/equivariance? | ✅ Translation invariance tested; rotation invariance via augmentation |
| Does it test resolution behavior? | ⚠️ ROI resolution fixed at 224×224; no multi-resolution test |
| Does it test feature quality? | ✅ Linear probe benchmark implemented |
| Does it compare against a baseline? | ⚠️ Baseline configs specified but need real data for comparison |
| Does it handle variable joint counts? | ✅ Padding + masking mechanism tested |
| Does it test multi-view fusion benefit? | ✅ Multi-view vs. single-view ablation implemented |

### Scientific ML Domain
| Question | Status |
|----------|--------|
| Does it test explanation faithfulness? | ⚠️ Dice scores can be computed; Deletion/Insertion TODO |
| Does it test calibration? | ❌ ECE analysis not yet implemented |
| Does it test robustness to data shift? | ❌ No domain shift evaluation |
| Does it provide clinically interpretable outputs? | ✅ Per-joint attention + top-k joint explanations |
| Are anatomical priors validated? | ✅ Prior construction documented; loss computes Dice alignment |
