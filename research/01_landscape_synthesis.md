# AI-Based Arthritis Imaging Analysis — Landscape Synthesis

## Domain Identification

| Domain | Relevance | Rationale |
|---|---|---|
| **Computer Vision (CV)** | Primary | X-ray image classification, object detection (joint localization), feature extraction, explainable AI |
| **Scientific ML (SciML)** | Secondary | Clinical/medical application context, regulatory considerations, domain-specific evaluation |
| **Generative AI** | Supporting | Potential for data augmentation (diffusion-based synthetic X-rays) to address limited labeled data |

---

## 1. Current State of the Art (as of 2025–2026)

### 1.1 Disease-Specific Progress

| Disease | SOTA Performance | Key Methods | Maturity |
|---|---|---|---|
| **RA** | ICC 0.90–0.94 for Sharp/van der Heijde scoring; AUROC 0.93 for Ratingen score | ViT, YOLOv7+ViT, attention CNNs, multistage pipelines (autoscoRA, RADAR) | High — largest body of work; multiple clinical-grade systems |
| **OA** | AUROC 0.88 (Kellgren-Lawrence on hand DIP); F1 0.763 (knee KL grading) | ViT, EfficientNet, ConvNeXt, transfer learning | High — most active DL area; knee OA dominates, hand OA less explored |
| **PsA** | ICC >0.90 (erosion, JSN); very limited multi-class studies | ViT end-to-end (2024), modified Sharp score | **Low** — significantly understudied vs. RA/OA; only 9% of studies in systematic review |

*Sources: Hügle et al. EULAR 2025 (ViT multi-disease); autoscoRA 2026 (ICC 0.9); Systematic review Semin Arthritis Rheum 2025 (88% RA / 9% PsA)*

### 1.2 Key Architecture Families for This Problem

| Architecture | Strengths | Weaknesses | Relevant Papers |
|---|---|---|---|
| **Multistage pipeline (detect → classify)** | 99% joint localization; handles missing joints; interpretable per-stage | Error propagation; complex training pipeline | YOLOv7+ViT (Sci Rep 2025); RADAR (ACR 2025); PsA ViT (MLMI 2024) |
| **Vision Transformer (ViT)** | Global context; strong multi-disease performance; end-to-end possible | Data-hungry; needs ~1000+ patients; less effective with very small feature regions | Hügle et al. EULAR 2025; autoscoRA 2026 |
| **Attention CNN (EfficientNetV2 + attention)** | Strong with limited data; attention on small feature regions; outperformed ViT in some settings | Less global context; architecture is more bespoke | Attention-enhanced mTSS (JMBE 2025) |
| **Attention-based MIL** | Naturally fits per-joint structure; handles variable joint counts; interpretable via attention weights | Bag-level labeling loses fine-grained signal; aggregation may dilute local findings | Anatomy-aware MIL for RA scoring (MICCAI AMAI 2025) |
| **Foundation models (DINOv2, OrthoFoundation, SKELEX)** | Label-efficient (matches supervised with 50% data); strong cross-anatomy transfer; zero-shot localization | Not yet validated for RA/PsA/OA discrimination; large model size; domain gap (trained on mixed MSK data) | OrthoFoundation (arXiv Jan 2026); SKELEX (arXiv Feb 2026); DAX (MICCAI 2025) |

### 1.3 Explainable AI (XAI) Landscape

| Method | Prevalence | Strength for This Application | Limitation |
|---|---|---|---|
| **Grad-CAM** | ~28% of studies; dominant in MSK | Simple, widely understood, good for heatmap visualization | Anatomically imprecise; can focus on spurious correlations |
| **Attention weights (MIL/Transformer)** | Growing | Inherently interpretable; per-joint importance scores | Raw attention weights may not reflect actual importance (attention is not explanation) |
| **Expert-guided explanation loss** | Emerging (2025) | Aligns model focus with clinician-defined ROIs via Dice loss on heatmaps | Requires expert annotations for training; additional engineering |
| **Counterfactuals** | Novel | Clinically intuitive ("what would change this diagnosis?") | Computationally expensive; hard to generate realistic X-ray counterfactuals |

---

## 2. Complexity & Properties Table

| Property | Multistage Pipeline | ViT End-to-End | Attention CNN | Attention MIL | Foundation Model + Head |
|---|---|---|---|---|---|
| **Training complexity** | High (4 stages) | Moderate | Moderate | Moderate | Low (freeze backbone) |
| **Inference complexity** | O(N_detect + N_classify) | O(N²) quadratic | O(N) | O(N_detect + N_classify) | O(N²) for ViT backbone |
| **Data efficiency** | Moderate | Low | Moderate-High | High | **High** (best) |
| **Interpretability** | High (per-stage) | Moderate (attention maps) | High (Grad-CAM per region) | **High** (per-joint weights) | Moderate (via head or CAM) |
| **Handles variable joints?** | Yes (detection-based) | No (fixed grid) | No | **Yes** (bag structure) | Depends on head |
| **Multi-view fusion** | Challenging | Possible (concatenation) | Moderate | Moderate | Possible (via multimodal) |
| **Clinical adoption readiness** | Moderate-high | Low-moderate | Moderate | Low-moderate | Low (new paradigm) |

---

## 3. Data Characteristics & Challenges

The available data has specific properties that constrain architecture choices:

| Property | Implication |
|---|---|
| **Per-joint structural annotations** | Enables MIL or detection-based approaches; allows fine-grained labels |
| **Multiple views/sites** | Requires multi-view fusion strategy; view-specific preprocessing needed |
| **Limited labeled data (clinical)** | Prefers data-efficient methods (SSL pretraining, foundation models, MIL) |
| **Class imbalance across diseases** | RA most prevalent in literature; PsA and OA less common; needs rebalancing |
| **Subtle visual differences between RA/PsA/OA** | Requires high-resolution input; fine-grained discrimination capability |

---

## 4. Principal Novelty Gaps

These gaps are ordered by feasibility within a 6-month Master's thesis timeline:

### Gap A — Three-Way RA vs PsA vs OA Per-Joint Discrimination (HIGH PRIORITY)
- **Current state:** RA-only scoring dominates. OA studies focus on grading severity, not differential diagnosis with RA/PsA. PsA is severely understudied (9% of literature). **No existing system jointly classifies all three at per-joint granularity.**
- **Why it matters:** Clinical differential diagnosis between RA, PsA, and OA is a real-world need — they have overlapping presentations but different treatment pathways.
- **What remains missing:** A unified model that outputs per-joint probability distributions over {RA, PsA, OA, healthy} and aggregates to a patient-level diagnosis.

### Gap B — Anatomy-Grounded XAI for Per-Joint Arthritis Classification (HIGH PRIORITY)
- **Current state:** Most XAI in rheumatology imaging uses Grad-CAM at the image level. Only ~28% of studies use any explainability. Expert-guided explanation loss (Uddin et al. 2025) is promising but not applied to arthritis.
- **Why it matters:** Clinical adoption requires that predictions are grounded in pathologically relevant anatomy (e.g., MCP joints for RA, DIP joints for OA, PIP joints for PsA).
- **What remains missing:** Alignment of model explanations with disease-specific anatomical priors (e.g., which joints are typically affected by each disease).

### Gap C — Multi-View Fusion for Arthritis X-Ray Analysis
- **Current state:** Most arthritis X-ray models use single-view input (usually hand PA). Multi-view architectures like XFMamba (MICCAI 2025) exist for general MSK but haven't been applied to arthritis differential diagnosis.
- **Why it matters:** This dataset includes multiple views/sites, creating an opportunity to model disease patterns that appear differently across anatomical locations (e.g., hand vs. foot in PsA).
- **What remains missing:** A principled fusion strategy that accounts for the differential diagnostic value of each view/site.

### Gap D — Foundation Model Adaptation for Arthritis-Specific Features
- **Current state:** OrthoFoundation (1.2M knee images) and SKELEX (1.2M MSK radiographs) exist but were **not** evaluated on RA/PsA/OA discrimination. They were trained on general MSK data that may underrepresent arthritic erosion patterns.
- **What remains missing:** Systematic evaluation and fine-tuning of these foundation models for three-way arthritis classification. This is an immediately feasible contribution.

### Gap E — Longitudinal Modeling of Joint Damage Progression
- **Current state:** Scoring systems (Sharp/van der Heijde) inherently track change over time. DL systems (autoscoRA) can detect progression with 70% agreement with humans.
- **Feasibility note:** May require multiple timepoints per patient, which may not be available in this dataset. Marked as secondary direction.

---

## 5. Recommended Direction

### Primary Hypothesis

> **An anatomy-aware, per-joint multi-instance learning framework using a foundation model backbone (e.g., DINOv2 or OrthoFoundation) can jointly discriminate RA, PsA, and OA from hand X-rays at per-joint granularity while producing anatomically grounded explanations, matching or exceeding the performance of single-disease specialist models.**

### Justification

1. **Architecture fit:** MIL naturally maps to the per-joint annotation structure. Foundation model backbone addresses the limited-data constraint (most common problem in clinical theses).
2. **Gap alignment:** Directly fills Gap A (three-way discrimination) and Gap B (anatomy-grounded XAI) simultaneously.
3. **Feasibility (6 months):** MIL + frozen backbone requires training only a lightweight attention aggregator and classifier head. Detection can leverage off-the-shelf YOLOv7 or an anatomy-aware cropping strategy using available joint annotations.
4. **Novelty threshold:** No existing work jointly discriminates all three diseases with per-joint explanations. The closest work (Bo et al. 2025, MICCAI AMAI) addresses RA-only scoring.

### Expected Observable Behavior

- Per-joint classification accuracy >80% on held-out test set for the three-class discrimination task
- Attention weights show higher mass on disease-relevant joints (MCP/PIP for RA, DIP for OA, PIP/DIP for PsA)
- Patient-level aggregation (mean/max/learned pooling of joint logits) improves over image-level baseline by >5% F1
- Foundation model backbone yields >10% improvement over scratch-trained CNN with same annotation budget

### Falsification Condition

The hypothesis is falsified if:
1. Per-joint three-way accuracy does not significantly exceed a strong baseline (e.g., image-level ViT fine-tuned with same data), OR
2. Attention-based explanations do not consistently highlight disease-relevant joint groups above random chance (measured by overlap with anatomical priors), OR
3. The MIL framework fails to outperform a simple pooling baseline (e.g., average all detected joint logits), indicating the per-joint modeling adds no value.
