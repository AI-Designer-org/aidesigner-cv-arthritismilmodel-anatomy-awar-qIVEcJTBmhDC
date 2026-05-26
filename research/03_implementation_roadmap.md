# Implementation Roadmap — 6-Month Thesis Plan

## Recommended Architecture: Anatomy-Aware Per-Joint MIL with Foundation Model Backbone

### High-Level Design

```
Input X-ray (hand, multiple views)
        │
        ▼
┌─────────────────────────────┐
│  Joint Detection Module      │  YOLOv7 or anatomy-aware cropping
│  (detects N joint ROIs)      │  using available joint annotations
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Foundation Model Backbone   │  DINOv2 / OrthoFoundation / SKELEX
│  (per-joint feature extract) │  FROZEN or LoRA fine-tuned
└─────────────────────────────┘
        │
        ├── f_joint_1 ──┐
        ├── f_joint_2 ──┤
        ├── f_joint_3 ──┤  ...
        └── f_joint_N ──┘
        │                    │
        ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐
│  Attention MIL    │  │  Disease-specific    │
│  Aggregator       │──│  Classification Head │
│  (per-joint       │  │  {RA, PsA, OA, norm} │
│   attention       │  └─────────────────────┘
│   weights α_i)    │           │
└──────────────────┘           ▼
        │              ┌─────────────────────┐
        │              │  Patient-level       │
        │              │  Aggregation         │
        │              │  (mean / max /       │
        │              │   learned pooling)   │
        │              └─────────────────────┘
        │                       │
        ▼                       ▼
┌──────────────────────────────────────────┐
│  Explainability Outputs                  │
│  • Per-joint attention weights (α_i)      │
│  • Grad-CAM overlays on top-k joints      │
│  • Anatomical alignment score (Dice with  │
│    disease-specific joint priors)         │
└──────────────────────────────────────────┘
```

### Design Rationale

| Component | Choice Rationale |
|---|---|
| **Joint detection** | Enables per-joint classification; handles variable joint counts; detection-based MIL outperforms tile-sampling MIL when annotations exist |
| **Foundation model backbone** | Addresses limited labeled data; DINOv2 is publicly available (fallback); OrthoFoundation/SKELEX if weights are accessible |
| **Frozen backbone + attention aggregator** | Only trains a lightweight head — feasible in 6 months on a single GPU; avoids overfitting |
| **Disease-specific head** | Separate classification per disease allows modeling different radiographic signatures |
| **Patient-level aggregation** | Bridges per-joint predictions to clinical decision (patient diagnosis); learned pooling via gated attention |

---

## Monthly Milestones

### Month 1: Data & Infrastructure
| Week | Task | Deliverable |
|---|---|---|
| 1 | Data access agreement, IRB/ethics, data inventory | Signed agreements; dataset manifest |
| 2 | Data cleaning, normalization, quality checks | Cleaned dataset + exclusion log |
| 3 | Literature deep-dive; finalize architecture | Literature review section; architecture spec |
| 4 | Implement preprocessing pipeline (resize, normalization, augmentation) | Preprocessing code + notebook |

**Key decisions to make:** Joint annotation format (bbox vs. label), view/site heterogeneity assessment, foundation model availability

### Month 2: Baseline Models
| Week | Task | Deliverable |
|---|---|---|
| 1 | Joint detection module (YOLOv7 fine-tuning or coordinate extraction) | Detection model with >95% mAP |
| 2 | Image-level ViT baseline | Baseline results (accuracy, F1, AUROC) |
| 3 | Per-joint CNN baseline (YOLO + ResNet-50) | Baseline results reported |
| 4 | Single-disease specialist models (RA-only, PsA-only, OA-only) | Baseline comparison table |

### Month 3: Core Model Development
| Week | Task | Deliverable |
|---|---|---|
| 1 | Foundation model backbone integration (DINOv2) | Feature extraction pipeline |
| 2 | Attention MIL aggregator implementation | MIL aggregator module |
| 3 | Disease-specific classification head + patient-level aggregation | Full model forward pass working |
| 4 | Training loop, validation, hyperparameter tuning | Tuned model checkpoint |

### Month 4: Multi-Disease & XAI
| Week | Task | Deliverable |
|---|---|---|
| 1 | Three-way joint training (RA + PsA + OA) | Multi-disease model checkpoint |
| 2 | Explainability: attention weight analysis, Grad-CAM | Explanation maps + quantitative metrics |
| 3 | Anatomical priors integration and alignment evaluation | Explanation Dice scores |
| 4 | Ablation experiments (frozen vs. fine-tuned, MIL vs. pooling) | Ablation results table |

### Month 5: Evaluation & Validation
| Week | Task | Deliverable |
|---|---|---|
| 1 | Full evaluation suite (per-joint, patient-level, per-disease, per-view) | Evaluation results |
| 2 | Robustness analysis (stratified by severity, site, view) | Robustness report |
| 3 | Calibration analysis (ECE per disease) | Calibration curves |
| 4 | Error analysis (confusion matrix analysis, failure cases) | Error analysis report |

### Month 6: Write-Up & Defense
| Week | Task | Deliverable |
|---|---|---|
| 1 | Thesis draft (methods + results) | Methods + results chapters |
| 2 | Thesis draft (introduction + discussion) | Full draft |
| 3 | Revision, figures, tables | Final thesis |
| 4 | Defense preparation | Presentation + Q&A prep |

---

## Technical Stack Recommendation

| Component | Recommended Tool | Fallback |
|---|---|---|
| **Framework** | PyTorch 2.x + PyTorch Lightning | PyTorch + ignite |
| **Detection** | YOLOv7 (Ultralytics) | MMDetection |
| **Foundation model** | DINOv2 (ViT-L/14) via `torch.hub` | ResNet-152 ImageNet-pretrained |
| **MIL** | Custom: `torch.nn.MultiheadAttention` + learnable query | Attention MIL (Ilse et al., 2018) |
| **Explainability** | Captum (Grad-CAM, Integrated Gradients) | Zennit |
| **Preprocessing** | Albumentations + MONAI | torchvision.transforms |
| **Metrics** | scikit-learn (F1, AUROC, ECE, Dice) | Custom |
| **Visualization** | matplotlib + seaborn + ITK-Snap (for overlay) | napari |
| **Experiment tracking** | Weights & Biases (free academic) | MLflow, TensorBoard |
| **Hardware** | 1x NVIDIA RTX 4090 (24GB) or A100 (cloud) | Google Colab Pro |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Small dataset (<100 patients)** | Medium | High | Switch to one-vs-all classification (RA vs. rest); use stronger augmentation; SimCLR pretraining on unlabeled data |
| **Foundation model weights unavailable** | Low-Medium | Medium | Fall back to DINOv2 (public); or ImageNet-pretrained ViT with careful regularization |
| **PsA samples very limited** | High | High | Focal loss per class; synthetic sampling; PsA-only held-out (report as limitation) |
| **Low inter-annotator agreement** | Medium | Medium | Use majority vote labeling; report agreement statistics; consider soft-label training |
| **Multi-view heterogeneity** | Medium | Medium | Simple: concatenation or separate attention streams per view. Complex: XFMamba-style fusion |
| **Compute constraints** | Low | Medium | Use frozen backbone + small classifier; batch size adaptations; gradient checkpointing |

---

## Key References

1. Bo et al., "Interpretable Rheumatoid Arthritis Scoring via Anatomy-aware Multiple Instance Learning," MICCAI AMAI Workshop, 2025. [arXiv:2508.06218]
2. Hügle et al., "A Vision Transformer Based Application for the Combined Detection and Grading of Osteoarthritis, CPPD and Rheumatoid Arthritis on Hand Radiographs," EULAR 2025.
3. Multistage deep learning for Sharp score prediction in RA, *Scientific Reports* (Nature), 2025. [PMC11772782]
4. Vision Transformer Model for Automated Radiographic Assessment of Joint Damage in Psoriatic Arthritis, *MLMI* (Springer), 2024.
5. autoscoRA: Deep Learning to Automate Sharp/van der Heijde Scoring, *Arthritis & Rheumatology*, 2026.
6. Uddin et al., "Expert-Guided Explainable Few-Shot Learning for Medical Image Diagnosis," arXiv:2509.08007, 2025.
7. OrthoFoundation: A multimodal vision foundation model for generalizable knee pathology, arXiv:2601.18250, 2026.
8. SKELEX: A generalizable large-scale foundation model for musculoskeletal radiographs, arXiv:2602.03076, 2026.
9. XFMamba: Cross-Fusion Mamba for Multi-View Medical Image Classification, MICCAI 2025.
10. Bilgin E., "Current application, possibilities, and challenges of AI in RA, axSpA, and PsA," *Ther Adv Musculoskelet Dis*, 2025.
