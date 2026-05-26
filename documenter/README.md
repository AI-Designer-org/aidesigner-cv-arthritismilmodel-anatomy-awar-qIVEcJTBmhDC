# ArthritisMILModel — Anatomy-Aware Per-Joint Multi-Instance Learning for Arthritis Discrimination

A detection-based multi-instance learning framework for joint-level classification of rheumatoid arthritis, psoriatic arthritis, and osteoarthritis from hand X-rays, with anatomically grounded explanations via gated attention weights.

This architecture addresses the gap that no existing system jointly discriminates all three diseases at per-joint granularity from hand radiographs. The core mechanism is a YOLOv7-guided ROI pipeline feeding a frozen DINOv2 foundation model backbone, whose per-joint features are aggregated through gated attention MIL (Ilse et al., 2018) into a patient-level diagnosis. A dual-path classification head produces both per-joint and patient-level predictions, and an optional anatomy-guided explanation loss aligns attention weights with disease-specific joint priors (RA→MCP/PIP, OA→DIP, PsA→PIP/DIP).

> TODO: unverified — the architecture is implemented and mechanically verified on synthetic data, but requires real clinical X-ray data to assess per-joint classification accuracy, explanation fidelity, and comparison against baselines.

## Highlights

- **Three-way per-joint arthritis discrimination** — first known system to classify RA, PsA, and OA at per-joint granularity from hand X-rays; see [ARCHITECTURE.md#section-1](docs/ARCHITECTURE.md#1-motivation)
- **Anatomically grounded explanations** — gated attention weights double as per-joint importance scores, optionally supervised by disease-specific anatomical priors via a Dice loss; see [ARCHITECTURE.md#section-3](docs/ARCHITECTURE.md#3-the-core-component)
- **Data-efficient via frozen foundation model** — DINOv2 ViT-L/14 backbone is frozen by default, keeping trainable parameters below 2M to prevent overfitting on limited clinical data; see [ARCHITECTURE.md#section-5](docs/ARCHITECTURE.md#5-design-decisions)
- **Multi-view fusion** — supports concatenative or cross-attention fusion across multiple X-ray views (PA, oblique, lateral) with learned view embeddings; see [ARCHITECTURE.md#section-3](docs/ARCHITECTURE.md#3-the-core-component)

## Quick start

```bash
# Smoke test with tiny_debug backbone (no pretrained weights needed)
cd coder/
python smoke_test.py

# Full test suite
pip install -r requirements.txt
pytest ../validator/test_model.py -v --tb=short

# Ablation runner (synthetic data with tiny_debug)
python ../validator/run_ablations.py --epochs 2 --steps_per_epoch 10
```

## Repository layout

```
coder/
  config.py            ModelConfig — all hyperparameters in one dataclass
  model.py             ArthritisMILModel — end-to-end pipeline composition
  backbone.py          FoundationBackbone — DINOv2/ResNet/LoRA wrappers
  layers.py            Core blocks: JointDetection, ROIExtractor, MultiViewFusion, GatedAttentionMIL
  heads.py             ArthritisClassificationHead — dual-path per-joint + patient classifier
  explanation.py       AnatomyExplanationModule — attention-based XAI with anatomical priors
  losses.py            Focal loss and multi-objective compute_loss
  smoke_test.py        6-test smoke suite for shape/gradient/XAI correctness
validator/
  test_model.py        30+ pytest tests across shapes, gradients, numerics, CV properties
  run_ablations.py     Configurable ablation runner (7 ablations)
  profile_model.py     torch.profiler script for memory/FLOPs estimation
  research_eval/       Claim grounding, experiment coverage, scorecard, rubric
research/
  *_landscape_synthesis.md    Literature review and gap analysis
  *_lifecycle_contract.yaml   Novelty claims, baselines, evaluation requirements
  *_implementation_roadmap.md 6-month thesis plan
architect/
  *_model_config.py           ModelConfig spec
  *_core_blocks.py            Pseudocode for core components
  *_architecture_diagram.md   ASCII diagram and inductive biases
  *_design_justification.md   Traceability table, risk flags, ablations
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design, inductive biases, equations, shape evolution
- [docs/TRAINING.md](docs/TRAINING.md) — environment setup, hyperparameters, training recipe, troubleshooting
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — results, ablations, profiling, research-quality evaluation
- [docs/API.md](docs/API.md) — module-level API reference with shape contracts

## Citation

```bibtex
@misc{arthritis-mil-2026,
  title  = {Anatomy-Aware Per-Joint Multi-Instance Learning for
            Rheumatoid Arthritis, Psoriatic Arthritis, and
            Osteoarthritis Discrimination from Hand X-rays},
  author = {<TODO>},
  year   = {2026},
  note   = {Generated via ml-designer pipeline. Medical University of Vienna,
            Department of Rheumatology, AutoPiX consortium}
}
```
