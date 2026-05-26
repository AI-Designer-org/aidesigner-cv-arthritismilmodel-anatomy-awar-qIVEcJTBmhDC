# Changelog

## [0.1.0] — 2026-05-26
### Added
- Initial implementation of ArthritisMILModel: anatomy-aware per-joint multi-instance learning for three-way RA/PsA/OA discrimination from hand X-rays.
- Config module (`ModelConfig`) with all hyperparameters in a single dataclass, `__post_init__` validation, and auto-set `d_model` from backbone choice.
- Foundation model backbone module (`FoundationBackbone`) supporting DINOv2 (ViT-B/L/G), ResNet-152, and a tiny_debug backbone for smoke testing. Includes LoRA parameter-efficient fine-tuning.
- Joint detection module (`JointDetectionModule`) wrapping YOLOv7 via torch.hub, configurable confidence and IoU thresholds.
- ROI feature extractor (`ROIFeatureExtractor`) using `torchvision.ops.roi_align` with differentiable cropping and padding/masking for variable joint counts.
- Multi-view fusion module (`MultiViewFusion`) with learned view embeddings, concatenative and cross-attention fusion modes.
- Gated attention MIL aggregator (`GatedAttentionMIL`) with pre-norm tanh×sigmoid gating and masked softmax (Ilse et al., 2018).
- Dual-path classification head (`ArthritisClassificationHead`) supporting both unified softmax and disease-specific binary heads.
- Anatomy-guided explanation module (`AnatomyExplanationModule`) with built-in disease-joint priors (RA→MCP/PIP, PsA→PIP/DIP, OA→DIP/CMC) and a fully-vectorized Dice explanation loss.
- Multi-objective training loss (`compute_loss`, `focal_loss`) combining per-joint CE, patient-level CE, anatomy prior Dice loss, and attention entropy regularization.
- Unit test suite (30+ tests) covering shape correctness, gradient flow, numeric stability, CV invariance properties, backbone correctness, loss correctness, multi-view fusion, and ROI extraction.
- Synthetic benchmarks: linear probe on frozen features, mode collapse check, attention masking correctness, permutation invariance.
- Ablation runner with 7 configurable ablations: MIL→Average Pooling, Frozen→Full FT, Focal→CE, Three-way→One-vs-All, Anatomy Prior On/Off, Multi/Single-View, Detection→Tile Sampling.
- Profiling script (`profile_model.py`) using `torch.profiler` for memory/FLOPs estimation and operator breakdown.
- Research evaluation files: claim grounding report, experiment coverage report, quality rubric, and scorecard with blocking gaps.
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API reference, and this changelog.
