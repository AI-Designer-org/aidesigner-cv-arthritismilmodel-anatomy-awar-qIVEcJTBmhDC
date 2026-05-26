# Training & Reproduction

## Environment

Tested with:

- **Python:** 3.10+
- **PyTorch:** 2.x (tested with 2.1+)
- **CUDA:** 11.8+, tested on NVIDIA RTX 4090 (24 GB) and A100
- **torchvision:** 0.16+ (for roi_align, pretrained backbones)
- **Other:** YOLOv7 via `torch.hub` (WongKinYiu/yolov7), DINOv2 via `torch.hub` (facebookresearch/dinov2)

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytest numpy scikit-learn captum   # captum for Grad-CAM (optional)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Note: The tiny_debug backbone works on CPU for smoke-testing. DINOv2 ViT-L/14 requires a GPU (16+ GB VRAM recommended for B=16 with 30 joints).

## Default hyperparameters

All hyperparameters are defined in `coder/config.py::ModelConfig`. The table below shows defaults and their rationale.

| Field | Default | Rationale |
|---|---|---|
| `backbone` | `"dinov2_vitl14"` | Best balance of feature quality (1024 dim) and compute; frozen by default |
| `d_model` | 1024 | Auto-set from backbone choice (ViT-L/14) |
| `backbone_frozen` | `True` | Prevents overfitting on limited clinical data (~1–2M trainable params) |
| `use_lora` | `False` | LoRA fine-tuning adds ~2M params if domain adaptation is needed |
| `detection_model` | `"yolov7"` | Off-the-shelf joint localization; bypass with `"none"` if coordinates available |
| `max_joints_per_view` | 30 | Covers all hand joints (14 DIP+PIP+MCP + wrist + CMC + carpals) |
| `input_views` | `("PA",)` | Default single-view; extend to multi-view for fusion experiments |
| `multi_view_fusion` | `"concat"` | Concatenation + view embeddings; `"cross_attention"` for learned correspondence |
| `mil_gated` | `True` | Gated attention (Ilse et al.) for non-linear importance weighting |
| `mil_hidden_dim` | 512 | Hidden dim of attention network (V and U projections) |
| `n_classes` | 4 | RA, PsA, OA, normal |
| `patient_pooling` | `"attention"` | Attention-weighted pooling; alternatives: `"mean"`, `"max"` |
| `loss_type` | `"focal"` | Focal loss (γ=2.0) to address class imbalance |
| `focal_gamma` | 2.0 | Focusing parameter; higher = more focus on hard examples |
| `per_joint_loss_weight` | 1.0 | Weight for per-joint CE loss in total objective |
| `patient_loss_weight` | 1.0 | Weight for patient-level CE loss |
| `entropy_reg_weight` | 0.01 | Light entropy regularization — encourages attention diversity |
| `use_anatomy_prior_loss` | `False` | Disabled by default; enable with prior weight 0.1 for XAI experiments |
| `explanation_method` | `"attention"` | Inherent attention weights as primary explanation |

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Peak learning rate | 1e-4 | Linear warmup over 500 steps, cosine decay to 0 |
| Batch size | 16 | Gradient accumulation as needed for larger effective batch |
| Weight decay | 0.01 | Excluded from bias and LayerNorm parameters |
| Gradient clip | 1.0 | Global norm clipping |
| Precision | fp16 mixed | GradScaler recommended; bf16 also supported on Ampere+ |
| Epochs | 100 | Early stopping patience of 15 epochs |
| Warmup steps | 500 | Linear warmup to peak LR |
| Scheduler | Cosine annealing | Decays to 0 over remaining steps |

### CV-specific training details

| Setting | Value | Rationale |
|---|---|---|
| Augmentation | Random rotation ±10°, brightness/contrast ±10%, Gaussian noise σ=0.02 | Mild augmentation — aggressive transforms can distort fine erosion patterns |
| Elastic transform | Off | May distort anatomical structure; use only with clinical validation |
| ROI size | 224×224 | Standard DINOv2 input; 384×384 possible for fine detail (3× compute cost) |
| Per-image normalization | Before DINOv2 input | Removes acquisition-level variance in X-ray brightness/contrast |
| Patient-level data split | GroupKFold | Prevents same-patient images from leaking between train and test |

### Multi-objective loss balancing

The total loss combines four terms:

```python
L_total = w_j * L_joint + w_p * L_patient + w_a * L_anatomy + w_e * H(α)
```

Expected training dynamics:
- **Patient-level loss** converges first (stronger gradient signal from bag label)
- **Per-joint loss** takes longer (sparser per-joint labels, higher variance)
- **Anatomy prior loss** should decrease monotonically if attention aligns with priors; if it oscillates, reduce `anatomy_prior_loss_weight`
- **Attention entropy** should start high (uniform attention) and decrease as the model learns to focus on discriminative joints; if entropy drops to 0 too quickly (<10 epochs), increase `entropy_reg_weight`

## Expected behavior

> TODO: unverified — no reference training run on real data has been completed. The following are expected behaviors based on architecture design and synthetic validation.

- Frozen backbone training should converge in 50–100 epochs on a 200-patient dataset
- Per-joint macro-F1 expected >0.70 (hypothesis: >0.80 requires sufficient data)
- Attention entropy should stabilize around 1.5–2.5 nats (4-class, moderate sharpness)
- Anatomy prior Dice should be >0.6 when the loss is enabled

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16/fp16 float-sensitive op in attention softmax | Softmax is cast to float32 in `GatedAttentionMIL.forward` — verify GradScaler is used for fp16 |
| Attention mass on padded positions | Defective masked softmax | Check `masked_fill(mask == 0, -1e9)` — verify mask values are 1.0 for real joints |
| DINOv2 OOM | 30 ROIs × 16 batch × ViT-L is memory-heavy | Reduce batch size to 8; enable gradient checkpointing (`use_checkpoint=True`); cache features after epoch 1 |
| YOLOv7 not finding joints | Confidence threshold too high | Lower `detection_confidence` (default 0.5) to 0.3 for small or subtle joints |
| Multi-view fusion no improvement | Secondary view adds no discriminative signal | Single-view default is the safe baseline — run ablation to verify benefit before deploying multi-view |
| Anatomy loss not decreasing | `prior_weight` too high relative to CE losses | Reduce `anatomy_prior_loss_weight` from 0.1 to 0.05 or lower |
| Attention entropy collapses to 0 | Model attends to a single joint only | Increase `entropy_reg_weight` from 0.01 to 0.1 |
| Focal loss not helping PsA class | γ=2.0 may be too aggressive for very rare classes | Reduce to γ=1.0, or use class-weighted CE with inverse-frequency weights |
| Frozen backbone underperforms scratch CNN | DINOv2 domain gap too large for X-ray | Switch to LoRA (`use_lora=True`, rank=8) or full fine-tuning (`backbone_frozen=False`) |
| Gradients NaN in focal loss | `ignore_index=-100` may produce log(0) | Verify masked positions get -1e9 fill before softmax; fallback to CE if persistent |
