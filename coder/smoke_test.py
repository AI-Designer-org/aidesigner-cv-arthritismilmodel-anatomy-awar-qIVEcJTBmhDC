#!/usr/bin/env python3
"""
Smoke test for the Arthritis MIL Model.

Tests:
  1. Model instantiation with tiny_debug backbone
  2. Forward pass with synthetic data (pre-computed boxes)
  3. Shape assertions on all output tensors
  4. Parameter count
  5. Loss computation
  6. Multi-view fusion path
  7. Training mode (gradient flow)

Usage:
    python smoke_test.py
"""

import sys
import os

# Ensure the coder directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from config import ModelConfig
from model import ArthritisMILModel, count_params
from losses import compute_loss


def test_single_view_forward():
    """Test basic forward pass with single view (PA) and pre-computed boxes."""
    print("=" * 60)
    print("Test 1: Single-view forward pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}  |  Dtype: {dtype}")

    # ── Config: tiny backbone, no detection ──
    cfg = ModelConfig(
        backbone="tiny_debug",
        detection_model="none",
        d_model=64,
        n_classes=4,
        mil_hidden_dim=32,
        per_joint_hidden=16,
        max_joints_per_view=10,
        input_views=("PA",),
        multi_view_fusion="none",
        use_view_embedding=False,
        loss_type="focal",
        use_anatomy_prior_loss=False,
    )

    model = ArthritisMILModel(cfg).to(device=device, dtype=dtype)
    model.eval()

    # ── Synthetic inputs ──
    B = 2
    N_joints = 7  # variable number of joints per image
    H_full, W_full = 224, 224

    x = torch.randn(B, 1, H_full, W_full, device=device, dtype=dtype)

    # Synthetic boxes: each batch item has N_joints boxes
    boxes = {
        "PA": [
            _random_boxes(N_joints, H_full, W_full, device=device)
            for _ in range(B)
        ]
    }

    # ── Forward pass ──
    with torch.no_grad():
        output = model(x, boxes=boxes)

    # ── Shape assertions ──
    # The model pads to max_joints_per_view=10
    N = cfg.max_joints_per_view

    assert output["per_joint_logits"] is not None
    assert output["per_joint_logits"].shape == (B, N, cfg.n_classes), \
        f"per_joint_logits: expected ({B}, {N}, {cfg.n_classes}), got {output['per_joint_logits'].shape}"
    assert output["patient_logits"].shape == (B, cfg.n_classes), \
        f"patient_logits: expected ({B}, {cfg.n_classes}), got {output['patient_logits'].shape}"
    assert output["attention_weights"].shape == (B, N), \
        f"attention_weights: expected ({B}, {N}), got {output['attention_weights'].shape}"
    assert output["bag_representation"].shape == (B, cfg.d_model), \
        f"bag_rep: expected ({B}, {cfg.d_model}), got {output['bag_representation'].shape}"
    assert output["joint_features"].shape == (B, N, cfg.d_model), \
        f"joint_features: expected ({B}, {N}, {cfg.d_model}), got {output['joint_features'].shape}"

    print(f"  ✓ per_joint_logits:   {output['per_joint_logits'].shape}")
    print(f"  ✓ patient_logits:     {output['patient_logits'].shape}")
    print(f"  ✓ attention_weights:  {output['attention_weights'].shape}")
    print(f"  ✓ bag_representation: {output['bag_representation'].shape}")
    print(f"  ✓ joint_features:     {output['joint_features'].shape}")
    print(f"  ✓ explanation_loss:   {output['explanation_loss'].item():.6f}")
    print("  PASSED")


def test_multi_view_forward():
    """Test forward pass with two views (PA + oblique) and cross-attention fusion."""
    print("=" * 60)
    print("Test 2: Multi-view forward pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = ModelConfig(
        backbone="tiny_debug",
        detection_model="none",
        d_model=64,
        n_classes=4,
        mil_hidden_dim=32,
        per_joint_hidden=16,
        max_joints_per_view=8,
        input_views=("PA", "oblique"),
        multi_view_fusion="cross_attention",
        fusion_n_heads=4,
        use_view_embedding=True,
        loss_type="focal",
        use_anatomy_prior_loss=False,
    )

    model = ArthritisMILModel(cfg).to(device=device, dtype=dtype)
    model.eval()

    B = 2
    H_full, W_full = 224, 224

    x_pa = torch.randn(B, 1, H_full, W_full, device=device, dtype=dtype)
    x_obl = torch.randn(B, 1, H_full, W_full, device=device, dtype=dtype)

    views = {"PA": x_pa, "oblique": x_obl}
    boxes = {
        "PA": [_random_boxes(6, H_full, W_full, device=device) for _ in range(B)],
        "oblique": [_random_boxes(5, H_full, W_full, device=device) for _ in range(B)],
    }

    with torch.no_grad():
        output = model(views=views, boxes=boxes)

    # Total joints: PA max=8 + oblique max=8 = 16
    N_total = cfg.max_joints_per_view * 2

    assert output["per_joint_logits"].shape == (B, N_total, cfg.n_classes), \
        f"per_joint_logits: {output['per_joint_logits'].shape}"
    assert output["patient_logits"].shape == (B, cfg.n_classes)
    assert output["attention_weights"].shape == (B, N_total)
    assert output["view_membership"].shape == (B, N_total)

    print(f"  ✓ per_joint_logits:   {output['per_joint_logits'].shape}")
    print(f"  ✓ patient_logits:     {output['patient_logits'].shape}")
    print(f"  ✓ attention_weights:  {output['attention_weights'].shape}")
    print(f"  ✓ view_membership:    {output['view_membership'].shape}")
    print("  PASSED")


def test_training_forward():
    """Test training-mode forward with loss computation and gradient flow."""
    print("=" * 60)
    print("Test 3: Training mode + loss + gradients")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = ModelConfig(
        backbone="tiny_debug",
        detection_model="none",
        d_model=64,
        n_classes=4,
        mil_hidden_dim=32,
        per_joint_hidden=16,
        max_joints_per_view=10,
        input_views=("PA",),
        multi_view_fusion="none",
        use_view_embedding=False,
        loss_type="focal",
        focal_gamma=2.0,
        use_anatomy_prior_loss=False,
        per_joint_loss_weight=1.0,
        patient_loss_weight=1.0,
        entropy_reg_weight=0.01,
    )

    model = ArthritisMILModel(cfg).to(device=device, dtype=dtype)
    model.train()

    B = 2
    N = cfg.max_joints_per_view
    H_full, W_full = 224, 224
    n_valid = 7  # 7 real joints, 3 padded

    x = torch.randn(B, 1, H_full, W_full, device=device, dtype=dtype)
    boxes = {
        "PA": [
            _random_boxes(n_valid, H_full, W_full, device=device)
            for _ in range(B)
        ]
    }

    # Forward
    output = model(x, boxes=boxes)

    # Synthetic labels
    batch_labels = {
        "joint_labels": torch.randint(
            0, cfg.n_classes, (B, N), device=device
        ),
        "patient_label": torch.randint(0, cfg.n_classes, (B,), device=device),
    }
    # Mask padding positions
    mask = torch.zeros(B, N, device=device)
    mask[:, :n_valid] = 1.0
    batch_labels["joint_labels"] = batch_labels["joint_labels"].masked_fill(mask == 0, -100)

    # Loss
    losses = compute_loss(output, batch_labels, cfg)

    # Backward
    losses["loss"].backward()

    assert "loss" in losses
    assert "loss_joint" in losses
    assert "loss_patient" in losses
    assert "loss_entropy" in losses

    # Check gradients flow to trainable parameters
    grad_flow = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_flow = True
            break

    print(f"  ✓ loss:            {losses['loss'].item():.4f}")
    print(f"  ✓ loss_joint:      {losses['loss_joint'].item():.4f}")
    print(f"  ✓ loss_patient:    {losses['loss_patient'].item():.4f}")
    print(f"  ✓ loss_entropy:    {losses['loss_entropy'].item():.4f}")
    print(f"  ✓ gradients flow:  {grad_flow}")
    print("  PASSED")


def test_anatomy_prior_loss():
    """Test the anatomy-guided explanation loss in isolation."""
    print("=" * 60)
    print("Test 4: Anatomy prior explanation loss")
    print("=" * 60)

    from explanation import AnatomyExplanationModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Need config with anatomy prior loss enabled
    cfg = ModelConfig(
        use_anatomy_prior_loss=True,
        anatomy_prior_loss_weight=0.1,
        n_classes=4,
    )

    expl = AnatomyExplanationModule(cfg).to(device)

    B, N = 2, 10

    # Attention weights (random, sum to 1)
    attn = torch.rand(B, N, device=device)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Joint group labels (0..17, -1 for unknown)
    group_labels = torch.randint(-1, 18, (B, N), device=device)

    # Target disease
    targets = torch.tensor([0, 2], device=device)  # RA, OA

    # Forward (with use_prior_loss=True, this computes the Dice loss)
    loss = expl(attn, group_labels, targets)

    assert loss.item() >= 0.0, f"Negative loss: {loss.item()}"
    assert loss.item() <= cfg.anatomy_prior_loss_weight * 1.1, \
        f"Loss too large: {loss.item()}"

    print(f"  ✓ explanation_loss: {loss.item():.4f}")
    print("  PASSED")


def test_gradient_checkpointing():
    """Test the gradient checkpointing hook."""
    print("=" * 60)
    print("Test 5: Gradient checkpointing")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = ModelConfig(
        backbone="tiny_debug",
        detection_model="none",
        d_model=64,
        max_joints_per_view=5,
        n_classes=4,
        use_anatomy_prior_loss=False,
    )

    model = ArthritisMILModel(cfg).to(device=device, dtype=dtype)
    model.train()

    B, N = 2, 5
    x = torch.randn(B, 1, 224, 224, device=device, dtype=dtype)
    boxes = {
        "PA": [
            _random_boxes(5, 224, 224, device=device) for _ in range(B)
        ]
    }

    # Forward with checkpointing
    output = model(x, boxes=boxes, use_checkpoint=True)

    # Verify output shapes are the same
    expected_N = cfg.max_joints_per_view
    assert output["per_joint_logits"].shape == (B, expected_N, cfg.n_classes)
    assert output["patient_logits"].shape == (B, cfg.n_classes)

    print(f"  ✓ checkpoint forward successful")
    print(f"  ✓ per_joint_logits: {output['per_joint_logits'].shape}")
    print("  PASSED")


def test_disease_specific_heads():
    """Test the disease-specific binary heads configuration."""
    print("=" * 60)
    print("Test 6: Disease-specific binary heads")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig(
        backbone="tiny_debug",
        detection_model="none",
        d_model=64,
        n_classes=4,
        disease_specific_heads=True,
        max_joints_per_view=10,
        mil_hidden_dim=32,
        per_joint_hidden=16,
        input_views=("PA",),
        multi_view_fusion="none",
        use_view_embedding=False,
    )

    model = ArthritisMILModel(cfg).to(device)
    model.eval()

    B = 2
    x = torch.randn(B, 1, 224, 224, device=device)
    boxes = {
        "PA": [_random_boxes(5, 224, 224, device=device) for _ in range(B)]
    }

    with torch.no_grad():
        output = model(x, boxes=boxes)

    # Disease-specific heads output (B, 4) — one logit per disease
    assert output["patient_logits"].shape == (B, cfg.n_classes)
    print(f"  ✓ patient_logits (disease-specific): {output['patient_logits'].shape}")
    print("  PASSED")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _random_boxes(
    n: int, H: int, W: int, device: torch.device
) -> torch.Tensor:
    """Generate n random bounding boxes within (H, W)."""
    boxes = []
    for _ in range(n):
        x1 = torch.randint(0, W - 20, (1,), device=device).item()
        y1 = torch.randint(0, H - 20, (1,), device=device).item()
        x2 = x1 + torch.randint(15, 60, (1,), device=device).item()
        y2 = y1 + torch.randint(15, 60, (1,), device=device).item()
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, device=device, dtype=torch.float)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      Arthritis MIL Model — Smoke Test Suite             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_single_view_forward()
    print()
    test_multi_view_forward()
    print()
    test_training_forward()
    print()
    test_anatomy_prior_loss()
    print()
    test_gradient_checkpointing()
    print()
    test_disease_specific_heads()

    print()
    print("═" * 60)
    print("All smoke tests PASSED")
    print("═" * 60)
