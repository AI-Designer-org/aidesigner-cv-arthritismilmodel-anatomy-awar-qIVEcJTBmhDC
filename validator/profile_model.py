#!/usr/bin/env python3
"""
Profiling script for the Arthritis MIL Model.

Uses torch.profiler to measure:
  - Forward pass memory and time
  - Forward + backward (training) memory and time
  - Per-operator breakdown
  - Estimated FLOPs

Output:
  - Console table of top-K operations by time/memory
  - Estimated FLOPs (2× params for inference, 6× for training)
  - Memory allocation breakdown per module

Usage:
    python profile_model.py --mode forward           # inference profile
    python profile_model.py --mode train              # fwd+bwd profile
    python profile_model.py --mode both               # both (default)
    python profile_model.py --backbone dinov2_vitl14  # real backbone
    python profile_model.py --steps 20                # more profiling steps
"""

import argparse
import math
import sys
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))
from config import ModelConfig
from model import ArthritisMILModel
from losses import compute_loss


def build_sample_input(
    cfg: ModelConfig,
    device: torch.device,
    B: int = 2,
    N_joints: int = 10,
    n_views: int = 1,
) -> Tuple:
    """Build synthetic input for profiling.

    Args:
        cfg: ModelConfig
        device: torch device
        B: batch size
        N_joints: number of joints per view
        n_views: number of views

    Returns:
        Tuple of (x_or_views, boxes, batch_labels)
    """
    H, W = cfg.img_size, cfg.img_size

    if n_views > 1:
        views = {}
        boxes = {}
        for v_idx, v_name in enumerate(cfg.input_views[:n_views]):
            views[v_name] = torch.randn(B, 1, H * 2, W * 2, device=device)  # full X-ray size
            boxes[v_name] = [
                _random_boxes(N_joints, H * 2, W * 2, device)
                for _ in range(B)
            ]
        batch_labels = {
            "joint_labels": torch.randint(0, cfg.n_classes, (B, N_joints * n_views), device=device),
            "patient_label": torch.randint(0, cfg.n_classes, (B,), device=device),
        }
        # Mask some joints as padding
        batch_labels["joint_labels"][:, N_joints * n_views // 2:] = -100
        return (views, boxes, batch_labels)
    else:
        x = torch.randn(B, 1, H * 2, W * 2, device=device)
        boxes = {
            "PA": [
                _random_boxes(N_joints, H * 2, W * 2, device)
                for _ in range(B)
            ]
        }
        batch_labels = {
            "joint_labels": torch.randint(0, cfg.n_classes, (B, N_joints), device=device),
            "patient_label": torch.randint(0, cfg.n_classes, (B,), device=device),
        }
        batch_labels["joint_labels"][:, N_joints // 2:] = -100
        return (x, boxes, batch_labels)


def _random_boxes(n: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Generate n random bounding boxes within (H, W)."""
    boxes = []
    for _ in range(n):
        x1 = torch.randint(0, max(W - 20, 1), (1,)).item()
        y1 = torch.randint(0, max(H - 20, 1), (1,)).item()
        x2 = x1 + torch.randint(15, min(60, W - x1), (1,)).item()
        y2 = y1 + torch.randint(15, min(60, H - y1), (1,)).item()
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, device=device, dtype=torch.float)


def profile_forward(model: nn.Module, cfg: ModelConfig, device: torch.device,
                     steps: int = 10, warmup: int = 3):
    """Profile inference (forward only)."""
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_key = "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_time_total"

    n_views = len(cfg.input_views)
    sample = build_sample_input(cfg, device, B=cfg.batch_size, n_views=n_views)

    model.eval()

    # Warmup
    for _ in range(warmup):
        if n_views > 1:
            with torch.no_grad():
                _ = model(views=sample[0], boxes=sample[1])
        else:
            with torch.no_grad():
                _ = model(sample[0], boxes=sample[1])

    # Profile
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(steps):
            with record_function("forward_pass"):
                if n_views > 1:
                    out = model(views=sample[0], boxes=sample[1])
                else:
                    out = model(sample[0], boxes=sample[1])

    # Print results
    print(f"\n{'='*70}")
    print(f"FORWARD PROFILE  |  Backbone: {cfg.backbone}  |  "
          f"Batch: {cfg.batch_size}  |  Views: {n_views}")
    print(f"{'='*70}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # FLOP estimate
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    est_flops_fwd = 2 * total  # Kaplan et al.: ~2× params for forward
    print(f"\n  Parameters: {total:,} total ({trainable:,} trainable, {frozen:,} frozen)")
    print(f"  Est. forward FLOPs: {est_flops_fwd / 1e9:.2f}G  (2× params)")

    if device.type == "cuda":
        peak_mem = prof.key_averages().table(sort_by=sort_key, row_limit=1)
        print(f"  Peak CUDA memory: see 'self_cuda_memory_usage' column above")

    return prof


def profile_train(model: nn.Module, cfg: ModelConfig, device: torch.device,
                   steps: int = 10, warmup: int = 3):
    """Profile training (forward + backward)."""
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_key = "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_time_total"

    n_views = len(cfg.input_views)
    sample = build_sample_input(cfg, device, B=cfg.batch_size, n_views=n_views)

    model.train()

    # Warmup
    for _ in range(warmup):
        if n_views > 1:
            out = model(views=sample[0], boxes=sample[1])
        else:
            out = model(sample[0], boxes=sample[1])
        losses = compute_loss(out, sample[2], cfg)
        losses["loss"].backward()
        model.zero_grad()

    # Profile
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(steps):
            with record_function("train_step"):
                if n_views > 1:
                    out = model(views=sample[0], boxes=sample[1])
                else:
                    out = model(sample[0], boxes=sample[1])
                losses = compute_loss(out, sample[2], cfg)
                losses["loss"].backward()
                model.zero_grad()

    # Print results
    print(f"\n{'='*70}")
    print(f"TRAIN PROFILE (fwd + bwd)  |  Backbone: {cfg.backbone}  |  "
          f"Batch: {cfg.batch_size}  |  Views: {n_views}")
    print(f"{'='*70}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # FLOP estimate
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    est_flops_train = 6 * total  # Kaplan et al.: ~6× params for fwd+bwd
    print(f"\n  Parameters: {total:,} total ({trainable:,} trainable, {frozen:,} frozen)")
    print(f"  Est. train FLOPs: {est_flops_train / 1e9:.2f}G  (6× params)")

    return prof


def profile_memory_budget(cfg: ModelConfig, device: torch.device):
    """Estimate total memory budget for training."""
    if device.type != "cuda":
        print("Memory budget estimation requires CUDA")
        return

    model = ArthritisMILModel(cfg).to(device)
    model.train()

    n_views = len(cfg.input_views)
    sample = build_sample_input(cfg, device, B=cfg.batch_size, n_views=n_views)

    if n_views > 1:
        out = model(views=sample[0], boxes=sample[1])
    else:
        out = model(sample[0], boxes=sample[1])

    losses = compute_loss(out, sample[2], cfg)
    losses["loss"].backward()

    # Memory stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)

    model.zero_grad()

    print(f"\n{'='*70}")
    print(f"MEMORY BUDGET  |  Backbone: {cfg.backbone}  |  "
          f"Batch: {cfg.batch_size}  |  Views: {n_views}")
    print(f"{'='*70}")
    print(f"  Current allocated:  {allocated / 1024**3:.2f} GiB")
    print(f"  Current reserved:   {reserved / 1024**3:.2f} GiB")
    print(f"  Peak allocated:     {max_allocated / 1024**3:.2f} GiB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model params:       {total:,} total ({trainable:,} trainable)")

    # Estimate max batch size
    free, _ = torch.cuda.mem_get_info(device)
    print(f"  Free memory:        {free / 1024**3:.2f} GiB")
    est_max_batch = int(free / max(allocated / cfg.batch_size, 1))
    print(f"  Est. max batch:     {est_max_batch} (assuming linear scaling)")

    return {
        "allocated_gib": allocated / 1024**3,
        "reserved_gib": reserved / 1024**3,
        "peak_gib": max_allocated / 1024**3,
        "total_params": total,
        "trainable_params": trainable,
        "free_gib": free / 1024**3,
        "est_max_batch": est_max_batch,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Arthritis MIL Model")
    parser.add_argument("--mode", choices=["forward", "train", "both", "memory"],
                        default="both", help="Profiling mode")
    parser.add_argument("--backbone", type=str, default="tiny_debug",
                        help="Backbone: tiny_debug | dinov2_vitb14 | dinov2_vitl14 | resnet152")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of profiling steps")
    parser.add_argument("--views", type=int, default=1,
                        help="Number of X-ray views (1 or 2)")
    parser.add_argument("--multi_view_fusion", type=str, default="none",
                        choices=["none", "concat", "cross_attention"],
                        help="Multi-view fusion strategy")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine-tuning")
    parser.add_argument("--max_joints", type=int, default=10,
                        help="Max joints per view")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build config
    views = ("PA",) if args.views == 1 else ("PA", "oblique")
    cfg = ModelConfig(
        backbone=args.backbone,
        detection_model="none",
        n_classes=4,
        max_joints_per_view=args.max_joints,
        input_views=views,
        multi_view_fusion=args.multi_view_fusion if args.views > 1 else "none",
        use_view_embedding=args.views > 1,
        mil_hidden_dim=512,
        per_joint_hidden=256,
        per_joint_classifier=True,
        loss_type="focal",
        use_lora=args.use_lora,
        backbone_frozen=not args.use_lora,
        batch_size=args.batch_size,
    )

    # Adjust d_model for comparison
    if args.backbone in ("tiny_debug",):
        cfg.d_model = 64
        cfg.mil_hidden_dim = 32
        cfg.per_joint_hidden = 16

    print(f"Config: backbone={cfg.backbone}, batch={cfg.batch_size}, "
          f"views={args.views}, fusion={cfg.multi_view_fusion}")

    model = ArthritisMILModel(cfg).to(device)

    if args.mode in ("forward", "both"):
        profile_forward(model, cfg, device, steps=args.steps)

    if args.mode in ("train", "both"):
        profile_train(model, cfg, device, steps=args.steps)

    if args.mode == "memory":
        profile_memory_budget(cfg, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
