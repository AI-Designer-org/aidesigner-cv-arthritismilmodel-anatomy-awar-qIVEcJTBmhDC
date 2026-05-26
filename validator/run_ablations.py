#!/usr/bin/env python3
"""
Ablation runner for the Arthritis MIL Model.

Runs all single-field ModelConfig ablations from the architect's specification,
trains each variant for a fixed number of steps, and reports evaluation metrics.

Ablations implemented (each is one config change from baseline):
  1. MIL → Average Pooling      (mil_gated=True → patient_pooling="mean")
  2. Frozen Backbone → Full FT   (backbone_frozen=True → False)
  3. Focal Loss → Cross-Entropy  (loss_type="focal" → "ce")
  4. Three-Way → One-vs-All      (n_classes=4 → separate binary classifiers)
  5. Anatomy Prior Off            (use_anatomy_prior_loss=True → False)
  6. Multi-View → Single-View    (input_views=("PA","oblique") → ("PA",))
  7. Detection → Tile Sampling   (detection_model="yolov7" → "none" with grid)

Usage:
    python run_ablations.py                     # run all ablations
    python run_ablations.py --ablation mil_pool  # run specific ablation
    python run_ablations.py --epochs 5 --steps_per_epoch 20

Output:
    results/ablations.json     — metrics for each ablation
    results/ablations.csv      — human-readable table
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))
from config import ModelConfig
from model import ArthritisMILModel
from losses import compute_loss


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Config
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_CONFIG = ModelConfig(
    backbone="tiny_debug",          # Use tiny_debug for CI speed; swap to dinov2_vitl14 for real runs
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
    per_joint_loss_weight=1.0,
    patient_loss_weight=1.0,
    entropy_reg_weight=0.01,
    use_anatomy_prior_loss=False,
    mil_gated=True,
    patient_pooling="attention",
    per_joint_classifier=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Ablation Definitions
# ═══════════════════════════════════════════════════════════════════════════

def _make_ablations(base_cfg: ModelConfig) -> Dict[str, ModelConfig]:
    """Build all ablation configs from the baseline."""

    # Ablation 1: MIL → Average Pooling
    cfg_mil_pool = replace(base_cfg, mil_gated=False, patient_pooling="mean")

    # Ablation 2: Frozen Backbone → Full Fine-Tuning
    cfg_full_ft = replace(base_cfg, backbone_frozen=False, use_lora=False)

    # Ablation 3: Focal Loss → Cross-Entropy
    cfg_ce = replace(base_cfg, loss_type="ce")

    # Ablation 4: Three-Way → One-vs-All (n_classes=2 per binary model)
    # NOTE: The ModelConfig __post_init__ restricts n_classes to 3 or 4.
    # To run OVA, manually override the validation or use a separate binary
    # classification head. This ablation config is a placeholder that users
    # must adapt to their specific binary training pipeline.

    # Ablation 5: Anatomy Prior Off → On (already off in baseline; turn on)
    cfg_anatomy = replace(
        base_cfg,
        use_anatomy_prior_loss=True,
        anatomy_prior_loss_weight=0.1,
    )

    # Ablation 6: Multi-View → Single-View (baseline is single-view)
    # The multi-view variant adds a second view
    cfg_multi_view = replace(
        base_cfg,
        input_views=("PA", "oblique"),
        multi_view_fusion="concat",
        use_view_embedding=True,
        max_joints_per_view=8,
    )

    # Ablation 7: Detection → Tile Sampling (simulate by reducing max_joints)
    cfg_tile = replace(base_cfg, max_joints_per_view=20, detection_model="none")

    return {
        "baseline": base_cfg,
        "mil_pool": cfg_mil_pool,
        "full_ft": cfg_full_ft,
        "ce_loss": cfg_ce,
        # OVA ablations require a separate binary head (n_classes=2 not allowed by config).
        # To run: create a modified ModelConfig subclass or override __post_init__.
        # "ova_ra": cfg_ova_ra,
        # "ova_psa": cfg_ova_psa,
        # "ova_oa": cfg_ova_oa,
        "anatomy_on": cfg_anatomy,
        "multi_view": cfg_multi_view,
        "tile_sampling": cfg_tile,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticArthritisDataset(torch.utils.data.Dataset):
    """Minimal synthetic dataset for ablation testing.

    Generates random images + boxes + labels on the fly.
    For real experiments, replace with actual X-ray data.
    """

    def __init__(
        self,
        n_samples: int = 100,
        img_size: int = 224,
        max_joints: int = 10,
        n_classes: int = 4,
        device: str = "cpu",
    ):
        self.n_samples = n_samples
        self.img_size = img_size
        self.max_joints = max_joints
        self.n_classes = n_classes
        self.device = device

        # Pre-generate patient-level labels (roughly balanced)
        self.patient_labels = torch.randint(0, n_classes, (n_samples,))

        # Pre-generate per-joint labels with some disease-specific structure
        self.joint_labels = torch.randint(0, n_classes, (n_samples, max_joints))
        # Mask some joints as padding
        mask = torch.rand(n_samples, max_joints) > 0.2  # 80% valid
        self.joint_labels[~mask] = -100

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.randn(1, self.img_size, self.img_size)
        boxes = self._random_boxes()
        joint_labels = self.joint_labels[idx]
        patient_label = self.patient_labels[idx]
        return x, boxes, joint_labels, patient_label

    def _random_boxes(self):
        n = torch.randint(3, self.max_joints + 1, ()).item()
        boxes_list = []
        for _ in range(n):
            x1 = torch.randint(0, self.img_size - 40, ()).item()
            y1 = torch.randint(0, self.img_size - 40, ()).item()
            x2 = x1 + torch.randint(20, 60, ()).item()
            y2 = y1 + torch.randint(20, 60, ()).item()
            boxes_list.append([x1, y1, x2, y2])
        return torch.tensor(boxes_list, dtype=torch.float)


def _random_boxes(n: int, H: int, W: int) -> list:
    """Generate a single list of box tensors for batch construction."""
    boxes = []
    for _ in range(n):
        x1 = torch.randint(0, max(W - 20, 1), (1,)).item()
        y1 = torch.randint(0, max(H - 20, 1), (1,)).item()
        x2 = x1 + torch.randint(15, min(60, W - x1), (1,)).item()
        y2 = y1 + torch.randint(15, min(60, H - y1), (1,)).item()
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes)


def collate_boxes(batch):
    """Custom collate for variable-length boxes."""
    xs, boxes_list, joint_labels, patient_labels = zip(*batch)
    xs = torch.stack(xs)
    joint_labels = torch.stack(joint_labels)
    patient_labels = torch.stack(patient_labels)

    # boxes_list is list of (N_i, 4) tensors — wrap in dict
    boxes = {"PA": list(boxes_list)}
    return xs, boxes, joint_labels, patient_labels


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: ModelConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch and return average losses."""
    model.train()
    total_losses = {"loss": 0.0, "loss_joint": 0.0, "loss_patient": 0.0}
    n_batches = 0

    for x, boxes, joint_labels, patient_labels in loader:
        x = x.to(device)
        joint_labels = joint_labels.to(device)
        patient_labels = patient_labels.to(device)

        # Move boxes to device
        boxes_on_device = {
            "PA": [b.to(device) for b in boxes["PA"]]
        }

        optimizer.zero_grad()

        out = model(x, boxes=boxes_on_device)
        batch = {"joint_labels": joint_labels, "patient_label": patient_labels}
        losses = compute_loss(out, batch, cfg)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
        optimizer.step()

        total_losses["loss"] += losses["loss"].item()
        total_losses["loss_joint"] += losses["loss_joint"].item()
        total_losses["loss_patient"] += losses["loss_patient"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: ModelConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_joint_preds = []
    all_joint_labels = []

    for x, boxes, joint_labels, patient_labels in loader:
        x = x.to(device)
        joint_labels = joint_labels.to(device)
        patient_labels = patient_labels.to(device)
        boxes_on_device = {
            "PA": [b.to(device) for b in boxes["PA"]]
        }

        out = model(x, boxes=boxes_on_device)

        # Patient-level predictions
        preds = out["patient_logits"].argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(patient_labels.cpu())

        # Per-joint predictions (only on valid joints)
        if out["per_joint_logits"] is not None:
            jpreds = out["per_joint_logits"].argmax(dim=-1)
            all_joint_preds.append(jpreds.cpu())
            all_joint_labels.append(joint_labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Patient accuracy
    patient_acc = (all_preds == all_labels).float().mean().item()

    # Per-class recall (macro-averaged)
    n_classes = cfg.n_classes
    per_class_recall = []
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = (all_preds[mask] == c).float().mean().item()
            per_class_recall.append(recall)
    macro_recall = sum(per_class_recall) / max(len(per_class_recall), 1)

    # Per-joint accuracy
    if all_joint_preds and all_joint_labels:
        jpreds = torch.cat(all_joint_preds)
        jlabels = torch.cat(all_joint_labels)
        valid = jlabels != -100
        if valid.sum() > 0:
            joint_acc = (jpreds[valid] == jlabels[valid]).float().mean().item()
        else:
            joint_acc = 0.0
    else:
        joint_acc = 0.0

    return {
        "patient_accuracy": patient_acc,
        "macro_recall": macro_recall,
        "joint_accuracy": joint_acc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_ablations(args):
    """Run all or specified ablations and report results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override backbone if specified
    base_cfg = BASELINE_CONFIG
    if args.backbone:
        base_cfg = replace(base_cfg, backbone=args.backbone)

    # Use larger backbone for meaningful comparison
    if args.backbone in ("dinov2_vitb14", "dinov2_vitl14"):
        print("[WARN] Using DINOv2 backbone — expect longer runtimes")
        # Adjust batch size for larger backbone
        base_cfg = replace(base_cfg, batch_size=4, detection_input_size=224)

    ablations = _make_ablations(base_cfg)

    # Filter to specific ablation if requested
    if args.ablation:
        if args.ablation not in ablations:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ablations.keys())}")
            return
        ablations = {args.ablation: ablations[args.ablation]}

    # Create datasets
    train_dataset = SyntheticArthritisDataset(
        n_samples=args.n_train,
        max_joints=base_cfg.max_joints_per_view,
        n_classes=base_cfg.n_classes,
    )
    eval_dataset = SyntheticArthritisDataset(
        n_samples=args.n_eval,
        max_joints=base_cfg.max_joints_per_view,
        n_classes=base_cfg.n_classes,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=base_cfg.batch_size,
        shuffle=True, collate_fn=collate_boxes,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=base_cfg.batch_size,
        shuffle=False, collate_fn=collate_boxes,
    )

    results = {}

    for name, cfg in ablations.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")
        print(f"  Config changes from baseline:")
        for k, v in cfg.__dict__.items():
            base_v = base_cfg.__dict__.get(k)
            if v != base_v:
                print(f"    {k}: {base_v} → {v}")

        # Build model
        model = ArthritisMILModel(cfg).to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Train
        start = time.time()
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, cfg, optimizer, device)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_metrics['loss']:.4f}")

        elapsed = time.time() - start

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, cfg, device)
        eval_metrics["train_time_sec"] = elapsed
        eval_metrics["trainable_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        eval_metrics["total_params"] = sum(p.numel() for p in model.parameters())

        results[name] = eval_metrics

        print(f"  Results:")
        for k, v in eval_metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Save results
    os.makedirs("results", exist_ok=True)

    with open("results/ablations.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Pretty-print table
    print(f"\n{'='*60}")
    print("Ablation Results Summary")
    print(f"{'='*60}")
    header = f"{'Ablation':<20} {'Patient Acc':<14} {'Macro Recall':<14} {'Joint Acc':<12} {'Params':<10}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<20} {metrics['patient_accuracy']:<14.4f} "
            f"{metrics['macro_recall']:<14.4f} {metrics['joint_accuracy']:<12.4f} "
            f"{metrics['trainable_params']:<10,}"
        )

    # Save CSV
    with open("results/ablations.csv", "w") as f:
        metrics_keys = ["patient_accuracy", "macro_recall", "joint_accuracy",
                        "trainable_params", "total_params", "train_time_sec"]
        f.write("ablation," + ",".join(metrics_keys) + "\n")
        for name, metrics in results.items():
            vals = [str(metrics.get(k, "")) for k in metrics_keys]
            f.write(f"{name}," + ",".join(vals) + "\n")

    print(f"\nResults saved to results/ablations.json and results/ablations.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Arthritis MIL ablations")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Specific ablation to run (default: all)")
    parser.add_argument("--backbone", type=str, default="tiny_debug",
                        help="Backbone to use (default: tiny_debug)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per ablation")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Synthetic training samples")
    parser.add_argument("--n_eval", type=int, default=50,
                        help="Synthetic evaluation samples")

    args = parser.parse_args()
    run_ablations(args)
