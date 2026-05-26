"""
FoundationBackbone: per-joint feature extractor using pretrained foundation models.
Domain: Computer Vision

Abstract base class (BaseOperator) for the core feature extraction operator,
with concrete implementation via DINOv2, ResNet, or a tiny debug backbone.

Shape convention:
    Input:  (B, N, 1, H, W)    — batch, joints, channels, height, width
    Output: (B, N, d_model)     — batch, joints, feature_dim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional

from config import ModelConfig


class BaseOperator(ABC, nn.Module):
    """Abstract base class for the core novel operator in the architecture.

    The FoundationBackbone is the primary feature-extraction operator.
    Concrete subclasses must implement forward() with the same shape contract.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-instance feature vectors.

        Args:
            x: (B, N, C, H, W) — batch of instance crops (joint ROIs)

        Returns:
            (B, N, d_model) — per-instance feature vectors
        """
        pass


class FoundationBackbone(BaseOperator):
    """Shared (optionally frozen) backbone applied independently to each joint ROI.

    Supports:
      - DINOv2 ViT variants (ViT-B/14, ViT-L/14, ViT-G/14) via torch.hub
      - ResNet-152 via torchvision
      - "tiny_debug" for smoke testing (small CNN, no pretrained weights)

    The [CLS] token (ViT) or global average pool (CNN) serves as the joint
    representation.  Grayscale input is replicated to 3 channels and normalized
    using ImageNet statistics.

    Inductive bias:  DINOv2's self-supervised ViT features capture generic
    texture/shape primitives (bone contours, joint space width, erosion
    boundaries) that transfer to X-ray despite the ImageNet→radiology domain gap.

    Args:
        config: ModelConfig instance
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.frozen = config.backbone_frozen
        self.d_model = config.d_model

        # --- Load backbone ---
        if config.backbone.startswith("dinov2"):
            # DINOv2 from Facebook Research
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", config.backbone, pretrained=True
            )
            self.use_cls_token = True

        elif config.backbone == "resnet152":
            from torchvision.models import resnet152, ResNet152_Weights
            self.backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            # Remove classifier and avgpool; we'll do our own pooling
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.use_cls_token = False

        elif config.backbone == "tiny_debug":
            # Tiny CNN for smoke-testing: no pretrained weights needed
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (B*N, 16, H, W)
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=False),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B*N, 32, H, W)
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(1),                       # (B*N, 32, 1, 1)
            )
            self.projection = nn.Linear(32, config.d_model)    # (B*N, d_model)
            self.use_cls_token = False

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # --- Freeze backbone if requested ---
        if self.frozen and config.backbone != "tiny_debug":
            for param in self.backbone.parameters():
                param.requires_grad = False

        # --- Optional LoRA fine-tuning ---
        if config.use_lora and config.backbone.startswith("dinov2"):
            self._apply_lora(config)

        # --- ImageNet normalization (used for all backbones) ---
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora(self, config: ModelConfig) -> None:
        """Replace linear attention projections with LoRA-adapted versions."""
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Check if this module name contains a target projection
            if any(f".{proj}" in name for proj in config.lora_target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = self.backbone.get_submodule(parent_name)

                lora_mod = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                )
                # Copy pretrained weight and bias
                lora_mod.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_mod.bias.data = module.bias.data.clone()

                setattr(parent, child_name, lora_mod)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-joint features.

        Args:
            x: (B, N, 1, H, W) — joint ROI crops
               H == W == config.img_size (typically 224)

        Returns:
            (B, N, d_model) — feature vectors
        """
        B, N, C, H, W = x.shape                           # batch, joints, channels, height, width

        # Flatten batch and joint dims → one forward pass per joint
        x_flat = x.reshape(B * N, C, H, W)                # (B*N, 1, H, W)

        # Convert grayscale → 3-channel RGB by repeating
        x_rgb = x_flat.expand(-1, 3, -1, -1).float()       # (B*N, 3, H, W)

        # Normalize using ImageNet statistics
        x_rgb = (x_rgb - self.mean) / self.std             # (B*N, 3, H, W)

        # Forward through backbone
        if self.use_cls_token:
            # DINOv2 returns (B*N, d_model) — the [CLS] token
            features = self.backbone(x_rgb)                # (B*N, d_model)
        else:
            # CNN backbone: spatial feature map → global average pool
            spatial = self.backbone(x_rgb)                 # (B*N, C_feat, H', W')
            features = spatial.mean(dim=[-2, -1])          # (B*N, C_feat)

        # Project if needed (tiny_debug has a separate projection head)
        if hasattr(self, "projection"):
            features = self.projection(features)           # (B*N, d_model)

        # Restore batch and joint dims
        return features.view(B, N, self.d_model)           # (B, N, d_model)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (Hu et al., 2021) of a linear layer.

    W' = W + (B · A) · (α / r)
    where A ~ N(0, σ²), B = 0 initially, so fine-tuning starts from pretrained weights.

    Applied only to backbone attention projections when use_lora=True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        # Original weight (kept frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features)"""
        base = F.linear(x, self.weight, self.bias)         # (..., out_features)
        lora_update = x @ self.lora_A @ self.lora_B * self.scaling  # (..., out_features)
        return base + lora_update                          # (..., out_features)
