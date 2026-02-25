# cbDice — Centerline-Boundary Dice Loss
# Adapted from: https://github.com/PengchengShi1220/cbDice
# Original license: Apache-2.0
# Copyright (c) 2024 Pengcheng Shi et al.
#
# Adaptation notes:
# - Simplified for 3D binary/multi-class segmentation (B, C, D, H, W)
# - Unified forward(logits, labels) interface for our loss factory
# - Removed dataset-specific skeletonisation; uses soft distance transform proxy

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CenterlineBoundaryDiceLoss(nn.Module):
    """Centerline-boundary Dice loss combining standard Dice with
    distance-weighted components for centerline and boundary fidelity.

    Parameters
    ----------
    lambda_dice:
        Weight for the standard Dice component.
    lambda_cl:
        Weight for the centerline-aware component.
    lambda_bd:
        Weight for the boundary-aware component.
    smooth:
        Smoothing constant to avoid division by zero.
    softmax:
        Whether to apply softmax to logits before computing loss.
    """

    def __init__(
        self,
        lambda_dice: float = 0.4,
        lambda_cl: float = 0.3,
        lambda_bd: float = 0.3,
        *,
        smooth: float = 1e-5,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_cl = lambda_cl
        self.lambda_bd = lambda_bd
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cbDice loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.
        """
        probs = torch.softmax(logits, dim=1) if self.softmax else logits
        n_classes = logits.shape[1]

        # One-hot encode labels
        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        labels_onehot = torch.zeros_like(logits)
        for c in range(n_classes):
            labels_onehot[:, c] = (labels_squeeze == c).float()

        # Standard Dice
        spatial_dims = tuple(range(2, logits.ndim))
        intersection = (probs * labels_onehot).sum(dim=spatial_dims)
        union = probs.sum(dim=spatial_dims) + labels_onehot.sum(dim=spatial_dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # Centerline weighting: approximate centerline via erosion-like distance
        # Uses average pooling as a soft distance proxy
        cl_weight = self._soft_distance_weight(labels_onehot, kernel_size=5)
        cl_intersection = (probs * labels_onehot * cl_weight).sum(dim=spatial_dims)
        cl_union = (probs * cl_weight).sum(dim=spatial_dims) + (labels_onehot * cl_weight).sum(dim=spatial_dims)
        cl_dice = (2.0 * cl_intersection + self.smooth) / (cl_union + self.smooth)
        cl_loss = 1.0 - cl_dice.mean()

        # Boundary weighting: approximate boundary via gradient magnitude
        bd_weight = self._boundary_weight(labels_onehot)
        bd_intersection = (probs * labels_onehot * bd_weight).sum(dim=spatial_dims)
        bd_union = (probs * bd_weight).sum(dim=spatial_dims) + (labels_onehot * bd_weight).sum(dim=spatial_dims)
        bd_dice = (2.0 * bd_intersection + self.smooth) / (bd_union + self.smooth)
        bd_loss = 1.0 - bd_dice.mean()

        return (
            self.lambda_dice * dice_loss
            + self.lambda_cl * cl_loss
            + self.lambda_bd * bd_loss
        )

    @staticmethod
    def _soft_distance_weight(
        labels_onehot: torch.Tensor, kernel_size: int = 5
    ) -> torch.Tensor:
        """Approximate distance-to-boundary weight using average pooling."""
        b, c = labels_onehot.shape[:2]
        flat = labels_onehot.reshape(b * c, 1, *labels_onehot.shape[2:])
        pad = kernel_size // 2
        avg = F.avg_pool3d(flat, kernel_size=kernel_size, stride=1, padding=pad)
        # Centerline-like: high weight where label is present but far from boundary
        weight = avg * flat  # values near 1 at center, near 0 at edges
        return weight.reshape(b, c, *labels_onehot.shape[2:])

    @staticmethod
    def _boundary_weight(labels_onehot: torch.Tensor) -> torch.Tensor:
        """Approximate boundary weight via spatial gradient magnitude."""
        grad_d = torch.diff(labels_onehot, dim=2)
        grad_h = torch.diff(labels_onehot, dim=3)
        grad_w = torch.diff(labels_onehot, dim=4)

        # Pad back to original size
        pad_d = F.pad(grad_d.abs(), (0, 0, 0, 0, 0, 1))
        pad_h = F.pad(grad_h.abs(), (0, 0, 0, 1, 0, 0))
        pad_w = F.pad(grad_w.abs(), (0, 1, 0, 0, 0, 0))

        boundary = (pad_d + pad_h + pad_w).clamp(min=0, max=1)
        # Add small base weight so non-boundary regions aren't zero
        return boundary + 0.1
