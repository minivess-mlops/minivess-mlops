# Centerline Cross-Entropy Loss
# Adapted from: https://github.com/cesaracebes/centerline_CE
# Original license: Apache-2.0
# Copyright (c) 2023 Cesar Acebes et al.
#
# Adaptation notes:
# - Adapted for 3D volumes (B, C, D, H, W)
# - Unified forward(logits, labels) interface
# - Uses soft skeleton proxy instead of explicit skeletonisation

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CenterlineCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with centerline-aware weighting.

    Upweights voxels near vessel centerlines to penalise centerline
    disconnections more heavily than peripheral misclassifications.

    Parameters
    ----------
    lambda_ce:
        Weight for standard cross-entropy component.
    lambda_cl_ce:
        Weight for centerline-weighted cross-entropy.
    kernel_size:
        Pooling kernel size for soft skeleton approximation.
    """

    def __init__(
        self,
        lambda_ce: float = 0.5,
        lambda_cl_ce: float = 0.5,
        *,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_cl_ce = lambda_cl_ce
        self.kernel_size = kernel_size

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute centerline-aware cross-entropy loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.
        """
        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]

        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, labels_squeeze)

        # Centerline-weighted cross-entropy
        cl_weight_map = self._compute_centerline_weights(
            labels_squeeze, logits.shape[1]
        )

        # Per-voxel CE
        log_probs = F.log_softmax(logits, dim=1)
        per_voxel_ce = F.nll_loss(log_probs, labels_squeeze, reduction="none")

        # Weight and average
        weighted_ce = (per_voxel_ce * cl_weight_map).mean()

        return self.lambda_ce * ce_loss + self.lambda_cl_ce * weighted_ce

    def _compute_centerline_weights(
        self, labels: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Approximate centerline weight map using iterative erosion proxy.

        Voxels that survive erosion (far from boundary) get higher weights,
        approximating vessel centerline locations.
        """
        fg_mask = (labels > 0).float().unsqueeze(1)  # (B, 1, D, H, W)
        pad = self.kernel_size // 2

        # Soft erosion via min-pooling (approximated by -max_pool(-x))
        eroded = -F.max_pool3d(
            -fg_mask, kernel_size=self.kernel_size, stride=1, padding=pad
        )

        # Centerline weight: eroded mask has higher weight for center voxels
        # Normalize so mean weight is ~1.0
        weight = 1.0 + 2.0 * eroded.squeeze(1)  # (B, D, H, W)
        return weight
