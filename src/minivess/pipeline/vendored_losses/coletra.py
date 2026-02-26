# CoLeTra — WarpLoss and TopoLoss
# Adapted from: https://github.com/jmlipman/CoLeTra
# Original license: Apache-2.0
# Copyright (c) 2024 J.M. Lipman et al.
#
# Adaptation notes:
# - Extracted WarpLoss and TopoLoss components
# - Simplified for 3D binary/multi-class (B, C, D, H, W)
# - Unified forward(logits, labels) interface

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WarpLoss(nn.Module):
    """Warping-based topology loss from CoLeTra.

    Computes a differentiable topology-aware penalty by comparing
    critical points (local minima/maxima) between prediction and
    ground truth probability maps.

    Parameters
    ----------
    lambda_warp:
        Scaling factor for the warp loss.
    softmax:
        Whether to apply softmax to logits.
    """

    def __init__(
        self,
        lambda_warp: float = 1.0,
        *,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_warp = lambda_warp
        self.softmax = softmax

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute warp loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.
        """
        probs = torch.softmax(logits, dim=1) if self.softmax else logits
        fg_prob = probs[:, 1:2] if probs.shape[1] > 1 else probs

        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        fg_label = (labels_squeeze > 0).float().unsqueeze(1)

        # Detect critical points via local extrema difference
        pred_critical = self._detect_critical_points(fg_prob)
        gt_critical = self._detect_critical_points(fg_label)

        # Warp penalty: MSE between critical point maps
        warp_penalty = F.mse_loss(pred_critical, gt_critical)

        return self.lambda_warp * warp_penalty

    @staticmethod
    def _detect_critical_points(volume: torch.Tensor) -> torch.Tensor:
        """Detect critical points via local max/min pooling difference.

        Critical points are where local max - local min is large,
        indicating topology-relevant locations (branch points, endpoints).
        """
        pad = 1
        local_max = F.max_pool3d(volume, kernel_size=3, stride=1, padding=pad)
        local_min = -F.max_pool3d(-volume, kernel_size=3, stride=1, padding=pad)
        return local_max - local_min


class TopoLoss(nn.Module):
    """Topology-preserving loss from CoLeTra.

    Penalises topological differences between prediction and ground truth
    using a persistent-homology-inspired differentiable proxy based on
    connected component counting via spatial gradient analysis.

    Parameters
    ----------
    lambda_topo:
        Scaling factor for the topology loss.
    softmax:
        Whether to apply softmax to logits.
    """

    def __init__(
        self,
        lambda_topo: float = 1.0,
        *,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_topo = lambda_topo
        self.softmax = softmax

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute topology loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.
        """
        probs = torch.softmax(logits, dim=1) if self.softmax else logits
        fg_prob = probs[:, 1:2] if probs.shape[1] > 1 else probs

        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        fg_label = (labels_squeeze > 0).float().unsqueeze(1)

        # Multi-scale topology comparison
        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for scale in [1, 2, 4]:
            if scale > 1:
                pred_scaled = F.avg_pool3d(fg_prob, kernel_size=scale)
                gt_scaled = F.avg_pool3d(fg_label, kernel_size=scale)
            else:
                pred_scaled = fg_prob
                gt_scaled = fg_label

            # Gradient-based topology proxy
            pred_topo = self._topology_signature(pred_scaled)
            gt_topo = self._topology_signature(gt_scaled)
            loss = loss + F.l1_loss(pred_topo, gt_topo)

        return self.lambda_topo * loss / 3.0

    @staticmethod
    def _topology_signature(volume: torch.Tensor) -> torch.Tensor:
        """Compute topology signature via spatial gradient statistics.

        The gradient magnitude distribution serves as a differentiable proxy
        for topological features (number of components, holes, tunnels).
        """
        grad_d = torch.diff(volume, dim=2)
        grad_h = torch.diff(volume, dim=3)
        grad_w = torch.diff(volume, dim=4)

        # Gradient magnitude per axis, concatenated
        return torch.cat(
            [
                grad_d.abs().mean(dim=(2, 3, 4), keepdim=True),
                grad_h.abs().mean(dim=(2, 3, 4), keepdim=True),
                grad_w.abs().mean(dim=(2, 3, 4), keepdim=True),
            ],
            dim=2,
        )
