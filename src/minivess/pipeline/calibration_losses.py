"""Auxiliary calibration losses for training-time calibration improvement.

Contains the Hard L1-ACE (hL1-ACE) loss adapted from:
  Barfoot et al. (2025). "Average Calibration Losses for Improved Calibration
  of Medical Image Segmentation Networks." IEEE TMI. arXiv:2506.03942

  Original implementation: https://github.com/cai4cai/Average-Calibration-Losses
  License: CC BY 4.0 (Creative Commons Attribution 4.0 International)
  Copyright (c) 2025 Centre for Advanced Instrumentation (CAI4CAI), King's College London

The hL1-ACE loss penalizes miscalibration by binning softmax probabilities and
computing the L1 distance between per-bin mean confidence and accuracy. When used
as an auxiliary loss alongside a segmentation loss, it encourages models to produce
well-calibrated probability outputs during training.

Formula:
    hL1-ACE = (1/B) * sum_{b=1}^{B} |acc(b) - conf(b)|
    where B = number of bins (default 15)
"""

from __future__ import annotations

import torch
from torch import nn


class HL1ACELoss(nn.Module):
    """Hard-binned L1 Average Calibration Error loss.

    Bins softmax probabilities into ``n_bins`` uniform bins and computes
    the mean absolute difference between per-bin accuracy and confidence.
    Designed as an auxiliary loss term for training-time calibration.

    Parameters
    ----------
    n_bins:
        Number of calibration bins. Default 15 per Barfoot et al.
    softmax:
        Apply softmax to input logits before binning.
    to_onehot_y:
        Convert integer labels to one-hot encoding.
    """

    def __init__(
        self,
        n_bins: int = 15,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.softmax = softmax
        self.to_onehot_y = to_onehot_y

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute hL1-ACE calibration loss.

        Parameters
        ----------
        logits:
            Model output tensor of shape ``(B, C, *spatial)``.
        labels:
            Ground truth of shape ``(B, 1, *spatial)`` with integer class labels.

        Returns
        -------
        Scalar calibration loss (mean hL1-ACE across batches and channels).
        """
        n_classes = logits.shape[1]

        # Apply softmax to get probabilities
        if self.softmax and n_classes > 1:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits

        # Convert labels to one-hot
        if self.to_onehot_y and n_classes > 1:
            target = self._to_onehot(labels, n_classes=n_classes)
        else:
            target = labels.float()

        # Compute hard-binned calibration per batch and channel
        mean_p_per_bin, mean_gt_per_bin, bin_counts = self._hard_binned_calibration(
            probs, target
        )

        # L1-ACE = nanmean of |confidence - accuracy| across bins
        ace_per_bc = torch.nanmean(
            torch.abs(mean_p_per_bin - mean_gt_per_bin), dim=-1
        )  # (B, C)

        # Mask empty classes (no voxels of that class in the volume)
        spatial_dims = list(range(2, target.ndim))
        non_empty = target.sum(dim=spatial_dims) > 0  # (B, C)
        ace_per_bc = ace_per_bc * non_empty.float()

        # Mean across batch and channels
        return ace_per_bc.mean()

    @staticmethod
    def _to_onehot(labels: torch.Tensor, *, n_classes: int) -> torch.Tensor:
        """Convert integer labels (B, 1, *spatial) to one-hot (B, C, *spatial)."""
        labels_squeeze = labels.long()
        if labels_squeeze.ndim > 1 and labels_squeeze.shape[1] == 1:
            labels_squeeze = labels_squeeze[:, 0]  # (B, *spatial)

        # Build one-hot manually — avoids F.one_hot dimension gymnastics
        batch = labels_squeeze.shape[0]
        spatial = labels_squeeze.shape[1:]
        onehot = torch.zeros(
            batch,
            n_classes,
            *spatial,
            device=labels.device,
            dtype=torch.float32,
        )
        for c in range(n_classes):
            onehot[:, c] = (labels_squeeze == c).float()
        return onehot

    def _hard_binned_calibration(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-bin mean confidence and accuracy using hard binning.

        Parameters
        ----------
        probs:
            Softmax probabilities of shape ``(B, C, *spatial)``.
        target:
            One-hot labels of shape ``(B, C, *spatial)``.

        Returns
        -------
        Tuple of (mean_p_per_bin, mean_gt_per_bin, bin_counts), each
        shaped ``(B, C, n_bins)``.
        """
        batch_size, n_channels = probs.shape[:2]

        # Create bin boundaries [0, ..., 1 + eps] with n_bins+1 edges
        boundaries = torch.linspace(
            0.0,
            1.0 + torch.finfo(torch.float32).eps,
            self.n_bins + 1,
            device=probs.device,
        )

        mean_p_per_bin = torch.zeros(
            batch_size, n_channels, self.n_bins, device=probs.device
        )
        mean_gt_per_bin = torch.zeros_like(mean_p_per_bin)
        bin_counts = torch.zeros_like(mean_p_per_bin)

        # Flatten spatial dimensions
        p_flat = probs.flatten(start_dim=2).float()  # (B, C, N)
        t_flat = target.flatten(start_dim=2).float()  # (B, C, N)

        for b in range(batch_size):
            for c in range(n_channels):
                # Assign each voxel to a bin
                bin_idx = torch.bucketize(p_flat[b, c], boundaries[1:])

                # Count voxels per bin
                bin_counts[b, c] = torch.zeros(
                    self.n_bins, device=probs.device
                ).scatter_add(0, bin_idx, torch.ones_like(p_flat[b, c]))

                # Mean prediction (confidence) per bin
                mean_p_per_bin[b, c] = torch.empty(
                    self.n_bins, device=probs.device
                ).scatter_reduce(
                    0, bin_idx, p_flat[b, c], reduce="mean", include_self=False
                )

                # Mean ground truth (accuracy) per bin
                mean_gt_per_bin[b, c] = torch.empty(
                    self.n_bins, device=probs.device
                ).scatter_reduce(
                    0, bin_idx, t_flat[b, c], reduce="mean", include_self=False
                )

        # Set empty bins to NaN so nanmean ignores them
        empty_mask = bin_counts == 0
        mean_p_per_bin[empty_mask] = float("nan")
        mean_gt_per_bin[empty_mask] = float("nan")

        return mean_p_per_bin, mean_gt_per_bin, bin_counts
