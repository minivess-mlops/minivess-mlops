"""Steerable Pyramid Weighted (SPW) Loss for thin vessel structures.

Implements multi-scale adaptive weighting via steerable pyramid-like
decomposition for cross-entropy loss. Assigns higher weight to boundary
and thin-structure voxels at multiple scales.

References:
    Lu (2025), "Steerable Pyramid Weighted Loss for Thin Structure
    Segmentation", arXiv:2503.06604, Cornell University.

EXPERIMENTAL: Simplified proxy using multi-scale Laplacian decomposition.
The full SPW uses steerable pyramid with oriented bandpass filters;
this implementation uses isotropic Laplacian-of-Gaussian approximation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class SPWLoss(nn.Module):
    """Steerable Pyramid Weighted cross-entropy loss.

    Uses multi-scale decomposition of the ground truth to weight the
    cross-entropy loss, emphasizing boundary and thin-structure voxels.

    Parameters
    ----------
    num_scales:
        Number of scales in the pyramid (default: 3).
    base_weight:
        Minimum weight for uniform regions (default: 1.0).
    boundary_weight:
        Additional weight for high-frequency (boundary) voxels (default: 2.0).
    """

    def __init__(
        self,
        *,
        num_scales: int = 3,
        base_weight: float = 1.0,
        boundary_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.base_weight = base_weight
        self.boundary_weight = boundary_weight

    def _laplacian_weight_map(self, labels_onehot: Tensor) -> Tensor:
        """Compute multi-scale boundary weight map from one-hot labels.

        Uses average pooling at multiple scales to detect boundaries —
        voxels near class transitions have high response.

        Parameters
        ----------
        labels_onehot:
            One-hot encoded labels, shape (B, C, D, H, W).

        Returns
        -------
        Weight map (B, 1, D, H, W) with higher values at boundaries.
        """
        weight_map = torch.zeros(
            labels_onehot.shape[0],
            1,
            *labels_onehot.shape[2:],
            device=labels_onehot.device,
            dtype=labels_onehot.dtype,
        )

        for scale_idx in range(self.num_scales):
            kernel_size = 3 + 2 * scale_idx  # 3, 5, 7, ...
            padding = kernel_size // 2

            # Average pool at this scale — smooth the label boundaries
            smoothed = F.avg_pool3d(
                labels_onehot.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )

            # High-frequency residual: original - smoothed
            residual = (labels_onehot.float() - smoothed).abs()

            # Sum across class channels to get boundary magnitude
            boundary_response = residual.sum(dim=1, keepdim=True)
            weight_map = weight_map + boundary_response

        # Normalize to [0, 1] range
        w_max = weight_map.max()
        if w_max > 0:
            weight_map = weight_map / w_max

        return weight_map

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute SPW-weighted cross-entropy loss.

        Parameters
        ----------
        logits:
            Model output, shape (B, C, D, H, W).
        labels:
            Ground truth, shape (B, 1, D, H, W) integer class indices.

        Returns
        -------
        Scalar loss value.
        """
        num_classes = logits.shape[1]

        # One-hot encode labels for weight map computation
        labels_squeezed = labels.squeeze(1).long()  # (B, D, H, W)
        labels_onehot = F.one_hot(labels_squeezed, num_classes)  # (B, D, H, W, C)
        labels_onehot = labels_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        # Compute multi-scale boundary weight map
        weight_map = self._laplacian_weight_map(labels_onehot)  # (B, 1, D, H, W)

        # Scale weights: base_weight + boundary_weight * weight_map
        pixel_weights = self.base_weight + self.boundary_weight * weight_map
        pixel_weights = pixel_weights.squeeze(1)  # (B, D, H, W)

        # Weighted cross-entropy
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, D, H, W)
        nll = F.nll_loss(log_probs, labels_squeezed, reduction="none")  # (B, D, H, W)

        # Apply weights
        weighted_loss = (nll * pixel_weights).mean()
        return weighted_loss
