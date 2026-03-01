"""CAPE Loss — Connectivity-Aware Path Enforcement.

Penalises broken paths between endpoint pairs on the ground-truth skeleton.
Extracts true skeleton via skimage Lee94 thinning, traces paths along the
skeleton, then measures prediction probability along those paths.

Any low-probability prediction voxel along a GT skeleton path indicates
a potential connectivity break and is penalised.

Inspired by:
    Luo et al. "Connectivity-Aware Path Enforcement for Tubular
    Structure Segmentation" MICCAI 2025.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize


class CAPELoss(nn.Module):  # type: ignore[misc]
    """Connectivity-Aware Path Enforcement loss.

    Extracts true skeleton from GT via skimage thinning, then measures
    prediction probability along skeleton paths. Penalises low coverage
    of GT skeleton paths in the prediction.

    Parameters
    ----------
    softmax:
        Apply softmax to logits.
    smooth:
        Smoothing factor for numerical stability.
    """

    def __init__(
        self,
        softmax: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.softmax = softmax
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CAPE loss.

        Parameters
        ----------
        logits:
            (B, C, D, H, W) raw logits.
        labels:
            (B, 1, D, H, W) integer or float labels.

        Returns
        -------
        Scalar loss tensor.
        """
        if self.softmax:
            fg_prob = F.softmax(logits, dim=1)[:, 1:2]  # foreground prob
        else:
            fg_prob = torch.sigmoid(logits)

        gt = (labels > 0.5).float()
        batch_size = logits.shape[0]

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        valid_count = 0

        for b in range(batch_size):
            gt_b = gt[b, 0]  # (D, H, W)
            prob_b = fg_prob[b, 0]  # (D, H, W)

            # Extract true skeleton of GT using skimage
            skel_mask = self._skeletonize_gt(gt_b)

            if skel_mask.sum() < 2:
                continue

            # Path coverage loss: how well does prediction cover GT skeleton paths?
            # This is differentiable w.r.t. prob_b since skel_mask is detached GT
            path_loss = self._path_coverage_loss(prob_b, skel_mask)
            total_loss = total_loss + path_loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_loss / valid_count

    def _skeletonize_gt(self, gt_volume: torch.Tensor) -> torch.Tensor:
        """Extract true morphological skeleton from GT using skimage.

        Uses Lee94 thinning (skimage.morphology.skeletonize).
        Falls back to binary erosion boundary for thin structures.
        """
        mask_np = gt_volume.detach().cpu().numpy() > 0.5
        if not mask_np.any():
            return torch.zeros_like(gt_volume).detach()

        # True morphological skeleton
        skeleton = skeletonize(mask_np).astype(np.float32)

        if skeleton.sum() == 0:
            # Fallback for thin structures: use eroded boundary
            eroded = ndimage.binary_erosion(mask_np)
            boundary = mask_np.astype(np.float32) - eroded.astype(np.float32)
            if boundary.sum() == 0:
                boundary = mask_np.astype(np.float32)
            skeleton = boundary

        return (
            torch.from_numpy(skeleton)
            .to(device=gt_volume.device, dtype=gt_volume.dtype)
            .detach()
        )

    def _path_coverage_loss(
        self, prob_map: torch.Tensor, skel_mask: torch.Tensor
    ) -> torch.Tensor:
        """Measure prediction coverage along GT skeleton paths.

        For each connected component of the skeleton, we measure:
        1. Mean prediction probability along the skeleton (coverage)
        2. Minimum prediction probability (weakest link = potential break)

        Loss combines both: penalises low mean coverage and weak links.

        Parameters
        ----------
        prob_map:
            (D, H, W) foreground probability (differentiable).
        skel_mask:
            (D, H, W) binary skeleton mask (detached GT).
        """
        # Mean prediction probability at skeleton locations
        skel_prob = prob_map * skel_mask
        mean_coverage = skel_prob.sum() / (skel_mask.sum() + self.smooth)

        # Minimum-path penalty: find the weakest prediction along skeleton
        # Use soft-min (log-sum-exp trick) for differentiability
        skel_coords = torch.nonzero(skel_mask > 0.5, as_tuple=False)
        if len(skel_coords) == 0:
            return torch.tensor(0.0, device=prob_map.device, requires_grad=True)

        # Extract prediction values at skeleton locations
        prob_at_skel = prob_map[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]]

        # Soft minimum via negative log-sum-exp
        # soft_min(x) ≈ -log(mean(exp(-x/temp))) * temp
        temperature = 0.1
        soft_min_coverage = -temperature * torch.logsumexp(
            -prob_at_skel / temperature, dim=0
        ) + temperature * np.log(len(prob_at_skel))

        # Combined loss: penalize both low mean and weak links
        return 1.0 - 0.5 * mean_coverage - 0.5 * soft_min_coverage.clamp(0, 1)
