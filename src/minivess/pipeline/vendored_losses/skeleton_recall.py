"""Skeleton Recall Loss (Kirchhoff et al., ECCV 2024).

Penalises missed skeleton voxels — voxels that lie on the morphological
skeleton of the ground truth but are not covered by the prediction.

Differentiable via soft skeletonization: iterative morphological erosion
to approximate the skeleton, weighted by distance from boundary.

Reference:
    Kirchhoff et al. "Skeleton Recall Loss for Connectedness-Aware
    Segmentation" ECCV 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonRecallLoss(nn.Module):  # type: ignore[misc]
    """Skeleton recall loss for topology-preserving segmentation.

    Computes a soft skeleton of the ground truth via iterative erosion,
    then measures recall of skeleton voxels in the prediction.

    Loss = 1 - (sum(soft_skeleton * pred_prob) / (sum(soft_skeleton) + eps))
    """

    def __init__(
        self,
        n_erosion_iters: int = 10,
        softmax: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_erosion_iters = n_erosion_iters
        self.softmax = softmax
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute skeleton recall loss.

        Args:
            logits: (B, C, D, H, W) raw logits.
            labels: (B, 1, D, H, W) integer or float labels.

        Returns:
            Scalar loss tensor.
        """
        if self.softmax:
            pred_prob = F.softmax(logits, dim=1)[:, 1:2]  # foreground prob
        else:
            pred_prob = torch.sigmoid(logits)

        # Ensure labels are binary float
        gt = (labels > 0.5).float()

        # Compute soft skeleton of ground truth
        soft_skel = self._soft_skeleton(gt)

        # Skeleton recall: how much of the skeleton is covered by prediction
        recall_num = (soft_skel * pred_prob).sum()
        recall_den = soft_skel.sum() + self.smooth

        skeleton_recall = recall_num / recall_den

        return 1.0 - skeleton_recall

    def _soft_skeleton(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute soft skeleton via iterative erosion.

        Uses 3D min-pooling as a differentiable approximation of
        morphological erosion. The skeleton is approximated as the
        difference between successive erosions, weighted by iteration.

        Args:
            mask: (B, 1, D, H, W) binary float mask.

        Returns:
            Soft skeleton tensor, same shape as mask.
        """
        skeleton = torch.zeros_like(mask)
        current = mask.clone()

        for i in range(self.n_erosion_iters):
            # Morphological erosion via min-pooling
            eroded = -F.max_pool3d(-current, kernel_size=3, stride=1, padding=1)

            # Boundary = current - eroded (voxels removed by erosion)
            boundary = current - eroded

            # Weight by iteration (inner voxels get higher weight)
            weight = (i + 1) / self.n_erosion_iters
            skeleton = skeleton + weight * boundary

            current = eroded

            # Stop if fully eroded
            if current.sum() == 0:
                break

        # Add remaining core (innermost voxels)
        skeleton = skeleton + current

        # Normalize to [0, 1]
        skel_max = skeleton.max()
        if skel_max > 0:
            skeleton = skeleton / skel_max

        return skeleton
