"""Skeleton Recall Loss (Kirchhoff et al., ECCV 2024).

Penalises missed skeleton voxels — voxels that lie on the morphological
skeleton of the ground truth but are not covered by the prediction.

Uses skimage.morphology.skeletonize (Lee94 thinning) for true GT skeleton
computation, then measures differentiable recall of prediction against it.

Reference:
    Kirchhoff et al. "Skeleton Recall Loss for Connectedness-Aware
    Segmentation" ECCV 2024.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize


class SkeletonRecallLoss(nn.Module):  # type: ignore[misc]
    """Skeleton recall loss for topology-preserving segmentation.

    Computes the true morphological skeleton of the ground truth using
    skimage Lee94 thinning, then measures recall of skeleton voxels
    in the prediction.

    Loss = 1 - (sum(skeleton_mask * pred_prob) / (sum(skeleton_mask) + eps))
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
        """Compute skeleton recall loss.

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
            pred_prob = F.softmax(logits, dim=1)[:, 1:2]  # foreground prob
        else:
            pred_prob = torch.sigmoid(logits)

        # Ensure labels are binary float
        gt = (labels > 0.5).float()

        # Compute true skeleton of ground truth via skimage
        skel_mask = self._skeletonize_gt(gt)

        # Skeleton recall: how much of the skeleton is covered by prediction
        recall_num = (skel_mask * pred_prob).sum()
        recall_den = skel_mask.sum() + self.smooth

        skeleton_recall = recall_num / recall_den

        return 1.0 - skeleton_recall

    def _skeletonize_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """Compute true morphological skeleton of GT using skimage.

        Uses Lee94 thinning algorithm (skimage.morphology.skeletonize)
        which produces a 1-voxel-wide centerline. Non-differentiable,
        but that's fine since we only need gradients w.r.t. prediction.

        Parameters
        ----------
        gt:
            (B, 1, D, H, W) binary float mask.

        Returns
        -------
        Binary skeleton mask, same shape as gt. No gradient required.
        """
        device = gt.device
        dtype = gt.dtype
        batch_size = gt.shape[0]

        skel_list = []
        for b in range(batch_size):
            mask_np = gt[b, 0].detach().cpu().numpy() > 0.5
            if mask_np.any():
                skel_np = skeletonize(mask_np).astype(np.float32)
                if skel_np.sum() == 0:
                    # Fallback for thin structures: use the mask itself
                    skel_np = mask_np.astype(np.float32)
            else:
                skel_np = np.zeros_like(mask_np, dtype=np.float32)
            skel_list.append(torch.from_numpy(skel_np))

        skeleton = torch.stack(skel_list).unsqueeze(1).to(device=device, dtype=dtype)
        return skeleton.detach()  # No gradient through GT skeleton
