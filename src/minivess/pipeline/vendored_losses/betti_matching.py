"""Betti Matching Loss (Stucki et al., ICML 2023).

Differentiable topology matching via persistence-based topological features.
Penalises topological differences (extra/missing connected components, loops)
between prediction and ground truth.

When gudhi is available: uses cubical complex persistence diagrams + Hungarian
matching of persistence pairs.

When gudhi is unavailable: uses a differentiable proxy based on soft
connected-component counting via spatial gradient magnitude (same proxy
as BettiLoss but with matching-based penalty).

Reference:
    Stucki et al. "Topologically Faithful Image Segmentation via Induced
    Matching of Persistence Barcodes" ICML 2023.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check gudhi availability at module level
_GUDHI_AVAILABLE = False
try:
    import gudhi  # noqa: F401

    _GUDHI_AVAILABLE = True
except ImportError:
    pass


class BettiMatchingLoss(nn.Module):  # type: ignore[misc]
    """Betti matching loss for topology-preserving segmentation.

    Uses persistence diagram matching to penalise topological differences.
    Falls back to differentiable proxy when gudhi is unavailable.

    Parameters
    ----------
    softmax:
        Apply softmax to logits.
    lambda_topo:
        Weight for the topology matching term.
    smooth:
        Smoothing factor for numerical stability.
    """

    def __init__(
        self,
        softmax: bool = True,
        lambda_topo: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.softmax = softmax
        self.lambda_topo = lambda_topo
        self.smooth = smooth

        if not _GUDHI_AVAILABLE:
            logger.warning(
                "gudhi not installed; BettiMatchingLoss uses differentiable proxy"
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Betti matching loss.

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
            fg_prob = F.softmax(logits, dim=1)[:, 1:2]
        else:
            fg_prob = torch.sigmoid(logits)

        gt = (labels > 0.5).float()

        if _GUDHI_AVAILABLE:
            return self._gudhi_matching_loss(fg_prob, gt)
        return self._proxy_matching_loss(fg_prob, gt)

    def _proxy_matching_loss(
        self, fg_prob: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable proxy for Betti matching when gudhi unavailable.

        Uses spatial gradient magnitude as a proxy for topological complexity.
        Penalises differences in spatial fragmentation between pred and GT.
        Additionally applies a soft component-count proxy via multi-threshold
        level-set analysis.
        """
        batch_size = fg_prob.shape[0]
        total_loss = torch.tensor(0.0, device=fg_prob.device, dtype=fg_prob.dtype)

        for b in range(batch_size):
            pred_b = fg_prob[b, 0]  # (D, H, W)
            gt_b = gt[b, 0]  # (D, H, W)

            # 1. Gradient-based fragmentation proxy
            pred_frag = self._spatial_gradient_magnitude(pred_b)
            gt_frag = self._spatial_gradient_magnitude(gt_b)
            frag_diff = (pred_frag - gt_frag).abs()

            # 2. Multi-threshold level-set proxy for component count
            # At each threshold, count how much "boundary" exists
            thresholds = [0.3, 0.5, 0.7]
            level_set_diff = torch.tensor(
                0.0, device=fg_prob.device, dtype=fg_prob.dtype
            )
            for tau in thresholds:
                pred_binary = torch.sigmoid((pred_b - tau) * 10.0)
                gt_binary = (gt_b > tau).float()
                pred_boundary = self._spatial_gradient_magnitude(pred_binary)
                gt_boundary = self._spatial_gradient_magnitude(gt_binary)
                level_set_diff = level_set_diff + (pred_boundary - gt_boundary).abs()

            total_loss = total_loss + frag_diff + level_set_diff / len(thresholds)

        return self.lambda_topo * total_loss / batch_size

    def _gudhi_matching_loss(
        self, fg_prob: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Full persistence diagram matching loss using gudhi.

        Computes persistence diagrams for pred and GT, matches pairs
        via Hungarian algorithm, and penalises unmatched topological features.
        """
        import gudhi
        from scipy.optimize import linear_sum_assignment

        batch_size = fg_prob.shape[0]
        spatial_volume = float(fg_prob[0, 0].numel())
        total_loss = torch.tensor(0.0, device=fg_prob.device, dtype=fg_prob.dtype)

        for b in range(batch_size):
            pred_np = fg_prob[b, 0].detach().cpu().numpy()
            gt_np = gt[b, 0].detach().cpu().numpy()

            # Compute persistence diagrams
            pred_diag = self._persistence_diagram(pred_np, gudhi)
            gt_diag = self._persistence_diagram(gt_np, gudhi)

            if len(pred_diag) == 0 and len(gt_diag) == 0:
                continue

            # Matching penalty based on unmatched features
            n_pred_feat = len(pred_diag)
            n_gt_feat = len(gt_diag)
            penalty = float(abs(n_pred_feat - n_gt_feat))

            # If both have features, compute matching distance
            if n_pred_feat > 0 and n_gt_feat > 0:
                import numpy as np

                # Build persistence distance matrix
                dist_matrix = np.zeros((n_gt_feat, n_pred_feat))
                for i, (gb, gd) in enumerate(gt_diag):
                    for j, (pb, pd) in enumerate(pred_diag):
                        dist_matrix[i, j] = abs(gb - pb) + abs(gd - pd)

                gt_idx, pred_idx = linear_sum_assignment(dist_matrix)
                matching_cost = float(
                    sum(
                        dist_matrix[gi, pi]
                        for gi, pi in zip(gt_idx, pred_idx, strict=True)
                    )
                )
                penalty += matching_cost

            # Normalize penalty by spatial volume to keep scale comparable
            # to other losses (output in ~[0, 5] range instead of ~[0, 100])
            normalized_penalty = penalty / (spatial_volume**0.5)

            # Use differentiable proxy for gradient flow
            pred_b = fg_prob[b, 0]
            gt_b = gt[b, 0]
            grad_proxy = self._spatial_gradient_magnitude(pred_b - gt_b)

            total_loss = total_loss + normalized_penalty * grad_proxy + grad_proxy

        return self.lambda_topo * total_loss / batch_size

    @staticmethod
    def _persistence_diagram(
        mask: object, gudhi_module: object
    ) -> list[tuple[float, float]]:
        """Compute persistence diagram for a binary mask."""
        import gudhi
        import numpy as np

        mask_arr = np.asarray(mask, dtype=np.float64)
        filtration = 1.0 - mask_arr
        cc = gudhi.CubicalComplex(
            dimensions=list(mask_arr.shape),
            top_dimensional_cells=filtration.ravel().tolist(),
        )
        cc.persistence()
        pairs = cc.persistence_intervals_in_dimension(0)
        finite_pairs = pairs[np.isfinite(pairs).all(axis=1)]
        return [(float(b), float(d)) for b, d in finite_pairs]

    @staticmethod
    def _spatial_gradient_magnitude(volume: torch.Tensor) -> torch.Tensor:
        """Compute mean spatial gradient magnitude for a 3D volume."""
        grad_d = torch.diff(volume, dim=0)
        grad_h = torch.diff(volume, dim=1)
        grad_w = torch.diff(volume, dim=2)
        return grad_d.abs().mean() + grad_h.abs().mean() + grad_w.abs().mean()
