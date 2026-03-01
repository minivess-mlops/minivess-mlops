"""TopoSegNet-inspired topology loss via critical point detection.

EXPERIMENTAL — Simplified proxy for the TopoSegNet loss (Gupta & Essa, IJCV 2025).
The original paper uses discrete Morse theory to identify topological critical points
and penalizes misclassification at those locations. This implementation uses a
SIMPLIFIED PROXY based on multi-scale pooling to approximate critical points.

WARNING: This is NOT a faithful implementation of discrete Morse theory.
It uses max-pooling to find local maxima (approximate critical points) and
penalizes prediction errors at those locations. The proxy is differentiable
but may not capture the same topological features as true discrete Morse.

References:
    Gupta, A. & Essa, I. (2025). "Scalable Topological Loss for Segmentation
    via Discrete Morse Theory." IJCV.

See docs/planning/novel-loss-debugging-plan.xml for risk assessment.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)

_WARNED = False


class TopoSegLoss(nn.Module):  # type: ignore[misc]
    """Topology loss via critical point detection (simplified proxy).

    EXPERIMENTAL: Uses multi-scale max-pooling to approximate topological
    critical points, NOT true discrete Morse theory. Penalizes prediction
    errors at these approximate critical points.

    Parameters
    ----------
    lambda_topo:
        Weight for the topology penalty term.
    scales:
        Pooling kernel sizes for multi-scale critical point detection.
    softmax:
        Apply softmax to logits before computing loss.
    """

    def __init__(
        self,
        lambda_topo: float = 1.0,
        scales: tuple[int, ...] = (3, 5),
        *,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_topo = lambda_topo
        self.scales = scales
        self.softmax = softmax

        global _WARNED  # noqa: PLW0603
        if not _WARNED:
            logger.warning(
                "TopoSegLoss is EXPERIMENTAL — uses multi-scale max-pooling as a "
                "proxy for discrete Morse critical points. NOT a faithful implementation "
                "of Gupta & Essa (IJCV 2025). See novel-loss-debugging-plan.xml."
            )
            _WARNED = True

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute topology-aware loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.

        Returns
        -------
        Scalar loss: BCE at critical points + spatial gradient penalty.
        """
        probs = torch.softmax(logits, dim=1) if self.softmax else logits
        fg_prob = probs[:, 1:2]  # Foreground channel (B, 1, D, H, W)

        # Prepare GT foreground
        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        fg_label = (labels_squeeze > 0).float().unsqueeze(1)  # (B, 1, D, H, W)

        # Detect approximate critical points at multiple scales
        critical_mask = self._detect_critical_points(fg_prob, fg_label)

        # Penalize prediction errors at critical points
        if critical_mask.sum() == 0:
            # No critical points found — return zero preserving gradient chain
            return (logits * 0.0).sum()

        # BCE loss weighted by critical point mask
        eps = 1e-7
        fg_clamped = fg_prob.clamp(eps, 1.0 - eps)
        bce = -(
            fg_label * torch.log(fg_clamped)
            + (1 - fg_label) * torch.log(1 - fg_clamped)
        )

        # Weight by critical point importance
        weighted_bce = (bce * critical_mask).sum() / (critical_mask.sum() + eps)

        return self.lambda_topo * weighted_bce

    def _detect_critical_points(
        self,
        fg_prob: torch.Tensor,
        fg_label: torch.Tensor,
    ) -> torch.Tensor:
        """Detect approximate critical points via multi-scale max-pooling.

        Critical points are locations where the prediction is locally extreme
        (local maxima/minima) AND the prediction disagrees with GT. These are
        topologically important — they correspond to locations where the
        predicted topology differs from the GT topology.

        This is a SIMPLIFIED PROXY for discrete Morse critical points.
        """
        # Ensure spatial dims are large enough for pooling
        spatial = fg_prob.shape[2:]
        min_dim = min(spatial)

        critical_mask = torch.zeros_like(fg_prob)

        for kernel_size in self.scales:
            if kernel_size > min_dim:
                continue

            pad = kernel_size // 2

            # Local max of foreground probability
            local_max = F.max_pool3d(
                fg_prob, kernel_size=kernel_size, stride=1, padding=pad
            )
            # Ensure spatial dims match after pooling (handle edge cases)
            if local_max.shape != fg_prob.shape:
                local_max = F.interpolate(local_max, size=spatial, mode="nearest")

            # Local min (max of negated = negated min)
            local_min = -F.max_pool3d(
                -fg_prob, kernel_size=kernel_size, stride=1, padding=pad
            )
            if local_min.shape != fg_prob.shape:
                local_min = F.interpolate(local_min, size=spatial, mode="nearest")

            # Critical points: local max or local min
            is_local_max = (fg_prob >= local_max - 1e-6).float()
            is_local_min = (fg_prob <= local_min + 1e-6).float()

            # Focus on disagreement regions (pred != GT)
            disagreement = (fg_prob - fg_label).abs()

            # Combine: critical points with high disagreement
            scale_mask = (is_local_max + is_local_min) * disagreement
            critical_mask = critical_mask + scale_mask

        # Normalize
        if critical_mask.max() > 0:
            critical_mask = critical_mask / (critical_mask.max() + 1e-7)

        return critical_mask
