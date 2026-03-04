"""Pixel-level calibration for segmentation models.

Implements pixel-wise temperature scaling and Meta Temperature Network (MTN)
for cross-domain calibration of 3D segmentation models.

References:
    - Li et al. (2026), "DA-Cal: Cross-Domain Calibration", arXiv:2602.08060.
    - Guo et al. (2017), "On Calibration of Modern Neural Networks" (global T-scaling).

The global temperature_scale() from ensemble.calibration handles image-level
calibration. This module extends to pixel-level calibration via learned or
fixed temperature maps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def pixel_temperature_scale(
    logits: NDArray[np.float32],
    temperature_map: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply pixel-wise temperature scaling to logits.

    Parameters
    ----------
    logits:
        Raw model logits, shape (B, C, D, H, W).
    temperature_map:
        Per-pixel temperature, shape (B, 1, D, H, W). Must be > 0.

    Returns
    -------
    Calibrated probabilities after softmax(logits / T_pixel).
    """
    scaled = logits / temperature_map
    # Stable softmax along class axis (axis=1)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    result: NDArray[np.float32] = (exp / exp.sum(axis=1, keepdims=True)).astype(
        np.float32
    )
    return result


class MetaTemperatureNetwork(nn.Module):
    """Meta Temperature Network (MTN) for pixel-level calibration.

    A lightweight 3D CNN that takes logits as input and produces a
    per-pixel temperature map. Used for DA-Cal-style cross-domain
    calibration.

    Parameters
    ----------
    in_channels:
        Number of input channels (= number of classes in logits).
    hidden_channels:
        Hidden layer width.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
            nn.Softplus(),  # Ensure T > 0
        )
        # Initialize bias so initial temperature ~ 1.0
        # Softplus(0) = ln(2) ≈ 0.693, so we bias to get ~1.0
        with torch.no_grad():
            last_conv = self.net[-2]
            if hasattr(last_conv, "bias") and last_conv.bias is not None:
                last_conv.bias.fill_(0.5413)  # type: ignore[operator]  # softplus(0.5413) ≈ 1.0

    def forward(self, logits: Tensor) -> Tensor:
        """Produce pixel-level temperature map from logits.

        Parameters
        ----------
        logits:
            Model logits, shape (B, C, D, H, W).

        Returns
        -------
        Temperature map (B, 1, D, H, W), all values > 0.
        """
        result: Tensor = self.net(logits)
        return result
