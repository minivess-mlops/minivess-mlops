"""Synthetic drift generation for monitoring validation.

Produces controlled distribution shifts that downstream drift detection
tools (Evidently, Alibi-Detect) should catch. Each drift type has a
severity parameter (0.0 = no drift, 1.0 = extreme).
"""

from __future__ import annotations

from enum import StrEnum

import torch
import torch.nn.functional as F  # noqa: N812


class DriftType(StrEnum):
    """Supported drift types for synthetic generation."""

    INTENSITY_SHIFT = "intensity_shift"
    NOISE_INJECTION = "noise_injection"
    RESOLUTION_DEGRADATION = "resolution_degradation"
    TOPOLOGY_CORRUPTION = "topology_corruption"


def apply_drift(
    volume: torch.Tensor,
    drift_type: DriftType,
    severity: float = 0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply a controlled distribution shift to a 3D volume.

    Parameters
    ----------
    volume:
        Input tensor of shape (C, D, H, W) or (D, H, W).
    drift_type:
        Type of drift to apply.
    severity:
        Drift intensity from 0.0 (no drift) to 1.0 (extreme).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Drifted volume (same shape as input, detached copy).
    """
    if severity < 1e-9:
        return volume.clone()

    if seed is not None:
        torch.manual_seed(seed)

    dispatch = {
        DriftType.INTENSITY_SHIFT: _intensity_shift,
        DriftType.NOISE_INJECTION: _noise_injection,
        DriftType.RESOLUTION_DEGRADATION: _resolution_degradation,
        DriftType.TOPOLOGY_CORRUPTION: _topology_corruption,
    }

    fn = dispatch[drift_type]
    return fn(volume.clone(), severity)


def _intensity_shift(volume: torch.Tensor, severity: float) -> torch.Tensor:
    """Global brightness shift: multiply by (1 ± severity * 0.6)."""
    # Shift factor: severity=0.5 → ×1.3 or ×0.7
    shift = 1.0 + severity * 0.6
    return volume * shift


def _noise_injection(volume: torch.Tensor, severity: float) -> torch.Tensor:
    """Additive Gaussian noise with std proportional to severity."""
    std = severity * 0.2  # severity=1.0 → std=0.2
    noise = torch.randn_like(volume) * std
    return volume + noise


def _resolution_degradation(volume: torch.Tensor, severity: float) -> torch.Tensor:
    """Downsample + upsample to simulate resolution loss (blur effect).

    At severity=1.0, downsamples to 25% resolution then upsamples back.
    """
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)  # Add channel dim
        squeeze = True
    else:
        squeeze = False

    # Add batch dim for F.interpolate
    x = volume.unsqueeze(0)  # (1, C, D, H, W)
    original_size = x.shape[2:]

    # Scale factor: severity=0 → 1.0 (no change), severity=1 → 0.25
    scale = max(1.0 - severity * 0.75, 0.25)
    small_size = tuple(max(int(s * scale), 1) for s in original_size)

    # Downsample then upsample
    x = F.interpolate(x, size=small_size, mode="trilinear", align_corners=False)
    x = F.interpolate(x, size=original_size, mode="trilinear", align_corners=False)

    result = x.squeeze(0)
    if squeeze:
        result = result.squeeze(0)
    return result


def _topology_corruption(volume: torch.Tensor, severity: float) -> torch.Tensor:
    """Morphological erosion to break vessel connectivity.

    Uses average pooling as a differentiable erosion proxy: pool then
    threshold. Higher severity → more erosion iterations.
    """
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)  # Add channel dim
        squeeze = True
    else:
        squeeze = False

    x = volume.unsqueeze(0).float()  # (1, C, D, H, W)

    # Number of erosion passes proportional to severity
    n_passes = max(1, int(severity * 3))
    kernel_size = 3

    for _ in range(n_passes):
        # Average pooling as soft erosion: values < 1.0 after pooling
        # indicate boundary voxels → threshold removes them
        x = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=1)
        # Threshold: keep only voxels that were fully surrounded
        threshold = 0.5 + severity * 0.3  # Higher severity → stricter threshold
        x = (x > threshold).float()

    result = x.squeeze(0)
    if squeeze:
        result = result.squeeze(0)
    return result
