"""Tests for synthetic drift generation.

Verifies that each drift type produces statistically detectable distribution
shifts, so downstream monitoring (Evidently/Alibi-Detect) can catch them.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats


class TestIntensityDrift:
    """Intensity shift should change the pixel value distribution."""

    def test_intensity_drift_detectable(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        rng = np.random.default_rng(42)
        original = torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
        drifted = apply_drift(original, DriftType.INTENSITY_SHIFT, severity=0.5, seed=42)

        # KS test should detect the shift
        _, p_value = stats.ks_2samp(
            original.numpy().flatten(), drifted.numpy().flatten()
        )
        assert p_value < 0.05, f"Drift not detected: p={p_value:.4f}"

    def test_no_drift_preserves_distribution(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        rng = np.random.default_rng(42)
        original = torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
        same = apply_drift(original, DriftType.INTENSITY_SHIFT, severity=0.0, seed=42)

        _, p_value = stats.ks_2samp(
            original.numpy().flatten(), same.numpy().flatten()
        )
        assert p_value > 0.05, f"False drift detection: p={p_value:.4f}"


class TestNoiseDrift:
    """Gaussian noise injection should be detectable via variance change."""

    def test_noise_drift_increases_variance(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        rng = np.random.default_rng(42)
        original = torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
        noisy = apply_drift(original, DriftType.NOISE_INJECTION, severity=0.8, seed=42)

        orig_var = original.var().item()
        noisy_var = noisy.var().item()
        assert noisy_var > orig_var, "Noise should increase variance"


class TestResolutionDrift:
    """Downsampling + upsampling (blur) should be detectable."""

    def test_resolution_drift_reduces_high_freq(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        rng = np.random.default_rng(42)
        # Create volume with high-frequency content
        original = torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))
        blurred = apply_drift(
            original, DriftType.RESOLUTION_DEGRADATION, severity=0.7, seed=42
        )

        # High-frequency energy should decrease after blur
        orig_laplacian = torch.nn.functional.conv3d(
            original.unsqueeze(0),
            _laplacian_kernel(),
            padding=1,
        ).abs().mean().item()

        blur_laplacian = torch.nn.functional.conv3d(
            blurred.unsqueeze(0),
            _laplacian_kernel(),
            padding=1,
        ).abs().mean().item()

        assert blur_laplacian < orig_laplacian, "Blur should reduce high-frequency energy"


class TestTopologyDrift:
    """Morphological erosion should change vessel connectivity."""

    def test_topology_drift_reduces_foreground(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        # Create a binary mask with connected structures
        mask = torch.zeros(1, 32, 32, 8)
        mask[0, 10:20, 10:20, 2:6] = 1.0  # solid block

        eroded = apply_drift(mask, DriftType.TOPOLOGY_CORRUPTION, severity=0.5, seed=42)

        orig_fg = (mask > 0.5).sum().item()
        eroded_fg = (eroded > 0.5).sum().item()

        assert eroded_fg < orig_fg, "Erosion should reduce foreground"


class TestGradualDrift:
    """Gradual drift should produce monotonically increasing severity."""

    def test_gradual_drift_monotonic(self) -> None:
        from minivess.data.drift_synthetic import DriftType, apply_drift

        rng = np.random.default_rng(42)
        original = torch.tensor(rng.random((1, 32, 32, 8), dtype=np.float32))

        deviations = []
        for severity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            drifted = apply_drift(
                original, DriftType.INTENSITY_SHIFT, severity=severity, seed=42
            )
            mse = ((drifted - original) ** 2).mean().item()
            deviations.append(mse)

        # MSE should be monotonically non-decreasing with severity
        for i in range(1, len(deviations)):
            assert deviations[i] >= deviations[i - 1] - 1e-7, (
                f"Deviation should increase with severity: "
                f"{deviations[i]:.6f} < {deviations[i-1]:.6f}"
            )


def _laplacian_kernel() -> torch.Tensor:
    """3D Laplacian kernel for high-frequency energy measurement."""
    kernel = torch.zeros(1, 1, 3, 3, 3)
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return kernel
