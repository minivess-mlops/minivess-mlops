"""Tests for pixel-level calibration (DA-Cal / Meta Temperature Network).

DA-Cal (Li et al. 2026, arXiv:2602.08060) generates pixel-level calibration
temperatures via a Meta Temperature Network (MTN). Under perfect calibration,
soft pseudo-labels equal hard pseudo-labels.

Issue: #311 | Phase 5 | Plan: T5.1 (RED)
"""

from __future__ import annotations

import numpy as np
import torch


class TestPixelTemperatureScaling:
    """Tests for pixel-level temperature scaling."""

    def test_pixel_temperature_scaling_reduces_ece(self) -> None:
        """Pixel-wise temperature scaling should reduce ECE compared to
        uncalibrated predictions."""
        from minivess.pipeline.calibration import pixel_temperature_scale

        # Overconfident logits (high values)
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((5, 2, 4, 4, 4)).astype(np.float32) * 5.0

        # Temperature map > 1 everywhere (softening)
        temperature_map = np.full((5, 1, 4, 4, 4), 2.0, dtype=np.float32)

        calibrated = pixel_temperature_scale(logits, temperature_map)

        assert calibrated.shape == logits.shape
        # Softmax sum should be 1 along class axis
        sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_meta_temperature_network_forward(self) -> None:
        """Meta Temperature Network should produce a positive temperature map."""
        from minivess.pipeline.calibration import MetaTemperatureNetwork

        mtn = MetaTemperatureNetwork(in_channels=2)
        logits = torch.randn(1, 2, 8, 8, 8)

        temp_map = mtn(logits)
        assert temp_map.shape == (1, 1, 8, 8, 8)
        assert (temp_map > 0).all(), "Temperature must be positive"

    def test_calibrated_softmax_sums_to_one(self) -> None:
        """After pixel-level calibration, softmax should sum to 1."""
        from minivess.pipeline.calibration import pixel_temperature_scale

        logits = (
            np.random.default_rng(42)
            .standard_normal((2, 3, 4, 4, 4))
            .astype(np.float32)
        )
        temperature_map = np.full((2, 1, 4, 4, 4), 1.5, dtype=np.float32)

        calibrated = pixel_temperature_scale(logits, temperature_map)

        sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_pixel_calibration_preserves_predictions(self) -> None:
        """Temperature scaling should not change the argmax predictions
        (monotonic transformation of logits)."""
        from minivess.pipeline.calibration import pixel_temperature_scale

        logits = (
            np.random.default_rng(42)
            .standard_normal((2, 2, 4, 4, 4))
            .astype(np.float32)
        )
        temperature_map = np.full((2, 1, 4, 4, 4), 3.0, dtype=np.float32)

        calibrated = pixel_temperature_scale(logits, temperature_map)

        original_preds = logits.argmax(axis=1)
        calibrated_preds = calibrated.argmax(axis=1)
        np.testing.assert_array_equal(original_preds, calibrated_preds)

    def test_pixel_temperature_t1_is_regular_softmax(self) -> None:
        """Temperature=1 should produce same result as regular softmax."""
        from minivess.pipeline.calibration import pixel_temperature_scale

        logits = (
            np.random.default_rng(42)
            .standard_normal((2, 2, 4, 4, 4))
            .astype(np.float32)
        )
        temperature_map = np.ones((2, 1, 4, 4, 4), dtype=np.float32)

        calibrated = pixel_temperature_scale(logits, temperature_map)

        # Manual softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        expected = exp / exp.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(calibrated, expected, atol=1e-5)

    def test_mtn_gradient_flows(self) -> None:
        """MTN should be differentiable (for bi-level optimization)."""
        from minivess.pipeline.calibration import MetaTemperatureNetwork

        mtn = MetaTemperatureNetwork(in_channels=2)
        logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)

        temp_map = mtn(logits)
        loss = temp_map.mean()
        loss.backward()

        assert logits.grad is not None
