"""Tests for calibration metrics in SegmentationMetrics (Phase B1).

Validates that Tier 1 calibration metrics (ECE, Brier, NLL, etc.) are computed
during the validation loop when soft predictions are provided.
"""

from __future__ import annotations

import torch


class TestSegmentationMetricsCalibration:
    """SegmentationMetrics should compute Tier 1 calibration metrics during validation."""

    def test_compute_returns_calibration_keys(self) -> None:
        """compute() should include val_ece, val_brier, val_nll etc."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # Create soft predictions (B=2, C=2, D=4, H=4, W=4)
        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 1, ...] = 0.8  # High foreground probability
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)  # All foreground

        metrics.update(pred, target)
        result = metrics.compute()

        assert "val_ece" in result.values
        assert "val_brier" in result.values
        assert "val_nll" in result.values

    def test_calibration_metrics_in_valid_range(self) -> None:
        """All calibration metrics should be non-negative."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 0, ...] = 0.3
        pred[:, 1, ...] = 0.7
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        result = metrics.compute()

        for key, val in result.values.items():
            if key.startswith("val_"):
                assert val >= 0.0, f"{key} is negative: {val}"

    def test_all_tier1_keys_present(self) -> None:
        """All 7 Tier 1 calibration metric keys should appear."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 0, ...] = 0.4
        pred[:, 1, ...] = 0.6
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        result = metrics.compute()

        expected_keys = {
            "val_ece",
            "val_mce",
            "val_rmsce",
            "val_brier",
            "val_nll",
            "val_overconfidence_error",
            "val_debiased_ece",
        }
        actual_cal_keys = {k for k in result.values if k.startswith("val_")}
        assert expected_keys == actual_cal_keys, (
            f"Missing keys: {expected_keys - actual_cal_keys}, "
            f"Extra keys: {actual_cal_keys - expected_keys}"
        )

    def test_reset_clears_calibration_accumulators(self) -> None:
        """reset() should clear calibration data."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 1, ...] = 0.8
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        assert len(metrics._cal_probs) == 1

        metrics.reset()

        # After reset, accumulators should be empty
        assert len(metrics._cal_probs) == 0
        assert len(metrics._cal_labels) == 0

        # After feeding new data post-reset, compute works and includes cal metrics
        metrics.update(pred, target)
        result = metrics.compute()
        assert isinstance(result.values, dict)
        assert "val_ece" in result.values

    def test_no_calibration_without_soft_predictions(self) -> None:
        """Hard predictions (class indices) should not produce calibration metrics."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # Hard predictions: (B, D, H, W) class indices — no probabilities
        pred = torch.ones(2, 4, 4, 4, dtype=torch.long)
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        result = metrics.compute()

        # Should NOT have calibration metrics (no probabilities available)
        assert "val_ece" not in result.values
        assert "val_brier" not in result.values

    def test_single_channel_sigmoid_predictions(self) -> None:
        """Single-channel (B, 1, D, H, W) sigmoid predictions should work."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # Single-channel sigmoid output
        pred = torch.full((2, 1, 4, 4, 4), 0.9)
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        result = metrics.compute()

        # Should have calibration metrics
        assert "val_ece" in result.values
        assert "val_brier" in result.values

    def test_multi_batch_accumulation(self) -> None:
        """Calibration metrics should accumulate across multiple update() calls."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # Two separate batches
        for _ in range(3):
            pred = torch.zeros(1, 2, 4, 4, 4)
            pred[:, 1, ...] = 0.7
            target = torch.ones(1, 4, 4, 4, dtype=torch.long)
            metrics.update(pred, target)

        assert len(metrics._cal_probs) == 3
        result = metrics.compute()
        assert "val_ece" in result.values

    def test_well_calibrated_predictions_low_ece(self) -> None:
        """Well-calibrated predictions should have low ECE and Brier."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # Near-perfect calibration: predict 1.0 for foreground, 0.0 for background
        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 1, ...] = 0.99  # Very high foreground prob
        pred[:, 0, ...] = 0.01
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)  # All foreground

        metrics.update(pred, target)
        result = metrics.compute()

        assert result.values["val_brier"] < 0.05, (
            f"Brier should be low for well-calibrated predictions, "
            f"got {result.values['val_brier']}"
        )

    def test_dice_and_f1_still_computed(self) -> None:
        """Calibration metrics should not break existing dice/f1 computation."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        pred = torch.zeros(2, 2, 4, 4, 4)
        pred[:, 1, ...] = 0.8
        target = torch.ones(2, 4, 4, 4, dtype=torch.long)

        metrics.update(pred, target)
        result = metrics.compute()

        # Original metrics must still be present
        assert "dice" in result.values
        assert "f1_foreground" in result.values
        assert result.values["dice"] > 0.0
        assert result.values["f1_foreground"] > 0.0
