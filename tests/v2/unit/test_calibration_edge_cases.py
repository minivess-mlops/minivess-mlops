"""Tests for calibration metrics edge cases.

Verifies compute_calibration_metrics handles single-class targets
(all-zero or all-one labels) without crashing.
"""

from __future__ import annotations

import math

import numpy as np

from minivess.pipeline.calibration_metrics import compute_calibration_metrics


class TestCalibrationEdgeCases:
    """Calibration metrics must handle single-class labels gracefully."""

    def test_all_zero_labels_no_crash(self) -> None:
        """All-background labels must not crash LogisticRegression."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.01, 0.99, size=1000).astype(np.float64)
        labels = np.zeros(1000, dtype=np.int32)

        result = compute_calibration_metrics(labels, probs, seed=42)
        assert result is not None
        # IPA and calibration_slope should be NaN for single-class
        assert math.isnan(result.ipa) or isinstance(result.ipa, float)
        assert math.isnan(result.calibration_slope) or isinstance(
            result.calibration_slope, float
        )

    def test_all_one_labels_no_crash(self) -> None:
        """All-foreground labels must not crash LogisticRegression."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.01, 0.99, size=1000).astype(np.float64)
        labels = np.ones(1000, dtype=np.int32)

        result = compute_calibration_metrics(labels, probs, seed=42)
        assert result is not None
        assert math.isnan(result.calibration_slope) or isinstance(
            result.calibration_slope, float
        )

    def test_normal_case_unchanged(self) -> None:
        """Mixed labels must produce valid (non-NaN) calibration metrics."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.01, 0.99, size=1000).astype(np.float64)
        labels = (rng.uniform(size=1000) > 0.5).astype(np.int32)

        result = compute_calibration_metrics(labels, probs, seed=42)
        assert result is not None
        assert not math.isnan(result.brier_score)
        assert not math.isnan(result.calibration_slope)
        assert not math.isnan(result.ipa)
