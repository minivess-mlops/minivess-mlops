"""Tests for calibration metrics assessment (Phase 4, T4.2).

Validates Brier score, O/E ratio, and IPA computation for evaluating
how well predicted probabilities reflect true frequencies.

Reference: Van Calster et al. (2019) "Calibration: the Achilles heel
of predictive analytics." BMC Medicine.
"""

from __future__ import annotations

import numpy as np


def _make_calibration_data(
    *,
    n_samples: int = 100,
    well_calibrated: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic probability predictions and binary labels.

    Returns (predictions, labels).
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_samples).astype(float)

    if well_calibrated:
        # Predictions close to true probabilities
        predictions = labels * 0.7 + (1 - labels) * 0.3
        predictions += rng.normal(0, 0.1, size=n_samples)
        predictions = np.clip(predictions, 0.01, 0.99)
    else:
        # Overconfident predictions (always predicts ~0.5)
        predictions = np.full(n_samples, 0.5) + rng.normal(0, 0.02, size=n_samples)
        predictions = np.clip(predictions, 0.01, 0.99)

    return predictions, labels


class TestBrierScore:
    """Brier score computation."""

    def test_brier_score_returns_float(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_brier_score

        preds, labels = _make_calibration_data()
        score = compute_brier_score(preds, labels)
        assert isinstance(score, float)

    def test_brier_score_in_range(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_brier_score

        preds, labels = _make_calibration_data()
        score = compute_brier_score(preds, labels)
        assert 0.0 <= score <= 1.0

    def test_perfect_predictions_brier_zero(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_brier_score

        labels = np.array([1.0, 0.0, 1.0, 0.0])
        preds = np.array([1.0, 0.0, 1.0, 0.0])
        score = compute_brier_score(preds, labels)
        assert score == 0.0

    def test_well_calibrated_lower_than_poor(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_brier_score

        preds_good, labels = _make_calibration_data(well_calibrated=True)
        preds_bad, _ = _make_calibration_data(well_calibrated=False)
        score_good = compute_brier_score(preds_good, labels)
        score_bad = compute_brier_score(preds_bad, labels)
        assert score_good < score_bad


class TestOERatio:
    """Observed/Expected ratio."""

    def test_oe_ratio_returns_float(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_oe_ratio

        preds, labels = _make_calibration_data()
        ratio = compute_oe_ratio(preds, labels)
        assert isinstance(ratio, float)

    def test_perfect_calibration_oe_near_one(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_oe_ratio

        # Well-calibrated predictions → O/E ≈ 1
        preds, labels = _make_calibration_data(well_calibrated=True, n_samples=1000)
        ratio = compute_oe_ratio(preds, labels)
        assert 0.5 < ratio < 2.0, (
            f"O/E should be near 1.0 for calibrated preds, got {ratio}"
        )


class TestIPA:
    """Index of Prediction Accuracy (Brier skill score)."""

    def test_ipa_returns_float(self) -> None:
        from minivess.pipeline.biostatistics_calibration import compute_ipa

        preds, labels = _make_calibration_data()
        ipa = compute_ipa(preds, labels)
        assert isinstance(ipa, float)

    def test_ipa_positive_for_good_predictions(self) -> None:
        """IPA > 0 means predictions are better than the prevalence baseline."""
        from minivess.pipeline.biostatistics_calibration import compute_ipa

        preds, labels = _make_calibration_data(well_calibrated=True)
        ipa = compute_ipa(preds, labels)
        assert ipa > 0, f"IPA should be positive for good predictions, got {ipa}"


class TestCalibrationSummary:
    """Combined calibration summary function."""

    def test_summary_returns_all_metrics(self) -> None:
        from minivess.pipeline.biostatistics_calibration import (
            CalibrationSummary,
            compute_calibration_summary,
        )

        preds, labels = _make_calibration_data()
        summary = compute_calibration_summary(preds, labels)
        assert isinstance(summary, CalibrationSummary)
        assert isinstance(summary.brier_score, float)
        assert isinstance(summary.oe_ratio, float)
        assert isinstance(summary.ipa, float)
