"""Tests for calibration-under-shift framework (Issue #19)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: ShiftType enum
# ---------------------------------------------------------------------------


class TestShiftType:
    """Test distribution shift type enum."""

    def test_enum_values(self) -> None:
        """ShiftType should have three shift categories."""
        from minivess.ensemble.calibration_shift import ShiftType

        assert ShiftType.COVARIATE == "covariate"
        assert ShiftType.LABEL == "label"
        assert ShiftType.CONCEPT == "concept"


# ---------------------------------------------------------------------------
# T2: Synthetic shifts
# ---------------------------------------------------------------------------


class TestSyntheticShift:
    """Test synthetic domain shift application."""

    def test_intensity_shift(self) -> None:
        """apply_synthetic_shift with intensity should shift mean."""
        from minivess.ensemble.calibration_shift import apply_synthetic_shift

        rng = np.random.default_rng(42)
        data = rng.normal(0.5, 0.1, size=(100,))
        shifted = apply_synthetic_shift(data, shift_type="intensity", magnitude=0.2)
        assert abs(np.mean(shifted) - np.mean(data)) > 0.1

    def test_noise_shift(self) -> None:
        """apply_synthetic_shift with noise should increase variance."""
        from minivess.ensemble.calibration_shift import apply_synthetic_shift

        rng = np.random.default_rng(42)
        data = rng.normal(0.5, 0.1, size=(1000,))
        shifted = apply_synthetic_shift(
            data, shift_type="noise", magnitude=0.3, seed=42
        )
        assert np.std(shifted) > np.std(data)

    def test_zero_magnitude_no_change(self) -> None:
        """Zero magnitude should return data unchanged."""
        from minivess.ensemble.calibration_shift import apply_synthetic_shift

        data = np.array([0.5, 0.6, 0.7])
        shifted = apply_synthetic_shift(data, shift_type="intensity", magnitude=0.0)
        np.testing.assert_array_almost_equal(shifted, data)


# ---------------------------------------------------------------------------
# T3: Calibration transfer evaluation
# ---------------------------------------------------------------------------


class TestCalibrationTransfer:
    """Test calibration degradation under shift."""

    def test_evaluate_calibration_transfer(self) -> None:
        """evaluate_calibration_transfer should return ECE metrics."""
        from minivess.ensemble.calibration_shift import evaluate_calibration_transfer

        rng = np.random.default_rng(42)
        # Well-calibrated source
        source_confidences = rng.uniform(0.6, 1.0, size=(200,))
        source_accuracies = (rng.random(200) < source_confidences).astype(float)
        # Shifted target (miscalibrated)
        target_confidences = rng.uniform(0.7, 1.0, size=(200,))
        target_accuracies = (rng.random(200) < 0.5).astype(float)

        result = evaluate_calibration_transfer(
            source_confidences=source_confidences,
            source_accuracies=source_accuracies,
            target_confidences=target_confidences,
            target_accuracies=target_accuracies,
        )
        assert hasattr(result, "source_ece")
        assert hasattr(result, "target_ece")
        assert hasattr(result, "degradation")

    def test_degradation_positive_when_worse(self) -> None:
        """Degradation should be positive when target calibration is worse."""
        from minivess.ensemble.calibration_shift import evaluate_calibration_transfer

        # Source: perfectly calibrated
        source_conf = np.array([0.9] * 100)
        source_acc = np.ones(100)
        # Target: highly miscalibrated (high confidence, low accuracy)
        target_conf = np.array([0.9] * 100)
        target_acc = np.zeros(100)

        result = evaluate_calibration_transfer(
            source_confidences=source_conf,
            source_accuracies=source_acc,
            target_confidences=target_conf,
            target_accuracies=target_acc,
        )
        assert result.degradation > 0


# ---------------------------------------------------------------------------
# T4: CalibrationShiftAnalyzer
# ---------------------------------------------------------------------------


class TestCalibrationShiftAnalyzer:
    """Test multi-domain calibration shift analyzer."""

    def test_add_domain(self) -> None:
        """add_domain should register a domain."""
        from minivess.ensemble.calibration_shift import CalibrationShiftAnalyzer

        analyzer = CalibrationShiftAnalyzer()
        analyzer.add_domain(
            name="site_a",
            confidences=np.array([0.8, 0.9]),
            accuracies=np.array([1.0, 1.0]),
        )
        assert "site_a" in analyzer.domains

    def test_analyze_transfer_pairwise(self) -> None:
        """analyze_transfer should compute pairwise calibration shifts."""
        from minivess.ensemble.calibration_shift import CalibrationShiftAnalyzer

        rng = np.random.default_rng(42)
        analyzer = CalibrationShiftAnalyzer()
        for name in ("site_a", "site_b"):
            conf = rng.uniform(0.5, 1.0, size=(100,))
            acc = (rng.random(100) < conf).astype(float)
            analyzer.add_domain(name, conf, acc)

        results = analyzer.analyze_transfer()
        assert len(results) >= 1
        assert all(hasattr(r, "source_ece") for r in results)

    def test_to_markdown(self) -> None:
        """to_markdown should produce a readable report."""
        from minivess.ensemble.calibration_shift import CalibrationShiftAnalyzer

        rng = np.random.default_rng(42)
        analyzer = CalibrationShiftAnalyzer()
        for name in ("site_a", "site_b"):
            conf = rng.uniform(0.5, 1.0, size=(50,))
            acc = (rng.random(50) < conf).astype(float)
            analyzer.add_domain(name, conf, acc)

        md = analyzer.to_markdown()
        assert "Calibration" in md
        assert "site_a" in md
        assert "site_b" in md
