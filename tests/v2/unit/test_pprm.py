"""Tests for Prediction-Powered Risk Monitoring (Issue #4)."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T1: RiskEstimate dataclass
# ---------------------------------------------------------------------------


class TestRiskEstimate:
    """Test RiskEstimate dataclass."""

    def test_construction(self) -> None:
        """RiskEstimate should store risk and CI bounds."""
        from minivess.observability.pprm import RiskEstimate

        est = RiskEstimate(
            risk=0.15,
            ci_lower=0.12,
            ci_upper=0.18,
            alarm=False,
            threshold=0.20,
            n_calibration=50,
            n_deployment=200,
        )
        assert est.risk == 0.15
        assert est.ci_lower == 0.12
        assert est.ci_upper == 0.18
        assert est.alarm is False

    def test_alarm_when_risk_exceeds_threshold(self) -> None:
        """alarm should be True when CI lower > threshold."""
        from minivess.observability.pprm import RiskEstimate

        est = RiskEstimate(
            risk=0.25,
            ci_lower=0.22,
            ci_upper=0.28,
            alarm=True,
            threshold=0.20,
            n_calibration=50,
            n_deployment=200,
        )
        assert est.alarm is True

    def test_to_dict(self) -> None:
        """to_dict should return flat dict for Prometheus/MLflow."""
        from minivess.observability.pprm import RiskEstimate

        est = RiskEstimate(
            risk=0.15,
            ci_lower=0.12,
            ci_upper=0.18,
            alarm=False,
            threshold=0.20,
            n_calibration=50,
            n_deployment=200,
        )
        d = est.to_dict()
        assert "pprm_risk" in d
        assert "pprm_ci_lower" in d
        assert "pprm_ci_upper" in d
        assert "pprm_alarm" in d
        assert d["pprm_alarm"] == 0.0  # False → 0.0


# ---------------------------------------------------------------------------
# T2: PPRMDetector
# ---------------------------------------------------------------------------


class TestPPRMDetector:
    """Test PPRM detector calibration and monitoring."""

    def test_calibrate(self) -> None:
        """calibrate should accept predictions and labels."""
        from minivess.observability.pprm import PPRMDetector

        detector = PPRMDetector(threshold=0.20, alpha=0.05)
        rng = np.random.default_rng(42)

        cal_predictions = rng.random((50,))
        cal_labels = rng.random((50,))

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        detector.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)
        assert detector.is_calibrated

    def test_monitor_requires_calibration(self) -> None:
        """monitor should raise if not calibrated."""
        from minivess.observability.pprm import PPRMDetector

        detector = PPRMDetector(threshold=0.20)
        with pytest.raises(RuntimeError, match="calibrat"):
            detector.monitor(np.random.default_rng(42).random((100,)))

    def test_monitor_returns_risk_estimate(self) -> None:
        """monitor should return RiskEstimate."""
        from minivess.observability.pprm import PPRMDetector, RiskEstimate

        detector = PPRMDetector(threshold=0.20, alpha=0.05)
        rng = np.random.default_rng(42)

        cal_pred = rng.random((50,))
        cal_label = rng.random((50,))

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        detector.calibrate(cal_pred, cal_label, risk_fn=risk_fn)
        result = detector.monitor(rng.random((200,)))
        assert isinstance(result, RiskEstimate)
        assert result.n_calibration == 50
        assert result.n_deployment == 200

    def test_low_risk_no_alarm(self) -> None:
        """Near-perfect predictions should not trigger alarm."""
        from minivess.observability.pprm import PPRMDetector

        detector = PPRMDetector(threshold=0.50, alpha=0.05)
        rng = np.random.default_rng(42)

        # Calibration: predictions very close to labels
        cal_labels = rng.random((100,))
        cal_predictions = cal_labels + rng.normal(0, 0.01, size=100)

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        detector.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)

        # Deployment: similarly good predictions
        deploy_predictions = rng.random((200,)) + rng.normal(0, 0.01, size=200)
        result = detector.monitor(deploy_predictions)
        assert result.alarm is False

    def test_high_risk_triggers_alarm(self) -> None:
        """Large prediction errors should trigger alarm."""
        from minivess.observability.pprm import PPRMDetector

        detector = PPRMDetector(threshold=0.10, alpha=0.05)
        rng = np.random.default_rng(42)

        # Calibration: predictions close to labels
        cal_labels = rng.random((100,))
        cal_predictions = cal_labels + rng.normal(0, 0.01, size=100)

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        detector.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)

        # Deployment: predictions are terrible (random, uncorrelated)
        deploy_predictions = rng.random((200,)) + 5.0
        result = detector.monitor(deploy_predictions)
        assert result.alarm is True

    def test_ci_width_decreases_with_more_data(self) -> None:
        """More deployment data should yield tighter CI."""
        from minivess.observability.pprm import PPRMDetector

        detector = PPRMDetector(threshold=0.50, alpha=0.05)
        rng = np.random.default_rng(42)

        cal_labels = rng.random((100,))
        cal_predictions = cal_labels + rng.normal(0, 0.05, size=100)

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        detector.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)

        small_deploy = rng.random((20,))
        large_deploy = rng.random((500,))

        result_small = detector.monitor(small_deploy)
        result_large = detector.monitor(large_deploy)

        ci_width_small = result_small.ci_upper - result_small.ci_lower
        ci_width_large = result_large.ci_upper - result_large.ci_lower
        assert ci_width_large < ci_width_small

    def test_configurable_alpha(self) -> None:
        """Lower alpha should give wider CIs."""
        from minivess.observability.pprm import PPRMDetector

        rng = np.random.default_rng(42)
        cal_labels = rng.random((100,))
        cal_predictions = cal_labels + rng.normal(0, 0.05, size=100)

        def risk_fn(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            return np.abs(pred - label)

        det_90 = PPRMDetector(threshold=0.50, alpha=0.10)
        det_90.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)
        deploy = rng.random((200,))
        result_90 = det_90.monitor(deploy)

        det_99 = PPRMDetector(threshold=0.50, alpha=0.01)
        det_99.calibrate(cal_predictions, cal_labels, risk_fn=risk_fn)
        result_99 = det_99.monitor(deploy)

        width_90 = result_90.ci_upper - result_90.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# T3: compute_prediction_risk
# ---------------------------------------------------------------------------


class TestComputePredictionRisk:
    """Test risk function for segmentation."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have zero risk."""
        from minivess.observability.pprm import compute_prediction_risk

        predictions = np.array([1.0, 1.0, 1.0])
        labels = np.array([1.0, 1.0, 1.0])
        risk = compute_prediction_risk(predictions, labels)
        assert np.allclose(risk, 0.0)

    def test_worst_predictions(self) -> None:
        """Maximally wrong predictions should have high risk."""
        from minivess.observability.pprm import compute_prediction_risk

        predictions = np.array([0.0, 0.0, 0.0])
        labels = np.array([1.0, 1.0, 1.0])
        risk = compute_prediction_risk(predictions, labels)
        assert np.all(risk > 0.5)

    def test_per_sample_output(self) -> None:
        """Should return one risk value per sample."""
        from minivess.observability.pprm import compute_prediction_risk

        predictions = np.array([0.9, 0.5, 0.1])
        labels = np.array([1.0, 1.0, 1.0])
        risk = compute_prediction_risk(predictions, labels)
        assert risk.shape == (3,)
        # Higher error → higher risk
        assert risk[2] > risk[0]
