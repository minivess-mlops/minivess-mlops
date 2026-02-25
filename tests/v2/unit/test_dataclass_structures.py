"""Dataclass structure tests (Issue #54 — R5.12).

Tests construction, field access, and derived properties for dataclasses
in observability, validation, and drift modules.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

# ---------------------------------------------------------------------------
# R5.12 T1: DriftResult (observability/drift.py)
# ---------------------------------------------------------------------------


class TestDriftResult:
    """Test DriftResult construction and fields."""

    def test_construction(self) -> None:
        """DriftResult should store all fields."""
        from minivess.observability.drift import DriftResult

        result = DriftResult(
            drift_detected=True,
            dataset_drift_score=0.75,
            feature_scores={"mean": 0.01, "std": 0.03},
            drifted_features=["mean"],
            n_features=2,
            n_drifted=1,
        )
        assert result.drift_detected is True
        assert result.dataset_drift_score == 0.75
        assert len(result.feature_scores) == 2
        assert result.n_drifted == 1

    def test_defaults(self) -> None:
        """DriftResult default fields should be empty."""
        from minivess.observability.drift import DriftResult

        result = DriftResult(
            drift_detected=False,
            dataset_drift_score=0.0,
        )
        assert result.feature_scores == {}
        assert result.drifted_features == []
        assert result.n_features == 0
        assert result.n_drifted == 0

    def test_timestamp_auto_generated(self) -> None:
        """DriftResult should have a timestamp set to ~now."""
        from minivess.observability.drift import DriftResult

        result = DriftResult(drift_detected=False, dataset_drift_score=0.0)
        assert isinstance(result.timestamp, datetime)
        # Should be within the last minute
        delta = datetime.now(UTC) - result.timestamp
        assert delta.total_seconds() < 60


# ---------------------------------------------------------------------------
# R5.12 T2: GateResult (validation/gates.py)
# ---------------------------------------------------------------------------


class TestGateResult:
    """Test GateResult construction and fields."""

    def test_construction_pass(self) -> None:
        """GateResult with passed=True should have no errors."""
        from minivess.validation.gates import GateResult

        result = GateResult(passed=True)
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []
        assert result.statistics == {}

    def test_construction_fail(self) -> None:
        """GateResult with errors should have passed=False."""
        from minivess.validation.gates import GateResult

        result = GateResult(
            passed=False,
            errors=["Missing column: shape_x", "Invalid voxel spacing"],
            warnings=["Low foreground ratio"],
            statistics={"n_rows": 100, "n_errors": 2},
        )
        assert result.passed is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.statistics["n_errors"] == 2


# ---------------------------------------------------------------------------
# R5.12 T3: DimensionScore and DataQualityReport (validation/data_care.py)
# ---------------------------------------------------------------------------


class TestDimensionScore:
    """Test DimensionScore dataclass."""

    def test_construction(self) -> None:
        """DimensionScore should store dimension, score, and issues."""
        from minivess.validation.data_care import DimensionScore, QualityDimension

        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.95,
            issues=["5% null in file_path"],
        )
        assert ds.dimension == QualityDimension.COMPLETENESS
        assert ds.score == 0.95
        assert ds.max_score == 1.0
        assert len(ds.issues) == 1


class TestDataQualityReport:
    """Test DataQualityReport dataclass."""

    def test_construction(self) -> None:
        """DataQualityReport should be constructible."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 0.9),
            DimensionScore(QualityDimension.CORRECTNESS, 0.8),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.85,
            passed=True,
            gate_threshold=0.7,
        )
        assert report.passed is True
        assert report.overall_score == 0.85

    def test_to_dict(self) -> None:
        """to_dict should flatten scores for logging."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 0.9),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.9,
            passed=True,
            gate_threshold=0.7,
        )
        d = report.to_dict()
        assert d["overall_score"] == 0.9
        assert d["completeness"] == 0.9


# ---------------------------------------------------------------------------
# R5.12 T4: ProfileDriftReport (validation/profiling.py)
# ---------------------------------------------------------------------------


class TestProfileDriftReport:
    """Test ProfileDriftReport dataclass."""

    def test_construction_empty(self) -> None:
        """Empty ProfileDriftReport should have empty lists."""
        from minivess.validation.profiling import ProfileDriftReport

        report = ProfileDriftReport()
        assert report.drifted_columns == []
        assert report.column_summaries == {}

    def test_construction_with_drift(self) -> None:
        """ProfileDriftReport should store drifted columns."""
        from minivess.validation.profiling import ProfileDriftReport

        report = ProfileDriftReport(
            drifted_columns=["mean", "std"],
            column_summaries={
                "mean": {"ref_mean": 0.5, "cur_mean": 0.8, "ref_stddev": 0.1},
            },
        )
        assert len(report.drifted_columns) == 2
        assert "mean" in report.column_summaries


# ---------------------------------------------------------------------------
# R5.12 T5: ConformalMetrics (ensemble/mapie_conformal.py)
# ---------------------------------------------------------------------------


class TestConformalMetrics:
    """Test ConformalMetrics dataclass."""

    def test_construction(self) -> None:
        """ConformalMetrics should store coverage and set size."""
        from minivess.ensemble.mapie_conformal import ConformalMetrics

        cm = ConformalMetrics(coverage=0.92, mean_set_size=1.3)
        assert cm.coverage == 0.92
        assert cm.mean_set_size == 1.3

    def test_to_dict(self) -> None:
        """to_dict should produce MLflow-compatible keys."""
        from minivess.ensemble.mapie_conformal import ConformalMetrics

        cm = ConformalMetrics(coverage=0.90, mean_set_size=1.5)
        d = cm.to_dict()
        assert d["conformal_coverage"] == 0.90
        assert d["conformal_mean_set_size"] == 1.5


# ---------------------------------------------------------------------------
# R5.12 T6: CalibrationResult (ensemble/calibration.py)
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_construction(self) -> None:
        """CalibrationResult should store ECE, MCE, and optional probs."""
        from minivess.ensemble.calibration import CalibrationResult

        result = CalibrationResult(ece=0.05, mce=0.12)
        assert result.ece == 0.05
        assert result.mce == 0.12
        assert result.calibrated_probs is None

    def test_with_calibrated_probs(self) -> None:
        """CalibrationResult should optionally carry calibrated_probs."""
        from minivess.ensemble.calibration import CalibrationResult

        probs = np.array([0.8, 0.9, 0.7])
        result = CalibrationResult(ece=0.03, mce=0.08, calibrated_probs=probs)
        assert result.calibrated_probs is not None
        assert len(result.calibrated_probs) == 3
