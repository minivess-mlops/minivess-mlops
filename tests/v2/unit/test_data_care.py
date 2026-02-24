"""Tests for DATA-CARE data quality assessment (Issue #11)."""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# T1: QualityDimension enum and DimensionScore dataclass
# ---------------------------------------------------------------------------


class TestQualityDimension:
    """Test DATA-CARE quality dimension enum."""

    def test_dimensions_exist(self) -> None:
        """All six DATA-CARE dimensions should be defined."""
        from minivess.validation.data_care import QualityDimension

        assert hasattr(QualityDimension, "COMPLETENESS")
        assert hasattr(QualityDimension, "CORRECTNESS")
        assert hasattr(QualityDimension, "CONSISTENCY")
        assert hasattr(QualityDimension, "UNIQUENESS")
        assert hasattr(QualityDimension, "TIMELINESS")
        assert hasattr(QualityDimension, "REPRESENTATIVENESS")


class TestDimensionScore:
    """Test per-dimension quality score dataclass."""

    def test_creation(self) -> None:
        """DimensionScore should store dimension name, score, and issues."""
        from minivess.validation.data_care import DimensionScore, QualityDimension

        score = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.95,
            max_score=1.0,
            issues=["5% null values in voxel_spacing_x"],
        )
        assert score.score == 0.95
        assert len(score.issues) == 1

    def test_perfect_score(self) -> None:
        """A perfect score should have no issues."""
        from minivess.validation.data_care import DimensionScore, QualityDimension

        score = DimensionScore(
            dimension=QualityDimension.UNIQUENESS,
            score=1.0,
            max_score=1.0,
            issues=[],
        )
        assert score.score == score.max_score
        assert len(score.issues) == 0


# ---------------------------------------------------------------------------
# T2: DataQualityReport
# ---------------------------------------------------------------------------


class TestDataQualityReport:
    """Test aggregate quality report."""

    def test_overall_score(self) -> None:
        """Overall score should be mean of dimension scores."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 1.0, 1.0, []),
            DimensionScore(QualityDimension.CORRECTNESS, 0.5, 1.0, ["issue"]),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.75,
            passed=True,
            gate_threshold=0.7,
        )
        assert report.overall_score == 0.75
        assert report.passed is True

    def test_to_dict(self) -> None:
        """Report should be serializable to dict for MLflow."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 0.9, 1.0, []),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.9,
            passed=True,
            gate_threshold=0.7,
        )
        d = report.to_dict()
        assert "overall_score" in d
        assert "completeness" in d


# ---------------------------------------------------------------------------
# T3: assess_nifti_quality — NIfTI metadata quality assessment
# ---------------------------------------------------------------------------


class TestAssessNiftiQuality:
    """Test NIfTI metadata quality assessment."""

    def _make_clean_df(self) -> pd.DataFrame:
        """Create a clean NIfTI metadata DataFrame."""
        return pd.DataFrame({
            "file_path": ["/data/vol001.nii.gz", "/data/vol002.nii.gz"],
            "shape_x": [128, 128],
            "shape_y": [128, 128],
            "shape_z": [32, 32],
            "voxel_spacing_x": [0.5, 0.5],
            "voxel_spacing_y": [0.5, 0.5],
            "voxel_spacing_z": [1.0, 1.0],
            "intensity_min": [0.0, 0.0],
            "intensity_max": [1.0, 1.0],
            "has_valid_affine": [True, True],
        })

    def test_clean_data_passes(self) -> None:
        """Clean NIfTI metadata should get high overall score."""
        from minivess.validation.data_care import assess_nifti_quality

        df = self._make_clean_df()
        report = assess_nifti_quality(df)
        assert report.overall_score >= 0.9
        assert report.passed is True

    def test_missing_values_lowers_completeness(self) -> None:
        """Null values should lower the completeness score."""
        from minivess.validation.data_care import QualityDimension, assess_nifti_quality

        df = self._make_clean_df()
        df.loc[0, "voxel_spacing_x"] = None

        report = assess_nifti_quality(df)
        completeness = next(
            s for s in report.dimension_scores
            if s.dimension == QualityDimension.COMPLETENESS
        )
        assert completeness.score < 1.0

    def test_duplicate_paths_lowers_uniqueness(self) -> None:
        """Duplicate file paths should lower uniqueness score."""
        from minivess.validation.data_care import QualityDimension, assess_nifti_quality

        df = self._make_clean_df()
        df.loc[1, "file_path"] = df.loc[0, "file_path"]  # duplicate

        report = assess_nifti_quality(df)
        uniqueness = next(
            s for s in report.dimension_scores
            if s.dimension == QualityDimension.UNIQUENESS
        )
        assert uniqueness.score < 1.0

    def test_invalid_spacing_lowers_correctness(self) -> None:
        """Out-of-range voxel spacing should lower correctness."""
        from minivess.validation.data_care import QualityDimension, assess_nifti_quality

        df = self._make_clean_df()
        df.loc[0, "voxel_spacing_x"] = -1.0  # invalid negative spacing

        report = assess_nifti_quality(df)
        correctness = next(
            s for s in report.dimension_scores
            if s.dimension == QualityDimension.CORRECTNESS
        )
        assert correctness.score < 1.0


# ---------------------------------------------------------------------------
# T4: assess_metrics_quality — Training metrics quality assessment
# ---------------------------------------------------------------------------


class TestAssessMetricsQuality:
    """Test training metrics quality assessment."""

    def _make_clean_metrics_df(self) -> pd.DataFrame:
        """Create a clean training metrics DataFrame."""
        return pd.DataFrame({
            "run_id": ["run_001", "run_001", "run_001"],
            "epoch": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "val_dice": [0.7, 0.75, 0.8],
            "learning_rate": [1e-4, 1e-4, 1e-4],
        })

    def test_clean_metrics_pass(self) -> None:
        """Clean training metrics should get high score."""
        from minivess.validation.data_care import assess_metrics_quality

        df = self._make_clean_metrics_df()
        report = assess_metrics_quality(df)
        assert report.overall_score >= 0.9
        assert report.passed is True

    def test_invalid_dice_lowers_correctness(self) -> None:
        """Dice > 1.0 should lower correctness."""
        from minivess.validation.data_care import (
            QualityDimension,
            assess_metrics_quality,
        )

        df = self._make_clean_metrics_df()
        df.loc[0, "val_dice"] = 1.5  # impossible

        report = assess_metrics_quality(df)
        correctness = next(
            s for s in report.dimension_scores
            if s.dimension == QualityDimension.CORRECTNESS
        )
        assert correctness.score < 1.0

    def test_negative_loss_lowers_correctness(self) -> None:
        """Negative loss should lower correctness."""
        from minivess.validation.data_care import (
            QualityDimension,
            assess_metrics_quality,
        )

        df = self._make_clean_metrics_df()
        df.loc[0, "train_loss"] = -0.1

        report = assess_metrics_quality(df)
        correctness = next(
            s for s in report.dimension_scores
            if s.dimension == QualityDimension.CORRECTNESS
        )
        assert correctness.score < 1.0


# ---------------------------------------------------------------------------
# T5: quality_gate — Convert report to GateResult
# ---------------------------------------------------------------------------


class TestQualityGate:
    """Test DATA-CARE to GateResult conversion."""

    def test_passing_report(self) -> None:
        """High-quality report should produce passing GateResult."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
            quality_gate,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 1.0, 1.0, []),
            DimensionScore(QualityDimension.CORRECTNESS, 0.95, 1.0, []),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.975,
            passed=True,
            gate_threshold=0.7,
        )
        gate = quality_gate(report)
        assert gate.passed is True
        assert len(gate.errors) == 0

    def test_failing_report(self) -> None:
        """Low-quality report should produce failing GateResult."""
        from minivess.validation.data_care import (
            DataQualityReport,
            DimensionScore,
            QualityDimension,
            quality_gate,
        )

        scores = [
            DimensionScore(QualityDimension.COMPLETENESS, 0.3, 1.0, ["70% null"]),
            DimensionScore(QualityDimension.CORRECTNESS, 0.2, 1.0, ["bad values"]),
        ]
        report = DataQualityReport(
            dimension_scores=scores,
            overall_score=0.25,
            passed=False,
            gate_threshold=0.7,
        )
        gate = quality_gate(report)
        assert gate.passed is False
        assert len(gate.errors) > 0
