"""Tests for validation pipeline: Pandera schemas, GE suites, drift, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_nifti_metadata() -> pd.DataFrame:
    """Valid NIfTI metadata DataFrame matching NiftiMetadataSchema."""
    return pd.DataFrame(
        {
            "file_path": ["/data/sub01.nii.gz", "/data/sub02.nii.gz"],
            "shape_x": [256, 256],
            "shape_y": [256, 256],
            "shape_z": [128, 64],
            "voxel_spacing_x": [0.5, 0.7],
            "voxel_spacing_y": [0.5, 0.7],
            "voxel_spacing_z": [1.0, 2.0],
            "intensity_min": [-1.0, 0.0],
            "intensity_max": [1.0, 255.0],
            "num_foreground_voxels": [5000, 3200],
            "has_valid_affine": [True, True],
        }
    )


@pytest.fixture()
def valid_training_metrics() -> pd.DataFrame:
    """Valid training metrics DataFrame matching TrainingMetricsSchema."""
    return pd.DataFrame(
        {
            "run_id": ["run_001", "run_001", "run_002"],
            "epoch": [0, 1, 0],
            "fold": [0, 0, 1],
            "train_loss": [0.8, 0.5, 0.9],
            "val_loss": [0.7, 0.4, 0.85],
            "val_dice": [0.3, 0.6, 0.25],
            "val_cldice": [0.2, 0.5, 0.2],
            "val_nsd": [0.4, 0.7, 0.35],
            "learning_rate": [1e-3, 1e-3, 1e-4],
        }
    )


@pytest.fixture()
def valid_annotation_quality() -> pd.DataFrame:
    """Valid annotation quality DataFrame matching AnnotationQualitySchema."""
    return pd.DataFrame(
        {
            "sample_id": ["s01", "s02", "s03"],
            "annotator_id": ["ann_A", "ann_B", "ann_A"],
            "num_connected_components": [1, 3, 2],
            "foreground_ratio": [0.05, 0.12, 0.08],
            "has_boundary_touching": [False, True, False],
            "inter_annotator_dice": [0.85, 0.72, None],
        }
    )


# ---------------------------------------------------------------------------
# T1: Pandera schemas
# ---------------------------------------------------------------------------


class TestNiftiMetadataSchema:
    """Test NIfTI metadata Pandera schema validation."""

    def test_valid_data_passes(self, valid_nifti_metadata: pd.DataFrame) -> None:
        from minivess.validation.schemas import NiftiMetadataSchema

        validated = NiftiMetadataSchema.validate(valid_nifti_metadata)
        assert len(validated) == 2

    def test_invalid_spacing_fails(self, valid_nifti_metadata: pd.DataFrame) -> None:
        from minivess.validation.schemas import NiftiMetadataSchema

        bad = valid_nifti_metadata.copy()
        bad.loc[0, "voxel_spacing_x"] = -0.5  # Negative spacing
        with pytest.raises(Exception):  # noqa: B017
            NiftiMetadataSchema.validate(bad)

    def test_invalid_affine_fails(self, valid_nifti_metadata: pd.DataFrame) -> None:
        from minivess.validation.schemas import NiftiMetadataSchema

        bad = valid_nifti_metadata.copy()
        bad.loc[0, "has_valid_affine"] = False
        with pytest.raises(Exception):  # noqa: B017
            NiftiMetadataSchema.validate(bad)

    def test_intensity_range_invalid(
        self, valid_nifti_metadata: pd.DataFrame
    ) -> None:
        from minivess.validation.schemas import NiftiMetadataSchema

        bad = valid_nifti_metadata.copy()
        bad.loc[0, "intensity_max"] = -2.0  # max < min
        with pytest.raises(Exception):  # noqa: B017
            NiftiMetadataSchema.validate(bad)

    def test_shape_out_of_bounds(self, valid_nifti_metadata: pd.DataFrame) -> None:
        from minivess.validation.schemas import NiftiMetadataSchema

        bad = valid_nifti_metadata.copy()
        bad.loc[0, "shape_z"] = 9999  # > 512
        with pytest.raises(Exception):  # noqa: B017
            NiftiMetadataSchema.validate(bad)


class TestTrainingMetricsSchema:
    """Test training metrics Pandera schema validation."""

    def test_valid_data_passes(self, valid_training_metrics: pd.DataFrame) -> None:
        from minivess.validation.schemas import TrainingMetricsSchema

        validated = TrainingMetricsSchema.validate(valid_training_metrics)
        assert len(validated) == 3

    def test_dice_above_one_fails(self, valid_training_metrics: pd.DataFrame) -> None:
        from minivess.validation.schemas import TrainingMetricsSchema

        bad = valid_training_metrics.copy()
        bad.loc[0, "val_dice"] = 1.5  # > 1.0
        with pytest.raises(Exception):  # noqa: B017
            TrainingMetricsSchema.validate(bad)

    def test_negative_loss_fails(self, valid_training_metrics: pd.DataFrame) -> None:
        from minivess.validation.schemas import TrainingMetricsSchema

        bad = valid_training_metrics.copy()
        bad.loc[0, "train_loss"] = -0.1
        with pytest.raises(Exception):  # noqa: B017
            TrainingMetricsSchema.validate(bad)


class TestAnnotationQualitySchema:
    """Test annotation quality Pandera schema validation."""

    def test_valid_data_passes(
        self, valid_annotation_quality: pd.DataFrame
    ) -> None:
        from minivess.validation.schemas import AnnotationQualitySchema

        validated = AnnotationQualitySchema.validate(valid_annotation_quality)
        assert len(validated) == 3

    def test_foreground_ratio_above_one_fails(
        self, valid_annotation_quality: pd.DataFrame
    ) -> None:
        from minivess.validation.schemas import AnnotationQualitySchema

        bad = valid_annotation_quality.copy()
        bad.loc[0, "foreground_ratio"] = 1.5
        with pytest.raises(Exception):  # noqa: B017
            AnnotationQualitySchema.validate(bad)


# ---------------------------------------------------------------------------
# T2: Great Expectations suite structure
# ---------------------------------------------------------------------------


class TestGESuites:
    """Test Great Expectations suite configuration."""

    def test_training_metrics_suite_structure(self) -> None:
        from minivess.validation.expectations import build_training_metrics_suite

        suite = build_training_metrics_suite()
        assert "expectation_suite_name" in suite
        assert "expectations" in suite
        assert len(suite["expectations"]) >= 5

    def test_nifti_metadata_suite_structure(self) -> None:
        from minivess.validation.expectations import build_nifti_metadata_suite

        suite = build_nifti_metadata_suite()
        assert "expectation_suite_name" in suite
        assert "expectations" in suite
        assert len(suite["expectations"]) >= 5

    def test_training_metrics_suite_has_required_columns(self) -> None:
        from minivess.validation.expectations import build_training_metrics_suite

        suite = build_training_metrics_suite()
        columns_checked = {
            exp["kwargs"]["column"]
            for exp in suite["expectations"]
            if "column" in exp.get("kwargs", {})
        }
        assert "val_dice" in columns_checked
        assert "train_loss" in columns_checked
        assert "run_id" in columns_checked

    def test_nifti_metadata_suite_has_required_columns(self) -> None:
        from minivess.validation.expectations import build_nifti_metadata_suite

        suite = build_nifti_metadata_suite()
        columns_checked = {
            exp["kwargs"]["column"]
            for exp in suite["expectations"]
            if "column" in exp.get("kwargs", {})
        }
        assert "file_path" in columns_checked
        assert "voxel_spacing_x" in columns_checked


# ---------------------------------------------------------------------------
# T3: Prediction drift detection
# ---------------------------------------------------------------------------


class TestPredictionDrift:
    """Test prediction drift detection (KS + PSI)."""

    def test_ks_no_drift(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        rng = np.random.default_rng(42)
        ref = rng.standard_normal(200)
        cur = rng.standard_normal(200)
        report = detect_prediction_drift(ref, cur, method="ks")
        assert not report.is_drifted

    def test_ks_drift_detected(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        rng = np.random.default_rng(42)
        ref = rng.standard_normal(200)
        cur = rng.standard_normal(200) + 3.0  # Large shift
        report = detect_prediction_drift(ref, cur, method="ks")
        assert report.is_drifted

    def test_psi_no_drift(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        rng = np.random.default_rng(42)
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500)
        report = detect_prediction_drift(ref, cur, method="psi")
        assert not report.is_drifted

    def test_psi_drift_detected(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        rng = np.random.default_rng(42)
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500) + 5.0
        report = detect_prediction_drift(ref, cur, method="psi")
        assert report.is_drifted

    def test_invalid_method_raises(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown drift method"):
            detect_prediction_drift(ref, cur, method="invalid")

    def test_drift_report_fields(self) -> None:
        from minivess.validation.drift import detect_prediction_drift

        rng = np.random.default_rng(42)
        ref = rng.standard_normal(100)
        cur = rng.standard_normal(100)
        report = detect_prediction_drift(ref, cur, method="ks")
        assert isinstance(report.is_drifted, bool)
        assert isinstance(report.dataset_drift_score, float)
        assert "prediction" in report.feature_scores
        assert "p_value" in report.details


# ---------------------------------------------------------------------------
# T4: Deepchecks config builders
# ---------------------------------------------------------------------------


class TestDeepchecksConfig:
    """Test Deepchecks suite configuration builders."""

    def test_data_integrity_suite_config(self) -> None:
        from minivess.validation.deepchecks_vision import build_data_integrity_suite

        config = build_data_integrity_suite()
        assert "suite_name" in config
        assert "checks" in config
        assert len(config["checks"]) >= 3

    def test_train_test_suite_config(self) -> None:
        from minivess.validation.deepchecks_vision import build_train_test_suite

        config = build_train_test_suite()
        assert "suite_name" in config
        assert "checks" in config
        check_names = [c["name"] for c in config["checks"]]
        assert "feature_drift" in check_names
        assert "label_drift" in check_names

    def test_evaluate_report_all_pass(self) -> None:
        from minivess.validation.deepchecks_vision import evaluate_report

        results = {
            "checks": [
                {"name": "check_a", "passed": True},
                {"name": "check_b", "passed": True},
            ]
        }
        report = evaluate_report(results)
        assert report.passed is True
        assert report.num_failed == 0

    def test_evaluate_report_with_failures(self) -> None:
        from minivess.validation.deepchecks_vision import evaluate_report

        results = {
            "checks": [
                {"name": "check_a", "passed": True},
                {"name": "check_b", "passed": False},
            ]
        }
        report = evaluate_report(results)
        assert report.passed is False
        assert report.num_failed == 1


# ---------------------------------------------------------------------------
# T5: whylogs profiling
# ---------------------------------------------------------------------------


class TestWhylogsProfiler:
    """Test whylogs profiling integration."""

    def test_profile_dataframe(self, valid_nifti_metadata: pd.DataFrame) -> None:
        from minivess.validation.profiling import profile_dataframe

        view = profile_dataframe(valid_nifti_metadata)
        assert view is not None
        columns = view.get_columns()
        assert "shape_x" in columns
        assert "voxel_spacing_x" in columns

    def test_profile_returns_column_stats(
        self, valid_nifti_metadata: pd.DataFrame
    ) -> None:
        from minivess.validation.profiling import profile_dataframe

        view = profile_dataframe(valid_nifti_metadata)
        columns = view.get_columns()
        # Should have stats for numeric columns
        assert len(columns) > 0

    def test_compare_profiles_no_drift(self) -> None:
        from minivess.validation.profiling import compare_profiles, profile_dataframe

        rng = np.random.default_rng(42)
        df1 = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
        df2 = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
        ref = profile_dataframe(df1)
        cur = profile_dataframe(df2)
        report = compare_profiles(ref, cur)
        assert report is not None
        assert isinstance(report.drifted_columns, list)

    def test_compare_profiles_with_drift(self) -> None:
        from minivess.validation.profiling import compare_profiles, profile_dataframe

        rng = np.random.default_rng(42)
        df1 = pd.DataFrame({"x": rng.standard_normal(200), "y": rng.standard_normal(200)})
        df2 = pd.DataFrame({"x": rng.standard_normal(200) + 10.0, "y": rng.standard_normal(200)})
        ref = profile_dataframe(df1)
        cur = profile_dataframe(df2)
        report = compare_profiles(ref, cur)
        assert len(report.drifted_columns) > 0


# ---------------------------------------------------------------------------
# T6: Validation gates
# ---------------------------------------------------------------------------


class TestValidationGates:
    """Test fail-fast validation gate functions."""

    def test_gate_passes_valid_metadata(
        self, valid_nifti_metadata: pd.DataFrame
    ) -> None:
        from minivess.validation.gates import validate_nifti_metadata

        result = validate_nifti_metadata(valid_nifti_metadata)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_gate_fails_invalid_metadata(
        self, valid_nifti_metadata: pd.DataFrame
    ) -> None:
        from minivess.validation.gates import validate_nifti_metadata

        bad = valid_nifti_metadata.copy()
        bad.loc[0, "voxel_spacing_x"] = -1.0
        result = validate_nifti_metadata(bad)
        assert result.passed is False
        assert len(result.errors) > 0

    def test_gate_passes_valid_metrics(
        self, valid_training_metrics: pd.DataFrame
    ) -> None:
        from minivess.validation.gates import validate_training_metrics

        result = validate_training_metrics(valid_training_metrics)
        assert result.passed is True

    def test_gate_fails_invalid_metrics(
        self, valid_training_metrics: pd.DataFrame
    ) -> None:
        from minivess.validation.gates import validate_training_metrics

        bad = valid_training_metrics.copy()
        bad.loc[0, "val_dice"] = 5.0  # Way above 1.0
        result = validate_training_metrics(bad)
        assert result.passed is False

    def test_gate_result_has_fields(
        self, valid_nifti_metadata: pd.DataFrame
    ) -> None:
        from minivess.validation.gates import validate_nifti_metadata

        result = validate_nifti_metadata(valid_nifti_metadata)
        assert hasattr(result, "passed")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
