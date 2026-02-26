"""Tests for Validation Gate Refactoring — R5.4 (Issue #57).

R5.4: Extract _validate_with_schema() generic wrapper that replaces the
repeated pandera try/except/format-error pattern in gates.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from minivess.validation.gates import GateResult

# =========================================================================
# R5.4: _validate_with_schema() generic wrapper
# =========================================================================


class TestValidateWithSchema:
    """Test the generic _validate_with_schema() wrapper."""

    def test_valid_nifti_passes(self) -> None:
        """_validate_with_schema should pass for valid NIfTI data."""
        from minivess.validation.gates import _validate_with_schema
        from minivess.validation.schemas import NiftiMetadataSchema

        df = pd.DataFrame(
            {
                "file_path": ["/data/sub01.nii.gz"],
                "shape_x": [256],
                "shape_y": [256],
                "shape_z": [128],
                "voxel_spacing_x": [0.5],
                "voxel_spacing_y": [0.5],
                "voxel_spacing_z": [1.0],
                "intensity_min": [-1.0],
                "intensity_max": [1.0],
                "num_foreground_voxels": [5000],
                "has_valid_affine": [True],
            }
        )
        result = _validate_with_schema(df, NiftiMetadataSchema, "nifti_metadata")
        assert isinstance(result, GateResult)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_invalid_data_fails(self) -> None:
        """_validate_with_schema should fail for invalid data."""
        from minivess.validation.gates import _validate_with_schema
        from minivess.validation.schemas import NiftiMetadataSchema

        df = pd.DataFrame(
            {
                "file_path": ["/data/sub01.nii.gz"],
                "shape_x": [256],
                "shape_y": [256],
                "shape_z": [9999],  # Out of bounds (> 512)
                "voxel_spacing_x": [0.5],
                "voxel_spacing_y": [0.5],
                "voxel_spacing_z": [1.0],
                "intensity_min": [-1.0],
                "intensity_max": [1.0],
                "num_foreground_voxels": [5000],
                "has_valid_affine": [True],
            }
        )
        result = _validate_with_schema(df, NiftiMetadataSchema, "nifti_metadata")
        assert result.passed is False
        assert len(result.errors) > 0

    def test_valid_metrics_passes(self) -> None:
        """_validate_with_schema should pass for valid training metrics."""
        from minivess.validation.gates import _validate_with_schema
        from minivess.validation.schemas import TrainingMetricsSchema

        df = pd.DataFrame(
            {
                "run_id": ["run_001"],
                "epoch": [0],
                "fold": [0],
                "train_loss": [0.8],
                "val_loss": [0.7],
                "val_dice": [0.3],
                "val_cldice": [0.2],
                "val_nsd": [0.4],
                "learning_rate": [1e-3],
            }
        )
        result = _validate_with_schema(df, TrainingMetricsSchema, "training_metrics")
        assert result.passed is True

    def test_works_with_annotation_schema(self) -> None:
        """_validate_with_schema should work with any Pandera schema."""
        from minivess.validation.gates import _validate_with_schema
        from minivess.validation.schemas import AnnotationQualitySchema

        df = pd.DataFrame(
            {
                "sample_id": ["s01"],
                "annotator_id": ["ann_A"],
                "num_connected_components": [1],
                "foreground_ratio": [0.05],
                "has_boundary_touching": [False],
                "inter_annotator_dice": [0.85],
            }
        )
        result = _validate_with_schema(
            df, AnnotationQualitySchema, "annotation_quality"
        )
        assert result.passed is True


class TestExistingGatesStillWork:
    """Verify existing gate functions still work after refactoring."""

    @pytest.fixture
    def valid_nifti_df(self) -> pd.DataFrame:
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

    @pytest.fixture
    def valid_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "run_id": ["run_001", "run_001"],
                "epoch": [0, 1],
                "fold": [0, 0],
                "train_loss": [0.8, 0.5],
                "val_loss": [0.7, 0.4],
                "val_dice": [0.3, 0.6],
                "val_cldice": [0.2, 0.5],
                "val_nsd": [0.4, 0.7],
                "learning_rate": [1e-3, 1e-3],
            }
        )

    def test_validate_nifti_metadata_passes(self, valid_nifti_df: pd.DataFrame) -> None:
        """validate_nifti_metadata should still work after refactor."""
        from minivess.validation.gates import validate_nifti_metadata

        result = validate_nifti_metadata(valid_nifti_df)
        assert result.passed is True

    def test_validate_nifti_metadata_fails(self, valid_nifti_df: pd.DataFrame) -> None:
        """validate_nifti_metadata should still detect errors."""
        from minivess.validation.gates import validate_nifti_metadata

        bad = valid_nifti_df.copy()
        bad.loc[0, "voxel_spacing_x"] = -1.0
        result = validate_nifti_metadata(bad)
        assert result.passed is False

    def test_validate_training_metrics_passes(
        self, valid_metrics_df: pd.DataFrame
    ) -> None:
        """validate_training_metrics should still work after refactor."""
        from minivess.validation.gates import validate_training_metrics

        result = validate_training_metrics(valid_metrics_df)
        assert result.passed is True

    def test_validate_training_metrics_fails(
        self, valid_metrics_df: pd.DataFrame
    ) -> None:
        """validate_training_metrics should still detect errors."""
        from minivess.validation.gates import validate_training_metrics

        bad = valid_metrics_df.copy()
        bad.loc[0, "val_dice"] = 5.0
        result = validate_training_metrics(bad)
        assert result.passed is False
