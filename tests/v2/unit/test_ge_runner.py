"""Tests for Great Expectations validation gate runner (Issue #45)."""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# T1: run_expectation_suite — core execution
# ---------------------------------------------------------------------------


class TestRunExpectationSuite:
    """Test GE suite execution against DataFrames."""

    def test_valid_data_passes(self) -> None:
        """Valid DataFrame should pass all expectations."""
        from minivess.validation.ge_runner import run_expectation_suite

        suite_config = {
            "expectation_suite_name": "test_suite",
            "expectations": [
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "name"},
                },
            ],
        }
        df = pd.DataFrame({"name": ["alice", "bob"]})
        result = run_expectation_suite(df, suite_config)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_invalid_data_fails(self) -> None:
        """DataFrame violating expectations should fail."""
        from minivess.validation.ge_runner import run_expectation_suite

        suite_config = {
            "expectation_suite_name": "test_suite",
            "expectations": [
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "name"},
                },
            ],
        }
        df = pd.DataFrame({"name": ["alice", None]})
        result = run_expectation_suite(df, suite_config)
        assert result.passed is False
        assert len(result.errors) >= 1

    def test_multiple_expectations_all_checked(self) -> None:
        """All expectations should be evaluated, not just the first."""
        from minivess.validation.ge_runner import run_expectation_suite

        suite_config = {
            "expectation_suite_name": "test_suite",
            "expectations": [
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "a"},
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "b", "min_value": 0, "max_value": 10},
                },
            ],
        }
        df = pd.DataFrame({"a": ["x", "y"], "b": [5, 7]})
        result = run_expectation_suite(df, suite_config)
        assert result.passed is True
        assert result.statistics["evaluated_expectations"] == 2

    def test_returns_gate_result(self) -> None:
        """Result should be a GateResult instance."""
        from minivess.validation.gates import GateResult
        from minivess.validation.ge_runner import run_expectation_suite

        suite_config = {
            "expectation_suite_name": "minimal",
            "expectations": [
                {
                    "expectation_type": "expect_table_row_count_to_be_between",
                    "kwargs": {"min_value": 1},
                },
            ],
        }
        df = pd.DataFrame({"x": [1]})
        result = run_expectation_suite(df, suite_config)
        assert isinstance(result, GateResult)

    def test_statistics_populated(self) -> None:
        """Result should include statistics about the validation run."""
        from minivess.validation.ge_runner import run_expectation_suite

        suite_config = {
            "expectation_suite_name": "test",
            "expectations": [
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "x"},
                },
            ],
        }
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = run_expectation_suite(df, suite_config)
        assert "evaluated_expectations" in result.statistics
        assert "successful_expectations" in result.statistics


# ---------------------------------------------------------------------------
# T2: NIfTI batch validation gate
# ---------------------------------------------------------------------------


class TestValidateNiftiBatch:
    """Test GE validation of NIfTI metadata batches."""

    def test_valid_nifti_batch_passes(self) -> None:
        """Valid NIfTI metadata should pass the nifti_metadata_suite."""
        from minivess.validation.ge_runner import validate_nifti_batch

        df = pd.DataFrame(
            {
                "file_path": ["/data/vol1.nii.gz", "/data/vol2.nii.gz"],
                "voxel_spacing_x": [1.0, 0.5],
                "voxel_spacing_y": [1.0, 0.5],
                "voxel_spacing_z": [1.0, 2.0],
                "has_valid_affine": [True, True],
            }
        )
        result = validate_nifti_batch(df)
        assert result.passed is True

    def test_invalid_voxel_spacing_fails(self) -> None:
        """NIfTI metadata with out-of-range voxel spacing should fail."""
        from minivess.validation.ge_runner import validate_nifti_batch

        df = pd.DataFrame(
            {
                "file_path": ["/data/vol1.nii.gz"],
                "voxel_spacing_x": [999.0],  # Way out of range
                "voxel_spacing_y": [1.0],
                "voxel_spacing_z": [1.0],
                "has_valid_affine": [True],
            }
        )
        result = validate_nifti_batch(df)
        assert result.passed is False

    def test_duplicate_file_paths_fail(self) -> None:
        """Duplicate file paths should fail uniqueness expectation."""
        from minivess.validation.ge_runner import validate_nifti_batch

        df = pd.DataFrame(
            {
                "file_path": ["/data/same.nii.gz", "/data/same.nii.gz"],
                "voxel_spacing_x": [1.0, 1.0],
                "voxel_spacing_y": [1.0, 1.0],
                "voxel_spacing_z": [1.0, 1.0],
                "has_valid_affine": [True, True],
            }
        )
        result = validate_nifti_batch(df)
        assert result.passed is False

    def test_empty_batch_fails(self) -> None:
        """Empty DataFrame should fail row count expectation."""
        from minivess.validation.ge_runner import validate_nifti_batch

        df = pd.DataFrame(
            {
                "file_path": pd.Series([], dtype=str),
                "voxel_spacing_x": pd.Series([], dtype=float),
                "voxel_spacing_y": pd.Series([], dtype=float),
                "voxel_spacing_z": pd.Series([], dtype=float),
                "has_valid_affine": pd.Series([], dtype=bool),
            }
        )
        result = validate_nifti_batch(df)
        assert result.passed is False


# ---------------------------------------------------------------------------
# T3: Training metrics batch validation gate
# ---------------------------------------------------------------------------


class TestValidateMetricsBatch:
    """Test GE validation of training metrics batches."""

    def test_valid_metrics_pass(self) -> None:
        """Valid training metrics should pass."""
        from minivess.validation.ge_runner import validate_metrics_batch

        df = pd.DataFrame(
            {
                "run_id": ["run_001", "run_001"],
                "epoch": [1, 2],
                "fold": [0, 0],
                "val_dice": [0.5, 0.7],
                "val_cldice": [0.4, 0.6],
                "train_loss": [1.0, 0.5],
                "val_loss": [1.2, 0.8],
                "learning_rate": [1e-4, 1e-4],
            }
        )
        result = validate_metrics_batch(df)
        assert result.passed is True

    def test_dice_out_of_range_fails(self) -> None:
        """Dice values > 1.0 should fail."""
        from minivess.validation.ge_runner import validate_metrics_batch

        df = pd.DataFrame(
            {
                "run_id": ["run_001"],
                "epoch": [1],
                "fold": [0],
                "val_dice": [1.5],  # Out of range
                "val_cldice": [0.5],
                "train_loss": [1.0],
                "val_loss": [1.0],
                "learning_rate": [1e-4],
            }
        )
        result = validate_metrics_batch(df)
        assert result.passed is False

    def test_null_run_id_fails(self) -> None:
        """Null run_id should fail not-null expectation."""
        from minivess.validation.ge_runner import validate_metrics_batch

        df = pd.DataFrame(
            {
                "run_id": [None],
                "epoch": [1],
                "fold": [0],
                "val_dice": [0.5],
                "val_cldice": [0.5],
                "train_loss": [1.0],
                "val_loss": [1.0],
                "learning_rate": [1e-4],
            }
        )
        result = validate_metrics_batch(df)
        assert result.passed is False
