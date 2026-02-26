from __future__ import annotations

from typing import Any


def build_training_metrics_suite() -> dict[str, Any]:
    """Build a Great Expectations expectation suite for training metrics.

    Returns a JSON-serializable dict representing a GE expectation suite.
    This is used for batch data quality validation of MLflow run summaries.

    GE validates the TABULAR metadata — not the 3D imaging data itself.
    For 3D data validation, use custom validators or Deepchecks Vision.
    """
    return {
        "expectation_suite_name": "training_metrics_suite",
        "expectations": [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "run_id"},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "val_dice", "min_value": 0.0, "max_value": 1.0},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "val_cldice", "min_value": 0.0, "max_value": 1.0},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "train_loss",
                    "min_value": 0.0,
                    "max_value": 100.0,
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "val_loss", "min_value": 0.0, "max_value": 100.0},
            },
            {
                "expectation_type": "expect_column_values_to_be_increasing",
                "kwargs": {"column": "epoch"},
                "meta": {
                    "notes": "Epochs should be monotonically increasing within a fold"
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "learning_rate",
                    "min_value": 1e-10,
                    "max_value": 1.0,
                },
            },
            {
                "expectation_type": "expect_column_pair_values_a_to_be_greater_than_b",
                "kwargs": {"column_A": "epoch", "column_B": "fold", "or_equal": True},
                "meta": {"notes": "Sanity: epoch >= 0 and fold >= 0"},
            },
        ],
        "meta": {
            "purpose": "Validate training metrics DataFrames from MLflow runs",
            "scope": "Tabular metadata only — NOT 3D imaging data",
            "project": "minivess-mlops-v2",
        },
    }


def build_nifti_metadata_suite() -> dict[str, Any]:
    """Build GE suite for NIfTI file metadata validation.

    Complements Pandera schema validation with batch-level expectations.
    """
    return {
        "expectation_suite_name": "nifti_metadata_suite",
        "expectations": [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "file_path"},
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "file_path"},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "voxel_spacing_x",
                    "min_value": 0.01,
                    "max_value": 10.0,
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "voxel_spacing_y",
                    "min_value": 0.01,
                    "max_value": 10.0,
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "voxel_spacing_z",
                    "min_value": 0.01,
                    "max_value": 50.0,
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_true",
                "kwargs": {"column": "has_valid_affine"},
            },
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1},
                "meta": {"notes": "At least one sample must be present"},
            },
        ],
        "meta": {
            "purpose": "Batch quality checks for NIfTI metadata at DVC pipeline boundaries",
            "scope": "Tabular metadata extracted from NIfTI headers",
            "project": "minivess-mlops-v2",
        },
    }
