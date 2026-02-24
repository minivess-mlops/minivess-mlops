"""Data validation — Pandera schemas, Great Expectations suites, and custom validators."""

from __future__ import annotations

from minivess.validation.deepchecks_vision import (
    ValidationReport,
    build_data_integrity_suite,
    build_train_test_suite,
    evaluate_report,
)
from minivess.validation.drift import (
    DriftReport,
    detect_prediction_drift,
)
from minivess.validation.expectations import (
    build_nifti_metadata_suite,
    build_training_metrics_suite,
)
from minivess.validation.gates import (
    GateResult,
    validate_nifti_metadata,
    validate_training_metrics,
)
from minivess.validation.profiling import (
    DatasetProfileView,
    ProfileDriftReport,
    compare_profiles,
    profile_dataframe,
)
from minivess.validation.schemas import (
    AnnotationQualitySchema,
    NiftiMetadataSchema,
    TrainingMetricsSchema,
)

__all__ = [
    "AnnotationQualitySchema",
    "DatasetProfileView",
    "DriftReport",
    "GateResult",
    "NiftiMetadataSchema",
    "ProfileDriftReport",
    "TrainingMetricsSchema",
    "ValidationReport",
    "build_data_integrity_suite",
    "build_nifti_metadata_suite",
    "build_train_test_suite",
    "build_training_metrics_suite",
    "compare_profiles",
    "detect_prediction_drift",
    "evaluate_report",
    "profile_dataframe",
    "validate_nifti_metadata",
    "validate_training_metrics",
]
