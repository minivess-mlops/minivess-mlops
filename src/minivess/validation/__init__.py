"""Data validation — Pandera schemas, Great Expectations suites, and custom validators."""

from __future__ import annotations

from minivess.validation.data_care import (
    DataQualityReport,
    DimensionScore,
    QualityDimension,
    assess_metrics_quality,
    assess_nifti_quality,
    quality_gate,
)
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
from minivess.validation.ge_runner import (
    run_expectation_suite,
    validate_metrics_batch,
    validate_nifti_batch,
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
from minivess.validation.vessqc import (
    CurationFlag,
    CurationReport,
    compute_error_detection_metrics,
    flag_uncertain_regions,
    rank_samples_by_uncertainty,
)

__all__ = [
    "AnnotationQualitySchema",
    "CurationFlag",
    "DataQualityReport",
    "DimensionScore",
    "CurationReport",
    "DatasetProfileView",
    "DriftReport",
    "GateResult",
    "NiftiMetadataSchema",
    "QualityDimension",
    "ProfileDriftReport",
    "TrainingMetricsSchema",
    "ValidationReport",
    "assess_metrics_quality",
    "assess_nifti_quality",
    "build_data_integrity_suite",
    "build_nifti_metadata_suite",
    "build_train_test_suite",
    "build_training_metrics_suite",
    "compare_profiles",
    "compute_error_detection_metrics",
    "detect_prediction_drift",
    "flag_uncertain_regions",
    "evaluate_report",
    "profile_dataframe",
    "quality_gate",
    "rank_samples_by_uncertainty",
    "run_expectation_suite",
    "validate_metrics_batch",
    "validate_nifti_batch",
    "validate_nifti_metadata",
    "validate_training_metrics",
]
