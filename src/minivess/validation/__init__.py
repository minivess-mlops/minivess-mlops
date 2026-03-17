"""Data validation — Pandera schemas, Great Expectations suites, and custom validators."""

from __future__ import annotations

# Core validation modules (always available)
from minivess.validation.enforcement import (
    DataQualityError,
    GateAction,
    enforce_gate,
    get_gate_severity,
)
from minivess.validation.gates import (
    GateResult,
    validate_nifti_metadata,
    validate_training_metrics,
)

# Optional dependency modules — gracefully degrade when deps not installed.
# This is an MLOps platform; not all users install all optional packages.
try:  # noqa: SIM105
    from minivess.validation.data_care import (
        DataQualityReport,
        DimensionScore,
        QualityDimension,
        assess_metrics_quality,
        assess_nifti_quality,
        quality_gate,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.deepchecks_vision import (
        ValidationReport,
        build_data_integrity_suite,
        build_train_test_suite,
        evaluate_report,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.drift import (
        DriftReport,
        detect_prediction_drift,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.expectations import (
        build_nifti_metadata_suite,
        build_training_metrics_suite,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.ge_runner import (
        run_expectation_suite,
        validate_metrics_batch,
        validate_nifti_batch,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.profiling import (
        DatasetProfileView,
        ProfileDriftReport,
        compare_profiles,
        profile_dataframe,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.schemas import (
        AnnotationQualitySchema,
        NiftiMetadataSchema,
        TrainingMetricsSchema,
    )
except ImportError:
    pass

try:  # noqa: SIM105
    from minivess.validation.vessqc import (
        CurationFlag,
        CurationReport,
        compute_error_detection_metrics,
        flag_uncertain_regions,
        rank_samples_by_uncertainty,
    )
except ImportError:
    pass

__all__ = [
    "AnnotationQualitySchema",
    "CurationFlag",
    "DataQualityError",
    "DataQualityReport",
    "DimensionScore",
    "CurationReport",
    "DatasetProfileView",
    "DriftReport",
    "GateAction",
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
    "enforce_gate",
    "get_gate_severity",
    "validate_metrics_batch",
    "validate_nifti_batch",
    "validate_nifti_metadata",
    "validate_training_metrics",
]
