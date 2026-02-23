from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of Deepchecks validation results."""

    passed: bool
    num_checks: int
    num_passed: int
    num_failed: int
    failures: list[dict[str, Any]] = field(default_factory=list)


def build_data_integrity_suite() -> dict[str, Any]:
    """Build a Deepchecks data integrity suite configuration.

    Returns a suite configuration dict that can be used to run
    data validation checks on image datasets.

    Checks include:
    - Image property outliers (brightness, contrast, aspect ratio)
    - Label distribution (class imbalance detection)
    - Duplicate image detection
    - Corrupted image detection
    """
    return {
        "suite_name": "minivess_data_integrity",
        "checks": [
            {
                "name": "image_property_outliers",
                "params": {"properties": ["brightness", "contrast"]},
            },
            {
                "name": "label_distribution",
                "params": {"max_imbalance_ratio": 10.0},
            },
            {
                "name": "duplicate_detection",
                "params": {"hash_size": 16},
            },
            {
                "name": "corrupted_images",
                "params": {},
            },
        ],
    }


def build_train_test_suite() -> dict[str, Any]:
    """Build a Deepchecks train-test validation suite configuration.

    Checks include:
    - Feature drift between train and test sets
    - Label drift
    - Image property drift
    - New labels in test set
    """
    return {
        "suite_name": "minivess_train_test_validation",
        "checks": [
            {
                "name": "feature_drift",
                "params": {"max_drift_score": 0.3},
            },
            {
                "name": "label_drift",
                "params": {"max_drift_score": 0.3},
            },
            {
                "name": "image_property_drift",
                "params": {"properties": ["brightness", "contrast", "area"]},
            },
            {
                "name": "new_labels",
                "params": {},
            },
        ],
    }


def evaluate_report(results: dict[str, Any]) -> ValidationReport:
    """Evaluate validation results into a summary report."""
    checks = results.get("checks", [])
    failures = [c for c in checks if not c.get("passed", True)]
    return ValidationReport(
        passed=len(failures) == 0,
        num_checks=len(checks),
        num_passed=len(checks) - len(failures),
        num_failed=len(failures),
        failures=failures,
    )
