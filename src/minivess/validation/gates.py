"""Validation gates: fail-fast data quality checks.

Combines Pandera schema validation with GE batch expectations
into simple pass/fail gates for pipeline boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a validation gate check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, int] = field(default_factory=dict)


def validate_nifti_metadata(df: pd.DataFrame) -> GateResult:
    """Validate NIfTI metadata DataFrame.

    Runs Pandera NiftiMetadataSchema validation and returns a
    structured gate result.

    Parameters
    ----------
    df:
        DataFrame with NIfTI metadata columns.

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    import pandera as pa

    from minivess.validation.schemas import NiftiMetadataSchema

    errors: list[str] = []
    try:
        NiftiMetadataSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for failure in exc.failure_cases.itertuples():
            errors.append(
                f"Column '{failure.column}': {failure.check} "
                f"(value={failure.failure_case})"
            )

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
    )


def validate_training_metrics(df: pd.DataFrame) -> GateResult:
    """Validate training metrics DataFrame.

    Runs Pandera TrainingMetricsSchema validation and returns a
    structured gate result.

    Parameters
    ----------
    df:
        DataFrame with training metrics columns.

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    import pandera as pa

    from minivess.validation.schemas import TrainingMetricsSchema

    errors: list[str] = []
    try:
        TrainingMetricsSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for failure in exc.failure_cases.itertuples():
            errors.append(
                f"Column '{failure.column}': {failure.check} "
                f"(value={failure.failure_case})"
            )

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
    )
