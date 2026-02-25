"""DATA-CARE comprehensive data quality assessment for medical AI.

Maps DATA-CARE quality dimensions (van Twist et al., 2026) to the
MinIVess data pipeline. Scores datasets across completeness, correctness,
consistency, uniqueness, timeliness, and representativeness.

Reference: van Twist et al. (2026). "DATA-CARE: Comprehensive Data Quality
Assessment Tool for Medical AI."

R5.18 assessment (333 lines): Six private scoring functions + two public
assessment entrypoints + one gate converter. All operate on the same
DataQualityReport/DimensionScore model — logically cohesive. No split needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from minivess.validation.gates import GateResult


class QualityDimension(StrEnum):
    """DATA-CARE quality dimensions for medical imaging datasets."""

    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    REPRESENTATIVENESS = "representativeness"


@dataclass
class DimensionScore:
    """Quality score for a single DATA-CARE dimension.

    Parameters
    ----------
    dimension:
        Quality dimension being assessed.
    score:
        Score between 0.0 (worst) and max_score (best).
    max_score:
        Maximum achievable score (default: 1.0).
    issues:
        List of specific quality issues found.
    """

    dimension: QualityDimension
    score: float
    max_score: float = 1.0
    issues: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Aggregate quality report across all DATA-CARE dimensions.

    Parameters
    ----------
    dimension_scores:
        Per-dimension scores.
    overall_score:
        Weighted mean of dimension scores (0-1).
    passed:
        Whether overall score meets gate_threshold.
    gate_threshold:
        Minimum acceptable overall score.
    """

    dimension_scores: list[DimensionScore]
    overall_score: float
    passed: bool
    gate_threshold: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for MLflow logging."""
        d: dict[str, float] = {"overall_score": self.overall_score}
        for ds in self.dimension_scores:
            d[ds.dimension.value] = ds.score
        return d


def _score_completeness(df: pd.DataFrame, columns: list[str]) -> DimensionScore:
    """Score completeness: fraction of non-null values."""
    issues: list[str] = []
    total_cells = len(df) * len(columns)
    if total_cells == 0:
        return DimensionScore(QualityDimension.COMPLETENESS, 1.0, 1.0, [])

    null_count = 0
    for col in columns:
        if col in df.columns:
            col_nulls = int(df[col].isna().sum())
            if col_nulls > 0:
                pct = col_nulls / len(df) * 100
                issues.append(f"{pct:.0f}% null values in {col}")
                null_count += col_nulls

    score = 1.0 - null_count / total_cells
    return DimensionScore(QualityDimension.COMPLETENESS, score, 1.0, issues)


def _score_uniqueness(df: pd.DataFrame, key_column: str) -> DimensionScore:
    """Score uniqueness: fraction of unique values in key column."""
    issues: list[str] = []
    if key_column not in df.columns or len(df) == 0:
        return DimensionScore(QualityDimension.UNIQUENESS, 1.0, 1.0, [])

    n_total = len(df)
    n_unique = df[key_column].nunique()
    n_dups = n_total - n_unique

    if n_dups > 0:
        issues.append(f"{n_dups} duplicate entries in {key_column}")

    score = n_unique / max(n_total, 1)
    return DimensionScore(QualityDimension.UNIQUENESS, score, 1.0, issues)


def _score_range_correctness(
    df: pd.DataFrame,
    range_checks: dict[str, tuple[float, float]],
) -> DimensionScore:
    """Score correctness: fraction of values within valid ranges."""
    issues: list[str] = []
    total = 0
    violations = 0

    for col, (low, high) in range_checks.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        n = len(values)
        if n == 0:
            continue
        total += n
        out_of_range = int(((values < low) | (values > high)).sum())
        if out_of_range > 0:
            violations += out_of_range
            issues.append(
                f"{out_of_range}/{n} values in {col} outside [{low}, {high}]"
            )

    score = 1.0 - violations / max(total, 1) if total > 0 else 1.0
    return DimensionScore(QualityDimension.CORRECTNESS, score, 1.0, issues)


def _score_consistency(
    df: pd.DataFrame,
    bool_columns: list[str],
) -> DimensionScore:
    """Score consistency: fraction of True values in boolean validity columns."""
    issues: list[str] = []
    total = 0
    invalid = 0

    for col in bool_columns:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        n = len(values)
        if n == 0:
            continue
        total += n
        n_false = int((~values.astype(bool)).sum())
        if n_false > 0:
            invalid += n_false
            issues.append(f"{n_false}/{n} records failed {col} check")

    score = 1.0 - invalid / max(total, 1) if total > 0 else 1.0
    return DimensionScore(QualityDimension.CONSISTENCY, score, 1.0, issues)


def assess_nifti_quality(
    df: pd.DataFrame,
    *,
    gate_threshold: float = 0.7,
) -> DataQualityReport:
    """Assess NIfTI metadata quality across DATA-CARE dimensions.

    Parameters
    ----------
    df:
        NIfTI metadata DataFrame (file_path, shape_*, voxel_spacing_*, etc.).
    gate_threshold:
        Minimum overall score to pass.

    Returns
    -------
    DataQualityReport with per-dimension scores.
    """
    metadata_cols = [
        "file_path", "shape_x", "shape_y", "shape_z",
        "voxel_spacing_x", "voxel_spacing_y", "voxel_spacing_z",
        "intensity_min", "intensity_max", "has_valid_affine",
    ]

    scores = [
        _score_completeness(df, metadata_cols),
        _score_range_correctness(df, {
            "voxel_spacing_x": (0.01, 10.0),
            "voxel_spacing_y": (0.01, 10.0),
            "voxel_spacing_z": (0.01, 50.0),
            "intensity_min": (-10000.0, 10000.0),
            "intensity_max": (-10000.0, 10000.0),
        }),
        _score_consistency(df, ["has_valid_affine"]),
        _score_uniqueness(df, "file_path"),
        # Timeliness and representativeness default to 1.0 (require external info)
        DimensionScore(QualityDimension.TIMELINESS, 1.0, 1.0, []),
        DimensionScore(QualityDimension.REPRESENTATIVENESS, 1.0, 1.0, []),
    ]

    overall = float(np.mean([s.score for s in scores]))
    return DataQualityReport(
        dimension_scores=scores,
        overall_score=overall,
        passed=overall >= gate_threshold,
        gate_threshold=gate_threshold,
    )


def assess_metrics_quality(
    df: pd.DataFrame,
    *,
    gate_threshold: float = 0.7,
) -> DataQualityReport:
    """Assess training metrics quality across DATA-CARE dimensions.

    Parameters
    ----------
    df:
        Training metrics DataFrame (run_id, epoch, train_loss, etc.).
    gate_threshold:
        Minimum overall score to pass.

    Returns
    -------
    DataQualityReport with per-dimension scores.
    """
    metric_cols = [
        "run_id", "epoch", "train_loss", "val_loss",
        "val_dice", "learning_rate",
    ]

    scores = [
        _score_completeness(df, metric_cols),
        _score_range_correctness(df, {
            "val_dice": (0.0, 1.0),
            "train_loss": (0.0, float("inf")),
            "val_loss": (0.0, float("inf")),
            "learning_rate": (0.0, 1.0),
        }),
        # Consistency: epochs should be monotonically increasing per run
        _score_epoch_consistency(df),
        # Uniqueness: not applicable for metrics (multiple epochs per run)
        DimensionScore(QualityDimension.UNIQUENESS, 1.0, 1.0, []),
        DimensionScore(QualityDimension.TIMELINESS, 1.0, 1.0, []),
        DimensionScore(QualityDimension.REPRESENTATIVENESS, 1.0, 1.0, []),
    ]

    overall = float(np.mean([s.score for s in scores]))
    return DataQualityReport(
        dimension_scores=scores,
        overall_score=overall,
        passed=overall >= gate_threshold,
        gate_threshold=gate_threshold,
    )


def _score_epoch_consistency(df: pd.DataFrame) -> DimensionScore:
    """Check that epochs are monotonically increasing per run."""
    issues: list[str] = []
    if "epoch" not in df.columns or "run_id" not in df.columns:
        return DimensionScore(QualityDimension.CONSISTENCY, 1.0, 1.0, [])

    violations = 0
    total_runs = 0

    for run_id, group in df.groupby("run_id"):
        total_runs += 1
        epochs = group["epoch"].tolist()
        if epochs != sorted(epochs):
            violations += 1
            issues.append(f"Non-monotonic epochs in run {run_id}")

    score = 1.0 - violations / max(total_runs, 1) if total_runs > 0 else 1.0
    return DimensionScore(QualityDimension.CONSISTENCY, score, 1.0, issues)


def quality_gate(report: DataQualityReport) -> GateResult:
    """Convert a DataQualityReport to a pipeline GateResult.

    Parameters
    ----------
    report:
        DATA-CARE quality assessment report.

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    from minivess.validation.gates import GateResult

    errors: list[str] = []
    warnings: list[str] = []

    for ds in report.dimension_scores:
        if ds.score < 0.5:
            errors.extend(ds.issues)
        elif ds.score < 0.8:
            warnings.extend(ds.issues)

    if not report.passed:
        errors.insert(
            0,
            f"Overall quality score {report.overall_score:.2f} "
            f"below threshold {report.gate_threshold:.2f}",
        )

    return GateResult(
        passed=report.passed,
        errors=errors,
        warnings=warnings,
        statistics={
            "overall_score_pct": int(report.overall_score * 100),
            "dimensions_assessed": len(report.dimension_scores),
        },
    )
