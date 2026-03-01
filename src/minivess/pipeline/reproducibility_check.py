"""Reproducibility verification: training metrics must match inference.

Confirms deterministic reproducibility by comparing training-logged metrics
against freshly computed inference metrics on the same validation volumes.

MLflow metric file format: ``<timestamp> <value> <step>`` per line.
Fold metrics are stored as ``eval_fold{N}_{metric_name}``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReproducibilityResult:
    """Comparison result for a single metric on a single fold."""

    run_id: str
    fold_id: int
    metric_name: str
    training_value: float
    inference_value: float
    absolute_diff: float
    is_reproducible: bool


@dataclass
class ReproducibilityReport:
    """Aggregate report from a reproducibility verification."""

    results: list[ReproducibilityResult] = field(default_factory=list)
    tolerance: float = 1e-5
    all_pass: bool = True
    summary: str = ""


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------


def compare_metric_values(
    run_id: str,
    fold_id: int,
    metric_name: str,
    training_value: float,
    inference_value: float,
    tolerance: float = 1e-5,
) -> ReproducibilityResult:
    """Compare a training metric against its inference counterpart.

    Parameters
    ----------
    run_id:
        MLflow run ID.
    fold_id:
        Cross-validation fold index.
    metric_name:
        Name of the metric being compared.
    training_value:
        Value logged during training.
    inference_value:
        Value computed by fresh inference.
    tolerance:
        Maximum absolute difference for reproducibility.

    Returns
    -------
    :class:`ReproducibilityResult` with comparison outcome.
    """
    if math.isnan(training_value) or math.isnan(inference_value):
        return ReproducibilityResult(
            run_id=run_id,
            fold_id=fold_id,
            metric_name=metric_name,
            training_value=training_value,
            inference_value=inference_value,
            absolute_diff=float("nan"),
            is_reproducible=False,
        )

    diff = abs(training_value - inference_value)
    is_reproducible = diff < tolerance

    return ReproducibilityResult(
        run_id=run_id,
        fold_id=fold_id,
        metric_name=metric_name,
        training_value=training_value,
        inference_value=inference_value,
        absolute_diff=diff,
        is_reproducible=is_reproducible,
    )


# ---------------------------------------------------------------------------
# Filesystem metric reading
# ---------------------------------------------------------------------------


def read_training_metric_for_fold(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    fold_id: int,
    metric_name: str,
) -> float:
    """Read a training-logged fold metric from the MLflow filesystem.

    Reads ``eval_fold{fold_id}_{metric_name}`` and returns the last-logged
    value (last line, second field).

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    run_id:
        MLflow run ID.
    fold_id:
        Cross-validation fold index.
    metric_name:
        Base metric name (without the ``eval_fold{N}_`` prefix).

    Returns
    -------
    Last-logged float value.

    Raises
    ------
    FileNotFoundError:
        If the run directory or metric file does not exist.
    """
    run_dir = mlruns_dir / experiment_id / run_id
    if not run_dir.is_dir():
        msg = f"Run directory does not exist: {run_dir}"
        raise FileNotFoundError(msg)

    metric_file = run_dir / "metrics" / f"eval_fold{fold_id}_{metric_name}"
    if not metric_file.is_file():
        msg = f"Metric file does not exist: {metric_file}"
        raise FileNotFoundError(msg)

    content = metric_file.read_text(encoding="utf-8").strip()
    if not content:
        msg = f"Metric file is empty: {metric_file}"
        raise FileNotFoundError(msg)

    # MLflow format: "<timestamp> <value> <step>"
    last_line = content.splitlines()[-1]
    parts = last_line.split()
    return float(parts[1])


# ---------------------------------------------------------------------------
# Report creation
# ---------------------------------------------------------------------------


def create_reproducibility_report(
    results: list[ReproducibilityResult],
    tolerance: float = 1e-5,
) -> ReproducibilityReport:
    """Create a reproducibility report from comparison results.

    Parameters
    ----------
    results:
        List of per-fold, per-metric comparison results.
    tolerance:
        Tolerance used for the comparisons.

    Returns
    -------
    :class:`ReproducibilityReport` with summary.
    """
    all_pass = all(r.is_reproducible for r in results)

    n_pass = sum(1 for r in results if r.is_reproducible)
    n_total = len(results)

    lines = [
        f"Reproducibility Report (tolerance={tolerance})",
        f"Results: {n_pass}/{n_total} pass",
    ]

    if results:
        # Group by metric
        metrics = sorted({r.metric_name for r in results})
        for metric in metrics:
            metric_results = [r for r in results if r.metric_name == metric]
            n_metric_pass = sum(1 for r in metric_results if r.is_reproducible)
            lines.append(f"  {metric}: {n_metric_pass}/{len(metric_results)} pass")

    if not all_pass:
        lines.append("")
        lines.append("Failures:")
        for r in results:
            if not r.is_reproducible:
                lines.append(
                    f"  {r.metric_name} fold={r.fold_id}: "
                    f"train={r.training_value:.6f} vs infer={r.inference_value:.6f} "
                    f"(diff={r.absolute_diff:.6f})"
                )

    summary = "\n".join(lines)

    return ReproducibilityReport(
        results=results,
        tolerance=tolerance,
        all_pass=all_pass,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# CV-mean reproducibility
# ---------------------------------------------------------------------------


def verify_cv_mean_reproducibility(
    fold_results: dict[int, list[ReproducibilityResult]],
    cv_mean_metrics: dict[str, float],
    tolerance: float = 1e-4,
) -> ReproducibilityReport:
    """Verify that CV-mean metrics match aggregated per-fold values.

    Computes the mean of inference values across folds for each metric,
    then compares against the expected CV-mean.

    Parameters
    ----------
    fold_results:
        Per-fold results: ``{fold_id: [ReproducibilityResult, ...]}``.
    cv_mean_metrics:
        Expected CV-mean values: ``{metric_name: float}``.
    tolerance:
        Maximum absolute difference for CV-mean matching.

    Returns
    -------
    :class:`ReproducibilityReport` for CV-mean comparisons.
    """
    if not fold_results or not cv_mean_metrics:
        return ReproducibilityReport(
            results=[],
            tolerance=tolerance,
            all_pass=True,
            summary="No CV-mean metrics to verify.",
        )

    # Collect inference values per metric across folds
    metric_fold_values: dict[str, list[float]] = {}
    has_nan = set()

    for _fold_id, results in fold_results.items():
        for r in results:
            if r.metric_name not in metric_fold_values:
                metric_fold_values[r.metric_name] = []
            if math.isnan(r.inference_value) or math.isnan(r.training_value):
                has_nan.add(r.metric_name)
            metric_fold_values[r.metric_name].append(r.inference_value)

    comparison_results: list[ReproducibilityResult] = []

    for metric_name, expected_mean in cv_mean_metrics.items():
        if metric_name in has_nan:
            comparison_results.append(
                ReproducibilityResult(
                    run_id="cv_mean",
                    fold_id=-1,
                    metric_name=metric_name,
                    training_value=expected_mean,
                    inference_value=float("nan"),
                    absolute_diff=float("nan"),
                    is_reproducible=False,
                )
            )
            continue

        fold_values = metric_fold_values.get(metric_name, [])
        if not fold_values:
            continue

        computed_mean = sum(fold_values) / len(fold_values)
        comparison_results.append(
            compare_metric_values(
                run_id="cv_mean",
                fold_id=-1,
                metric_name=metric_name,
                training_value=expected_mean,
                inference_value=computed_mean,
                tolerance=tolerance,
            )
        )

    return create_reproducibility_report(comparison_results, tolerance=tolerance)
