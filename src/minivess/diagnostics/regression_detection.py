"""Performance regression detection helpers.

Compares latest run metrics against baseline (median of prior runs
on the same GPU, filtered by normalized GPU name from T3.1).

NOT wired into flows — utility library for dashboards/scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineStats:
    """Aggregated baseline statistics from prior runs."""

    metric_name: str
    count: int
    median: float
    min: float
    max: float
    std: float
    gpu_model: str


@dataclass
class RegressionResult:
    """Result of a single regression check."""

    is_regression: bool
    baseline_value: float
    latest_value: float
    delta_pct: float
    metric_name: str
    gpu_model: str


def detect_regression(
    baseline: BaselineStats,
    latest_value: float,
    threshold_pct: float = 5.0,
) -> RegressionResult:
    """Compare latest value against baseline median.

    Parameters
    ----------
    baseline:
        Aggregated statistics from prior runs.
    latest_value:
        Metric value from the current run.
    threshold_pct:
        Percentage above baseline median to flag as regression.

    Returns
    -------
    RegressionResult with is_regression flag and delta_pct.
    """
    if baseline.count < 2 or baseline.median == 0.0:
        return RegressionResult(
            is_regression=False,
            baseline_value=baseline.median,
            latest_value=latest_value,
            delta_pct=0.0,
            metric_name=baseline.metric_name,
            gpu_model=baseline.gpu_model,
        )

    delta_pct = (latest_value - baseline.median) / baseline.median * 100.0
    is_regression = delta_pct > threshold_pct

    return RegressionResult(
        is_regression=is_regression,
        baseline_value=baseline.median,
        latest_value=latest_value,
        delta_pct=delta_pct,
        metric_name=baseline.metric_name,
        gpu_model=baseline.gpu_model,
    )


def format_regression_report(result: RegressionResult) -> str:
    """Format a human-readable regression report."""
    status = "REGRESSION DETECTED" if result.is_regression else "No regression"
    return (
        f"{status}\n"
        f"  Metric: {result.metric_name}\n"
        f"  GPU: {result.gpu_model}\n"
        f"  Baseline: {result.baseline_value:.0f}\n"
        f"  Latest: {result.latest_value:.0f} ({result.delta_pct:+.1f}%)"
    )
