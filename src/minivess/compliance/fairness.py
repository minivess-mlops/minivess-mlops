"""CyclOps-inspired healthcare ML fairness auditing.

Evaluates model fairness across demographic subgroups and generates
compliance audit reports. Implements disparity analysis following
CyclOps (Krishnan et al., 2022) patterns without external dependency.

Reference: Krishnan et al. (2022). "CyclOps: Toolkit for Healthcare
ML Auditing and Monitoring." NeurIPS 2022 Workshop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class SubgroupMetrics:
    """Performance metrics for a single demographic subgroup.

    Parameters
    ----------
    subgroup_name:
        Identifier for the subgroup (e.g., "age_65_plus").
    subgroup_size:
        Number of samples in the subgroup.
    metrics:
        Metric name → value mapping (e.g., {"dice": 0.85}).
    """

    subgroup_name: str
    subgroup_size: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FairnessReport:
    """Aggregate fairness evaluation across demographic subgroups.

    Parameters
    ----------
    subgroup_metrics:
        Per-subgroup metric results.
    disparity_scores:
        Metric name → max-min disparity.
    passed:
        Whether all disparities are below threshold.
    threshold:
        Maximum acceptable disparity.
    """

    subgroup_metrics: list[SubgroupMetrics]
    disparity_scores: dict[str, float]
    passed: bool
    threshold: float

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for MLflow logging."""
        d: dict[str, float] = {}
        for metric_name, disparity in self.disparity_scores.items():
            d[f"disparity_{metric_name}"] = disparity
        for sm in self.subgroup_metrics:
            for metric_name, value in sm.metrics.items():
                d[f"{sm.subgroup_name}_{metric_name}"] = value
        return d


def compute_disparity(
    subgroup_metrics: list[SubgroupMetrics],
    metric_name: str,
) -> float:
    """Compute max-min disparity for a metric across subgroups.

    Parameters
    ----------
    subgroup_metrics:
        Per-subgroup metrics.
    metric_name:
        Name of the metric to compute disparity for.

    Returns
    -------
    Max-min disparity (0.0 if metric not found in any subgroup).
    """
    values = [
        sm.metrics[metric_name] for sm in subgroup_metrics if metric_name in sm.metrics
    ]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def evaluate_subgroup_fairness(
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    subgroups: NDArray[np.str_] | list[str],
    metric_fn: Callable[[NDArray[np.int64], NDArray[np.int64]], dict[str, float]],
    *,
    threshold: float = 0.1,
) -> FairnessReport:
    """Evaluate model fairness across demographic subgroups.

    Parameters
    ----------
    predictions:
        Model predictions (N,).
    labels:
        Ground truth labels (N,).
    subgroups:
        Subgroup assignments (N,).
    metric_fn:
        Function (predictions, labels) → dict of metric values.
    threshold:
        Maximum acceptable max-min disparity for any metric.

    Returns
    -------
    FairnessReport with per-subgroup metrics and disparity analysis.
    """
    subgroups_arr = np.asarray(subgroups)
    unique_groups = np.unique(subgroups_arr)

    all_metrics: list[SubgroupMetrics] = []
    for group in unique_groups:
        mask = subgroups_arr == group
        group_preds = predictions[mask]
        group_labels = labels[mask]
        metrics = metric_fn(group_preds, group_labels)
        all_metrics.append(
            SubgroupMetrics(
                subgroup_name=str(group),
                subgroup_size=int(mask.sum()),
                metrics=metrics,
            )
        )

    # Compute disparity for each metric
    metric_names: set[str] = set()
    for sm in all_metrics:
        metric_names.update(sm.metrics.keys())

    disparity_scores = {
        name: compute_disparity(all_metrics, name) for name in sorted(metric_names)
    }

    # Check if any disparity exceeds threshold
    passed = all(d <= threshold for d in disparity_scores.values())

    return FairnessReport(
        subgroup_metrics=all_metrics,
        disparity_scores=disparity_scores,
        passed=passed,
        threshold=threshold,
    )


def generate_audit_report(
    report: FairnessReport,
    product_name: str,
) -> str:
    """Generate markdown fairness audit report.

    Parameters
    ----------
    report:
        FairnessReport from evaluate_subgroup_fairness.
    product_name:
        Name of the product being audited.

    Returns
    -------
    Markdown string with fairness audit findings.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    status = "PASS" if report.passed else "FAIL"

    lines = [
        "# Fairness Audit Report\n",
        f"**Product:** {product_name}",
        f"**Date:** {now}",
        f"**Status:** {status}",
        f"**Disparity Threshold:** {report.threshold}\n",
    ]

    # Subgroup performance table
    lines.append("## Subgroup Performance\n")
    if report.subgroup_metrics:
        metric_names = sorted({k for sm in report.subgroup_metrics for k in sm.metrics})
        header = "| Subgroup | Size | " + " | ".join(metric_names) + " |"
        separator = (
            "|----------|------|" + "|".join("------" for _ in metric_names) + "|"
        )
        lines.append(header)
        lines.append(separator)
        for sm in report.subgroup_metrics:
            values = " | ".join(f"{sm.metrics.get(m, 0.0):.4f}" for m in metric_names)
            lines.append(f"| {sm.subgroup_name} | {sm.subgroup_size} | {values} |")

    # Disparity analysis
    lines.append("\n## Disparity Analysis\n")
    lines.append("| Metric | Max-Min Disparity | Status |")
    lines.append("|--------|-------------------|--------|")
    for metric_name, disparity in sorted(report.disparity_scores.items()):
        d_status = "PASS" if disparity <= report.threshold else "FAIL"
        lines.append(f"| {metric_name} | {disparity:.4f} | {d_status} |")

    # Recommendations
    if not report.passed:
        lines.append("\n## Recommendations\n")
        lines.append(
            "Significant performance disparities detected. "
            "Consider the following mitigations:\n"
        )
        lines.append("1. Investigate data representation across subgroups")
        lines.append("2. Apply fairness-aware training techniques")
        lines.append("3. Increase minority subgroup sample sizes")
        lines.append("4. Review annotation quality per subgroup")

    return "\n".join(lines)
