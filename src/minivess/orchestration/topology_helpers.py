"""Topology-aware segmentation helpers for orchestration flows.

Provides helper functions used by analysis (Flow 3) and dashboard (Flow 5)
flows for topology metrics extraction and comparison table generation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_per_head_metrics(
    run_data: dict[str, Any],
) -> dict[str, float]:
    """Extract per-head metrics (SDF/CL MAE/RMSE) from an MLflow run.

    Args:
        run_data: Dict with "metrics" key containing logged metric values.

    Returns:
        Dict of per-head metrics found in the run.
    """
    metrics = run_data.get("metrics", {})
    per_head: dict[str, float] = {}

    for key, value in metrics.items():
        # Per-head metrics follow pattern: {head_name}/{metric_type}
        if "/" in key and not key.startswith("val_") and not key.startswith("loss/"):
            per_head[key] = float(value)

    return per_head


def build_topology_comparison(
    condition_results: dict[str, list[dict[str, float]]],
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build topology comparison table from condition results.

    Args:
        condition_results: Condition name -> list of fold metric dicts.
        metric_names: Metrics to include in comparison.

    Returns:
        Nested dict: condition -> metric -> {mean, std}.
    """
    from minivess.pipeline.topology_comparison import TopologyComparisonEvaluator

    evaluator = TopologyComparisonEvaluator(condition_results, metric_names)
    return evaluator.compute_summary()


def extract_multitask_training_curves(
    run_history: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Extract per-component loss curves from run history.

    Args:
        run_history: List of step dicts with metric values per epoch.

    Returns:
        Dict mapping "loss/{component}" to list of values over epochs.
    """
    curves: dict[str, list[float]] = {}

    for step in run_history:
        for key, value in step.items():
            if key.startswith("loss/"):
                if key not in curves:
                    curves[key] = []
                curves[key].append(float(value))

    return curves


def build_topology_comparison_data(
    condition_results: dict[str, list[dict[str, float]]],
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Build bar chart data for topology comparison.

    Args:
        condition_results: Condition name -> list of fold metric dicts.
        metric_names: Metrics to include.

    Returns:
        Dict: condition -> metric -> mean value.
    """
    if not condition_results:
        logger.info("No multitask runs found, skipping topology comparison")
        return {}

    from minivess.pipeline.topology_comparison import TopologyComparisonEvaluator

    evaluator = TopologyComparisonEvaluator(condition_results, metric_names)
    summary = evaluator.compute_summary()

    # Flatten to condition -> metric -> mean only (for charting)
    chart_data: dict[str, dict[str, float]] = {}
    for condition, metrics in summary.items():
        chart_data[condition] = {m: stats["mean"] for m, stats in metrics.items()}

    return chart_data
