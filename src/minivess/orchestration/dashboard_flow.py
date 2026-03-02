"""Flow 5 (Dashboard) — multi-task training curves and topology comparison.

Best-effort flow: failure does not block pipeline.
Extends dashboard with multi-task training curves and topology comparison charts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
