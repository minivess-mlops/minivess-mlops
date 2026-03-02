"""Flow 3 (Analysis) — topology metrics integration.

Extends analysis flow with per-head validation metrics extraction and
topology comparison table generation from MLflow runs.
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
