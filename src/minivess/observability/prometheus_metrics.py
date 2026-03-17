"""Shared Prometheus Gauges for training cost metrics.

Provides module-level Gauge instances that are registered in the
default prometheus_client CollectorRegistry. BentoML automatically
exposes these via its /metrics endpoint.

Data flow:
    train_flow.py -> compute_cost_analysis() -> update_cost_gauges()
    train_flow.py -> estimate_cost_from_first_epoch() -> update_estimated_cost_gauges()
    BentoML /metrics -> Prometheus scrapes -> Grafana dashboards

Issue: #747
"""

from __future__ import annotations

import logging

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gauge definitions — all prefixed with minivess_training_ to avoid
# collision with BentoML built-in metrics (bentoml_ prefix).
# ---------------------------------------------------------------------------

_GAUGE_DEFS: dict[str, str] = {
    "minivess_training_cost_total_usd": "Total training cost in USD",
    "minivess_training_cost_per_epoch_usd": "Cost per epoch in USD",
    "minivess_training_gpu_utilization": "GPU utilization fraction (training / total)",
    "minivess_training_setup_fraction": "Setup fraction of total cost",
    "minivess_training_effective_gpu_rate_usd": "Effective GPU rate in USD/hr",
    "minivess_training_estimated_total_cost_usd": "Estimated total cost from epoch-0 prediction",
}

GAUGES: dict[str, Gauge] = {}
for _name, _doc in _GAUGE_DEFS.items():
    GAUGES[_name] = Gauge(_name, _doc)

# ---------------------------------------------------------------------------
# Mapping from compute_cost_analysis() dict keys to Gauge names
# ---------------------------------------------------------------------------

_COST_KEY_TO_GAUGE: dict[str, str] = {
    "cost/total_usd": "minivess_training_cost_total_usd",
    "cost/effective_gpu_rate": "minivess_training_effective_gpu_rate_usd",
    "cost/gpu_utilization_fraction": "minivess_training_gpu_utilization",
    "cost/setup_fraction": "minivess_training_setup_fraction",
}

# ---------------------------------------------------------------------------
# Mapping from estimate_cost_from_first_epoch() dict keys to Gauge names
# ---------------------------------------------------------------------------

_ESTIMATE_KEY_TO_GAUGE: dict[str, str] = {
    "est/total_cost": "minivess_training_estimated_total_cost_usd",
    "est/cost_per_epoch": "minivess_training_cost_per_epoch_usd",
}


def update_cost_gauges(cost: dict[str, float]) -> None:
    """Set Gauge values from a compute_cost_analysis() output dict.

    Missing keys are silently skipped — Gauges retain their previous value.

    Parameters
    ----------
    cost:
        Dict with keys from compute_cost_analysis() output.
    """
    for cost_key, gauge_name in _COST_KEY_TO_GAUGE.items():
        if cost_key in cost:
            GAUGES[gauge_name].set(float(cost[cost_key]))
        else:
            logger.debug(
                "Missing cost key '%s' — gauge %s not updated", cost_key, gauge_name
            )


def update_estimated_cost_gauges(estimate: dict[str, float]) -> None:
    """Set Gauge values from an estimate_cost_from_first_epoch() output dict.

    Missing keys are silently skipped.

    Parameters
    ----------
    estimate:
        Dict with keys from estimate_cost_from_first_epoch() output.
    """
    for est_key, gauge_name in _ESTIMATE_KEY_TO_GAUGE.items():
        if est_key in estimate:
            GAUGES[gauge_name].set(float(estimate[est_key]))
        else:
            logger.debug(
                "Missing estimate key '%s' — gauge %s not updated", est_key, gauge_name
            )
