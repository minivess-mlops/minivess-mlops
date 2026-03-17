"""Cloud instance cost parameter logging and savings computation.

Collects cloud instance metadata from SkyPilot environment variables
or config overrides. All values use the ``cost/`` slash-prefix
convention for MLflow param logging.

IMPORTANT: No hardcoded rates. Rates come from config or env vars.

PR-E T1 (Issue #830), T2 (Issue #831).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# SkyPilot env var → cost param key mapping
_SKYPILOT_ENV_MAP: dict[str, str] = {
    "SKYPILOT_INSTANCE_TYPE": "cost/instance_type",
    "SKYPILOT_ACCELERATOR_TYPE": "cost/gpu_type",
    "SKYPILOT_USE_SPOT": "cost/spot_enabled",
    "SKYPILOT_REGION": "cost/region",
    "SKYPILOT_CLOUD": "cost/provider",
    "SKYPILOT_TASK_ID": "cost/task_id",
    "SKYPILOT_CLUSTER_NAME": "cost/cluster_name",
    "SKYPILOT_ZONE": "cost/zone",
    "SKYPILOT_NUM_GPUS_PER_NODE": "cost/num_gpus",
}

# Default values when neither env nor config provides a value
_DEFAULTS: dict[str, str] = {
    "cost/instance_type": "unknown",
    "cost/gpu_type": "unknown",
    "cost/spot_enabled": "False",
    "cost/region": "unknown",
    "cost/provider": "unknown",
    "cost/task_id": "unknown",
    "cost/cluster_name": "unknown",
    "cost/zone": "unknown",
    "cost/num_gpus": "1",
    "cost/spot_hourly_rate": "unknown",
    "cost/ondemand_hourly_rate": "unknown",
}

# Config override key → cost param key mapping
_CONFIG_KEY_MAP: dict[str, str] = {
    "instance_type": "cost/instance_type",
    "gpu_type": "cost/gpu_type",
    "spot_enabled": "cost/spot_enabled",
    "region": "cost/region",
    "provider": "cost/provider",
    "task_id": "cost/task_id",
    "spot_hourly_rate": "cost/spot_hourly_rate",
    "ondemand_hourly_rate": "cost/ondemand_hourly_rate",
}


def collect_cost_params(
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Collect cloud instance cost parameters for MLflow logging.

    Priority order:
    1. SkyPilot environment variables (highest)
    2. Config overrides dict
    3. Default "unknown" values (lowest)

    Parameters
    ----------
    config_overrides:
        Optional dict with cost config values. Keys should match
        the short names in ``_CONFIG_KEY_MAP`` (e.g., ``instance_type``).

    Returns
    -------
    Dict with ``cost/*`` keys and string values for MLflow param logging.
    """
    params = dict(_DEFAULTS)

    # Layer 1: Apply config overrides (lower priority)
    if config_overrides:
        for config_key, cost_key in _CONFIG_KEY_MAP.items():
            if config_key in config_overrides:
                params[cost_key] = str(config_overrides[config_key])

    # Layer 2: Apply SkyPilot env vars (higher priority)
    for env_var, cost_key in _SKYPILOT_ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            params[cost_key] = value

    logger.info(
        "Collected cost params: provider=%s instance=%s gpu=%s spot=%s",
        params["cost/provider"],
        params["cost/instance_type"],
        params["cost/gpu_type"],
        params["cost/spot_enabled"],
    )

    return params


# ---------------------------------------------------------------------------
# Cost summary and spot savings (PR-E T2)
# ---------------------------------------------------------------------------


@dataclass
class CostSummary:
    """Aggregated cost summary for the manuscript appendix.

    Parameters
    ----------
    total_spot_cost_usd:
        Total cost using spot instances.
    total_ondemand_cost_usd:
        Hypothetical cost using on-demand instances.
    savings_pct:
        Percentage saved by using spot: ``(ondemand - spot) / ondemand * 100``.
    cost_by_phase:
        Cost breakdown by phase (e.g., ``{"training": 5.20, "debug": 0.40}``).
    cost_by_model:
        Cost breakdown by model family (e.g., ``{"dynunet": 2.80}``).
    total_gpu_hours:
        Total GPU hours across all runs.
    """

    total_spot_cost_usd: float = 0.0
    total_ondemand_cost_usd: float = 0.0
    savings_pct: float = 0.0
    cost_by_phase: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    total_gpu_hours: float = 0.0


def compute_spot_savings(
    records: list[dict[str, Any]],
) -> CostSummary:
    """Compute spot vs on-demand savings from cost records.

    Parameters
    ----------
    records:
        List of cost record dicts. Each must have:
        ``spot_cost_usd``, ``gpu_hours``, ``ondemand_hourly_rate``,
        ``model``, ``phase``.

    Returns
    -------
    :class:`CostSummary` with aggregated cost data.
    """
    total_spot = 0.0
    total_ondemand = 0.0
    total_hours = 0.0
    by_phase: dict[str, float] = {}
    by_model: dict[str, float] = {}

    for rec in records:
        spot_cost = float(rec["spot_cost_usd"])
        hours = float(rec["gpu_hours"])
        ondemand_rate = float(rec["ondemand_hourly_rate"])
        model = str(rec.get("model", "unknown"))
        phase = str(rec.get("phase", "unknown"))

        total_spot += spot_cost
        total_ondemand += hours * ondemand_rate
        total_hours += hours

        by_phase[phase] = by_phase.get(phase, 0.0) + spot_cost
        by_model[model] = by_model.get(model, 0.0) + spot_cost

    savings_pct = (
        (total_ondemand - total_spot) / total_ondemand * 100.0
        if total_ondemand > 0
        else 0.0
    )

    summary = CostSummary(
        total_spot_cost_usd=total_spot,
        total_ondemand_cost_usd=total_ondemand,
        savings_pct=savings_pct,
        cost_by_phase=by_phase,
        cost_by_model=by_model,
        total_gpu_hours=total_hours,
    )

    logger.info(
        "Cost summary: spot=$%.2f ondemand=$%.2f savings=%.1f%% hours=%.1f",
        total_spot,
        total_ondemand,
        savings_pct,
        total_hours,
    )

    return summary
