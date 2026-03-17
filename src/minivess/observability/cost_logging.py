"""Cloud instance cost parameter logging.

Collects cloud instance metadata from SkyPilot environment variables
or config overrides. All values use the ``cost/`` slash-prefix
convention for MLflow param logging.

IMPORTANT: No hardcoded rates. Rates come from config or env vars.

PR-E T1 (Issue #830).
"""

from __future__ import annotations

import logging
import os
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
