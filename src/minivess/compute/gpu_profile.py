"""GPU benchmark cache integration — read cache, log to MLflow.

Reads benchmark YAML cache from gpu_benchmark module and logs
params with bench/ prefix (#790) to MLflow via ExperimentTracker.

Non-blocking: missing or invalid cache logs a warning, never crashes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_benchmark_params(cache_path: Path) -> dict[str, Any]:
    """Load benchmark cache and return params with sys_bench_ prefix.

    Returns empty dict if cache is missing or invalid.
    """
    if not cache_path.exists():
        logger.warning("GPU benchmark cache not found: %s", cache_path)
        return {}

    try:
        data = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("GPU benchmark cache invalid: %s — %s", cache_path, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("GPU benchmark cache has unexpected format: %s", cache_path)
        return {}

    params: dict[str, Any] = {}

    # Instance info
    instance = data.get("instance_info", {})
    if isinstance(instance, dict):
        if "gpu_model" in instance:
            params["bench/gpu_model"] = instance["gpu_model"]
        if "total_vram_mb" in instance:
            params["bench/total_vram_mb"] = instance["total_vram_mb"]
        if "driver_version" in instance:
            params["bench/driver_version"] = instance["driver_version"]
        if "cuda_version" in instance:
            params["bench/cuda_version"] = instance["cuda_version"]

    # Per-model benchmarks
    benchmarks = data.get("benchmarks", {})
    if isinstance(benchmarks, dict):
        for model_name, bench in benchmarks.items():
            if not isinstance(bench, dict):
                continue
            prefix = f"bench/{model_name}"
            if "peak_vram_mb" in bench:
                params[f"{prefix}/vram_peak_mb"] = bench["peak_vram_mb"]
            if "throughput_img_per_sec" in bench:
                params[f"{prefix}/throughput"] = bench["throughput_img_per_sec"]
            if "forward_ms" in bench:
                params[f"{prefix}/forward_ms"] = bench["forward_ms"]

    # Capabilities (feasibility flags)
    capabilities = data.get("capabilities", {})
    if isinstance(capabilities, dict):
        for model_name, cap in capabilities.items():
            if isinstance(cap, dict) and "feasible" in cap:
                params[f"bench/{model_name}/feasible"] = cap["feasible"]

    return params


def log_benchmark_to_mlflow(
    cache_path: Path,
    *,
    tracker: Any,
) -> None:
    """Load benchmark cache and log params to MLflow.

    Non-blocking: missing/invalid cache produces a warning, not an error.
    """
    params = load_benchmark_params(cache_path)
    if not params:
        return

    try:
        tracker.log_params(params)
        logger.info("Logged %d GPU benchmark params to MLflow", len(params))
    except Exception:
        logger.warning("Failed to log GPU benchmark params", exc_info=True)
