"""GPU benchmark module — run-once capability probing with YAML cache.

Probes GPU info, normalizes model name to canonical form (RC11),
runs per-model micro-benchmarks, and writes a YAML cache that is
invalidated on driver/CUDA version change.

Metrics logged with sys_bench_ prefix (RC8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Cache schema version — bump when format changes
SCHEMA_VERSION = 1

# VRAM feasibility threshold: model must use less than this fraction
_FEASIBILITY_THRESHOLD = 0.85


@dataclass
class GpuInfo:
    """GPU hardware information."""

    name: str
    normalized_name: str
    total_vram_mb: float
    driver_version: str
    cuda_version: str


@dataclass
class BenchmarkResult:
    """Per-model benchmark measurement."""

    model_family: str
    peak_vram_mb: float
    throughput_img_per_sec: float
    forward_ms: float
    feasible: bool


def normalize_gpu_name(raw_name: str) -> str:
    """Normalize GPU model name to canonical form (RC11).

    Examples
    --------
    >>> normalize_gpu_name("NVIDIA GeForce RTX 2070 SUPER")
    'rtx_2070_super'
    >>> normalize_gpu_name("NVIDIA A100-SXM4-80GB")
    'a100_sxm4_80gb'
    """
    if not raw_name.strip():
        return "unknown_gpu"

    name = raw_name.strip()
    # Strip common prefixes
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Lowercase, replace spaces and hyphens with underscores
    name = name.lower().replace(" ", "_").replace("-", "_")

    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def check_feasibility(*, peak_vram_mb: float, total_vram_mb: float) -> bool:
    """Check if a model fits within VRAM budget.

    A model is feasible if peak_vram < total_vram * threshold.
    """
    return peak_vram_mb < total_vram_mb * _FEASIBILITY_THRESHOLD


def write_benchmark_yaml(
    cache_path: Path,
    *,
    gpu_info: GpuInfo,
    results: list[BenchmarkResult],
) -> None:
    """Write benchmark results to YAML cache file."""
    capabilities: dict[str, Any] = {}
    benchmarks: dict[str, Any] = {}

    for r in results:
        capabilities[r.model_family] = {
            "feasible": r.feasible,
        }
        benchmarks[r.model_family] = {
            "peak_vram_mb": r.peak_vram_mb,
            "throughput_img_per_sec": r.throughput_img_per_sec,
            "forward_ms": r.forward_ms,
            "feasible": r.feasible,
        }

    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "instance_info": {
            "gpu_model": gpu_info.normalized_name,
            "gpu_name_raw": gpu_info.name,
            "total_vram_mb": gpu_info.total_vram_mb,
            "driver_version": gpu_info.driver_version,
            "cuda_version": gpu_info.cuda_version,
        },
        "capabilities": capabilities,
        "benchmarks": benchmarks,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Wrote GPU benchmark cache to %s", cache_path)


def is_cache_valid(
    cache_path: Path,
    *,
    current_driver: str,
    current_cuda: str,
) -> bool:
    """Check if existing cache matches current GPU driver/CUDA versions."""
    if not cache_path.exists():
        return False

    try:
        data = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return False

    if not isinstance(data, dict):
        return False

    instance = data.get("instance_info", {})
    return bool(
        instance.get("driver_version") == current_driver
        and instance.get("cuda_version") == current_cuda
    )
