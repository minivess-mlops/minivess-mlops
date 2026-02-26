"""Adaptive compute profile generation based on hardware and dataset characteristics.

Replaces static profile lookup with dynamic computation that accounts for:
- Available GPU VRAM and RAM
- Dataset size and volume dimensions
- Model-specific patch divisibility requirements

Target hardware: 8 GB VRAM / 32 GB RAM desktop without OOM.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.data.profiler import DatasetProfile

# Model-specific patch size divisors
_MODEL_DIVISORS: dict[str, int] = {
    "dynunet": 8,
    "segresnet": 8,
    "vista3d": 16,
    "swinunetr": 32,
}

# Batch sizes per GPU tier
_TIER_BATCH_SIZES: dict[str, int] = {
    "cpu": 1,
    "gpu_low": 2,
    "gpu_high": 4,
    "gpu_extreme": 8,
}

# Cache RAM headroom factor: use 60% of available RAM for caching
_CACHE_RAM_FACTOR: float = 0.6


@dataclass
class HardwareBudget:
    """Detected hardware resource budget.

    Parameters
    ----------
    ram_total_mb:
        Total system RAM in megabytes.
    ram_available_mb:
        Currently available system RAM in megabytes.
    gpu_vram_mb:
        GPU VRAM in megabytes (0 if no GPU detected).
    gpu_name:
        GPU device name string (empty if no GPU detected).
    cpu_count:
        Number of logical CPU cores.
    swap_used_mb:
        Currently used swap space in megabytes.
    """

    ram_total_mb: int
    ram_available_mb: int
    gpu_vram_mb: int
    gpu_name: str
    cpu_count: int
    swap_used_mb: int

    @property
    def gpu_tier(self) -> str:
        """Map GPU VRAM to compute tier.

        Returns
        -------
        str
            One of ``"cpu"``, ``"gpu_low"``, ``"gpu_high"``, or ``"gpu_extreme"``.
        """
        if self.gpu_vram_mb == 0:
            return "cpu"
        if self.gpu_vram_mb <= 8192:
            return "gpu_low"
        if self.gpu_vram_mb <= 16384:
            return "gpu_high"
        return "gpu_extreme"


@dataclass(frozen=True)
class AdaptiveComputeProfile:
    """Hardware-detected and dataset-constrained compute profile.

    Extends static ``ComputeProfile`` with ``cache_rate`` for dataset caching.

    Parameters
    ----------
    name:
        Descriptive auto-generated profile name.
    batch_size:
        Training batch size adapted to GPU VRAM.
    patch_size:
        3D patch dimensions constrained by dataset volume sizes and model divisor.
    num_workers:
        DataLoader worker count based on CPU cores.
    mixed_precision:
        Whether to use AMP; disabled for CPU-only runs.
    gradient_accumulation_steps:
        Steps to accumulate before optimizer update.
    cache_rate:
        Fraction of dataset to cache in RAM (0.0–1.0), computed from
        available RAM vs. dataset size.
    """

    name: str
    batch_size: int
    patch_size: tuple[int, int, int]
    num_workers: int
    mixed_precision: bool
    gradient_accumulation_steps: int
    cache_rate: float


def _read_proc_meminfo() -> dict[str, int]:
    """Read /proc/meminfo and return values in megabytes.

    Returns
    -------
    dict[str, int]
        Mapping of key names to values in MB. Returns empty dict if
        /proc/meminfo is unavailable.
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return {}
    result: dict[str, int] = {}
    with meminfo_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    # Values in /proc/meminfo are in kB
                    value_kb = int(parts[1])
                    result[key] = value_kb // 1024
                except ValueError:
                    continue
    return result


def _detect_gpu() -> tuple[int, str]:
    """Detect GPU VRAM via nvidia-smi.

    Returns
    -------
    tuple[int, str]
        ``(vram_mb, gpu_name)`` where ``vram_mb`` is 0 and ``gpu_name`` is
        ``""`` when no GPU is found or nvidia-smi is unavailable.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0, ""
        # Take the first GPU only
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) >= 2:
            gpu_name = parts[0]
            vram_mb = int(parts[1])
            return vram_mb, gpu_name
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0, ""


def detect_hardware() -> HardwareBudget:
    """Detect available hardware resources.

    Reads:
    - RAM from ``/proc/meminfo`` (MemTotal, MemAvailable, SwapTotal, SwapFree)
    - GPU VRAM from ``nvidia-smi``
    - CPU count from ``os.cpu_count()``

    Returns
    -------
    HardwareBudget
        Populated hardware budget with all fields set. Falls back to
        conservative defaults when detection fails.
    """
    mem = _read_proc_meminfo()
    ram_total_mb = mem.get("MemTotal", 4096)
    ram_available_mb = mem.get("MemAvailable", ram_total_mb // 2)
    swap_total_mb = mem.get("SwapTotal", 0)
    swap_free_mb = mem.get("SwapFree", 0)
    swap_used_mb = swap_total_mb - swap_free_mb

    gpu_vram_mb, gpu_name = _detect_gpu()

    cpu_count = os.cpu_count() or 1

    return HardwareBudget(
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        gpu_vram_mb=gpu_vram_mb,
        gpu_name=gpu_name,
        cpu_count=cpu_count,
        swap_used_mb=max(0, swap_used_mb),
    )


def _compute_patch_size(
    dataset_profile: DatasetProfile,
    model_divisor: int,
    max_patch_xy: int = 128,
) -> tuple[int, int, int]:
    """Compute safe patch size from dataset min shape and model divisor.

    Parameters
    ----------
    dataset_profile:
        Dataset profile containing minimum volume shapes.
    model_divisor:
        Required patch size divisibility.
    max_patch_xy:
        Maximum patch size in the XY plane.

    Returns
    -------
    tuple[int, int, int]
        Patch size (x, y, z) guaranteed to be divisible by ``model_divisor``
        and no larger than the smallest volume dimension.
    """
    min_x, min_y, min_z = (
        dataset_profile.min_shape[0],
        dataset_profile.min_shape[1],
        dataset_profile.min_shape[2],
    )

    # Clamp XY to max_patch_xy
    clamped_x = min(min_x, max_patch_xy)
    clamped_y = min(min_y, max_patch_xy)
    clamped_z = min_z

    def _floor_to_divisor(value: int, divisor: int) -> int:
        """Floor value to largest multiple of divisor that fits within value.

        If value < divisor, return value unchanged (volume is smaller than
        one divisor step; caller must handle this edge case).
        """
        if value < divisor:
            return value
        return (value // divisor) * divisor

    safe_x = _floor_to_divisor(clamped_x, model_divisor)
    safe_y = _floor_to_divisor(clamped_y, model_divisor)
    safe_z = _floor_to_divisor(clamped_z, model_divisor)

    # Ensure at least 1 in each dimension
    safe_x = max(1, safe_x)
    safe_y = max(1, safe_y)
    safe_z = max(1, safe_z)

    return (safe_x, safe_y, safe_z)


def _compute_cache_rate(
    ram_available_mb: int,
    total_size_bytes: int,
) -> float:
    """Compute dataset cache rate based on available RAM.

    Uses 60% of available RAM for caching, returns value clamped to [0.0, 1.0].

    Parameters
    ----------
    ram_available_mb:
        Available RAM in megabytes.
    total_size_bytes:
        Total dataset size in bytes.

    Returns
    -------
    float
        Cache rate between 0.0 and 1.0.
    """
    if total_size_bytes <= 0:
        return 1.0
    total_dataset_mb = total_size_bytes / (1024 * 1024)
    cache_budget_mb = ram_available_mb * _CACHE_RAM_FACTOR
    rate = cache_budget_mb / total_dataset_mb
    return min(1.0, max(0.0, rate))


def _compute_num_workers(cpu_count: int, gpu_tier: str) -> int:
    """Compute DataLoader worker count.

    Parameters
    ----------
    cpu_count:
        Number of logical CPU cores.
    gpu_tier:
        GPU tier string.

    Returns
    -------
    int
        Number of workers (capped relative to CPU count).
    """
    if gpu_tier == "cpu":
        return min(cpu_count, 2)
    # Use half of CPU cores, minimum 2, maximum 16
    return min(max(2, cpu_count // 2), 16)


def _compute_gradient_accumulation(batch_size: int, gpu_tier: str) -> int:
    """Compute gradient accumulation steps to maintain effective batch size.

    Parameters
    ----------
    batch_size:
        Actual mini-batch size.
    gpu_tier:
        GPU tier string.

    Returns
    -------
    int
        Number of gradient accumulation steps.
    """
    if gpu_tier == "cpu":
        return 4
    if batch_size <= 1:
        return 4
    if batch_size <= 2:
        return 2
    return 1


def compute_adaptive_profile(
    budget: HardwareBudget,
    dataset_profile: DatasetProfile,
    model_name: str = "dynunet",
    override_patch_size: tuple[int, int, int] | None = None,
) -> AdaptiveComputeProfile:
    """Compute an adaptive profile tuned to hardware and dataset.

    Algorithm:
    1. Determine model patch divisor from ``model_name``.
    2. Compute safe patch size from dataset min shapes, clamped and divisible.
    3. Apply ``override_patch_size`` if provided.
    4. Determine batch size from GPU tier.
    5. Compute cache rate from available RAM vs. dataset size.
    6. Set mixed precision based on GPU availability.
    7. Build descriptive profile name.

    Parameters
    ----------
    budget:
        Detected hardware budget.
    dataset_profile:
        Dataset statistics from the profiler.
    model_name:
        Model architecture name for divisor lookup.
    override_patch_size:
        If provided, overrides computed patch size entirely.

    Returns
    -------
    AdaptiveComputeProfile
        Fully specified adaptive compute profile.
    """
    model_divisor = _MODEL_DIVISORS.get(model_name, 8)

    # Compute patch size
    if override_patch_size is not None:
        patch_size = override_patch_size
    else:
        patch_size = _compute_patch_size(dataset_profile, model_divisor)

    # Batch size from GPU tier
    batch_size = _TIER_BATCH_SIZES.get(budget.gpu_tier, 1)

    # Cache rate from RAM availability
    cache_rate = _compute_cache_rate(
        budget.ram_available_mb, dataset_profile.total_size_bytes
    )

    # Mixed precision only when GPU is available
    mixed_precision = budget.gpu_vram_mb > 0

    # Worker count
    num_workers = _compute_num_workers(budget.cpu_count, budget.gpu_tier)

    # Gradient accumulation
    gradient_accumulation_steps = _compute_gradient_accumulation(
        batch_size, budget.gpu_tier
    )

    # Descriptive name: auto_{model}_{tier}
    name = f"auto_{model_name}_{budget.gpu_tier}"

    return AdaptiveComputeProfile(
        name=name,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cache_rate=cache_rate,
    )
