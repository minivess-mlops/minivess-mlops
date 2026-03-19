"""Grid partitioning for parallel factorial HPO studies.

Assigns grid cell indices to workers using modular arithmetic:
cell_index % total_workers == worker_id.

This produces disjoint, exhaustive partitions without any coordination
between workers — each worker can independently compute its cell set
from (worker_id, total_workers, total_cells).

Phase 1 (this module): Simple grid partitioning without Optuna.
Phase 2 (deferred): Optuna ask-tell API + PostgreSQL distributed trials.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def partition_grid_cells(
    *,
    total_cells: int,
    worker_id: int,
    total_workers: int,
) -> list[int]:
    """Partition grid cell indices across workers using modular arithmetic.

    Each worker gets cells where cell_index % total_workers == worker_id.
    This guarantees disjoint, exhaustive coverage with zero coordination.

    Parameters
    ----------
    total_cells:
        Total number of cells in the factorial grid.
    worker_id:
        Zero-based worker index (0 <= worker_id < total_workers).
    total_workers:
        Total number of parallel workers.

    Returns
    -------
    Sorted list of cell indices assigned to this worker.

    Raises
    ------
    ValueError
        If worker_id >= total_workers or worker_id < 0.
    """
    if worker_id < 0 or worker_id >= total_workers:
        msg = (
            f"worker_id={worker_id} out of range: "
            f"must be 0 <= worker_id < total_workers={total_workers}"
        )
        raise ValueError(msg)

    return [i for i in range(total_cells) if i % total_workers == worker_id]


def expand_grid_cell(
    *,
    cell_index: int,
    factors: dict[str, list[Any]],
) -> dict[str, Any]:
    """Expand a cell index to a hyperparameter dict via mixed-radix decomposition.

    Given factors {"model": ["a","b"], "loss": ["x","y","z"]}, cell_index=0
    maps to {"model": "a", "loss": "x"}, cell_index=1 to {"model": "a", "loss": "y"}, etc.

    The decomposition uses the factor order from dict iteration (Python 3.7+
    guarantees insertion order). The rightmost factor varies fastest.

    Parameters
    ----------
    cell_index:
        Zero-based index into the Cartesian product of all factors.
    factors:
        Ordered dict mapping factor names to their value lists.

    Returns
    -------
    Dict mapping factor names to the selected value for this cell.

    Raises
    ------
    ValueError
        If cell_index >= product of all factor sizes, or < 0.
    """
    if not factors:
        msg = "factors dict must not be empty"
        raise ValueError(msg)

    total_cells = math.prod(len(v) for v in factors.values())
    if cell_index < 0 or cell_index >= total_cells:
        msg = (
            f"cell_index={cell_index} out of range: "
            f"must be 0 <= cell_index < {total_cells}"
        )
        raise ValueError(msg)

    # Mixed-radix decomposition: rightmost factor varies fastest
    result: dict[str, Any] = {}
    remaining = cell_index
    factor_items = list(factors.items())

    # Iterate in reverse so rightmost factor is extracted first
    for name, values in reversed(factor_items):
        n_values = len(values)
        idx = remaining % n_values
        remaining //= n_values
        result[name] = values[idx]

    return result


def compute_factorial_size(factors: dict[str, list[Any]]) -> int:
    """Compute the total number of cells in a factorial grid.

    Parameters
    ----------
    factors:
        Dict mapping factor names to their value lists.

    Returns
    -------
    Product of all factor list lengths.
    """
    return math.prod(len(v) for v in factors.values())


# ---------------------------------------------------------------------------
# Provenance tag helpers (T1.5)
# ---------------------------------------------------------------------------


def compute_config_hash(config_path: Path) -> str:
    """Compute SHA-256 hash of a YAML config file.

    Used as an MLflow tag to link training runs to the exact grid config
    that produced them. Deterministic: same file content → same hash.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    64-character hex SHA-256 digest.
    """
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def get_git_sha() -> str:
    """Get the current git commit SHA.

    Returns ``"unknown"`` when git is not available (e.g., inside Docker
    runner images that don't ship git).

    Returns
    -------
    40-character hex SHA or ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def get_docker_image_digest() -> str:
    """Get the Docker image digest from DOCKER_IMAGE_DIGEST env var.

    The digest is set at build time in the Docker image or injected by
    the orchestration layer (SkyPilot, Docker Compose).

    Returns ``"unknown"`` when the env var is not set.

    Returns
    -------
    Docker image digest string (e.g., ``"sha256:abc123..."``).
    """
    return os.environ.get("DOCKER_IMAGE_DIGEST", "unknown")


def build_provenance_tags(config_path: Path) -> dict[str, str]:
    """Build provenance tags for MLflow runs.

    Returns a dict suitable for passing to ``mlflow.set_tags()`` or
    ``mlflow.start_run(tags=...)``.

    Parameters
    ----------
    config_path:
        Path to the factorial YAML config file.

    Returns
    -------
    Dict with keys: ``grid_config_hash``, ``git_sha``, ``docker_image_digest``.
    """
    return {
        "grid_config_hash": compute_config_hash(config_path),
        "git_sha": get_git_sha(),
        "docker_image_digest": get_docker_image_digest(),
    }
