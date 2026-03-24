"""DVC drift simulation setup — VesselNN batch partitioning (T-A2).

Partitions 12 VesselNN volumes into 6 batches of 2 for drift
simulation, generates config files, and produces git tag names.

Usage:
    config = generate_drift_simulation_config(seed=42)
    save_drift_simulation_config(config, Path("configs/splits/vesselnn_drift_simulation.json"))
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# VesselNN has 12 volumes, partitioned into 6 batches of 2
_N_VESSELNN_VOLUMES = 12
_N_BATCHES = 6
_BATCH_SIZE = 2


def partition_vesselnn_batches(
    volume_ids: list[str],
    *,
    seed: int,
) -> list[list[str]]:
    """Partition VesselNN volumes into 6 batches of 2.

    Parameters
    ----------
    volume_ids:
        List of 12 volume identifiers.
    seed:
        Random seed for deterministic partitioning.

    Returns
    -------
    List of 6 batches, each containing 2 volume IDs.

    Raises
    ------
    ValueError
        If the number of volumes is not 12.
    """
    if len(volume_ids) != _N_VESSELNN_VOLUMES:
        msg = f"Expected {_N_VESSELNN_VOLUMES} volumes, got {len(volume_ids)}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    shuffled = list(volume_ids)
    rng.shuffle(shuffled)

    batches = [
        shuffled[i : i + _BATCH_SIZE] for i in range(0, len(shuffled), _BATCH_SIZE)
    ]

    logger.info("Partitioned %d volumes into %d batches", len(volume_ids), len(batches))
    return batches


def generate_drift_simulation_config(*, seed: int) -> dict[str, Any]:
    """Generate a drift simulation config for VesselNN.

    Parameters
    ----------
    seed:
        Random seed for batch partitioning.

    Returns
    -------
    Config dict with batches, n_batches, seed, and volume_ids.
    """
    volume_ids = [f"vol_{i:03d}" for i in range(_N_VESSELNN_VOLUMES)]
    batches = partition_vesselnn_batches(volume_ids, seed=seed)

    return {
        "dataset": "vesselnn",
        "n_volumes": _N_VESSELNN_VOLUMES,
        "n_batches": _N_BATCHES,
        "batch_size": _BATCH_SIZE,
        "seed": seed,
        "batches": [
            {"batch_id": i, "volume_ids": batch} for i, batch in enumerate(batches)
        ],
    }


def save_drift_simulation_config(config: dict[str, Any], path: Path) -> Path:
    """Save drift simulation config to JSON.

    Parameters
    ----------
    config:
        Config dict from generate_drift_simulation_config().
    path:
        Output path for the JSON file.

    Returns
    -------
    Path to the saved config file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Drift simulation config saved to %s", path)
    return path


def load_drift_simulation_config(path: Path) -> dict[str, Any]:
    """Load drift simulation config from JSON.

    Parameters
    ----------
    path:
        Path to the JSON config file.

    Returns
    -------
    Config dict.
    """
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def generate_batch_tags() -> list[str]:
    """Generate git tag names for drift simulation batches.

    Returns
    -------
    List of 6 git tag strings (data/vesselnn/batch-1 through batch-6).
    """
    return [f"data/vesselnn/batch-{i + 1}" for i in range(_N_BATCHES)]
