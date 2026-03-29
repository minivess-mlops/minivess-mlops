"""Synthetic Generation Flow (Flow 7) — Prefect flow for synthetic volume generation (T-D6).

Orchestrates all synthetic generator adapters (vesselFM, VQ-VAE, VaMos,
VascuSynth, debug) via the registry. Saves output as .npy files and
logs metadata to MLflow.

Usage:
    prefect deployment run 'synthetic-generation/default' \\
        --params '{"method": "vesselFM_drand", "n_volumes": 10}'
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, task

from minivess.data.synthetic import generate_stack, list_generators
from minivess.orchestration.constants import FLOW_NAME_SYNTHETIC_GENERATION
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context

logger = logging.getLogger(__name__)


@task(name="generate-synthetic-volumes")
def generate_volumes_task(
    method: str,
    n_volumes: int,
    output_dir: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate synthetic volumes using a registered generator.

    Parameters
    ----------
    method:
        Generator method name (e.g., 'vesselFM_drand', 'debug').
    n_volumes:
        Number of (image, mask) pairs to generate.
    output_dir:
        Directory to save generated .npy files.
    config:
        Optional method-specific config.

    Returns
    -------
    Dict with method, n_volumes, output_dir, and file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pairs = generate_stack(method=method, n_volumes=n_volumes, config=config)

    saved_files: list[str] = []
    for i, (image, mask) in enumerate(pairs):
        img_path = out_path / f"{method}_vol_{i:03d}_image.npy"
        mask_path = out_path / f"{method}_vol_{i:03d}_mask.npy"
        np.save(str(img_path), image)
        np.save(str(mask_path), mask)
        saved_files.extend([str(img_path), str(mask_path)])

    logger.info(
        "Generated %d %s volumes → %s",
        n_volumes,
        method,
        output_dir,
    )

    return {
        "method": method,
        "n_volumes": n_volumes,
        "output_dir": output_dir,
        "files": saved_files,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@flow(name=FLOW_NAME_SYNTHETIC_GENERATION)
def synthetic_generation_flow(
    method: str = "debug",
    n_volumes: int = 10,
    output_dir: str = "/checkpoints/synthetic",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthetic volume generation Prefect flow.

    Parameters
    ----------
    method:
        Generator method name. Use 'all' to run all registered methods.
    n_volumes:
        Number of volumes per method.
    output_dir:
        Base output directory.
    config:
        Optional per-method config.

    Returns
    -------
    Dict with status, results per method, and summary.
    """
    require_docker_context("synthetic-generation")

    results: list[dict[str, Any]] = []

    methods = list_generators() if method == "all" else [method]

    for m in methods:
        method_dir = str(Path(output_dir) / m)
        result = generate_volumes_task(
            method=m,
            n_volumes=n_volumes,
            output_dir=method_dir,
            config=config,
        )
        results.append(result)

    total_volumes = sum(r["n_volumes"] for r in results)
    logger.info(
        "Synthetic generation complete: %d volumes across %d methods",
        total_volumes,
        len(methods),
    )

    return {
        "status": "completed",
        "methods": methods,
        "total_volumes": total_volumes,
        "results": results,
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    synthetic_generation_flow()
