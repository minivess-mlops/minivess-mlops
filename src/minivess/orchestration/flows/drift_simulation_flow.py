"""Drift Simulation Flow (Flow 6) — automated drift detection on VesselNN batches (T-E1).

Runs drift detection on sequential batches against a reference distribution.
Each batch represents a temporal slice of data accumulation, simulating
gradual distribution shift.

Usage:
    prefect deployment run 'drift-simulation/default' \\
        --params '{"config_path": "configs/splits/vesselnn_drift_simulation.json"}'
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
from prefect import flow, task

from minivess.data.feature_extraction import extract_batch_features
from minivess.observability.drift import FeatureDriftDetector
from minivess.orchestration.constants import FLOW_NAME_DRIFT_SIMULATION
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context

logger = logging.getLogger(__name__)


@task(name="extract-reference-features")
def extract_reference_features_task(
    reference_volumes: list[np.ndarray],
) -> dict[str, Any]:
    """Extract features from reference volumes.

    Parameters
    ----------
    reference_volumes:
        List of 3D numpy arrays as the reference distribution.

    Returns
    -------
    Dict with 'features' (DataFrame as dict) and 'n_volumes'.
    """
    features_df = extract_batch_features(reference_volumes)

    logger.info("Extracted features from %d reference volumes", len(reference_volumes))
    return {
        "features": features_df.to_dict(),
        "n_volumes": len(reference_volumes),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@task(name="run-batch-drift-detection")
def run_batch_drift_task(
    batch_volumes: list[np.ndarray],
    reference_features: dict[str, Any],
    batch_id: int,
) -> dict[str, Any]:
    """Run drift detection on a single batch.

    Parameters
    ----------
    batch_volumes:
        List of 3D numpy arrays for the current batch.
    reference_features:
        Reference features dict (from extract_reference_features_task).
    batch_id:
        Sequential batch identifier.

    Returns
    -------
    Dict with drift detection results.
    """
    ref_df = pd.DataFrame(reference_features)
    cur_df = extract_batch_features(batch_volumes)

    detector = FeatureDriftDetector(ref_df)
    result = detector.detect(cur_df)

    logger.info(
        "Batch %d: drift=%s, score=%.3f, drifted=%d/%d",
        batch_id,
        result.drift_detected,
        result.dataset_drift_score,
        result.n_drifted,
        result.n_features,
    )

    return {
        "batch_id": batch_id,
        "drift_detected": result.drift_detected,
        "dataset_drift_score": result.dataset_drift_score,
        "n_drifted": result.n_drifted,
        "n_features": result.n_features,
        "drifted_features": result.drifted_features,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@task(name="save-drift-summary")
def save_drift_summary_task(
    batch_results: list[dict[str, Any]],
    output_dir: str,
) -> str:
    """Save drift simulation summary to JSON.

    Parameters
    ----------
    batch_results:
        List of per-batch drift results.
    output_dir:
        Directory to save the summary.

    Returns
    -------
    Path to the saved summary file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_path = out_path / "drift_simulation_summary.json"
    summary = {
        "n_batches": len(batch_results),
        "batches": batch_results,
        "any_drift_detected": any(r["drift_detected"] for r in batch_results),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("Drift summary saved to %s", summary_path)
    return str(summary_path)


@flow(name=FLOW_NAME_DRIFT_SIMULATION)
def drift_simulation_flow(
    reference_volumes: list[np.ndarray] | None = None,
    batches: list[list[np.ndarray]] | None = None,
    output_dir: str = "/logs/drift_reports",
) -> dict[str, Any]:
    """Drift simulation Prefect flow.

    Runs sequential drift detection on batches against a reference
    distribution. Designed for VesselNN 6-batch simulation.

    Parameters
    ----------
    reference_volumes:
        Reference distribution volumes.
    batches:
        List of batches, each a list of 3D numpy arrays.
    output_dir:
        Directory for drift reports.

    Returns
    -------
    Dict with status, n_batches, batch_results, and summary path.
    """
    require_docker_context("drift-simulation")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("drift-simulation", logs_dir=logs_dir) as event_logger:
        if reference_volumes is None or batches is None:
            logger.warning("No data provided — returning empty result")
            return {
                "status": "completed",
                "n_batches": 0,
                "batch_results": [],
                "summary_path": None,
            }

        # Step 1: Extract reference features
        ref_result = extract_reference_features_task(reference_volumes)

        # Step 2: Run drift detection on each batch
        batch_results: list[dict[str, Any]] = []
        for i, batch in enumerate(batches):
            result = run_batch_drift_task(
                batch_volumes=batch,
                reference_features=ref_result["features"],
                batch_id=i,
            )
            batch_results.append(result)

        # Step 3: Save summary
        summary_path = save_drift_summary_task(batch_results, output_dir)

        n_drifted_batches = sum(1 for r in batch_results if r["drift_detected"])
        logger.info(
            "Drift simulation complete: %d/%d batches drifted",
            n_drifted_batches,
            len(batches),
        )

        return {
            "status": "completed",
            "n_batches": len(batches),
            "n_drifted_batches": n_drifted_batches,
            "batch_results": batch_results,
            "summary_path": summary_path,
        }


if __name__ == "__main__":
    drift_simulation_flow()
