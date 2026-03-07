"""Auto-resume: discover already-completed training configurations in MLflow.

When running a sweep (e.g., 128 configs), the training flow checks MLflow
for existing FINISHED runs with matching config fingerprints and skips them.

Config fingerprint = deterministic hash of:
  loss_name, model_family, fold_id, max_epochs, batch_size, patch_size
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_config_fingerprint(
    loss_name: str,
    model_family: str,
    fold_id: int,
    max_epochs: int,
    batch_size: int,
    patch_size: tuple[int, ...] | None = None,
) -> str:
    """Compute a deterministic fingerprint for a training configuration.

    Parameters
    ----------
    loss_name:
        Loss function name.
    model_family:
        Model architecture family.
    fold_id:
        Cross-validation fold index.
    max_epochs:
        Maximum training epochs.
    batch_size:
        Training batch size.
    patch_size:
        Training patch dimensions (optional).

    Returns
    -------
    16-character hex hash of the config.
    """
    config_dict = {
        "loss_name": loss_name,
        "model_family": model_family,
        "fold_id": fold_id,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patch_size": list(patch_size) if patch_size else None,
    }
    content = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def find_completed_config(
    tracking_uri: str,
    experiment_name: str,
    config_fingerprint: str,
) -> str | None:
    """Find an existing FINISHED MLflow run matching this config fingerprint.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI.
    experiment_name:
        MLflow experiment to search.
    config_fingerprint:
        Config fingerprint to match against run tags.

    Returns
    -------
    run_id if found, None if this config needs training.
    """
    try:
        import mlflow
        from mlflow.entities import ViewType

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"tags.config_fingerprint = '{config_fingerprint}' "
                "and attributes.status = 'FINISHED'"
            ),
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            output_format="list",
        )

        if runs:
            run_id = runs[0].info.run_id
            logger.info(
                "Found completed run %s for fingerprint %s — skipping",
                run_id,
                config_fingerprint,
            )
            return str(run_id)

    except Exception:
        logger.debug(
            "MLflow resume discovery failed for fingerprint %s",
            config_fingerprint,
            exc_info=True,
        )

    return None


def load_fold_result_from_mlflow(
    tracking_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Load fold result metrics from an existing MLflow run.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI.
    run_id:
        MLflow run ID to load results from.

    Returns
    -------
    Dict with fold results (best_val_loss, metrics, etc.).
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)

        return {
            "run_id": run_id,
            "status": "resumed",
            "best_val_loss": float(run.data.metrics.get("best_val_loss", float("inf"))),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }

    except Exception:
        logger.warning("Failed to load fold result from run %s", run_id, exc_info=True)
        return {
            "run_id": run_id,
            "status": "resume_failed",
            "best_val_loss": float("inf"),
        }
