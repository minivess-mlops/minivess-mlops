"""MLflow model artifact logger for segmentation models.

Logs single models and ensembles as ``mlflow.pyfunc`` artifacts with correct
signatures, checkpoints, and ensemble manifests.  This is the bridge between
the training loop (which saves ``.pth`` checkpoints) and the MLflow Model
Registry (which expects pyfunc artifacts).

References
----------
* MLflow pyfunc custom models: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

from minivess.serving.mlflow_wrapper import (
    MiniVessEnsembleModel,
    MiniVessSegModel,
    get_model_signature,
)

if TYPE_CHECKING:
    from minivess.ensemble.builder import EnsembleSpec

logger = logging.getLogger(__name__)


def create_ensemble_manifest(ensemble_spec: EnsembleSpec) -> dict[str, Any]:
    """Convert an EnsembleSpec to a JSON-serializable manifest dict.

    The manifest is consumed by
    :meth:`MiniVessEnsembleModel.load_context` to load all ensemble
    members from their checkpoint files.

    Parameters
    ----------
    ensemble_spec:
        Built ensemble specification with loaded members.

    Returns
    -------
    JSON-serializable dict with keys: ``name``, ``strategy``,
    ``n_members``, ``members``.
    """
    members_list: list[dict[str, Any]] = []
    for member in ensemble_spec.members:
        members_list.append(
            {
                "checkpoint_path": str(member.checkpoint_path),
                "run_id": member.run_id,
                "loss_type": member.loss_type,
                "fold_id": member.fold_id,
                "metric_name": member.metric_name,
            }
        )

    return {
        "name": ensemble_spec.name,
        "strategy": ensemble_spec.strategy.value,
        "n_members": len(ensemble_spec.members),
        "members": members_list,
    }


def log_single_model(
    *,
    checkpoint_path: Path,
    model_config_dict: dict[str, Any],
    artifact_path: str = "model",
) -> Any:
    """Log a single segmentation model as an MLflow pyfunc artifact.

    Calls ``mlflow.pyfunc.log_model()`` with :class:`MiniVessSegModel`,
    the checkpoint file, and a JSON-serialized model config as artifacts.

    Must be called within an active ``mlflow.start_run()`` context.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint file.
    model_config_dict:
        Model architecture configuration dict (serialized to JSON).
    artifact_path:
        MLflow artifact path for the logged model.

    Returns
    -------
    ``mlflow.models.model.ModelInfo`` from ``log_model``.
    """
    # Write model config to a temp JSON file for MLflow artifact tracking
    config_json_path = Path(tempfile.mktemp(suffix=".json", prefix="model_config_"))
    config_json_path.write_text(
        json.dumps(model_config_dict, indent=2),
        encoding="utf-8",
    )

    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=MiniVessSegModel(),
        artifacts={
            "checkpoint": str(checkpoint_path),
            "model_config": str(config_json_path),
        },
        signature=get_model_signature(),
    )

    logger.info(
        "Logged single model from %s as pyfunc artifact '%s'",
        checkpoint_path,
        artifact_path,
    )
    return model_info


def log_ensemble_model(
    *,
    ensemble_spec: EnsembleSpec,
    model_config_dict: dict[str, Any],
    artifact_path: str = "ensemble_model",
) -> Any:
    """Log an ensemble model as an MLflow pyfunc artifact.

    Creates an ensemble manifest JSON from the :class:`EnsembleSpec`,
    then calls ``mlflow.pyfunc.log_model()`` with
    :class:`MiniVessEnsembleModel`.

    Must be called within an active ``mlflow.start_run()`` context.

    Parameters
    ----------
    ensemble_spec:
        Built ensemble specification with loaded members.
    model_config_dict:
        Model architecture configuration dict (shared by all members).
    artifact_path:
        MLflow artifact path for the logged model.

    Returns
    -------
    ``mlflow.models.model.ModelInfo`` from ``log_model``.
    """
    # Create ensemble manifest JSON
    manifest = create_ensemble_manifest(ensemble_spec)
    manifest_path = Path(
        tempfile.mktemp(suffix=".json", prefix="ensemble_manifest_")
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Write model config JSON
    config_json_path = Path(
        tempfile.mktemp(suffix=".json", prefix="model_config_")
    )
    config_json_path.write_text(
        json.dumps(model_config_dict, indent=2),
        encoding="utf-8",
    )

    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=MiniVessEnsembleModel(),
        artifacts={
            "ensemble_manifest": str(manifest_path),
            "model_config": str(config_json_path),
        },
        signature=get_model_signature(),
    )

    logger.info(
        "Logged ensemble '%s' (%d members) as pyfunc artifact '%s'",
        ensemble_spec.name,
        len(ensemble_spec.members),
        artifact_path,
    )
    return model_info
