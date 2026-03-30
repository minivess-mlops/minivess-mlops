"""MLflow factorial tag utilities.

Retroactively tags MLflow runs with factorial metadata (with_aux_calib,
post_training_method, recalibration, ensemble_strategy, is_zero_shot)
and provides helpers for training flow tag logging.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# All factorial tags that must be present on every training run
_FACTORIAL_TAGS = (
    "loss_function",
    "model_family",
    "with_aux_calib",
    "post_training_method",
    "recalibration",
    "ensemble_strategy",
    "is_zero_shot",
)

# Default values for Layer B+C tags when not explicitly provided
_LAYER_BC_DEFAULTS: dict[str, str] = {
    "post_training_method": "none",
    "recalibration": "none",
    "ensemble_strategy": "none",
    "is_zero_shot": "false",
}


def set_factorial_tags(
    client: MlflowClient,
    config_map: dict[str, dict[str, Any]],
) -> None:
    """Apply factorial metadata tags to MLflow runs.

    Parameters
    ----------
    client:
        MLflow tracking client (local or remote).
    config_map:
        Mapping of run_id → config dict. Each config dict must have:
          - config_name: str — experiment config name (used to infer with_aux_calib)
          - loss_function: str
          - model_family: str
        And optionally:
          - post_training_method: str (default "none")
          - recalibration: str (default "none")
          - ensemble_strategy: str (default "none")
          - is_zero_shot: str (default "false")
    """
    for run_id, config in config_map.items():
        config_name = config["config_name"]

        # Infer with_aux_calib from config name
        has_aux = "auxcalib" in config_name.lower()

        tags: dict[str, str] = {
            "loss_function": config["loss_function"],
            "model_family": config["model_family"],
            "with_aux_calib": str(has_aux).lower(),
        }

        # Layer B+C: use provided values or defaults
        for tag_name, default_value in _LAYER_BC_DEFAULTS.items():
            tags[tag_name] = config.get(tag_name, default_value)

        # Apply all tags
        for tag_name, tag_value in tags.items():
            client.set_tag(run_id, tag_name, tag_value)

        logger.info(
            "Tagged run %s: loss=%s, aux_calib=%s, model=%s",
            run_id,
            tags["loss_function"],
            tags["with_aux_calib"],
            tags["model_family"],
        )


def build_training_tags(
    *,
    loss_function: str,
    model_family: str,
    with_aux_calib: bool,
    fold_id: int,
    post_training_method: str = "none",
    recalibration: str = "none",
    ensemble_strategy: str = "none",
    is_zero_shot: bool = False,
) -> dict[str, str]:
    """Build the complete set of factorial tags for a training run.

    Used by the training flow to log tags on new runs.

    Returns
    -------
    dict mapping tag name → string value, ready for mlflow.set_tag().
    """
    return {
        "loss_function": loss_function,
        "model_family": model_family,
        "with_aux_calib": str(with_aux_calib).lower(),
        "fold_id": str(fold_id),
        "post_training_method": post_training_method,
        "recalibration": recalibration,
        "ensemble_strategy": ensemble_strategy,
        "is_zero_shot": str(is_zero_shot).lower(),
    }
