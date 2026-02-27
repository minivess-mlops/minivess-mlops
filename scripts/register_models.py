"""Post-hoc pyfunc model registration for production checkpoints.

Registers existing training checkpoints as MLflow pyfunc models so they
can be loaded via ``mlflow.pyfunc.load_model()`` and served for inference.

This script bridges the gap between training (which saves ``.pth``
checkpoints) and the MLflow Model Registry (which expects pyfunc
artifacts).

Usage::

    # Register all production checkpoints (default: best_val_compound_masd_cldice.pth)
    uv run python scripts/register_models.py

    # Register specific checkpoint types
    uv run python scripts/register_models.py --checkpoints best_val_dice.pth last.pth

    # Custom target experiment name
    uv run python scripts/register_models.py --experiment minivess_registered_models

References
----------
* MLflow pyfunc: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
* MiniVessSegModel: src/minivess/serving/mlflow_wrapper.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc

from minivess.pipeline.mlruns_inspector import (
    get_production_runs,
    get_run_tags,
)
from minivess.serving.mlflow_wrapper import (
    MiniVessSegModel,
    get_model_signature,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_EXPERIMENT_ID: str = "843896622863223169"
_DEFAULT_TARGET_EXPERIMENT: str = "minivess_registered_models"
_DEFAULT_CHECKPOINT_NAMES: list[str] = [
    "best_val_compound_masd_cldice.pth",
]
_DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "family": "dynunet",
    "name": "dynunet",
    "in_channels": 1,
    "out_channels": 2,
    "architecture_params": {
        "filters": [32, 64, 128, 256],
    },
}


# ---------------------------------------------------------------------------
# Core registration functions
# ---------------------------------------------------------------------------


def register_checkpoint_as_pyfunc(
    *,
    checkpoint_path: Path,
    model_config: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    loss_type: str,
    checkpoint_name: str,
    tags: dict[str, str] | None = None,
) -> Any:
    """Register a single checkpoint as an MLflow pyfunc model.

    Creates a new MLflow run in the target experiment, logs the checkpoint
    as a pyfunc artifact using :class:`MiniVessSegModel`, and returns the
    ``ModelInfo`` object.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint file.
    model_config:
        Model architecture configuration dict (serialized to JSON as artifact).
    tracking_uri:
        MLflow tracking URI (file path or server URL).
    experiment_name:
        Name of the MLflow experiment to create the run in.
    run_name:
        Human-readable name for the MLflow run.
    loss_type:
        Loss function name (logged as tag).
    checkpoint_name:
        Checkpoint filename without path (logged as tag).
    tags:
        Additional tags to set on the MLflow run.

    Returns
    -------
    ``mlflow.models.model.ModelInfo`` from ``mlflow.pyfunc.log_model``.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Write model config to a temp JSON file
    config_json_path = Path(tempfile.mktemp(suffix=".json", prefix="model_config_"))
    config_json_path.write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )

    run_tags: dict[str, str] = {
        "loss_type": loss_type,
        "checkpoint_name": checkpoint_name,
        "source_checkpoint": str(checkpoint_path),
        "registration_type": "post_hoc",
    }
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=run_tags):
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MiniVessSegModel(),
            artifacts={
                "checkpoint": str(checkpoint_path),
                "model_config": str(config_json_path),
            },
            signature=get_model_signature(),
        )

        logger.info(
            "Registered checkpoint %s as pyfunc model: %s",
            checkpoint_path.name,
            model_info.model_uri,
        )

    return model_info


def register_all_production_checkpoints(
    *,
    source_mlruns_dir: Path,
    experiment_id: str,
    model_config: dict[str, Any],
    tracking_uri: str,
    target_experiment_name: str,
    checkpoint_names: list[str] | None = None,
) -> list[Any]:
    """Register all production checkpoints as pyfunc models.

    Discovers production runs in the source MLflow experiment, then
    registers the specified checkpoints from each run.

    Parameters
    ----------
    source_mlruns_dir:
        Root mlruns directory containing the source experiment.
    experiment_id:
        MLflow experiment ID string for the source training experiment.
    model_config:
        Model architecture configuration dict.
    tracking_uri:
        MLflow tracking URI for the target (registration) backend.
    target_experiment_name:
        Name of the target experiment to create registered models in.
    checkpoint_names:
        List of checkpoint filenames to register per run.
        Defaults to ``["best_val_compound_masd_cldice.pth"]``.

    Returns
    -------
    List of ``ModelInfo`` objects, one per registered checkpoint.
    """
    if checkpoint_names is None:
        checkpoint_names = list(_DEFAULT_CHECKPOINT_NAMES)

    production_runs = get_production_runs(source_mlruns_dir, experiment_id)
    if not production_runs:
        logger.warning(
            "No production runs found in %s/%s",
            source_mlruns_dir,
            experiment_id,
        )
        return []

    logger.info(
        "Found %d production runs, registering %d checkpoint(s) each",
        len(production_runs),
        len(checkpoint_names),
    )

    results: list[Any] = []
    for run_id in production_runs:
        tags = get_run_tags(source_mlruns_dir, experiment_id, run_id)
        loss_type = tags.get("loss_function", "unknown")

        checkpoints_dir = (
            source_mlruns_dir / experiment_id / run_id / "artifacts" / "checkpoints"
        )

        for ckpt_name in checkpoint_names:
            ckpt_path = checkpoints_dir / ckpt_name
            if not ckpt_path.is_file():
                logger.warning(
                    "Checkpoint not found: %s (run %s, loss=%s)",
                    ckpt_path,
                    run_id,
                    loss_type,
                )
                continue

            # Derive a clean checkpoint label (without .pth extension)
            ckpt_label = ckpt_name.removesuffix(".pth")
            run_name = f"{loss_type}_{ckpt_label}"

            model_info = register_checkpoint_as_pyfunc(
                checkpoint_path=ckpt_path,
                model_config=model_config,
                tracking_uri=tracking_uri,
                experiment_name=target_experiment_name,
                run_name=run_name,
                loss_type=loss_type,
                checkpoint_name=ckpt_name,
                tags={
                    "source_run_id": run_id,
                    "source_experiment_id": experiment_id,
                },
            )
            results.append(model_info)

    logger.info(
        "Registered %d pyfunc models in experiment '%s'",
        len(results),
        target_experiment_name,
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Register production checkpoints as MLflow pyfunc models.",
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "mlruns",
        help="Root mlruns directory (default: repo_root/mlruns)",
    )
    parser.add_argument(
        "--experiment-id",
        default=_DEFAULT_EXPERIMENT_ID,
        help=f"Source MLflow experiment ID (default: {_DEFAULT_EXPERIMENT_ID})",
    )
    parser.add_argument(
        "--target-experiment",
        default=_DEFAULT_TARGET_EXPERIMENT,
        help=f"Target experiment name (default: {_DEFAULT_TARGET_EXPERIMENT})",
    )
    parser.add_argument(
        "--tracking-uri",
        default="mlruns",
        help="MLflow tracking URI for registration (default: mlruns)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=_DEFAULT_CHECKPOINT_NAMES,
        help="Checkpoint filenames to register (default: best_val_compound_masd_cldice.pth)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for checkpoint registration."""
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mlruns_dir: Path = args.mlruns_dir
    if not mlruns_dir.is_dir():
        logger.error("mlruns directory not found: %s", mlruns_dir)
        return 1

    experiment_dir = mlruns_dir / args.experiment_id
    if not experiment_dir.is_dir():
        logger.error("Experiment directory not found: %s", experiment_dir)
        return 1

    results = register_all_production_checkpoints(
        source_mlruns_dir=mlruns_dir,
        experiment_id=args.experiment_id,
        model_config=_DEFAULT_MODEL_CONFIG,
        tracking_uri=args.tracking_uri,
        target_experiment_name=args.target_experiment,
        checkpoint_names=args.checkpoints,
    )

    if not results:
        logger.warning("No models were registered")
        return 1

    logger.info("Successfully registered %d models", len(results))
    for info in results:
        logger.info("  Model URI: %s", info.model_uri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
