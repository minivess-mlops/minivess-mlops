"""Training Prefect flow — model training as a managed flow.

Flow 2: Wraps the training logic from scripts/train_monitored.py
as a Prefect @flow with tasks for each phase (data loading, training,
evaluation, checkpointing).

This enables:
- Prefect UI visibility into training progress
- Work pool routing (GPU pool)
- Retry and failure handling
- Integration with the trigger chain
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minivess.orchestration._prefect_compat import flow, task

logger = logging.getLogger(__name__)


@dataclass
class TrainingFlowResult:
    """Result returned by training_flow().

    Replaces the plain-dict stub pattern. Downstream flows can consume
    this via MLflow tags (mlflow_run_id, upstream_data_run_id).
    """

    flow_name: str = "train"
    status: str = "completed"
    mlflow_run_id: str | None = None
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    upstream_data_run_id: str | None = None
    n_folds: int = 0


# ---------------------------------------------------------------------------
# Focused tasks
# ---------------------------------------------------------------------------


@task(name="load-fold-splits")
def load_fold_splits_task(splits_dir: Path) -> list[dict[str, Any]]:
    """Load fold splits from SPLITS_DIR/splits.json.

    Parameters
    ----------
    splits_dir:
        Directory containing splits.json (from SPLITS_DIR env var).

    Returns
    -------
    List of fold dicts, each with 'train' and 'val' lists.

    Raises
    ------
    FileNotFoundError
        When splits.json does not exist. Run the data flow first.
    """
    from minivess.data.splits import load_splits

    splits_file = splits_dir / "splits.json"
    if not splits_file.exists():
        msg = (
            f"No splits.json at {splits_file}. "
            "Run the data flow first to generate fold splits."
        )
        raise FileNotFoundError(msg)
    return load_splits(splits_file)  # type: ignore[return-value]


@task(name="train-one-fold")
def train_one_fold_task(
    fold_id: int,
    fold_split: dict[str, Any],
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Train one fold and return results.

    Builds DataLoaders, model, loss, and SegmentationTrainer from config,
    then calls trainer.fit() with the given checkpoint directory.

    Parameters
    ----------
    fold_id:
        Zero-based fold index.
    fold_split:
        Dict with 'train' and 'val' lists of volume dicts.
    config:
        Flat training config dict (loss_name, model_family, max_epochs, etc.).
    checkpoint_dir:
        Where to save fold checkpoints (must be on a persistent volume).

    Returns
    -------
    Fold result dict from SegmentationTrainer.fit().
    """
    import torch

    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import (
        DataConfig,
        ModelConfig,
        ModelFamily,
        TrainingConfig,
    )
    from minivess.data.loader import build_train_loader, build_val_loader
    from minivess.pipeline.loss_functions import build_loss_function
    from minivess.pipeline.metrics import SegmentationMetrics
    from minivess.pipeline.trainer import SegmentationTrainer

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    loss_name: str = config.get("loss_name", "cbdice_cldice")
    debug: bool = config.get("debug", False)
    max_epochs: int = config.get("max_epochs", 100)
    num_folds: int = config.get("num_folds", 3)
    batch_size: int = config.get("batch_size", 2)
    model_family_str: str = config.get("model_family", "dynunet")
    cache_rate: float = 0.0 if debug else 0.5

    # Build DataConfig (dataset_name required, cache_rate passed to loaders)
    patch_size: tuple[int, int, int] = tuple(  # type: ignore[assignment]
        config.get("patch_size", (64, 64, 16))
    )
    data_dir: Path = Path(config.get("data_dir", "data/raw"))
    num_workers: int = 0 if debug else config.get("num_workers", 4)

    data_config = DataConfig(
        dataset_name=config.get("dataset_name", "minivess"),
        data_dir=data_dir,
        patch_size=patch_size,
        num_workers=num_workers,
    )

    # Build ModelConfig (family must be ModelFamily enum)
    model_family = ModelFamily(model_family_str)
    arch_params: dict[str, Any] = dict(config.get("architecture_params", {}))
    model_config = ModelConfig(
        family=model_family,
        name=model_family_str,
        in_channels=1,
        out_channels=2,
        architecture_params=arch_params,
    )

    # Build TrainingConfig
    training_config = TrainingConfig(
        max_epochs=1 if debug else max_epochs,
        num_folds=num_folds,
        batch_size=batch_size,
        warmup_epochs=0 if debug else 5,
        early_stopping_patience=1 if debug else 20,
    )

    # Extract volume dicts from the fold split
    train_dicts: list[Any] = fold_split.get("train", [])
    val_dicts: list[Any] = fold_split.get("val", [])

    if debug:
        from minivess.config.debug import DEBUG_MAX_VOLUMES

        train_dicts = train_dicts[:DEBUG_MAX_VOLUMES]
        val_dicts = val_dicts[: max(2, DEBUG_MAX_VOLUMES // 3)]

    logger.info(
        "Fold %d: loss=%s, device=%s, train=%d, val=%d",
        fold_id,
        loss_name,
        device_str,
        len(train_dicts),
        len(val_dicts),
    )

    train_loader = build_train_loader(
        train_dicts,
        data_config,
        batch_size=training_config.batch_size,
        cache_rate=cache_rate,
    )
    val_loader = build_val_loader(
        val_dicts,
        data_config,
        cache_rate=cache_rate,
    )

    model = build_adapter(model_config)
    criterion = build_loss_function(loss_name)
    metrics = SegmentationMetrics(
        num_classes=model_config.out_channels,
        device=device_str,
    )

    _is_sam3 = model_family_str.startswith("sam3_")
    trainer = SegmentationTrainer(
        model,
        training_config,
        device=device_str,
        metrics=metrics,
        criterion=criterion,
        val_roi_size=data_config.patch_size,
        sw_batch_size=1 if _is_sam3 else 4,
        fold_label=f"f #{fold_id + 1}/{num_folds}",
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return trainer.fit(train_loader, val_loader, checkpoint_dir=checkpoint_dir)


@task(name="log-fold-results")
def log_fold_results_task(
    fold_id: int,
    result: dict[str, Any],
    mlflow_run_id: str | None,
) -> None:
    """Log fold training results to MLflow.

    Parameters
    ----------
    fold_id:
        Zero-based fold index.
    result:
        Fold result dict from SegmentationTrainer.fit().
    mlflow_run_id:
        Active MLflow run ID to log into.
    """
    if mlflow_run_id is None:
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        best_val_loss = result.get("best_val_loss", float("nan"))
        final_epoch = result.get("final_epoch", 0)
        client.log_metric(mlflow_run_id, f"fold_{fold_id}_best_val_loss", best_val_loss)
        client.log_metric(
            mlflow_run_id, f"fold_{fold_id}_final_epoch", float(final_epoch)
        )
        # Log per-epoch val_loss history for the fold
        for epoch, val_loss in enumerate(
            result.get("history", {}).get("val_loss", []), start=1
        ):
            client.log_metric(
                mlflow_run_id, f"fold_{fold_id}_val_loss", val_loss, step=epoch
            )
        # Also log val_loss (without fold prefix) for cross-fold visibility
        if result.get("history", {}).get("val_loss"):
            for epoch, val_loss in enumerate(result["history"]["val_loss"], start=1):
                client.log_metric(mlflow_run_id, "val_loss", val_loss, step=epoch)
    except Exception:
        logger.warning(
            "Failed to log fold %d results to MLflow", fold_id, exc_info=True
        )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="training-flow")
def training_flow(
    *,
    loss_name: str = "cbdice_cldice",
    model_family: str = "dynunet",
    compute: str = "auto",
    debug: bool = False,
    experiment_name: str = "minivess_training",
    trigger_source: str = "manual",
    num_folds: int = 3,
    max_epochs: int = 100,
    batch_size: int = 2,
    **kwargs: Any,
) -> TrainingFlowResult:
    """Training Prefect flow — orchestrates model training.

    Reads fold splits from SPLITS_DIR env var, trains each fold via
    train_one_fold_task(), and logs results to MLflow under the
    'minivess_training' experiment.

    Parameters
    ----------
    loss_name:
        Loss function name (e.g., 'cbdice_cldice', 'dice_ce').
    model_family:
        Model family string (e.g., 'dynunet', 'sam3_vanilla').
    compute:
        Compute profile name.
    debug:
        If True, use debug overrides (1 epoch, reduced data).
    experiment_name:
        MLflow experiment name.
    trigger_source:
        What triggered this flow (for logging).
    num_folds:
        Number of cross-validation folds.
    max_epochs:
        Maximum training epochs per fold.
    batch_size:
        Batch size per GPU.

    Returns
    -------
    TrainingFlowResult with fold results, MLflow run ID, and upstream link.
    """
    logger.info("Training flow started (trigger: %s)", trigger_source)

    # Resolve environment variables
    splits_dir = Path(os.environ.get("SPLITS_DIR", "configs/splits"))
    checkpoint_base = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

    # Find upstream data run
    upstream_data_run_id: str = "no_upstream"
    try:
        from minivess.orchestration.flow_contract import FlowContract

        contract = FlowContract(tracking_uri=tracking_uri)
        upstream = contract.find_upstream_run(
            experiment_name="minivess_data",
            upstream_flow="data",
        )
        if upstream:
            upstream_data_run_id = upstream["run_id"]
            logger.info("Upstream data run: %s", upstream_data_run_id)
    except Exception:
        logger.warning("Could not find upstream data run", exc_info=True)

    # Load fold splits (outside MLflow run — no side effects)
    splits = load_fold_splits_task(splits_dir)
    folds_to_run = splits[:num_folds]

    # Build per-fold config dict
    config: dict[str, Any] = {
        "loss_name": loss_name,
        "model_family": model_family,
        "compute": compute,
        "debug": debug,
        "max_epochs": max_epochs,
        "num_folds": num_folds,
        "batch_size": batch_size,
    }

    # Train each fold and collect results (outside MLflow context — avoids re-entry issues)
    fold_results: list[dict[str, Any]] = []
    for fold_id, fold_split in enumerate(folds_to_run):
        checkpoint_dir = checkpoint_base / f"fold_{fold_id}"
        fold_result = train_one_fold_task(fold_id, fold_split, config, checkpoint_dir)
        fold_results.append(fold_result)

    # Open MLflow run, log everything, then close cleanly
    mlflow_run_id: str | None = None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(
            tags={
                "flow_name": "train",
                "upstream_data_run_id": upstream_data_run_id,
                "loss_name": loss_name,
                "model_family": model_family,
            }
        ) as active_run:
            mlflow_run_id = active_run.info.run_id
            logger.info("MLflow run opened: %s", mlflow_run_id)

            # Log fold results inside the run context
            for fold_id, fold_result in enumerate(fold_results):
                log_fold_results_task(fold_id, fold_result, mlflow_run_id)

            mlflow.log_metric("n_folds_completed", float(len(fold_results)))

        # Log flow completion tag after run is closed
        from minivess.orchestration.flow_contract import FlowContract

        contract = FlowContract(tracking_uri=tracking_uri)
        contract.log_flow_completion(
            flow_name="train",
            run_id=mlflow_run_id,
        )
    except Exception:
        logger.warning("Failed to open/finalize MLflow run", exc_info=True)

    result = TrainingFlowResult(
        flow_name="train",
        status="completed",
        mlflow_run_id=mlflow_run_id,
        fold_results=fold_results,
        upstream_data_run_id=upstream_data_run_id,
        n_folds=len(fold_results),
    )
    logger.info(
        "Training flow complete: %d folds, run_id=%s",
        len(fold_results),
        mlflow_run_id,
    )
    return result


# ---------------------------------------------------------------------------
# Deprecated stub — kept for backward compatibility
# ---------------------------------------------------------------------------


@task(name="load-training-config")
def load_training_config(
    *,
    loss_name: str = "cbdice_cldice",
    model_family: str = "dynunet",
    compute: str = "auto",
    debug: bool = False,
    experiment_name: str = "dynunet_loss_variation",
) -> dict[str, Any]:
    """Load and validate training configuration.

    .. deprecated::
        Use training_flow() directly. This task is kept for backward
        compatibility only and will be removed in a future release.
    """
    config = {
        "loss_name": loss_name,
        "model_family": model_family,
        "compute": compute,
        "debug": debug,
        "experiment_name": experiment_name,
    }
    logger.info("Training config loaded: %s", config)
    return config


@task(name="run-training")
def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute model training.

    .. deprecated::
        This stub is no longer valid. Use training_flow() which calls
        train_one_fold_task() → SegmentationTrainer.fit() directly.
        See docs/planning/prefect-container-production-grade-hardening-plan.xml T-06.

    Raises
    ------
    NotImplementedError
        Always. This function is deprecated.
    """
    logger.warning(
        "run_training() is DEPRECATED. "
        "Use training_flow() which calls train_one_fold_task() directly. "
        "This stub previously returned: "
        "{'status': 'configured', 'message': 'Training flow configured. "
        "Use scripts/train_monitored.py for execution.'}. "
        "That message no longer reflects the system design. "
        "See T-06 in docs/planning/prefect-container-production-grade-hardening-plan.xml."
    )
    msg = (
        "run_training() is deprecated. "
        "Call training_flow() instead, which orchestrates SegmentationTrainer.fit()."
    )
    raise NotImplementedError(msg)
