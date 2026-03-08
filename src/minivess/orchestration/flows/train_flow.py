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

import yaml
from prefect import flow, task

from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import FLOW_NAME_TRAIN
from minivess.orchestration.mlflow_helpers import (
    find_upstream_safely,
    log_completion_safe,
)
from minivess.pipeline.resume_discovery import (
    compute_config_fingerprint,
    find_completed_config,
    load_fold_result_from_mlflow,
)

logger = logging.getLogger(__name__)


def _require_docker_context() -> None:
    """Raise RuntimeError if not running inside a Docker container.

    Checks for /.dockerenv (standard Docker marker) or DOCKER_CONTAINER env var.
    Escape hatch: MINIVESS_ALLOW_HOST=1 for pytest only — never in scripts.

    See: docs/planning/minivess-vision-enforcement-plan.md (T-00)
    """
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        "Training flow must run inside a Docker container.\n"
        "Run: docker compose -f deployment/docker-compose.flows.yml run train\n"
        "Or for SAM3: docker compose -f deployment/docker-compose.flows.yml "
        "run -e MODEL_FAMILY=sam3_vanilla train\n\n"
        "Escape hatch (pytest ONLY): export MINIVESS_ALLOW_HOST=1\n"
        "See: docs/planning/minivess-vision-enforcement-plan.md"
    )


def _validate_training_env() -> None:
    """Validate required environment variables for training flow.

    Raises
    ------
    RuntimeError
        When SPLITS_DIR or CHECKPOINT_DIR is not set, with actionable instructions.
    """
    missing = [v for v in ("SPLITS_DIR", "CHECKPOINT_DIR") if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {missing}\n"
            "Set them before running the training flow:\n"
            "  export SPLITS_DIR=/path/to/configs/splits\n"
            "  export CHECKPOINT_DIR=/path/to/checkpoints\n"
            "Or configure them in your .env file."
        )


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


@task(name="check-resume-state")
def check_resume_state_task(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Check for epoch_latest.yaml to determine if this is a resumed run.

    Uses yaml.safe_load() exclusively — NO regex (CLAUDE.md Rule #16).

    Parameters
    ----------
    checkpoint_dir:
        Directory where epoch_latest.yaml would be written by SegmentationTrainer.

    Returns
    -------
    State dict if a valid RUNNING MLflow run is found, None otherwise.
    """
    latest_path = checkpoint_dir / "epoch_latest.yaml"
    if not latest_path.exists():
        return None

    with latest_path.open(encoding="utf-8") as f:
        state = yaml.safe_load(f)

    if not isinstance(state, dict):
        logger.warning("epoch_latest.yaml is not a dict — ignoring")
        return None

    run_id = state.get("mlflow_run_id")
    if not run_id:
        return None

    # Validate the referenced MLflow run is still RUNNING
    try:
        import mlflow

        run = mlflow.get_run(run_id)
        if run.info.status == "RUNNING":
            return state
    except Exception:
        logger.debug("Could not fetch MLflow run %s — treating as stale", run_id)

    return None


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
    _is_sam3 = model_family_str.startswith("sam3_")
    # SAM3: each z-slice goes through ViT-32L encoder at 1008×1008 — limit depth
    # to 3 slices and batch=1 to fit on 8 GB GPUs. MONAI's RandCropByPosNegLabeld
    # with num_samples=4 produces 4 crops, so effective batch is 4× — this is fine
    # with small patch depth.
    default_patch = (64, 64, 3) if _is_sam3 else (64, 64, 16)
    patch_size: tuple[int, int, int] = tuple(  # type: ignore[assignment]
        config.get("patch_size", default_patch)
    )
    if _is_sam3:
        batch_size = min(batch_size, 1)
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
    # SAM3: validation with sliding_window_inference is very slow (~2h per epoch
    # on RTX 2070 Super) because each window requires 3 encoder forward passes
    # through the 454M-param ViT-32L. Validate every 10 epochs to keep training
    # practical (50 epochs × 3 folds ≈ 16h with val_interval=10, vs 300h without).
    val_interval = 10 if _is_sam3 and not debug else 1
    training_config = TrainingConfig(
        max_epochs=1 if debug else max_epochs,
        num_folds=num_folds,
        batch_size=batch_size,
        warmup_epochs=0 if debug else 5,
        early_stopping_patience=1 if debug else 20,
        val_interval=val_interval,
    )

    # Extract volume dicts from the fold split (FoldSplit dataclass or dict)
    if hasattr(fold_split, "train"):
        train_dicts: list[Any] = fold_split.train  # type: ignore[attr-defined]
        val_dicts: list[Any] = fold_split.val  # type: ignore[attr-defined]
    else:
        train_dicts = fold_split.get("train", [])
        val_dicts = fold_split.get("val", [])

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

    # Create ExperimentTracker for per-fold MLflow epoch-level logging
    from minivess.config.models import ExperimentConfig
    from minivess.observability.tracking import ExperimentTracker

    tracking_uri: str = config.get("tracking_uri", "mlruns")
    experiment_name: str = config.get("experiment_name", "minivess_training")
    exp_config = ExperimentConfig(
        experiment_name=experiment_name,
        run_name=f"fold_{fold_id}_{loss_name}",
        data=data_config,
        model=model_config,
        training=training_config,
    )
    tracker = ExperimentTracker(exp_config, tracking_uri=tracking_uri)

    _is_sam3 = model_family_str.startswith("sam3_")
    # SAM3 validation: use full-slice ROI (512,512,3) instead of training
    # patch (64,64,3). The ViT-32L encoder always resizes to 1008×1008
    # regardless of input size, so larger patches cost the same per-window
    # but reduce window count by ~121× (11×11 spatial grid eliminated).
    # sw_batch_size=1 for SAM3 to keep VRAM low with large validation patches.
    val_roi = (512, 512, 3) if _is_sam3 else data_config.patch_size
    val_sw_batch = 1 if _is_sam3 else 4
    trainer = SegmentationTrainer(
        model,
        training_config,
        device=device_str,
        metrics=metrics,
        criterion=criterion,
        val_roi_size=val_roi,
        sw_batch_size=val_sw_batch,
        fold_label=f"f #{fold_id + 1}/{num_folds}",
        tracker=tracker,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Compute config fingerprint for auto-resume discovery
    from minivess.pipeline.resume_discovery import compute_config_fingerprint as _fp

    fingerprint = _fp(
        loss_name=loss_name,
        model_family=model_family_str,
        fold_id=fold_id,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patch_size=patch_size,
    )

    with tracker.start_run(
        tags={
            "fold_id": str(fold_id),
            "loss_name": loss_name,
            "config_fingerprint": fingerprint,
        }
    ):
        return trainer.fit(
            train_loader, val_loader, checkpoint_dir=checkpoint_dir, fold_id=fold_id
        )


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


@flow(name=FLOW_NAME_TRAIN)
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
    upstream_data_run_id: str | None = None,
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

    # Preflight: Docker context gate (CLAUDE.md Rule #19 — STOP Protocol)
    _require_docker_context()

    # Preflight: validate required environment variables
    _validate_training_env()

    # Resolve environment variables
    splits_dir = Path(os.environ.get("SPLITS_DIR", "configs/splits"))
    checkpoint_base = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
    tracking_uri = resolve_tracking_uri()

    # Find upstream data run (explicit param takes priority over auto-discovery)
    if upstream_data_run_id is None:
        upstream = find_upstream_safely(
            tracking_uri=tracking_uri,
            experiment_name="minivess_data",
            upstream_flow="data",
        )
        upstream_data_run_id = upstream["run_id"] if upstream else None
    if upstream_data_run_id:
        logger.info("Upstream data run: %s", upstream_data_run_id)

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
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
    }

    # Train each fold, skipping already-completed configs (auto-resume)
    fold_results: list[dict[str, Any]] = []
    for fold_id, fold_split in enumerate(folds_to_run):
        # Check for already-completed run with matching config fingerprint
        fingerprint = compute_config_fingerprint(
            loss_name=loss_name,
            model_family=model_family,
            fold_id=fold_id,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        existing_run_id = find_completed_config(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            config_fingerprint=fingerprint,
        )
        if existing_run_id is not None:
            logger.info(
                "Fold %d: resuming from completed run %s", fold_id, existing_run_id
            )
            fold_result = load_fold_result_from_mlflow(tracking_uri, existing_run_id)
            fold_results.append(fold_result)
            continue

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
                "flow_name": FLOW_NAME_TRAIN,
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

    except Exception:
        logger.warning("Failed to open/finalize MLflow run", exc_info=True)

    # Log flow completion tag (best-effort, non-blocking)
    log_completion_safe(
        flow_name=FLOW_NAME_TRAIN,
        tracking_uri=tracking_uri,
        run_id=mlflow_run_id,
    )

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Prefect flow")
    parser.add_argument(
        "--model-family", default=os.environ.get("MODEL_FAMILY", "dynunet")
    )
    parser.add_argument(
        "--loss-name", default=os.environ.get("LOSS_NAME", "cbdice_cldice")
    )
    parser.add_argument(
        "--max-epochs", type=int, default=int(os.environ.get("MAX_EPOCHS", "100"))
    )
    parser.add_argument(
        "--num-folds", type=int, default=int(os.environ.get("NUM_FOLDS", "3"))
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "2"))
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("EXPERIMENT_NAME", "minivess_training"),
    )
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG", "") == "1"
    )
    args = parser.parse_args()

    training_flow(
        model_family=args.model_family,
        loss_name=args.loss_name,
        max_epochs=args.max_epochs,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        debug=args.debug,
    )
