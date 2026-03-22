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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import yaml
from prefect import flow, task

from minivess.compliance.audit import AuditTrail
from minivess.observability.infrastructure_timing import (
    estimate_cost_from_first_epoch,
    generate_timing_jsonl,
    get_hourly_rate_usd,
)
from minivess.observability.lineage import LineageEmitter, emit_flow_lineage
from minivess.observability.prometheus_metrics import update_estimated_cost_gauges
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    FLOW_NAME_POST_TRAINING_SUBFLOW,
    FLOW_NAME_TRAIN,
    FLOW_NAME_TRAINING_SUBFLOW,
)
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


def log_epoch0_cost_estimate(
    *,
    training_time_seconds: float,
    max_epochs: int,
    num_folds: int,
    hourly_rate_usd: float,
) -> None:
    """Log epoch-0 cost estimate to MLflow and update Prometheus gauges.

    Skipped when max_epochs <= 1 (no point predicting from the only epoch)
    or when training_time_seconds <= 0 (no timing data).

    Parameters
    ----------
    training_time_seconds:
        Total training time in seconds so far.
    max_epochs:
        Total planned epochs.
    num_folds:
        Number of cross-validation folds.
    hourly_rate_usd:
        Instance hourly cost in USD.
    """
    if max_epochs <= 1:
        return
    if training_time_seconds <= 0:
        return

    # Derive per-epoch time from total training time / max_epochs
    epoch_seconds = training_time_seconds / max_epochs

    estimate = estimate_cost_from_first_epoch(
        epoch_seconds=epoch_seconds,
        max_epochs=max_epochs,
        num_folds=num_folds,
        hourly_rate_usd=hourly_rate_usd,
    )

    mlflow.log_metrics(estimate, step=0)
    update_estimated_cost_gauges(estimate)
    logger.info(
        "Epoch-0 cost estimate: $%.4f total, %.2fh, $%.4f/epoch",
        estimate["est/total_cost"],
        estimate["est/total_hours"],
        estimate["est/cost_per_epoch"],
    )


def log_timing_jsonl_artifact(
    *,
    setup_durations: dict[str, float],
    training_seconds: float,
    epoch_count: int,
    hourly_rate_usd: float,
) -> None:
    """Generate timing JSONL and log it as an MLflow artifact.

    No-op when there is no timing data (setup_durations empty AND
    training_seconds <= 0).

    Parameters
    ----------
    setup_durations:
        Operation name -> duration in seconds (from parse_setup_timing).
    training_seconds:
        Total training time in seconds.
    epoch_count:
        Number of epochs completed.
    hourly_rate_usd:
        Instance hourly cost in USD.
    """
    if not setup_durations and training_seconds <= 0:
        return

    timing_jsonl = generate_timing_jsonl(
        setup_durations=setup_durations,
        training_seconds=training_seconds,
        epoch_count=epoch_count,
        hourly_rate_usd=hourly_rate_usd,
    )
    mlflow.log_text(timing_jsonl, "timing/timing_report.jsonl")
    logger.info("Logged timing JSONL artifact (%d bytes)", len(timing_jsonl))


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


def prepare_training_data(
    *,
    data_dir: Path | None = None,
    dvc_remote: str | None = None,
) -> dict[str, Any]:
    """Ensure training data is available, pulling from DVC if needed.

    Checks if DATA_DIR has MiniVess data files. If empty, runs
    ``dvc pull -r <remote>`` to fetch data from the configured remote.

    Parameters
    ----------
    data_dir:
        Data directory to check. Defaults to DATA_DIR env var.
    dvc_remote:
        DVC remote name. Defaults to DVC_REMOTE env var or "minio".

    Returns
    -------
    Dict with keys: ``pulled`` (bool), ``remote`` (str), ``duration_s`` (float).
    """
    import subprocess
    import time

    if data_dir is None:
        data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))

    remote = dvc_remote or os.environ.get("DVC_REMOTE", "minio")

    # Check if data already exists (volume-mounted in Docker)
    images_dir = data_dir / "raw" / "minivess" / "imagesTr"
    if images_dir.exists() and any(images_dir.iterdir()):
        logger.info("Training data found at %s — skipping DVC pull", images_dir)
        return {"pulled": False, "remote": remote, "duration_s": 0.0}

    # Data not found — pull from DVC
    logger.info("Training data not found. Pulling from DVC remote '%s'...", remote)
    t0 = time.monotonic()
    cmd = ["dvc", "pull", "-r", remote]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"DVC pull failed (exit {result.returncode}):\n{result.stderr}\n"
            f"Command: {' '.join(cmd)}\n"
            "Check DVC_S3_* variables in .env and run: "
            "uv run python scripts/configure_dvc_remote.py"
        )

    # Verify data was actually pulled
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(
            f"DVC pull completed but no data at {images_dir}. "
            "Verify DVC tracking file (data/minivess.dvc) is correct."
        )

    logger.info("DVC pull complete in %.1fs from remote '%s'", duration, remote)
    return {"pulled": True, "remote": remote, "duration_s": duration}


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
    loss_name: str = "cbdice_cldice"
    model_family: str = "dynunet"
    checkpoint_dirs: dict[int, Path] = field(default_factory=dict)
    # Phase 2: post-training metadata
    post_training_method: str = "none"
    post_training_run_id: str | None = None


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
    from minivess.diagnostics.pre_training_checks import run_pre_training_checks
    from minivess.diagnostics.weight_diagnostics import run_weightwatcher
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
    _is_sam3_hybrid = model_family_str == "sam3_hybrid"
    _is_vesselfm = model_family_str == "vesselfm"
    # SAM3: each z-slice goes through ViT-32L encoder at 1008×1008 — limit depth
    # to 3 slices and batch=1 to fit on 8 GB GPUs. MONAI's RandCropByPosNegLabeld
    # with num_samples=4 produces 4 crops, so effective batch is 4× — this is fine
    # with small patch depth.
    # VesselFM: 6-level DynUNet with 5 stride-2 downsamplings needs Z >= 2^5 = 32.
    # Z=16 causes skip connection shape mismatch at level 4 (#711).
    if _is_sam3:
        default_patch = (64, 64, 3)
    elif _is_vesselfm:
        default_patch = (64, 64, 32)
    else:
        default_patch = (64, 64, 16)
    # patch_size=null in config means "use model-adaptive default" (set above).
    _patch_raw = config.get("patch_size")
    patch_size: tuple[int, int, int] = (  # type: ignore[assignment]
        default_patch if _patch_raw is None else tuple(_patch_raw)
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
    # Validation interval strategy:
    # 1. If config explicitly sets val_interval, respect it (cloud smoke tests).
    # 2. Otherwise, apply model-based heuristics:
    #    sam3_hybrid debug: skip validation (OOM on 8 GB GPU).
    #    sam3 production: validate every 10 epochs (slow inference).
    #    Other models: validate every epoch.
    _config_val_interval = config.get("val_interval")
    if _config_val_interval is not None:
        val_interval = int(_config_val_interval)
    elif _is_sam3_hybrid and debug:
        val_interval = max_epochs + 1  # never validate: OOM on 8 GB GPU
    elif _is_sam3 and not debug:
        val_interval = 10  # sparse validation in production (slow inference)
    else:
        val_interval = 1
    # mixed_precision: respect config if set, default True.
    # MONAI maintainers confirm AMP + 3D ops can produce NaN
    # (Project-MONAI/MONAI#4243) — allow disabling per experiment.
    _mixed_precision = config.get("mixed_precision", True)
    training_config = TrainingConfig(
        max_epochs=max_epochs,
        num_folds=num_folds,
        batch_size=batch_size,
        warmup_epochs=0 if debug else 5,
        early_stopping_patience=1 if debug else 20,
        val_interval=val_interval,
        mixed_precision=bool(_mixed_precision),
    )

    # Extract volume dicts from the fold split (FoldSplit dataclass or dict)
    if hasattr(fold_split, "train"):
        train_dicts: list[Any] = fold_split.train  # type: ignore[attr-defined]
        val_dicts: list[Any] = fold_split.val  # type: ignore[attr-defined]
    else:
        train_dicts = fold_split.get("train", [])
        val_dicts = fold_split.get("val", [])

    # Config-driven data subsetting (max_train_volumes / max_val_volumes from Hydra)
    max_train_volumes: int | None = config.get("max_train_volumes")
    max_val_volumes: int | None = config.get("max_val_volumes")
    if max_train_volumes is not None:
        train_dicts = train_dicts[:max_train_volumes]
    elif debug:
        from minivess.config.debug import DEBUG_MAX_VOLUMES

        train_dicts = train_dicts[:DEBUG_MAX_VOLUMES]
    if max_val_volumes is not None:
        val_dicts = val_dicts[:max_val_volumes]
    elif debug:
        from minivess.config.debug import DEBUG_MAX_VOLUMES

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

    # Auxiliary calibration loss (hL1-ACE) — factorial factor c=2
    with_aux_calib: bool = config.get("with_aux_calib", False)
    aux_calib_weight: float = config.get("aux_calib_weight", 1.0)
    criterion = build_loss_function(
        loss_name,
        with_aux_calib=with_aux_calib,
        aux_calib_weight=aux_calib_weight,
    )
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
    _is_sam3_hybrid = model_family_str == "sam3_hybrid"
    # SAM3 validation: use full-slice ROI (512,512,3) instead of training
    # patch (64,64,3). The ViT-32L encoder always resizes to 1008×1008
    # regardless of input size, so larger patches cost the same per-window
    # but reduce window count by ~121× (11×11 spatial grid eliminated).
    # sw_batch_size=1 for SAM3 to keep VRAM low with large validation patches.
    #
    # sam3_hybrid exception: model weights = 6.65 GiB, leaving only ~1 GiB
    # CUDA budget. val_roi=(512,512,3) needs 5+ GiB extra → always OOM on 8 GB GPU.
    # Use patch_size for sam3_hybrid to avoid OOM — even in production.
    # sam3_vanilla (2.9 GiB weights): (512,512,3) fits fine (4.5 GiB free) and
    # is ~100× faster than patch-sized val_roi (fewer sliding-window patches).
    _sam3_val_roi = patch_size if _is_sam3_hybrid else (512, 512, 3)
    val_roi = _sam3_val_roi if _is_sam3 else data_config.patch_size
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
            "loss_function": loss_name,
            "config_fingerprint": fingerprint,
        }
    ):
        tracker.log_hydra_config(config)

        # --- Pre-training diagnostics (RC17: always run, not gated by profiling) ---
        sample_batch = next(iter(train_loader))
        device = torch.device(device_str)
        sample_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }
        pre_check_results = run_pre_training_checks(
            model=model,
            sample_batch=sample_on_device,
            criterion=criterion,
            expected_channels=model_config.out_channels,
        )
        errors = [
            r for r in pre_check_results if not r.passed and r.severity == "error"
        ]
        if errors:
            msg = "; ".join(f"{r.name}: {r.message}" for r in errors)
            raise RuntimeError(f"Pre-training check failed: {msg}")

        # --- Check for spot preemption resume ---
        resume_state = check_resume_state_task(checkpoint_dir)
        start_epoch = 0
        if resume_state is not None:
            resume_epoch = resume_state.get("epoch", 0)
            epoch_pth = checkpoint_dir / "epoch_latest.pth"
            if epoch_pth.exists():
                logger.info(
                    "Spot recovery: loading epoch_latest.pth (epoch %d)", resume_epoch
                )
                state_dict = torch.load(
                    epoch_pth, map_location=device_str, weights_only=True
                )
                model.load_state_dict(state_dict)
                start_epoch = resume_epoch + 1
                logger.info("Resuming training from epoch %d", start_epoch)

        # --- Training ---
        fit_result = trainer.fit(
            train_loader,
            val_loader,
            checkpoint_dir=checkpoint_dir,
            fold_id=fold_id,
            start_epoch=start_epoch,
        )

        # --- Post-training diagnostics (RC17: always run) ---
        ww_summary = run_weightwatcher(model)
        logger.info("WeightWatcher: %s", ww_summary)

        # T09: Print VRAM sentinel for Ralph Loop log parsing (#744)
        # Multi-fold: Ralph Loop takes max across all sentinels.
        _vram_mb = fit_result.get("vram_peak_mb", 0)
        _vram_gb = fit_result.get("vram_peak_gb", 0.0)
        if _vram_mb > 0:
            print(f"VRAM_PEAK_MB={_vram_mb}")  # noqa: T201
            print(f"VRAM_PEAK_GB={_vram_gb}")  # noqa: T201
            logger.info("T09 VRAM: %d MB (%.2f GB) peak allocated", _vram_mb, _vram_gb)

        return fit_result


@task(name="log-fold-results")
def log_fold_results_task(
    fold_id: int,
    result: dict[str, Any],
    mlflow_run_id: str | None,
    checkpoint_dir: Path | None = None,
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
    checkpoint_dir:
        Optional path to fold checkpoint directory. When provided, writes a
        ``checkpoint_dir_fold_{fold_id}`` tag so post-training flow can
        discover checkpoints without filesystem scanning.
    """
    if mlflow_run_id is None:
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        best_val_loss = result.get("best_val_loss", float("nan"))
        final_epoch = result.get("final_epoch", 0)
        client.log_metric(mlflow_run_id, f"fold/{fold_id}/best_val_loss", best_val_loss)
        client.log_metric(
            mlflow_run_id, f"fold/{fold_id}/final_epoch", float(final_epoch)
        )
        # Log per-epoch val_loss history for the fold
        for epoch, val_loss in enumerate(
            result.get("history", {}).get("val_loss", []), start=1
        ):
            client.log_metric(
                mlflow_run_id, f"fold/{fold_id}/val_loss", val_loss, step=epoch
            )
        # Also log val/loss (without fold prefix) for cross-fold visibility
        if result.get("history", {}).get("val_loss"):
            for epoch, val_loss in enumerate(result["history"]["val_loss"], start=1):
                client.log_metric(mlflow_run_id, "val/loss", val_loss, step=epoch)
        # T09: Log VRAM peak to MLflow (#744)
        _vram_mb = result.get("vram_peak_mb", 0)
        if _vram_mb > 0:
            client.log_metric(mlflow_run_id, "vram/peak_mb", float(_vram_mb))

        # Tag checkpoint_dir so post-training flow can discover it via FlowContract
        if checkpoint_dir is not None:
            client.set_tag(
                mlflow_run_id, f"checkpoint_dir_fold_{fold_id}", str(checkpoint_dir)
            )
    except Exception:
        logger.warning(
            "Failed to log fold %d results to MLflow", fold_id, exc_info=True
        )


# ---------------------------------------------------------------------------
# Training sub-flow (Phase 1: extracted from training_flow)
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_TRAINING_SUBFLOW)
def training_subflow(
    *,
    folds_to_run: list[Any],
    config: dict[str, Any],
    tracking_uri: str,
    checkpoint_base: Path,
    upstream_data_run_id: str | None = None,
    fold_id: int | None = None,
) -> TrainingFlowResult:
    """Training sub-flow — executes fold training and logs to MLflow.

    Called by the parent training_flow(). Contains the training loop,
    MLflow logging, cost analysis, and lineage emission.

    In Phase 2, the parent will also call post_training_subflow() after
    this sub-flow completes, reusing the GPU and DataLoaders.

    Parameters
    ----------
    folds_to_run:
        List of fold split dicts (each with 'train' and 'val' lists).
    config:
        Merged training config dict (loss_name, model_family, etc.).
    tracking_uri:
        MLflow tracking URI.
    checkpoint_base:
        Base directory for fold checkpoints (must be on persistent volume).
    upstream_data_run_id:
        Optional upstream data flow run ID for MLflow linking.
    fold_id:
        Specific fold ID when running a single fold (None = all folds).

    Returns
    -------
    TrainingFlowResult with fold results, MLflow run ID, and upstream link.
    """
    loss_name: str = config["loss_name"]
    model_family: str = config["model_family"]
    experiment_name: str = config["experiment_name"]
    max_epochs: int = config["max_epochs"]
    num_folds: int = config["num_folds"]
    batch_size: int = config.get("batch_size", 2)
    with_aux_calib: bool = config.get("with_aux_calib", False)

    # FDA audit trail: log data access (Issue #821 — IEC 62304 traceability)
    dataset_name: str = config.get("dataset_name", "minivess")
    file_paths: list[str] = []
    for fold_split in folds_to_run:
        # fold_split is FoldSplit dataclass (has .train/.val attrs) or dict
        _train = (
            fold_split.get("train", [])
            if isinstance(fold_split, dict)
            else getattr(fold_split, "train", [])
        )
        _val = (
            fold_split.get("val", [])
            if isinstance(fold_split, dict)
            else getattr(fold_split, "val", [])
        )
        for entry in _train:
            if isinstance(entry, dict) and "image" in entry:
                file_paths.append(entry["image"])
        for entry in _val:
            if isinstance(entry, dict) and "image" in entry:
                file_paths.append(entry["image"])
    audit = AuditTrail()
    audit.log_data_access(
        dataset_name=dataset_name,
        file_paths=file_paths,
        actor="train-flow",
    )
    logger.info(
        "Audit: logged data access for %s (%d files)", dataset_name, len(file_paths)
    )

    # Train each fold, skipping already-completed configs (auto-resume)
    fold_results: list[dict[str, Any]] = []
    fold_checkpoint_dirs: dict[int, Path] = {}
    for fold_idx, fold_split in enumerate(folds_to_run):
        # Actual fold ID: if a specific fold was requested, use it; otherwise enumerate
        actual_fold_id = fold_id if fold_id is not None else fold_idx
        # Check for already-completed run with matching config fingerprint
        fingerprint = compute_config_fingerprint(
            loss_name=loss_name,
            model_family=model_family,
            fold_id=actual_fold_id,
            max_epochs=max_epochs,
            batch_size=batch_size,
            with_aux_calib=with_aux_calib,
        )
        existing_run_id = find_completed_config(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            config_fingerprint=fingerprint,
        )
        if existing_run_id is not None:
            logger.info(
                "Fold %d: resuming from completed run %s",
                actual_fold_id,
                existing_run_id,
            )
            fold_result = load_fold_result_from_mlflow(tracking_uri, existing_run_id)
            fold_results.append(fold_result)
            continue

        # Namespace checkpoints by condition (model+loss) to prevent collisions
        # when concurrent factorial jobs share the same GCS-backed checkpoint mount.
        condition_name = f"{model_family}_{loss_name}"
        checkpoint_dir = checkpoint_base / condition_name / f"fold_{actual_fold_id}"
        fold_checkpoint_dirs[actual_fold_id] = checkpoint_dir
        fold_result = train_one_fold_task(
            actual_fold_id, fold_split, config, checkpoint_dir
        )
        fold_results.append(fold_result)

    # Open MLflow run, log everything, then close cleanly
    mlflow_run_id: str | None = None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        # MLflow tags: flow_name is inline so AST checks in test_flow_name_tags.py pass.
        # Optional upstream_data_run_id is injected via dict unpacking to avoid None
        # values that raise TypeError during protobuf serialization (tag.to_proto()).
        with mlflow.start_run(
            tags={
                "flow_name": FLOW_NAME_TRAIN,
                "loss_function": loss_name,
                "model_family": model_family,
                **(
                    {"upstream_data_run_id": upstream_data_run_id}
                    if upstream_data_run_id is not None
                    else {}
                ),
            }
        ) as active_run:
            mlflow_run_id = active_run.info.run_id
            mlflow.set_tag("parent_run_id", mlflow_run_id)
            logger.info("MLflow run opened: %s", mlflow_run_id)

            # Log infrastructure timing from setup phase (#683)
            from minivess.observability.infrastructure_timing import (
                log_cost_analysis,
                log_infrastructure_timing,
            )

            log_infrastructure_timing(timing_dir=Path.cwd())

            # Log fold results inside the run context.
            # Use fold_checkpoint_dirs keys (actual fold IDs) not enumerate indices.
            logged_fold_ids = sorted(fold_checkpoint_dirs.keys())
            for log_idx, fold_result in enumerate(fold_results):
                log_fold_id = (
                    logged_fold_ids[log_idx]
                    if log_idx < len(logged_fold_ids)
                    else log_idx
                )
                log_fold_results_task(
                    log_fold_id,
                    fold_result,
                    mlflow_run_id,
                    checkpoint_dir=fold_checkpoint_dirs.get(log_fold_id),
                )

            mlflow.log_metric("fold/n_completed", float(len(fold_results)))

            # Log cost analysis after training (#683)
            _total_training_seconds = sum(
                fr.get("training_time_seconds", 0.0)
                for fr in fold_results
                if isinstance(fr, dict)
            )
            _setup_seconds = 0.0
            _setup_durations: dict[str, float] = {}
            try:
                from minivess.observability.infrastructure_timing import (
                    parse_setup_timing,
                )

                _setup_durations = parse_setup_timing(Path.cwd() / "timing_setup.txt")
                _setup_seconds = _setup_durations.get("setup_total", 0.0)
            except Exception:
                pass
            if _total_training_seconds > 0 or _setup_seconds > 0:
                log_cost_analysis(
                    setup_seconds=_setup_seconds,
                    training_seconds=_total_training_seconds,
                    epoch_count=max_epochs * len(fold_results),
                )

            # Log epoch-0 cost estimate (#717 — wire dead code)
            if _total_training_seconds > 0:
                log_epoch0_cost_estimate(
                    training_time_seconds=_total_training_seconds,
                    max_epochs=max_epochs,
                    num_folds=num_folds,
                    hourly_rate_usd=get_hourly_rate_usd(),
                )

            # Log timing JSONL artifact (#683 — wire dead code)
            log_timing_jsonl_artifact(
                setup_durations=_setup_durations,
                training_seconds=_total_training_seconds,
                epoch_count=max_epochs * len(fold_results),
                hourly_rate_usd=get_hourly_rate_usd(),
            )

    except Exception:
        logger.warning("Failed to open/finalize MLflow run", exc_info=True)

    # Log flow completion tag (best-effort, non-blocking)
    log_completion_safe(
        flow_name=FLOW_NAME_TRAIN,
        tracking_uri=tracking_uri,
        run_id=mlflow_run_id,
    )

    # OpenLineage lineage emission (Issue #799 — IEC 62304 §8 traceability)
    try:
        _emitter = LineageEmitter(namespace="minivess")
        emit_flow_lineage(
            emitter=_emitter,
            job_name="train-flow",
            inputs=[{"namespace": "minivess", "name": "raw_volumes"}],
            outputs=[
                {"namespace": "minivess", "name": "checkpoints"},
                {"namespace": "minivess", "name": "mlflow_metrics"},
            ],
        )
    except Exception:
        logger.warning("OpenLineage emission failed (non-blocking)", exc_info=True)

    result = TrainingFlowResult(
        flow_name="train",
        status="completed",
        mlflow_run_id=mlflow_run_id,
        fold_results=fold_results,
        upstream_data_run_id=upstream_data_run_id,
        n_folds=len(fold_results),
        loss_name=loss_name,
        model_family=model_family,
        checkpoint_dirs=fold_checkpoint_dirs,
    )
    logger.info(
        "Training sub-flow complete: %d folds, run_id=%s",
        len(fold_results),
        mlflow_run_id,
    )
    return result


# ---------------------------------------------------------------------------
# Post-training sub-flow (Phase 2: SWAG, checkpoint averaging, etc.)
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_POST_TRAINING_SUBFLOW)
def post_training_subflow(
    *,
    post_training_method: str,
    training_result: TrainingFlowResult,
    config: dict[str, Any],
    tracking_uri: str,
) -> str | None:
    """Post-training sub-flow — runs SWAG or other post-training methods.

    Called by the parent training_flow() when post_training_method != "none".
    Shares the same GPU session as training — no extra SkyPilot job needed.

    Parameters
    ----------
    post_training_method:
        Method name: "swag", "checkpoint_averaging", etc.
    training_result:
        Result from training_subflow() — contains checkpoint dirs and run ID.
    config:
        Merged training config dict (includes SWAG-specific keys).
    tracking_uri:
        MLflow tracking URI.

    Returns
    -------
    MLflow run ID for the post-training run, or None on failure.
    """
    loss_name: str = config["loss_name"]
    model_family: str = config["model_family"]
    experiment_name: str = config["experiment_name"]

    logger.info(
        "Post-training sub-flow started: method=%s, upstream_run=%s",
        post_training_method,
        training_result.mlflow_run_id,
    )

    # Load SWAG config from Hydra post_training group
    swag_config: dict[str, Any] = config.get("post_training", {}).get("swag", {})
    if not swag_config and post_training_method == "swag":
        # Fallback to top-level swag key or defaults from SWAGPluginConfig
        from minivess.config.post_training_config import SWAGPluginConfig

        swag_config = SWAGPluginConfig().model_dump()

    # Open MLflow run for post-training
    post_training_run_id: str | None = None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(
            tags={
                "flow_name": FLOW_NAME_POST_TRAINING_SUBFLOW,
                "post_training_method": post_training_method,
                "loss_function": loss_name,
                "model_family": model_family,
                **(
                    {"upstream_training_run_id": training_result.mlflow_run_id}
                    if training_result.mlflow_run_id is not None
                    else {}
                ),
            }
        ) as active_run:
            post_training_run_id = active_run.info.run_id
            logger.info(
                "Post-training MLflow run opened: %s (method=%s)",
                post_training_run_id,
                post_training_method,
            )

            if post_training_method == "swag":
                _run_swag_post_training(
                    training_result=training_result,
                    swag_config=swag_config,
                    config=config,
                )
            else:
                logger.warning(
                    "Unknown post-training method: %s — skipping",
                    post_training_method,
                )

    except Exception:
        logger.warning("Post-training sub-flow failed (non-blocking)", exc_info=True)

    logger.info(
        "Post-training sub-flow complete: method=%s, run_id=%s",
        post_training_method,
        post_training_run_id,
    )
    return post_training_run_id


def _run_swag_post_training(
    *,
    training_result: TrainingFlowResult,
    swag_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Execute SWAG post-training using the SWAGPlugin.

    Discovers checkpoints from training_result, loads the model,
    and runs SWAG posterior approximation.

    Parameters
    ----------
    training_result:
        Result from training sub-flow (has checkpoint_dirs).
    swag_config:
        SWAG-specific config (swa_lr, swa_epochs, max_rank, etc.).
    config:
        Full training config dict.
    """
    from minivess.pipeline.post_training_plugin import PluginInput
    from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

    # Collect checkpoint paths from all folds
    checkpoint_paths: list[Path] = []
    for fold_id, ckpt_dir in sorted(training_result.checkpoint_dirs.items()):
        best_ckpt = ckpt_dir / "best_val_loss.pth"
        if best_ckpt.exists():
            checkpoint_paths.append(best_ckpt)
        else:
            # Fallback to any .pth file in the fold dir
            pth_files = sorted(ckpt_dir.glob("*.pth"))
            if pth_files:
                checkpoint_paths.append(pth_files[0])
            else:
                logger.warning(
                    "No checkpoint found in %s for fold %d", ckpt_dir, fold_id
                )

    if not checkpoint_paths:
        raise ValueError(
            "SWAG requires at least one checkpoint — none found in "
            f"checkpoint_dirs: {training_result.checkpoint_dirs}"
        )

    # Rebuild model + DataLoader from config for SWAG resumed training.
    # The training sub-flow already used these — we rebuild from the same config
    # to share the GPU session without modifying TrainingFlowResult (avoids
    # Prefect serialization issues with nn.Module + DataLoader).
    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import DataConfig, ModelConfig, ModelFamily
    from minivess.data.loader import build_train_loader
    from minivess.data.splits import load_splits

    model_family_str: str = config.get("model_family", "dynunet")
    model_family = ModelFamily(model_family_str)
    model_config = ModelConfig(
        family=model_family,
        name=model_family_str,
        in_channels=1,
        out_channels=2,
        architecture_params=dict(config.get("architecture_params", {})),
    )
    adapter = build_adapter(model_config)

    # Build train DataLoader from the same fold split used in training
    splits_dir = Path(os.environ.get("SPLITS_DIR", "configs/splits"))
    splits_file = splits_dir / "splits.json"
    if not splits_file.exists():
        # Fallback to 3fold_seed42.json
        splits_file = splits_dir / "3fold_seed42.json"
    splits = load_splits(splits_file)
    pt_fold_id: int = int(config.get("fold_id", 0))
    fold_split = splits[pt_fold_id] if pt_fold_id < len(splits) else splits[0]
    train_dicts: list[dict[str, str]] = (
        fold_split.train  # type: ignore[union-attr]
        if hasattr(fold_split, "train")
        else fold_split["train"]  # type: ignore[index]
    )

    # Respect max_train_volumes from config (debug = half data)
    max_train_volumes = config.get("max_train_volumes")
    if max_train_volumes is not None:
        train_dicts = train_dicts[:max_train_volumes]

    _is_sam3 = model_family_str.startswith("sam3_")
    _is_vesselfm = model_family_str == "vesselfm"
    if _is_sam3:
        default_patch = (64, 64, 3)
    elif _is_vesselfm:
        default_patch = (64, 64, 32)
    else:
        default_patch = (64, 64, 16)
    _patch_raw = config.get("patch_size")
    patch_size = default_patch if _patch_raw is None else tuple(_patch_raw)
    batch_size = (
        min(config.get("batch_size", 2), 1) if _is_sam3 else config.get("batch_size", 2)
    )

    data_config = DataConfig(
        dataset_name=config.get("dataset_name", "minivess"),
        data_dir=Path(config.get("data_dir", "data/raw")),
        patch_size=patch_size,
        num_workers=0,  # Debug: no multiprocessing for SWAG
    )
    train_loader = build_train_loader(
        train_dicts,
        data_config,
        batch_size=batch_size,
        cache_rate=0.0,
    )

    calibration_data = {
        "train_loader": train_loader,
        "model": adapter,  # ModelAdapter IS an nn.Module (has .net inside)
    }

    plugin = SWAGPlugin()
    plugin_input = PluginInput(
        checkpoint_paths=checkpoint_paths,
        config=swag_config,
        calibration_data=calibration_data,
    )

    # Validate — raise on errors (Rule 25: loud failures, never silent skip)
    errors = plugin.validate_inputs(plugin_input)
    if errors:
        raise ValueError(f"SWAG validation failed: {errors}")

    output = plugin.execute(plugin_input)

    # Log SWAG metrics to MLflow
    for metric_name, metric_value in output.metrics.items():
        mlflow.log_metric(f"post_training/swag/{metric_name}", metric_value)

    # Log SWAG model artifact
    for model_path in output.model_paths:
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="post_training")

    logger.info(
        "SWAG post-training complete: %d metrics, %d model files",
        len(output.metrics),
        len(output.model_paths),
    )


# ---------------------------------------------------------------------------
# Parent flow (orchestrates training + post-training sub-flows)
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
    config_dict: dict[str, Any] | None = None,
    # Factorial experiment parameters
    fold_id: int | None = None,
    with_aux_calib: bool = False,
    max_train_volumes: int = 0,
    max_val_volumes: int = 0,
    zero_shot: bool = False,
    eval_dataset: str = "minivess",
    post_training_method: str = "none",
    **kwargs: Any,
) -> TrainingFlowResult:
    """Parent training flow — orchestrates training + optional post-training.

    Handles preflight checks, config resolution, and fold split loading,
    then delegates to training_subflow(). If post_training_method != "none",
    also calls post_training_subflow() (SWAG, etc.) in the same GPU session.

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
    post_training_method:
        Post-training method: "none" (default), "swag", etc.

    Returns
    -------
    TrainingFlowResult with fold results, MLflow run ID, and upstream link.
    """
    logger.info("Training flow started (trigger: %s)", trigger_source)

    # Wire JSONL log handler so training output reaches: terminal + Prefect UI + volume file
    # Issue #503: double-logging to durable JSONL on the logs volume
    import os

    from minivess.observability.flow_logging import configure_flow_logging

    configure_flow_logging(logs_dir=Path(os.environ.get("LOGS_DIR", "/app/logs")))

    # Extract params from config_dict when provided (Hydra-zen bridge, Rule #23)
    if config_dict is not None:
        losses = config_dict.get("losses", [loss_name])
        loss_name = losses[0] if losses else loss_name
        model_family = str(config_dict.get("model", model_family))
        debug = bool(config_dict.get("debug", debug))
        max_epochs = int(config_dict.get("max_epochs", max_epochs))
        num_folds = int(config_dict.get("num_folds", num_folds))
        batch_size = int(config_dict.get("batch_size", batch_size))
        experiment_name = str(config_dict.get("experiment_name", experiment_name))
        logger.info(
            "Config loaded from Hydra dict: experiment=%s model=%s loss=%s epochs=%d",
            experiment_name,
            model_family,
            loss_name,
            max_epochs,
        )

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

    # Zero-shot early return: skip training, just log that evaluation is needed.
    # The actual zero-shot evaluation will be done by the analysis flow.
    if zero_shot and max_epochs == 0:
        logger.info(
            "Zero-shot mode: model=%s, eval_dataset=%s — skipping training loop. "
            "Evaluation will be handled by the analysis flow.",
            model_family,
            eval_dataset,
        )
        return TrainingFlowResult(
            status="ZERO_SHOT",
            fold_results=[],
            loss_name=loss_name,
            model_family=model_family,
            mlflow_run_id=None,
        )

    # Load fold splits (outside MLflow run — no side effects)
    splits = load_fold_splits_task(splits_dir)
    if fold_id is not None:
        # Specific fold requested (e.g., --fold 2 → run only fold 2)
        if fold_id >= len(splits):
            msg = f"Requested fold_id={fold_id} but only {len(splits)} folds in splits file"
            raise ValueError(msg)
        folds_to_run = [splits[fold_id]]
    else:
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
        # Factorial experiment parameters (Glitch #8/#9/#10/#12 fix)
        "with_aux_calib": with_aux_calib,
        "max_train_volumes": max_train_volumes if max_train_volumes > 0 else None,
        "max_val_volumes": max_val_volumes if max_val_volumes > 0 else None,
        "zero_shot": zero_shot,
        "eval_dataset": eval_dataset,
    }

    # Merge full Hydra config so train_one_fold_task can log it and use all keys
    if config_dict is not None:
        merged = dict(config_dict)
        merged.update(config)  # individual params take precedence
        config = merged

    # --- Sub-flow 1: Training ---
    result = training_subflow(
        folds_to_run=folds_to_run,
        config=config,
        tracking_uri=tracking_uri,
        checkpoint_base=checkpoint_base,
        upstream_data_run_id=upstream_data_run_id,
        fold_id=fold_id,
    )

    # --- Sub-flow 2: Post-training (iterate over comma-separated methods) ---
    # post_training_method can be comma-separated: "none,swag"
    # "none" cell is already produced by training sub-flow (free).
    # Non-"none" methods run in post-training sub-flow (same GPU session).
    pt_methods = [m.strip() for m in post_training_method.split(",")]
    pt_run_ids: list[str | None] = []
    for pt_method in pt_methods:
        if pt_method != "none":
            pt_run_id = post_training_subflow(
                post_training_method=pt_method,
                training_result=result,
                config=config,
                tracking_uri=tracking_uri,
            )
            pt_run_ids.append(pt_run_id)

    # Store the methods and run IDs in result
    result.post_training_method = ",".join(pt_methods)
    result.post_training_run_id = (
        pt_run_ids[0]
        if len(pt_run_ids) == 1
        else ",".join(str(rid) for rid in pt_run_ids if rid)
        if pt_run_ids
        else None
    )

    logger.info(
        "Training flow complete: %d folds, run_id=%s, post_training=%s",
        result.n_folds,
        result.mlflow_run_id,
        result.post_training_method,
    )
    return result


if __name__ == "__main__":
    # CLAUDE.md DG1.7: Suppress non-actionable third-party noise at entry point.
    # MetricsReloaded SyntaxWarnings fire during import; MONAI/CUDA warnings at runtime.
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="MetricsReloaded")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda.cudart.*")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*non-tuple sequence for multidimensional indexing.*",
    )

    # Hydra-zen bridge (Rule #23): EXPERIMENT env var → compose_experiment_config()
    # → resolved config dict → training_flow(config_dict=resolved)
    _experiment = os.environ.get("EXPERIMENT")
    _hydra_overrides_str = os.environ.get("HYDRA_OVERRIDES", "")
    # Bracket-aware split: preserve commas inside [...] (e.g. patch_size=[32,32,3]).
    # Plain str.split(",") would break list overrides like patch_size=[32,32,3].
    _hydra_overrides: list[str] = []
    _current: list[str] = []
    _depth = 0
    for _ch in _hydra_overrides_str:
        if _ch == "[":
            _depth += 1
            _current.append(_ch)
        elif _ch == "]":
            _depth -= 1
            _current.append(_ch)
        elif _ch == "," and _depth == 0:
            _part = "".join(_current).strip()
            if _part:
                _hydra_overrides.append(_part)
            _current = []
        else:
            _current.append(_ch)
    _part = "".join(_current).strip()
    if _part:
        _hydra_overrides.append(_part)

    if _experiment:
        from minivess.config.compose import compose_experiment_config

        _config = compose_experiment_config(
            experiment_name=_experiment,
            overrides=_hydra_overrides,
        )
        training_flow(config_dict=_config)
    else:
        # Backward compat: build from individual env vars when EXPERIMENT not set.
        # This path is retained for existing docker-compose configs using old env vars.
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
        # Factorial experiment parameters (set by run_factorial.sh via SkyPilot env vars)
        parser.add_argument(
            "--fold",
            type=int,
            default=int(os.environ.get("FOLD_ID", "-1")),
            help="Specific fold ID to run (-1 = all folds via num_folds)",
        )
        parser.add_argument(
            "--with-aux-calib",
            default=os.environ.get("WITH_AUX_CALIB", "false"),
            help="Enable auxiliary calibration loss (true/false)",
        )
        parser.add_argument(
            "--max-train-volumes",
            type=int,
            default=int(os.environ.get("MAX_TRAIN_VOLUMES", "0")),
            help="Max training volumes (0 = full dataset)",
        )
        parser.add_argument(
            "--max-val-volumes",
            type=int,
            default=int(os.environ.get("MAX_VAL_VOLUMES", "0")),
            help="Max validation volumes (0 = full dataset)",
        )
        parser.add_argument(
            "--zero-shot",
            default=os.environ.get("ZERO_SHOT", "false"),
            help="Zero-shot evaluation only (true/false)",
        )
        parser.add_argument(
            "--eval-dataset",
            default=os.environ.get("EVAL_DATASET", "minivess"),
            help="Evaluation dataset (minivess/deepvess)",
        )
        parser.add_argument(
            "--post-training-method",
            default=os.environ.get(
                "POST_TRAINING_METHODS",
                os.environ.get("POST_TRAINING_METHOD", "none,swag"),
            ),
            help="Comma-separated post-training methods: none,swag (default: none,swag)",
        )
        args = parser.parse_args()

        # If a specific fold is requested, override num_folds to 1
        fold_id = args.fold if args.fold >= 0 else None

        training_flow(
            model_family=args.model_family,
            loss_name=args.loss_name,
            max_epochs=args.max_epochs,
            num_folds=1 if fold_id is not None else args.num_folds,
            batch_size=args.batch_size,
            experiment_name=args.experiment_name,
            debug=args.debug,
            fold_id=fold_id,
            with_aux_calib=args.with_aux_calib.lower() == "true",
            max_train_volumes=args.max_train_volumes,
            max_val_volumes=args.max_val_volumes,
            zero_shot=args.zero_shot.lower() == "true",
            eval_dataset=args.eval_dataset,
            post_training_method=args.post_training_method,
        )
