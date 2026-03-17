"""Canonical metric/param key definitions and migration mapping.

Single source of truth for ALL MLflow key names used across the platform.
Used by ExperimentTracker, biostatistics_flow, analytics.py.

Convention: MLflow 2.11+ auto-groups metrics by slash prefix in UI.
    - train/loss, val/dice → grouped under train/, val/
    - sys/gpu_model, sys/torch_version → grouped under sys/
    - cost/total_usd, cost/setup_fraction → grouped under cost/

CI encoding: val/dice/ci95_lo, val/dice/ci95_hi
Fold encoding: fold/0/best_val_loss, fold/1/best_val_loss

Issue: #790
"""

from __future__ import annotations


class MetricKeys:
    """Canonical metric key constants using slash-prefix convention.

    All MLflow metric and param keys should be referenced from this class
    to prevent drift between logging and querying code.
    """

    # --- Training metrics (per-epoch) ---
    TRAIN_LOSS = "train/loss"
    TRAIN_DICE = "train/dice"
    TRAIN_F1_FOREGROUND = "train/f1_foreground"

    # --- Validation metrics (per-epoch) ---
    VAL_LOSS = "val/loss"
    VAL_DICE = "val/dice"
    VAL_F1_FOREGROUND = "val/f1_foreground"
    VAL_CLDICE = "val/cldice"
    VAL_MASD = "val/masd"
    VAL_COMPOUND_MASD_CLDICE = "val/compound_masd_cldice"

    # --- Optimizer (per-epoch) ---
    OPTIM_LR = "optim/lr"
    OPTIM_GRAD_SCALE = "optim/grad_scale"

    # --- Gradient monitoring (per-epoch, T3) ---
    GRAD_NORM_MEAN = "grad/norm_mean"
    GRAD_NORM_MAX = "grad/norm_max"
    GRAD_CLIP_COUNT = "grad/clip_count"

    # --- GPU monitoring (per-epoch) ---
    GPU_PREFIX = "gpu/"

    # --- Profiling (logged once) ---
    PROF_FIRST_EPOCH_SECONDS = "prof/first_epoch_seconds"
    PROF_STEADY_EPOCH_SECONDS = "prof/steady_epoch_seconds"
    PROF_OVERHEAD_PCT = "prof/overhead_pct"
    PROF_TRACE_SIZE_MB = "prof/trace_size_mb"
    PROF_DATA_TO_DEVICE_FRACTION = "prof/data_to_device_fraction"
    PROF_FORWARD_FRACTION = "prof/forward_fraction"
    PROF_BACKWARD_FRACTION = "prof/backward_fraction"
    PROF_VAL_SECONDS = "prof/val_seconds"
    PROF_TRAIN_SECONDS = "prof/train_seconds"

    # --- Fold-level metrics ---
    FOLD_N_COMPLETED = "fold/n_completed"
    VRAM_PEAK_MB = "vram/peak_mb"
    VRAM_PEAK_GB = "vram/peak_gb"

    # --- Cost / FinOps ---
    COST_TOTAL_WALL_SECONDS = "cost/total_wall_seconds"
    COST_TOTAL_USD = "cost/total_usd"
    COST_SETUP_USD = "cost/setup_usd"
    COST_TRAINING_USD = "cost/training_usd"
    COST_EFFECTIVE_GPU_RATE = "cost/effective_gpu_rate"
    COST_SETUP_FRACTION = "cost/setup_fraction"
    COST_GPU_UTILIZATION_FRACTION = "cost/gpu_utilization_fraction"
    COST_EPOCHS_TO_AMORTIZE_SETUP = "cost/epochs_to_amortize_setup"
    COST_BREAK_EVEN_EPOCHS = "cost/break_even_epochs"

    # --- Cost estimates (epoch 0) ---
    EST_TOTAL_COST = "est/total_cost"
    EST_TOTAL_HOURS = "est/total_hours"
    EST_COST_PER_EPOCH = "est/cost_per_epoch"
    EST_EPOCH_SECONDS = "est/epoch_seconds"

    # --- Inference latency (T4) ---
    INFER_LATENCY_MS_PER_VOLUME = "infer/latency_ms_per_volume"
    INFER_THROUGHPUT_VOLUMES_PER_SEC = "infer/throughput_volumes_per_sec"

    # --- Checkpoint metadata (T7) ---
    CHECKPOINT_SIZE_MB = "checkpoint/size_mb"
    CHECKPOINT_EPOCH = "checkpoint/epoch"

    # --- Early stopping (T6) ---
    TRAIN_PATIENCE_COUNTER = "train/patience_counter"
    TRAIN_STOPPED_EARLY = "train/stopped_early"

    # --- Data augmentation (T5) ---
    DATA_AUGMENTATION_PIPELINE = "data/augmentation_pipeline"


# ---------------------------------------------------------------------------
# Migration mapping: old underscore keys -> new slash-prefix keys
# ---------------------------------------------------------------------------

MIGRATION_MAP: dict[str, str] = {
    # Per-epoch training/validation metrics
    "train_loss": "train/loss",
    "val_loss": "val/loss",
    "learning_rate": "optim/lr",
    "train_dice": "train/dice",
    "train_f1_foreground": "train/f1_foreground",
    "val_dice": "val/dice",
    "val_f1_foreground": "val/f1_foreground",
    "val_cldice": "val/cldice",
    "val_masd": "val/masd",
    "val_compound_masd_cldice": "val/compound_masd_cldice",
    # Profiling metrics
    "prof_first_epoch_seconds": "prof/first_epoch_seconds",
    "prof_steady_epoch_seconds": "prof/steady_epoch_seconds",
    "prof_overhead_pct": "prof/overhead_pct",
    "prof_trace_size_mb": "prof/trace_size_mb",
    "prof_data_to_device_fraction": "prof/data_to_device_fraction",
    "prof_forward_fraction": "prof/forward_fraction",
    "prof_backward_fraction": "prof/backward_fraction",
    # System info params
    "sys_python_version": "sys/python_version",
    "sys_os": "sys/os",
    "sys_os_kernel": "sys/os_kernel",
    "sys_hostname": "sys/hostname",
    "sys_total_ram_gb": "sys/total_ram_gb",
    "sys_cpu_model": "sys/cpu_model",
    "sys_torch_version": "sys/torch_version",
    "sys_cuda_version": "sys/cuda_version",
    "sys_cudnn_version": "sys/cudnn_version",
    "sys_monai_version": "sys/monai_version",
    "sys_mlflow_version": "sys/mlflow_version",
    "sys_numpy_version": "sys/numpy_version",
    "sys_gpu_count": "sys/gpu_count",
    "sys_gpu_model": "sys/gpu_model",
    "sys_gpu_vram_mb": "sys/gpu_vram_mb",
    "sys_git_commit": "sys/git_commit",
    "sys_git_commit_short": "sys/git_commit_short",
    "sys_git_branch": "sys/git_branch",
    "sys_git_dirty": "sys/git_dirty",
    "sys_dvc_version": "sys/dvc_version",
    # Data params
    "data_n_volumes": "data/n_volumes",
    "data_total_size_gb": "data/total_size_gb",
    "data_min_shape": "data/min_shape",
    "data_max_shape": "data/max_shape",
    "data_median_shape": "data/median_shape",
    "data_min_spacing": "data/min_spacing",
    "data_max_spacing": "data/max_spacing",
    "data_median_spacing": "data/median_spacing",
    "data_n_outlier_volumes": "data/n_outlier_volumes",
    # Config params (Dynaconf)
    "cfg_project_name": "cfg/project_name",
    "cfg_data_dir": "cfg/data_dir",
    "cfg_dvc_remote": "cfg/dvc_remote",
    "cfg_mlflow_tracking_uri": "cfg/mlflow_tracking_uri",
    # Setup timing params
    "setup_python_install_seconds": "setup/python_install_seconds",
    "setup_uv_install_seconds": "setup/uv_install_seconds",
    "setup_uv_sync_seconds": "setup/uv_sync_seconds",
    "setup_dvc_config_seconds": "setup/dvc_config_seconds",
    "setup_dvc_pull_seconds": "setup/dvc_pull_seconds",
    "setup_model_weights_seconds": "setup/model_weights_seconds",
    "setup_verification_seconds": "setup/verification_seconds",
    "setup_total_seconds": "setup/total_seconds",
    # Cost metrics
    "cost_total_wall_seconds": "cost/total_wall_seconds",
    "cost_total_usd": "cost/total_usd",
    "cost_setup_usd": "cost/setup_usd",
    "cost_training_usd": "cost/training_usd",
    "cost_effective_gpu_rate": "cost/effective_gpu_rate",
    "cost_setup_fraction": "cost/setup_fraction",
    "cost_gpu_utilization_fraction": "cost/gpu_utilization_fraction",
    "cost_epochs_to_amortize_setup": "cost/epochs_to_amortize_setup",
    "cost_break_even_epochs": "cost/break_even_epochs",
    # Cost estimates
    "estimated_total_cost": "est/total_cost",
    "estimated_total_hours": "est/total_hours",
    "cost_per_epoch": "est/cost_per_epoch",
    "epoch_seconds": "est/epoch_seconds",
    # Fold-level metrics
    "vram_peak_mb": "vram/peak_mb",
    "n_folds_completed": "fold/n_completed",
    # Benchmark params
    "sys_bench_gpu_model": "bench/gpu_model",
    "sys_bench_total_vram_mb": "bench/total_vram_mb",
    # Training config params
    "model_family": "model/family",
    "model_name": "model/name",
    "in_channels": "model/in_channels",
    "out_channels": "model/out_channels",
    "batch_size": "train/batch_size",
    # Note: config param "learning_rate" -> "train/learning_rate" handled
    # directly in _log_config(), not here, to avoid collision with metric key.
    "max_epochs": "train/max_epochs",
    "optimizer": "train/optimizer",
    "scheduler": "train/scheduler",
    "seed": "train/seed",
    "num_folds": "train/num_folds",
    "mixed_precision": "train/mixed_precision",
    "weight_decay": "train/weight_decay",
    "warmup_epochs": "train/warmup_epochs",
    "gradient_clip_val": "train/gradient_clip_val",
    "gradient_checkpointing": "train/gradient_checkpointing",
    "early_stopping_patience": "train/early_stopping_patience",
    "trainable_parameters": "model/trainable_params",
    "split_mode": "data/split_mode",
}


def normalize_metric_key(key: str) -> str:
    """Map an old underscore-convention key to new slash-prefix convention.

    If the key is already in slash-prefix format (contains '/'), it is
    returned unchanged. Unknown keys are also returned unchanged.

    Parameters
    ----------
    key:
        Metric or param key, possibly in legacy underscore format.

    Returns
    -------
    Key in the new slash-prefix convention, or the original key if
    no mapping exists.
    """
    if "/" in key:
        return key
    return MIGRATION_MAP.get(key, key)


def normalize_metric_dict(metrics: dict[str, object]) -> dict[str, object]:
    """Normalize all keys in a metric/param dict from old to new convention.

    Useful when reading legacy MLflow runs that use underscore-convention
    keys. Values are preserved unchanged.

    Parameters
    ----------
    metrics:
        Dictionary with potentially old-convention keys.

    Returns
    -------
    New dictionary with all keys normalized to slash-prefix convention.
    """
    return {normalize_metric_key(k): v for k, v in metrics.items()}
