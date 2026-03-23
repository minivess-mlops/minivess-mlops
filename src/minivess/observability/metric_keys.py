"""Canonical metric/param key definitions.

Single source of truth for ALL MLflow key names used across the platform.
Used by ExperimentTracker, biostatistics_flow, analytics.py.

Convention: MLflow 2.11+ auto-groups metrics by slash prefix in UI.
    - train/loss, val/dice -> grouped under train/, val/
    - sys/gpu_model, sys/torch_version -> grouped under sys/
    - cost/total_usd, cost/setup_fraction -> grouped under cost/

Eval metric CI encoding uses underscore suffix to avoid filesystem
conflicts with MLflow file-based tracking:
    eval/minivess/all/dsc_ci95_lo, eval/minivess/all/dsc_ci95_hi

Fold encoding: fold/0/best_val_loss, fold/1/best_val_loss

Greenfield: No backward compatibility layer. MIGRATION_MAP and
normalize_metric_key() have been deleted.

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

    # --- Performance instrumentation (infrastructure timing) ---
    # Logged per-job for data-driven optimization and regression detection.
    # Issue #913, Plan: infrastructure-performance-audit.xml Phase 4.
    PERF_SETUP_TOTAL_SECONDS = "perf/setup_total_seconds"
    PERF_TRAINING_SECONDS = "perf/training_seconds"
    PERF_SWAG_SECONDS = "perf/swag_seconds"
    PERF_SWAG_OVERHEAD_RATIO = "perf/swag_overhead_ratio"
    PERF_TOTAL_JOB_SECONDS = "perf/total_job_seconds"
    PERF_GPU_UTILIZATION_FRACTION = "perf/gpu_utilization_fraction"

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
