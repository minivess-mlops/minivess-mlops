#!/usr/bin/env python
"""Memory-safe, crash-resistant training wrapper with system monitoring.

Wraps scripts/train.py with:
- Real-time system resource monitoring (RAM, GPU, swap)
- Per-fold checkpointing for cold-resume after crashes
- Reduced cache_rate to prevent OOM (root cause of terminal crashes)
- Explicit memory cleanup between folds via gc.collect()
- Configurable memory thresholds with graceful abort

Usage::

    # Debug smoke test
    uv run python scripts/train_monitored.py \\
        --compute gpu_low --loss dice_ce --debug \\
        --log-dir logs/debug_smoke

    # Full training with monitoring
    uv run python scripts/train_monitored.py \\
        --compute gpu_low --loss dice_ce \\
        --log-dir logs/dice_ce_run1 \\
        --memory-limit-gb 50

    # Resume after crash (reads checkpoint from log-dir)
    uv run python scripts/train_monitored.py \\
        --compute gpu_low --loss dice_ce \\
        --log-dir logs/dice_ce_run1 \\
        --resume

See docs/planning/dynunet-evaluation-plan.xml for the full plan.
"""

from __future__ import annotations

# Warning routing: BEFORE any heavy imports.
#
# Rule: warnings from site-packages OR frozen stdlib → DEBUG logger.
#       warnings from our own code (minivess/, scripts/) → WARNING logger.
#
# Researchers only see warnings they can act on.
# To see all warnings: set log level to DEBUG or PYTHONWARNINGS=always.
import contextlib
import logging as _logging
import os
import warnings


@contextlib.contextmanager  # type: ignore[misc]
def _suppress_fd2():  # type: ignore[return]
    """Redirect OS-level stderr (fd 2) to /dev/null for the duration.

    Required for ONNX Runtime: its C++ device discovery warning fires at
    .so load time, before Python's warnings module is consulted, so the
    only reliable suppression is at the file-descriptor level.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


# Pre-import onnxruntime with fd 2 redirected so the C++ device-discovery
# warning never reaches the terminal. Subsequent imports are no-ops (cached).
with _suppress_fd2(), contextlib.suppress(ImportError):
    import onnxruntime as _ort  # noqa: F401


def _route_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: object = None,
    line: str | None = None,
) -> None:
    """Route third-party warnings to DEBUG; project warnings to WARNING.

    "Third-party" = anything from site-packages or Python's frozen stdlib
    (e.g. <frozen importlib._bootstrap_external>).
    """
    fname = str(filename)
    is_external = "site-packages" in fname or fname.startswith("<frozen ")
    level = _logging.DEBUG if is_external else _logging.WARNING
    _logging.getLogger("py.warnings").log(
        level, "%s:%d: %s: %s", filename, lineno, category.__name__, message
    )


warnings.showwarning = _route_warning  # type: ignore[assignment]

# MONAI sliding-window: fires once per inference *window* (hundreds/epoch).
# Even routing to DEBUG would flood debug logs — suppress entirely.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*non-tuple sequence for multidimensional indexing.*",
)

import argparse
import gc
import json
import logging
import sys

# Load .env early so HF_TOKEN (and other secrets) are available before
# any model imports. Environment variables always override .env values.
from minivess.utils.hf_auth import load_dotenv_if_present

load_dotenv_if_present(".env")
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minivess.adapters.model_builder import build_adapter
from minivess.config.compute_profiles import apply_profile, get_compute_profile
from minivess.config.debug import (
    DEBUG_CACHE_RATE,
    DEBUG_MAX_VOLUMES,
    apply_debug_overrides,
)
from minivess.config.models import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    TrackedMetricConfig,
    TrainingConfig,
)
from minivess.data.loader import build_train_loader, build_val_loader
from minivess.data.splits import (
    generate_kfold_splits_from_dir,
    load_splits,
    save_splits,
)
from minivess.data.transforms import build_train_transforms, build_val_transforms
from minivess.observability.tracking import ExperimentTracker
from minivess.pipeline.evaluation import EvaluationRunner, FoldResult
from minivess.pipeline.inference import SlidingWindowInferenceRunner
from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.metrics import SegmentationMetrics
from minivess.pipeline.trainer import SegmentationTrainer

# Import system_monitor from same directory (works when run as script)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from system_monitor import MonitorConfig, SystemMonitor

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "configs" / "splits" / "3fold_seed42.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# With native resolution (no Spacingd) + ThreadDataLoader (no fork), full caching is safe.
# Total dataset at native res: ~4.5 GB (70 volumes × ~65 MB average).
# ThreadDataLoader uses threading, not multiprocessing, so no fork memory duplication.
SAFE_TRAIN_CACHE_RATE = 1.0
SAFE_VAL_CACHE_RATE = 1.0
SAFE_NUM_WORKERS = 2  # For CacheDataset initialization only
SAFE_EVAL_CACHE_RATE = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (superset of train.py args + monitoring args)."""
    parser = argparse.ArgumentParser(
        description="Memory-safe DynUNet training with system monitoring"
    )
    # Training args (same as train.py)
    parser.add_argument("--compute", type=str, default="cpu")
    parser.add_argument(
        "--model-family",
        type=str,
        default="dynunet",
        help="Model family name (e.g. dynunet, sam3_vanilla, sam3_topolora, sam3_hybrid)",
    )
    parser.add_argument("--loss", type=str, default="cbdice_cldice")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-file", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="dynunet_loss_variation")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patch-size", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from compute profile (e.g. --batch-size 1 for COMMA Mamba)",
    )

    # Monitoring args
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for monitor logs and checkpoints (auto-generated if not set)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=10.0,
        help="System monitor sampling interval in seconds",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=50.0,
        help="RAM usage warning threshold in GB (abort at limit+8)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint in --log-dir",
    )

    return parser.parse_args(argv)


def _build_configs(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Build configuration objects with memory-safe overrides."""
    data_config = DataConfig(dataset_name="minivess", data_dir=args.data_dir)
    arch_params = getattr(args, "architecture_params", None) or {}

    # Model-agnostic: use --model-family to dispatch to correct adapter
    model_family_str = getattr(args, "model_family", "dynunet")
    model_family = ModelFamily(model_family_str)

    # Extract LoRA params from architecture_params if present (SAM3 TopoLoRA)
    lora_rank = arch_params.pop("lora_rank", 16)
    lora_alpha = arch_params.pop("lora_alpha", 32.0)
    lora_dropout = arch_params.pop("lora_dropout", 0.1)

    model_config = ModelConfig(
        family=model_family,
        name=model_family_str,
        in_channels=1,
        out_channels=2,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        architecture_params=arch_params,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        num_folds=args.num_folds,
    )

    # Apply compute profile
    profile = get_compute_profile(args.compute)
    apply_profile(profile, data_config, training_config)

    # MEMORY FIX: Override num_workers to prevent forked-memory explosion
    if not args.debug:
        data_config.num_workers = min(data_config.num_workers, SAFE_NUM_WORKERS)
        logger.info(
            "Memory safety: num_workers capped at %d (profile wanted %d)",
            data_config.num_workers,
            profile.num_workers,
        )

    if args.max_epochs is not None:
        training_config.max_epochs = args.max_epochs

    if args.patch_size is not None:
        parts = [int(x) for x in args.patch_size.split("x")]
        if len(parts) == 3:
            data_config.patch_size = (parts[0], parts[1], parts[2])

    if getattr(args, "batch_size", None) is not None:
        training_config.batch_size = args.batch_size
        logger.info("Batch size overridden to %d via --batch-size", args.batch_size)

    if args.debug:
        apply_debug_overrides(training_config, data_config)

    # Parse checkpoint config from YAML/CLI override.
    # Multi-metric tracking is the ONLY mode — "single metric" is just a YAML
    # with one entry in tracked_metrics. There is no separate single-metric path.
    if hasattr(args, "checkpoint_config") and args.checkpoint_config:
        ckpt_cfg = args.checkpoint_config  # dict from YAML
        tracked = [
            TrackedMetricConfig(**m) for m in ckpt_cfg.get("tracked_metrics", [])
        ]
        if tracked:
            training_config.checkpoint = CheckpointConfig(
                tracked_metrics=tracked,
                early_stopping_strategy=ckpt_cfg.get("early_stopping_strategy", "all"),
                primary_metric=ckpt_cfg.get("primary_metric", "val_loss"),
                min_delta=ckpt_cfg.get("min_delta", 1e-4),
                min_epochs=ckpt_cfg.get("min_epochs", 0),
                save_last=ckpt_cfg.get("save_last", True),
                save_history=ckpt_cfg.get("save_history", True),
            )
    else:
        logger.warning(
            "No checkpoint config provided (--checkpoint-config or via experiment YAML). "
            "Using default single val_loss tracker. For multi-metric checkpointing, "
            "use run_experiment.py with a YAML config that includes a checkpoint section."
        )

    return data_config, model_config, training_config


def _load_or_generate_splits(
    args: argparse.Namespace,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> list:
    """Load splits from file or generate from data directory.

    When ``split_mode`` is ``"file"`` (default), always loads from the
    canonical splits JSON file for reproducibility. When ``"random"``,
    generates fresh splits from the seed (useful for ablation studies).
    """
    num_folds = training_config.num_folds
    split_mode = getattr(args, "split_mode", "file")

    # In "random" mode, always regenerate from seed
    if split_mode == "random":
        logger.info(
            "split_mode=random: Generating fresh %d-fold splits (seed=%d)",
            num_folds,
            training_config.seed,
        )
        splits = generate_kfold_splits_from_dir(
            data_config.data_dir,
            num_folds=num_folds,
            seed=training_config.seed,
        )
        save_splits(splits, args.splits_file)
        return splits

    # Default: "file" mode — load from canonical splits file
    if args.splits_file.exists():
        logger.info("Loading splits from %s (split_mode=file)", args.splits_file)
        splits = load_splits(args.splits_file)
        if len(splits) < num_folds:
            logger.warning(
                "Split file has %d folds but %d requested, regenerating",
                len(splits),
                num_folds,
            )
        else:
            return splits[:num_folds]

    logger.info("Generating %d-fold splits from %s", num_folds, data_config.data_dir)
    splits = generate_kfold_splits_from_dir(
        data_config.data_dir,
        num_folds=num_folds,
        seed=training_config.seed,
    )
    save_splits(splits, args.splits_file)
    return splits


def _cleanup_memory(tag: str = "") -> None:
    """Force memory cleanup between folds/losses.

    This is the primary OOM mitigation: ensures MONAI CacheDataset
    and PyTorch tensors are freed before the next allocation spike.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()  # Second pass to catch weak references freed by first pass

    # Log memory state after cleanup
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    logger.info(
                        "[MEMORY CLEANUP %s] Process RSS after gc: %.1f GB",
                        tag,
                        rss_kb / (1024 * 1024),
                    )
                    break
    except OSError:
        pass


class CheckpointManager:
    """Manages per-phase checkpoints for crash recovery.

    Checkpoint format::

        {
            "loss_name": "dice_ce",
            "completed_folds": [0, 1],
            "completed_losses": ["dice_ce"],
            "current_fold": 2,
            "current_loss": "cbdice",
            "mlflow_run_ids": {"dice_ce": "abc123"},
            "fold_results": {"dice_ce": [...]},
            "last_update": "2026-02-25T14:30:00+00:00",
        }
    """

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self._state: dict = {}

    def load(self) -> dict:
        """Load checkpoint from disk. Returns empty dict if not found."""
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._state = json.load(f)
            logger.info("Loaded checkpoint: %s", self.path)
            logger.info(
                "  Completed losses: %s",
                self._state.get("completed_losses", []),
            )
            logger.info(
                "  Current: loss=%s, fold=%s",
                self._state.get("current_loss", "none"),
                self._state.get("current_fold", "none"),
            )
        return self._state

    def save(self) -> None:
        """Persist checkpoint to disk."""
        self._state["last_update"] = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via temp file
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, default=str)
        tmp_path.rename(self.path)

    def mark_fold_start(self, loss_name: str, fold_id: int) -> None:
        """Record that a fold is starting."""
        self._state["current_loss"] = loss_name
        self._state["current_fold"] = fold_id
        self.save()

    def mark_fold_complete(self, loss_name: str, fold_id: int) -> None:
        """Record that a fold completed successfully."""
        key = f"{loss_name}_completed_folds"
        if key not in self._state:
            self._state[key] = []
        if fold_id not in self._state[key]:
            self._state[key].append(fold_id)
        self.save()

    def mark_loss_complete(self, loss_name: str, run_id: str | None = None) -> None:
        """Record that all folds for a loss function completed."""
        if "completed_losses" not in self._state:
            self._state["completed_losses"] = []
        if loss_name not in self._state["completed_losses"]:
            self._state["completed_losses"].append(loss_name)
        if run_id:
            if "mlflow_run_ids" not in self._state:
                self._state["mlflow_run_ids"] = {}
            self._state["mlflow_run_ids"][loss_name] = run_id
        self._state["current_loss"] = None
        self._state["current_fold"] = None
        self.save()

    def is_fold_complete(self, loss_name: str, fold_id: int) -> bool:
        """Check if a specific fold was already completed."""
        key = f"{loss_name}_completed_folds"
        return fold_id in self._state.get(key, [])

    def is_loss_complete(self, loss_name: str) -> bool:
        """Check if all folds for a loss function were completed."""
        return loss_name in self._state.get("completed_losses", [])


def run_fold_safe(
    fold_id: int,
    fold_split,
    *,
    loss_name: str,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tracker: ExperimentTracker | None = None,
    debug: bool = False,
    log_model_info: bool = False,
    condition: dict[str, Any] | None = None,
    precomputed_dir: Path | None = None,
    compute: str = "auto",
    system_monitor: Any | None = None,
) -> dict:
    """Train and evaluate a single fold with memory-safe cache rates.

    Identical to train.py's run_fold but with reduced cache_rate values
    and explicit memory cleanup. Supports optional condition-based model
    wrapping and multi-task auxiliary target loading.

    Parameters
    ----------
    condition:
        Optional condition config dict with 'wrappers' and 'd2c_enabled' keys.
        When provided, the model and loss are built via condition_builder
        and auxiliary targets are loaded into the data pipeline.
    precomputed_dir:
        Directory containing precomputed auxiliary NIfTI files.
    compute:
        Compute profile name (determines resource budget, not device).
        Device is always auto-detected (CUDA if available, else CPU).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_name = condition["name"] if condition else "none"
    logger.info(
        "=== Fold %d: loss=%s, condition=%s, device=%s ===",
        fold_id,
        loss_name,
        cond_name,
        device,
    )

    # Optionally limit data for debug
    train_dicts = fold_split.train
    val_dicts = fold_split.val
    if debug:
        train_dicts = train_dicts[:DEBUG_MAX_VOLUMES]
        val_dicts = val_dicts[: max(2, DEBUG_MAX_VOLUMES // 3)]

    # MEMORY FIX: Use reduced cache rates
    train_cache = DEBUG_CACHE_RATE if debug else SAFE_TRAIN_CACHE_RATE
    val_cache = DEBUG_CACHE_RATE if debug else SAFE_VAL_CACHE_RATE

    logger.info(
        "Cache rates: train=%.2f (%d vols), val=%.2f (%d vols)",
        train_cache,
        len(train_dicts),
        val_cache,
        len(val_dicts),
    )

    # Resolve auxiliary target configs for multi-task conditions
    aux_configs = None
    if condition:
        for wrapper in condition.get("wrappers", []):
            if wrapper.get("type") == "multitask":
                # Build AuxTargetConfig objects for the data pipeline
                from minivess.data.multitask_targets import AuxTargetConfig
                from minivess.pipeline.sdf_generation import compute_sdf_from_mask

                _AUX_COMPUTE_FNS: dict[str, Any] = {
                    "sdf": compute_sdf_from_mask,
                }
                try:
                    from minivess.adapters.centreline_head import (
                        compute_centreline_distance_map,
                    )

                    _AUX_COMPUTE_FNS["centerline_dist"] = (
                        compute_centreline_distance_map
                    )
                except ImportError:
                    pass

                aux_configs = []
                for head in wrapper.get("auxiliary_heads", []):
                    gt_key = head.get("gt_key", head["name"])
                    compute_fn = _AUX_COMPUTE_FNS.get(gt_key)
                    if compute_fn is not None:
                        aux_configs.append(
                            AuxTargetConfig(
                                name=gt_key,
                                suffix=gt_key,
                                compute_fn=compute_fn,
                            )
                        )
                break

        # Apply D2C config from condition
        if condition.get("d2c_enabled", False):
            data_config = data_config.model_copy()
            data_config.d2c_enabled = True
            data_config.d2c_probability = condition.get("d2c_probability", 0.3)

    train_loader = build_train_loader(
        train_dicts,
        data_config,
        batch_size=training_config.batch_size,
        cache_rate=train_cache,
        transforms=build_train_transforms(
            data_config,
            aux_configs=aux_configs,
            precomputed_dir=precomputed_dir,
        )
        if aux_configs
        else None,
    )
    val_loader = build_val_loader(
        val_dicts,
        data_config,
        cache_rate=val_cache,
        transforms=build_val_transforms(
            data_config,
            aux_configs=aux_configs,
            precomputed_dir=precomputed_dir,
        )
        if aux_configs
        else None,
    )

    # Build model — model-agnostic via build_adapter() factory
    base_model = build_adapter(model_config)

    if condition and condition.get("wrappers"):
        from minivess.pipeline.condition_builder import build_condition_model

        model = build_condition_model(base_model, condition)
    else:
        model = base_model

    # Log model info on first fold (architecture details + trainable params)
    if log_model_info and tracker is not None:
        try:
            tracker.log_model_info(model)
        except Exception:
            logger.warning("Failed to log model info", exc_info=True)

    # Build loss
    if condition and condition.get("wrappers"):
        from minivess.pipeline.condition_builder import build_condition_loss

        criterion = build_condition_loss(loss_name, condition)
    else:
        criterion = build_loss_function(loss_name)

    # Build metrics
    metrics = SegmentationMetrics(
        num_classes=model_config.out_channels,
        device=device,
    )

    # Build trainer — use sliding window inference for validation
    # because full 512x512xZ volumes don't fit in 8 GB VRAM
    # SAM3 models need sw_batch_size=1 to avoid OOM during sliding window
    # (each patch does slice-by-slice 1008x1008 processing)
    _is_sam3 = model_config.family.value.startswith("sam3_")
    _sw_bs = 1 if _is_sam3 else 4

    trainer = SegmentationTrainer(
        model,
        training_config,
        device=device,
        tracker=tracker,
        metrics=metrics,
        criterion=criterion,
        val_roi_size=data_config.patch_size,
        sw_batch_size=_sw_bs,
        fold_label=f"f #{fold_id + 1}/{training_config.num_folds}",
        system_monitor=system_monitor,
    )

    # Create checkpoint directory — resolved from CHECKPOINT_DIR env var (never /tmp)
    checkpoint_dir = (
        Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")) / f"fold_{fold_id}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Train ===
    result = trainer.fit(train_loader, val_loader, checkpoint_dir=checkpoint_dir)

    logger.info(
        "Fold %d training complete: best_val_loss=%.4f, epochs=%d",
        fold_id,
        result["best_val_loss"],
        result["final_epoch"],
    )

    # MEMORY FIX: Free training data before loading eval data
    del train_loader
    del trainer
    del metrics
    del criterion
    _cleanup_memory(f"post-train-fold{fold_id}")

    # === Phase 2: Post-training evaluation with MetricsReloaded ===
    # Use primary metric checkpoint (e.g. best_val_loss.pth)
    primary_metric = training_config.checkpoint.primary_metric
    safe_name = primary_metric.replace("/", "_")
    best_checkpoint = checkpoint_dir / f"best_{safe_name}.pth"
    if best_checkpoint.exists():
        logger.info("Loading best checkpoint for evaluation: %s", best_checkpoint)
        model.load_checkpoint(best_checkpoint)

        # Build inference runner
        inference_runner = SlidingWindowInferenceRunner(
            roi_size=data_config.patch_size,
            num_classes=model_config.out_channels,
            overlap=0.25 if not debug else 0.0,
            sw_batch_size=_sw_bs,
        )

        # MEMORY FIX: Rebuild val_loader with reduced cache for inference
        eval_cache = DEBUG_CACHE_RATE if debug else SAFE_EVAL_CACHE_RATE
        eval_loader = build_val_loader(
            val_dicts,
            data_config,
            cache_rate=eval_cache,
        )

        # Run inference
        logger.info(
            "Running sliding window inference on %d validation volumes",
            len(val_dicts),
        )
        predictions, labels = inference_runner.infer_dataset(
            model, eval_loader, device=device
        )

        # MEMORY FIX: Free inference runner and eval loader
        del eval_loader
        del inference_runner
        _cleanup_memory(f"post-infer-fold{fold_id}")

        # Run MetricsReloaded evaluation
        logger.info("Running MetricsReloaded evaluation for fold %d", fold_id)
        runner = EvaluationRunner(include_expensive=False)
        fold_result = runner.evaluate_fold(
            predictions,
            labels,
            n_resamples=100 if debug else 1000,
            seed=42,
        )

        # Log summary
        for metric_name, ci in fold_result.aggregated.items():
            logger.info(
                "  Fold %d %s: %.4f [%.4f, %.4f]",
                fold_id,
                metric_name,
                ci.point_estimate,
                ci.lower,
                ci.upper,
            )

        # Log to MLflow tracker if available
        if tracker is not None:
            tracker.log_evaluation_results(
                fold_result, fold_id=fold_id, loss_name=loss_name
            )

        result["evaluation"] = fold_result

        # MEMORY FIX: Free evaluation data
        del predictions
        del labels
        del fold_result
        del runner
    else:
        logger.warning("No best checkpoint found, skipping evaluation")

    # MEMORY FIX: Free model last
    del model
    _cleanup_memory(f"end-fold{fold_id}")

    # Checkpoints persist on the mounted volume (CHECKPOINT_DIR) — no cleanup needed.

    return result


def run_monitored_experiment(args: argparse.Namespace) -> dict:
    """Run the full experiment with monitoring and checkpointing."""
    # Setup log directory
    if args.log_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        args.log_dir = PROJECT_ROOT / "logs" / f"train_{ts}"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging (in addition to console)
    file_handler = logging.FileHandler(args.log_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("MONITORED TRAINING SESSION")
    logger.info("  Log dir: %s", args.log_dir)
    logger.info("  Resume: %s", args.resume)
    logger.info("=" * 70)

    # Check swap usage — heavy swap from previous OOM crashes is a red flag
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        swap_used_gb = (meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)) / (
            1024 * 1024
        )
        ram_available_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)
        if swap_used_gb > 5.0:
            logger.warning(
                "HIGH SWAP USAGE: %.1f GB in swap. This likely remains from previous OOM crashes. "
                "Consider clearing swap before training: sudo swapoff -a && sudo swapon -a "
                "(will move ~%.1f GB back to RAM, need %.1f GB available, have %.1f GB).",
                swap_used_gb,
                swap_used_gb,
                swap_used_gb,
                ram_available_gb,
            )
        logger.info(
            "Pre-training memory: %.1f GB RAM available, %.1f GB swap used",
            ram_available_gb,
            swap_used_gb,
        )
    except OSError:
        pass

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(args.log_dir / "checkpoint.json")
    if args.resume:
        checkpoint_mgr.load()

    # Initialize system monitor — use process RSS for abort, not system-wide RAM
    # (system-wide is misleading when swap is heavily used from previous crashes)
    monitor_config = MonitorConfig(
        interval_sec=args.monitor_interval,
        log_dir=args.log_dir / "monitor",
        memory_warn_gb=args.memory_limit_gb,
        memory_abort_gb=args.memory_limit_gb + 8,
        enable_gpu=True,
        track_pid=os.getpid(),
    )

    abort_requested = {"value": False}

    def _on_memory_abort(snapshot):
        # Use process RSS for abort decision, not system-wide RAM
        # System-wide can be inflated by swap from previous OOM crashes
        if snapshot.process_rss_gb > 0 and snapshot.process_rss_gb < 35.0:
            logger.warning(
                "System RAM at %.1f GB but process RSS only %.1f GB — "
                "swap from previous crashes inflating system metric. Continuing.",
                snapshot.ram_used_gb,
                snapshot.process_rss_gb,
            )
            return
        abort_requested["value"] = True
        logger.critical(
            "MEMORY ABORT: %.1f GB system RAM, %.1f GB process RSS. "
            "Saving checkpoint and stopping.",
            snapshot.ram_used_gb,
            snapshot.process_rss_gb,
        )

    monitor = SystemMonitor(
        config=monitor_config,
        on_memory_abort=_on_memory_abort,
    )
    monitor.start()

    try:
        return _run_experiment_inner(args, checkpoint_mgr, abort_requested, monitor)
    except Exception:
        logger.exception("TRAINING FAILED WITH EXCEPTION")
        # Save crash info to checkpoint
        checkpoint_mgr._state["crash_info"] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "traceback": traceback.format_exc(),
        }
        checkpoint_mgr.save()
        raise
    finally:
        summary = monitor.stop()
        logger.info("Monitor summary: %s", json.dumps(summary, indent=2))

        # Save final state
        final_state = {
            "completed_at": datetime.now(UTC).isoformat(),
            "monitor_summary": summary,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        with open(args.log_dir / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2)


def _run_experiment_inner(
    args: argparse.Namespace,
    checkpoint_mgr: CheckpointManager,
    abort_requested: dict,
    monitor: SystemMonitor | None = None,
) -> dict:
    """Inner experiment loop with checkpoint awareness."""
    data_config, model_config, training_config = _build_configs(args)
    loss_names = [name.strip() for name in args.loss.split(",")]

    logger.info(
        "Experiment: losses=%s, folds=%d, epochs=%d, debug=%s",
        loss_names,
        training_config.num_folds,
        training_config.max_epochs,
        args.debug,
    )

    # Load or generate splits
    splits = _load_or_generate_splits(args, data_config, training_config)

    # Build experiment config for tracker
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        data=data_config,
        model=model_config,
        training=training_config,
    )
    tracker = ExperimentTracker(experiment_config)

    all_results: dict[str, list[dict]] = {}
    all_eval_results: dict[str, list[FoldResult]] = {}

    for loss_name in loss_names:
        # Skip completed losses on resume
        if args.resume and checkpoint_mgr.is_loss_complete(loss_name):
            logger.info("SKIP (already complete): loss=%s", loss_name)
            continue

        if abort_requested["value"]:
            logger.warning("Aborting due to memory pressure")
            break

        logger.info("--- Loss function: %s ---", loss_name)
        fold_results: list[dict] = []
        fold_eval_results: list[FoldResult] = []

        run_name = f"{loss_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        import time as _time_mod

        _run_start = _time_mod.monotonic()
        with tracker.start_run(
            run_name=run_name,
            tags={"loss_function": loss_name, "num_folds": str(len(splits))},
        ):
            # Log loss as param (not just tag) + fold info + device + split file
            import mlflow as _mlflow

            _device_str = "cuda" if torch.cuda.is_available() else "cpu"
            _extra_params: dict[str, str] = {
                "loss_name": loss_name,
                "fold_ids": ",".join(str(i) for i in range(len(splits))),
                "device": _device_str,
            }
            # Volume counts from first split (all folds have same total)
            if splits:
                _extra_params["data_n_train_volumes"] = str(len(splits[0].train))
                _extra_params["data_n_val_volumes"] = str(len(splits[0].val))
                _extra_params["data_dataset_name"] = data_config.dataset_name
            _mlflow.log_params(_extra_params)

            # Log fold splits (artifact + per-fold tags + split_mode)
            _split_mode = getattr(args, "split_mode", "file")
            _splits_path = (
                Path(args.splits_file)
                if hasattr(args, "splits_file") and args.splits_file
                else None
            )
            tracker.log_fold_splits(
                splits,
                splits_file=_splits_path,
                split_mode=_split_mode,
            )

            _model_info_logged = False
            for fold_id, fold_split in enumerate(splits):
                # Skip completed folds on resume
                if args.resume and checkpoint_mgr.is_fold_complete(loss_name, fold_id):
                    logger.info(
                        "SKIP (already complete): loss=%s fold=%d",
                        loss_name,
                        fold_id,
                    )
                    continue

                if abort_requested["value"]:
                    logger.warning("Aborting due to memory pressure")
                    break

                checkpoint_mgr.mark_fold_start(loss_name, fold_id)

                fold_result = run_fold_safe(
                    fold_id,
                    fold_split,
                    loss_name=loss_name,
                    data_config=data_config,
                    model_config=model_config,
                    training_config=training_config,
                    tracker=tracker,
                    debug=args.debug,
                    log_model_info=(not _model_info_logged),
                    condition=getattr(args, "condition", None),
                    precomputed_dir=getattr(args, "precomputed_dir", None),
                    compute=getattr(args, "compute", "auto"),
                    system_monitor=monitor,
                )
                _model_info_logged = True
                fold_results.append(fold_result)

                if "evaluation" in fold_result:
                    fold_eval_results.append(fold_result["evaluation"])

                checkpoint_mgr.mark_fold_complete(loss_name, fold_id)

                # MEMORY FIX: Force cleanup between folds
                _cleanup_memory(f"between-folds-{loss_name}")

            # Log training time
            _training_time = _time_mod.monotonic() - _run_start
            _mlflow.log_param("training_time_seconds", str(round(_training_time, 1)))

            # T8: Upload system monitor CSV + JSONL to MLflow artifact store
            if monitor is not None:
                if monitor.csv_path is not None and monitor.csv_path.exists():
                    _mlflow.log_artifact(str(monitor.csv_path), "system_monitor")
                if monitor.jsonl_path is not None and monitor.jsonl_path.exists():
                    _mlflow.log_artifact(str(monitor.jsonl_path), "system_monitor")

        all_results[loss_name] = fold_results
        all_eval_results[loss_name] = fold_eval_results
        checkpoint_mgr.mark_loss_complete(loss_name)

        # MEMORY FIX: Force cleanup between loss functions
        _cleanup_memory(f"between-losses-after-{loss_name}")

    # Build cross-loss comparison
    from train import build_comparison_summary

    comparison = build_comparison_summary(all_eval_results)

    return {
        "training": all_results,
        "evaluation": all_eval_results,
        "comparison": comparison,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = parse_args(argv)
    results = run_monitored_experiment(args)

    # Training summary
    for loss_name, fold_results in results["training"].items():
        best_losses = [r["best_val_loss"] for r in fold_results]
        if best_losses:
            mean_loss = sum(best_losses) / len(best_losses)
            logger.info(
                "Loss=%s: mean_best_val_loss=%.4f across %d folds",
                loss_name,
                mean_loss,
                len(fold_results),
            )

    # Evaluation comparison summary
    comparison = results.get("comparison", {})
    if comparison:
        logger.info("=== Cross-Loss Comparison ===")
        for loss_name, metrics in comparison.items():
            for metric_name, stats in metrics.items():
                logger.info(
                    "  %s / %s: mean=%.4f [%.4f, %.4f] std=%.4f",
                    loss_name,
                    metric_name,
                    stats["mean"],
                    stats["ci_lower"],
                    stats["ci_upper"],
                    stats["std"],
                )


if __name__ == "__main__":
    # Deprecation gate — direct invocation is no longer supported.
    # Set ALLOW_STANDALONE_TRAINING=1 to bypass (e.g. in CI or migration).
    warnings.warn(
        "\n\n"
        "DEPRECATED ENTRY POINT: scripts/train_monitored.py\n"
        "This script is NOT the supported way to run training.\n"
        "The correct entry point is:\n"
        "  prefect deployment run 'training-flow/default' --params '{...}'\n"
        "Or use the shell wrapper: scripts/run_training.sh <loss> <model>\n"
        "This script will be removed in a future release.\n"
        "Set ALLOW_STANDALONE_TRAINING=1 to suppress this warning and continue.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not os.environ.get("ALLOW_STANDALONE_TRAINING"):
        sys.exit(1)
    main()
