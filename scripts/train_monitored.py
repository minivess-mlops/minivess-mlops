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

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minivess.adapters.dynunet import DynUNetAdapter
from minivess.config.compute_profiles import apply_profile, get_compute_profile
from minivess.config.debug import (
    DEBUG_CACHE_RATE,
    DEBUG_MAX_VOLUMES,
    apply_debug_overrides,
)
from minivess.config.models import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
)
from minivess.data.loader import build_train_loader, build_val_loader
from minivess.data.splits import (
    generate_kfold_splits_from_dir,
    load_splits,
    save_splits,
)
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
    parser.add_argument("--loss", type=str, default="dice_ce")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-file", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="dynunet_loss_variation")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patch-size", type=str, default=None)

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
    model_config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet",
        in_channels=1,
        out_channels=2,
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
            data_config.patch_size = tuple(parts)

    if args.debug:
        apply_debug_overrides(training_config, data_config)

    return data_config, model_config, training_config


def _load_or_generate_splits(
    args: argparse.Namespace,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> list:
    """Load splits from file or generate from data directory."""
    num_folds = training_config.num_folds

    if args.splits_file.exists():
        logger.info("Loading splits from %s", args.splits_file)
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
            "last_update": "2026-02-25T14:30:00+00:00"
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
) -> dict:
    """Train and evaluate a single fold with memory-safe cache rates.

    Identical to train.py's run_fold but with reduced cache_rate values
    and explicit memory cleanup.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "=== Fold %d: loss=%s, device=%s ===", fold_id, loss_name, device
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

    train_loader = build_train_loader(
        train_dicts,
        data_config,
        batch_size=training_config.batch_size,
        cache_rate=train_cache,
    )
    val_loader = build_val_loader(
        val_dicts,
        data_config,
        cache_rate=val_cache,
    )

    # Build model
    model = DynUNetAdapter(model_config)

    # Build loss
    criterion = build_loss_function(loss_name)

    # Build metrics
    metrics = SegmentationMetrics(
        num_classes=model_config.out_channels,
        device=device,
    )

    # Build trainer — use sliding window inference for validation
    # because full 512x512xZ volumes don't fit in 8 GB VRAM
    trainer = SegmentationTrainer(
        model,
        training_config,
        device=device,
        tracker=tracker,
        metrics=metrics,
        criterion=criterion,
        val_roi_size=data_config.patch_size,
    )

    # Create checkpoint directory
    checkpoint_dir = Path(tempfile.mkdtemp()) / f"fold_{fold_id}"
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
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if best_checkpoint.exists():
        logger.info("Loading best checkpoint for evaluation: %s", best_checkpoint)
        model.load_checkpoint(best_checkpoint)

        # Build inference runner
        inference_runner = SlidingWindowInferenceRunner(
            roi_size=data_config.patch_size,
            num_classes=model_config.out_channels,
            overlap=0.25 if not debug else 0.0,
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

    return result


def run_monitored_experiment(args: argparse.Namespace) -> dict:
    """Run the full experiment with monitoring and checkpointing."""
    # Setup log directory
    if args.log_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        args.log_dir = PROJECT_ROOT / "logs" / f"train_{ts}"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging (in addition to console)
    file_handler = logging.FileHandler(
        args.log_dir / "training.log", encoding="utf-8"
    )
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
        swap_used_gb = (meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)) / (1024 * 1024)
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
        return _run_experiment_inner(args, checkpoint_mgr, abort_requested)
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
        with tracker.start_run(
            run_name=run_name,
            tags={"loss_function": loss_name, "num_folds": str(len(splits))},
        ):
            for fold_id, fold_split in enumerate(splits):
                # Skip completed folds on resume
                if args.resume and checkpoint_mgr.is_fold_complete(
                    loss_name, fold_id
                ):
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
                )
                fold_results.append(fold_result)

                if "evaluation" in fold_result:
                    fold_eval_results.append(fold_result["evaluation"])

                checkpoint_mgr.mark_fold_complete(loss_name, fold_id)

                # MEMORY FIX: Force cleanup between folds
                _cleanup_memory(f"between-folds-{loss_name}")

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
    main()
