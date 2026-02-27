#!/usr/bin/env python
"""DynUNet loss variation experiment training script.

Entry point for running 3D vessel segmentation experiments with
K-fold cross-validation and configurable loss functions.

Usage::

    # Full training
    uv run python scripts/train.py compute=gpu_low loss=dice_ce

    # Debug mode (1 epoch, data subset, fast)
    uv run python scripts/train.py compute=cpu loss=dice_ce --debug

    # Multi-loss sweep
    uv run python scripts/train.py compute=gpu_low loss=dice_ce,cbdice,dice_ce_cldice
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

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

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "configs" / "splits" / "3fold_seed42.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="DynUNet loss variation experiment")
    parser.add_argument(
        "--compute",
        type=str,
        default="cpu",
        help="Compute profile name (cpu, gpu_low, gpu_high, dgx_spark, cloud_single, cloud_multi)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cbdice_cldice",
        help="Loss function name(s), comma-separated for sweep (e.g. dice_ce,cbdice)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (1 epoch, data subset, fast CI)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to data directory with NIfTI files",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to splits JSON file (generated if not present)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dynunet_loss_variation",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (optional)",
    )
    parser.add_argument(
        "--patch-size",
        type=str,
        default=None,
        help="Override patch size as DxHxW (e.g. 64x64x16)",
    )
    return parser.parse_args(argv)


def _build_configs(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Build configuration objects from CLI args."""
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

    # Apply max_epochs override if provided
    if args.max_epochs is not None:
        training_config.max_epochs = args.max_epochs

    # Apply patch size override if provided
    if args.patch_size is not None:
        parts = [int(x) for x in args.patch_size.split("x")]
        if len(parts) == 3:
            data_config.patch_size = tuple(parts)

    # Apply debug overrides last (overrides profile settings)
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

    # Generate from data directory
    logger.info("Generating %d-fold splits from %s", num_folds, data_config.data_dir)
    splits = generate_kfold_splits_from_dir(
        data_config.data_dir,
        num_folds=num_folds,
        seed=training_config.seed,
    )
    save_splits(splits, args.splits_file)
    return splits


def evaluate_fold_and_log(
    *,
    predictions: list[np.ndarray],
    labels: list[np.ndarray],
    tracker: ExperimentTracker | None = None,
    fold_id: int,
    loss_name: str,
    include_expensive: bool = False,
    n_resamples: int = 1000,
    seed: int = 42,
) -> FoldResult:
    """Run MetricsReloaded evaluation on fold predictions and log to MLflow.

    Parameters
    ----------
    predictions:
        List of binary prediction arrays (D, H, W).
    labels:
        List of binary ground truth arrays (D, H, W).
    tracker:
        Optional MLflow tracker (metrics logged if provided).
    fold_id:
        Fold index (0-based).
    loss_name:
        Loss function name (for logging context).
    include_expensive:
        If True, compute HD95 and NSD (slower).
    n_resamples:
        Bootstrap resamples for CI computation.
    seed:
        Random seed for bootstrap.

    Returns
    -------
    FoldResult
        Per-volume metrics and aggregated CIs.
    """
    runner = EvaluationRunner(include_expensive=include_expensive)
    fold_result = runner.evaluate_fold(
        predictions,
        labels,
        n_resamples=n_resamples,
        seed=seed,
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

    return fold_result


def build_comparison_summary(
    eval_results: dict[str, list[FoldResult]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build a cross-loss comparison summary from per-fold evaluation results.

    Averages per-fold point estimates and CI bounds across folds
    for each loss function.

    Parameters
    ----------
    eval_results:
        Dict mapping loss_name to list of FoldResult (one per fold).

    Returns
    -------
    dict
        Nested dict: ``{loss_name: {metric_name: {mean, ci_lower, ci_upper}}}``.
    """
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for loss_name, fold_results in eval_results.items():
        loss_summary: dict[str, dict[str, float]] = {}

        # Collect all metric names from the first fold
        if not fold_results:
            continue
        metric_names = list(fold_results[0].aggregated.keys())

        for metric_name in metric_names:
            point_estimates = []
            lowers = []
            uppers = []

            for fr in fold_results:
                if metric_name in fr.aggregated:
                    ci = fr.aggregated[metric_name]
                    point_estimates.append(ci.point_estimate)
                    lowers.append(ci.lower)
                    uppers.append(ci.upper)

            if point_estimates:
                loss_summary[metric_name] = {
                    "mean": float(np.mean(point_estimates)),
                    "ci_lower": float(np.mean(lowers)),
                    "ci_upper": float(np.mean(uppers)),
                    "std": float(np.std(point_estimates)),
                }

        summary[loss_name] = loss_summary

    return summary


def run_fold(
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
    """Train and evaluate a single fold.

    Trains the model, then loads the best checkpoint and runs sliding
    window inference + MetricsReloaded evaluation on the validation set.

    Parameters
    ----------
    fold_id:
        Fold index (0-based).
    fold_split:
        FoldSplit with train/val data dicts.
    loss_name:
        Loss function name for build_loss_function().
    data_config:
        Data configuration.
    model_config:
        Model configuration.
    training_config:
        Training configuration.
    tracker:
        MLflow experiment tracker.
    debug:
        Whether debug mode is active.

    Returns
    -------
    dict
        Training summary with best_val_loss, final_epoch, metrics,
        and evaluation FoldResult.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== Fold %d: loss=%s, device=%s ===", fold_id, loss_name, device)

    # Optionally limit data for debug
    train_dicts = fold_split.train
    val_dicts = fold_split.val
    if debug:
        train_dicts = train_dicts[:DEBUG_MAX_VOLUMES]
        val_dicts = val_dicts[: max(2, DEBUG_MAX_VOLUMES // 3)]

    # Build data loaders
    cache_rate = DEBUG_CACHE_RATE if debug else 0.5
    train_loader = build_train_loader(
        train_dicts,
        data_config,
        batch_size=training_config.batch_size,
        cache_rate=cache_rate,
    )
    val_loader = build_val_loader(
        val_dicts,
        data_config,
        cache_rate=1.0 if not debug else DEBUG_CACHE_RATE,
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
        )

        # Rebuild val_loader for full-volume inference (batch_size=1, no crop)
        eval_loader = build_val_loader(
            val_dicts,
            data_config,
            cache_rate=1.0 if not debug else DEBUG_CACHE_RATE,
        )

        # Run inference
        logger.info(
            "Running sliding window inference on %d validation volumes", len(val_dicts)
        )
        predictions, labels = inference_runner.infer_dataset(
            model, eval_loader, device=device
        )

        # Run MetricsReloaded evaluation
        logger.info("Running MetricsReloaded evaluation for fold %d", fold_id)
        eval_result = evaluate_fold_and_log(
            predictions=predictions,
            labels=labels,
            tracker=tracker,
            fold_id=fold_id,
            loss_name=loss_name,
            n_resamples=100 if debug else 1000,
            seed=42,
        )
        result["evaluation"] = eval_result
    else:
        logger.warning("No best checkpoint found, skipping evaluation")

    return result


def run_experiment(args: argparse.Namespace) -> dict:
    """Run the full experiment (all loss functions x all folds).

    Returns
    -------
    dict
        Results with ``training`` (per-fold dicts) and ``evaluation``
        (per-fold FoldResults) keyed by loss_name, plus a
        ``comparison`` summary.
    """
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
        logger.info("--- Loss function: %s ---", loss_name)
        fold_results: list[dict] = []
        fold_eval_results: list[FoldResult] = []

        run_name = f"{loss_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        with tracker.start_run(
            run_name=run_name,
            tags={"loss_function": loss_name, "num_folds": str(len(splits))},
        ):
            for fold_id, fold_split in enumerate(splits):
                fold_result = run_fold(
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

                # Collect evaluation result if present
                if "evaluation" in fold_result:
                    fold_eval_results.append(fold_result["evaluation"])

        all_results[loss_name] = fold_results
        all_eval_results[loss_name] = fold_eval_results

    # Build cross-loss comparison summary
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
    results = run_experiment(args)

    # Training summary
    for loss_name, fold_results in results["training"].items():
        best_losses = [r["best_val_loss"] for r in fold_results]
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
