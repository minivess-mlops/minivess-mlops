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
        default="dice_ce",
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
        Training summary with best_val_loss, final_epoch, metrics.
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

    # Build trainer
    trainer = SegmentationTrainer(
        model,
        training_config,
        device=device,
        tracker=tracker,
        metrics=metrics,
        criterion=criterion,
    )

    # Create checkpoint directory
    checkpoint_dir = Path(tempfile.mkdtemp()) / f"fold_{fold_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    result = trainer.fit(train_loader, val_loader, checkpoint_dir=checkpoint_dir)

    logger.info(
        "Fold %d complete: best_val_loss=%.4f, epochs=%d",
        fold_id,
        result["best_val_loss"],
        result["final_epoch"],
    )

    return result


def run_experiment(args: argparse.Namespace) -> dict:
    """Run the full experiment (all loss functions x all folds).

    Returns
    -------
    dict
        Results keyed by loss_name containing per-fold results.
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

    for loss_name in loss_names:
        logger.info("--- Loss function: %s ---", loss_name)
        fold_results: list[dict] = []

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

        all_results[loss_name] = fold_results

    return all_results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = parse_args(argv)
    results = run_experiment(args)

    # Summary
    for loss_name, fold_results in results.items():
        best_losses = [r["best_val_loss"] for r in fold_results]
        mean_loss = sum(best_losses) / len(best_losses)
        logger.info(
            "Loss=%s: mean_best_val_loss=%.4f across %d folds",
            loss_name,
            mean_loss,
            len(fold_results),
        )


if __name__ == "__main__":
    main()
