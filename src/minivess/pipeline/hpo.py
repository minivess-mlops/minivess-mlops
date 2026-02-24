"""Optuna HPO integration for hyperparameter optimization.

Provides search space definitions, study creation, trial-to-config
conversion, and an orchestrator for running HPO with SegmentationTrainer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optuna

from minivess.config.models import TrainingConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Defines parameter ranges for Optuna HPO.

    Default ranges target DynUNet-style 3D segmentation networks.

    Parameters
    ----------
    lr_low, lr_high:
        Learning rate range (log scale).
    weight_decay_low, weight_decay_high:
        Weight decay range (log scale).
    batch_size_low, batch_size_high:
        Batch size range (integer).
    optimizers:
        List of optimizer names to search over.
    warmup_epochs_low, warmup_epochs_high:
        Warmup epoch range.
    gradient_clip_low, gradient_clip_high:
        Gradient clipping value range.
    """

    lr_low: float = 1e-5
    lr_high: float = 1e-2
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 1e-3
    batch_size_low: int = 1
    batch_size_high: int = 4
    optimizers: list[str] = field(default_factory=lambda: ["adamw", "sgd"])
    warmup_epochs_low: int = 0
    warmup_epochs_high: int = 10
    gradient_clip_low: float = 0.5
    gradient_clip_high: float = 2.0


def create_study(
    study_name: str = "hpo_study",
    *,
    storage: str | None = None,
    direction: str = "minimize",
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Create an Optuna study with MedianPruner by default.

    Parameters
    ----------
    study_name:
        Name for the study.
    storage:
        Optuna storage URL (e.g. ``sqlite:///hpo.db``). None for in-memory.
    direction:
        Optimization direction (``"minimize"`` or ``"maximize"``).
    pruner:
        Custom pruner. Defaults to MedianPruner.

    Returns
    -------
    optuna.Study
    """
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
    )


def build_trial_config(
    trial: optuna.Trial,
    space: SearchSpace,
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    """Convert an Optuna trial's suggestions into a TrainingConfig.

    Parameters
    ----------
    trial:
        Active Optuna trial providing parameter suggestions.
    space:
        SearchSpace defining parameter ranges.
    base_config:
        Optional base config whose non-HPO fields are preserved.

    Returns
    -------
    TrainingConfig with HPO-sampled parameters.
    """
    lr = trial.suggest_float("learning_rate", space.lr_low, space.lr_high, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", space.weight_decay_low, space.weight_decay_high, log=True
    )
    batch_size = trial.suggest_int(
        "batch_size", space.batch_size_low, space.batch_size_high
    )
    optimizer = trial.suggest_categorical("optimizer", space.optimizers)
    warmup_epochs = trial.suggest_int(
        "warmup_epochs", space.warmup_epochs_low, space.warmup_epochs_high
    )
    gradient_clip_val = trial.suggest_float(
        "gradient_clip_val", space.gradient_clip_low, space.gradient_clip_high
    )

    # Start from base config or defaults
    base_fields: dict[str, Any] = {}
    if base_config is not None:
        base_fields = base_config.model_dump()

    # Override with HPO-sampled values
    base_fields.update(
        {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "warmup_epochs": warmup_epochs,
            "gradient_clip_val": gradient_clip_val,
        }
    )

    return TrainingConfig(**base_fields)


def make_objective(
    train_fn: Callable[[TrainingConfig], dict[str, Any]],
    search_space: SearchSpace,
    base_config: TrainingConfig | None = None,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function from a training function.

    Parameters
    ----------
    train_fn:
        Function that takes a TrainingConfig and returns a dict with
        ``"best_val_loss"`` key.
    search_space:
        SearchSpace defining parameter ranges.
    base_config:
        Optional base config for non-HPO fields.

    Returns
    -------
    Callable that accepts an optuna.Trial and returns the objective value.
    """

    def objective(trial: optuna.Trial) -> float:
        config = build_trial_config(trial, search_space, base_config=base_config)
        result = train_fn(config)
        return float(result["best_val_loss"])

    return objective


def run_hpo(
    objective_fn: Callable[[optuna.Trial], float],
    search_space: SearchSpace,
    n_trials: int = 20,
    *,
    study_name: str = "hpo_study",
    storage: str | None = None,
) -> dict[str, Any]:
    """Run HPO optimization and return best parameters.

    Parameters
    ----------
    objective_fn:
        Objective function accepting an optuna.Trial.
    search_space:
        SearchSpace (used for study metadata).
    n_trials:
        Number of trials to run.
    study_name:
        Name for the Optuna study.
    storage:
        Optuna storage URL.

    Returns
    -------
    dict with ``best_params``, ``best_value``, and ``study``.
    """
    study = create_study(study_name=study_name, storage=storage)

    study.optimize(objective_fn, n_trials=n_trials)

    logger.info(
        "HPO complete: best_value=%.4f, best_params=%s",
        study.best_value,
        study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
