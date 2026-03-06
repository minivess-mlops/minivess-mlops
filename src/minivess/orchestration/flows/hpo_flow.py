"""HPO Prefect flow — hyperparameter optimization as a managed flow.

Flow 2.5: Wraps HPOEngine (Optuna + ASHA) as a Prefect @flow.
Each trial calls training_flow() with suggested hyperparameters and
reports val_loss back to Optuna.

This enables:
- Prefect UI visibility into HPO progress
- Work pool routing (GPU pool)
- Integration with the trigger chain
- Fault-tolerant parallel trial execution
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from minivess.orchestration._prefect_compat import flow, task
from minivess.orchestration.flows.train_flow import training_flow

logger = logging.getLogger(__name__)

# Suppress Optuna's INFO logs unless debugging — they're verbose
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Default HPO search space for DynUNet training
_DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [1, 2, 4]},
    "loss_name": {
        "type": "categorical",
        "choices": ["cbdice_cldice", "dice_ce", "dice_ce_cldice"],
    },
}


@task(name="run-hpo-trial")
def run_trial_task(
    trial_number: int,
    params: dict[str, Any],
    base_config: dict[str, Any],
) -> float:
    """Run one HPO trial and return the objective value (val_loss).

    Parameters
    ----------
    trial_number:
        Optuna trial number (for logging).
    params:
        Hyperparameters suggested by Optuna for this trial.
    base_config:
        Base training config dict (overridden by trial params).

    Returns
    -------
    Best val_loss across all folds for this trial.
    """
    config = {**base_config, **params}
    logger.info(
        "Trial %d: %s",
        trial_number,
        {k: v for k, v in params.items()},
    )

    raw_result: Any = training_flow(
        loss_name=config.get("loss_name", "cbdice_cldice"),
        model_family=config.get("model_family", "dynunet"),
        compute=config.get("compute", "auto"),
        debug=config.get("debug", False),
        experiment_name=config.get("experiment_name", "minivess_hpo"),
        num_folds=config.get("num_folds", 1),
        max_epochs=config.get("max_epochs", 20),
        batch_size=config.get("batch_size", 2),
        learning_rate=config.get("learning_rate", 1e-3),
    )

    # Extract objective value (minimize val_loss)
    fold_results: list[dict[str, Any]] = (
        raw_result.fold_results if hasattr(raw_result, "fold_results") else []
    )
    if fold_results:
        val_losses = [r.get("best_val_loss", float("inf")) for r in fold_results]
        return float(min(val_losses))
    return float("inf")


@flow(name="hpo-flow")
def hpo_flow(
    *,
    n_trials: int = 20,
    study_name: str = "minivess_hpo",
    sampler: str = "tpe",
    pruner: str | None = "hyperband",
    search_space: dict[str, dict[str, Any]] | None = None,
    base_config: dict[str, Any] | None = None,
    trigger_source: str = "manual",
) -> dict[str, Any]:
    """HPO Prefect flow — runs n_trials of Optuna optimization.

    Each trial calls training_flow() with Optuna-suggested hyperparameters
    and reports the resulting val_loss as the objective value.

    Parameters
    ----------
    n_trials:
        Number of Optuna trials to run.
    study_name:
        Optuna study name (used for persistence across runs).
    sampler:
        Optuna sampler: ``"tpe"`` (default), ``"cmaes"``, ``"random"``.
    pruner:
        Optuna pruner: ``"hyperband"`` (ASHA, default) or ``None``.
    search_space:
        Dict mapping param names to their search specs. Defaults to
        ``_DEFAULT_SEARCH_SPACE`` (learning_rate, batch_size, loss_name).
    base_config:
        Base training config dict merged with trial-suggested params.
    trigger_source:
        What triggered this flow (for logging).

    Returns
    -------
    Dict with ``best_params``, ``best_value``, ``n_trials``, ``study_name``.
    """
    logger.info(
        "HPO flow started: study=%r, n_trials=%d, sampler=%s (trigger: %s)",
        study_name,
        n_trials,
        sampler,
        trigger_source,
    )

    if search_space is None:
        search_space = _DEFAULT_SEARCH_SPACE
    if base_config is None:
        base_config = {}

    from minivess.optimization.hpo_engine import HPOEngine

    engine = HPOEngine(
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
    )
    study = engine.create_study(direction="minimize")

    def _objective(trial: optuna.Trial) -> float:
        params = engine.suggest_params(trial, search_space)
        return run_trial_task(trial.number, params, base_config)  # type: ignore[no-any-return]

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params if study.trials else {}
    best_value = study.best_value if study.trials else float("inf")

    logger.info(
        "HPO complete: best_value=%.4f, best_params=%s",
        best_value,
        best_params,
    )

    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "best_params": best_params,
        "best_value": best_value,
    }
