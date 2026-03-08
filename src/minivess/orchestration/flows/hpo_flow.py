"""HPO Prefect flow — hyperparameter optimization as a managed flow.

Flow 2.5: Wraps HPOEngine (Optuna + ASHA) as a Prefect @flow.
Each trial triggers a training deployment via Prefect's run_deployment() API,
ensuring each training run occurs in its own Docker container.

This enables:
- Prefect UI visibility into HPO progress
- Work pool routing (GPU pool for training, CPU for HPO orchestration)
- Integration with the trigger chain
- Fault-tolerant parallel trial execution
- Total Docker isolation between HPO controller and training runs
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
from prefect import flow, task
from prefect.deployments import run_deployment

from minivess.orchestration.constants import FLOW_NAME_HPO, FLOW_NAME_TRAIN

if TYPE_CHECKING:
    from prefect.client.schemas.objects import FlowRun

logger = logging.getLogger(__name__)


def _require_docker_context() -> None:
    """Require Docker container context or MINIVESS_ALLOW_HOST=1."""
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        "HPO flow must run inside a Docker container.\n"
        "Run: docker compose -f deployment/docker-compose.flows.yml run hpo\n"
        "Escape hatch for tests: MINIVESS_ALLOW_HOST=1"
    )


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

# Deployment name for the training flow (must match deployments.yaml)
_TRAINING_DEPLOYMENT = f"{FLOW_NAME_TRAIN}/default"


@task(name="run-hpo-trial")
def run_trial_task(
    trial_number: int,
    params: dict[str, Any],
    base_config: dict[str, Any],
) -> float:
    """Run one HPO trial via Prefect run_deployment() and return val_loss.

    Each trial triggers a separate training container via the Prefect worker.
    Results are read from the deployment return value (which comes from MLflow).

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

    # Build parameters for the training deployment
    training_params: dict[str, Any] = {
        "loss_name": config.get("loss_name", "cbdice_cldice"),
        "model_family": config.get("model_family", "dynunet"),
        "compute": config.get("compute", "auto"),
        "debug": config.get("debug", False),
        "experiment_name": config.get("experiment_name", "minivess_hpo"),
        "num_folds": config.get("num_folds", 1),
        "max_epochs": config.get("max_epochs", 20),
        "batch_size": config.get("batch_size", 2),
        "learning_rate": config.get("learning_rate", 1e-3),
    }

    # Trigger training in a separate Docker container via Prefect
    deployment_name = config.get("training_deployment", _TRAINING_DEPLOYMENT)
    timeout = config.get("trial_timeout", 86400)  # 24h default per trial

    # run_deployment() returns FlowRun in synchronous @task context
    flow_run: FlowRun = run_deployment(  # type: ignore[assignment]
        name=deployment_name,
        parameters=training_params,
        timeout=timeout,
    )

    # Extract objective value from the training flow result
    raw_result = flow_run.state.result() if flow_run.state else None
    if raw_result is None:
        logger.warning("Trial %d: no result from training deployment", trial_number)
        return float("inf")

    fold_results: list[dict[str, Any]] = (
        raw_result.fold_results if hasattr(raw_result, "fold_results") else []
    )
    if fold_results:
        val_losses = [r.get("best_val_loss", float("inf")) for r in fold_results]
        return float(min(val_losses))
    return float("inf")


@flow(name=FLOW_NAME_HPO)
def hpo_flow(
    *,
    n_trials: int = 20,
    study_name: str = "minivess_hpo",
    sampler: str = "tpe",
    pruner: str | None = "hyperband",
    search_space: dict[str, dict[str, Any]] | None = None,
    base_config: dict[str, Any] | None = None,
    trigger_source: str = "manual",
    allocation_strategy: str = "sequential",
) -> dict[str, Any]:
    """HPO Prefect flow — runs n_trials of Optuna optimization.

    Each trial triggers a training deployment via run_deployment(),
    ensuring Docker isolation between the HPO controller and training.

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
    allocation_strategy:
        Trial allocation strategy: ``"sequential"`` (default), ``"parallel"``,
        or ``"hybrid"``. See ``AllocationStrategy`` in hpo_engine.py.
        PostgreSQL storage is required for all strategies.

    Returns
    -------
    Dict with ``best_params``, ``best_value``, ``n_trials``, ``study_name``.
    """
    _require_docker_context()

    from minivess.optimization.hpo_engine import AllocationStrategy, HPOEngine

    strategy = AllocationStrategy(allocation_strategy.lower())

    if strategy == AllocationStrategy.PARALLEL:
        raise NotImplementedError(
            "PARALLEL HPO requires multiple worker containers. "
            "Use: docker compose -f deployment/docker-compose.flows.yml "
            "up --scale hpo-worker=N\n"
            "Each worker reads from the shared PostgreSQL Optuna study. "
            "The hpo_flow orchestrator should not run directly in PARALLEL mode — "
            "launch hpo-worker replicas instead."
        )

    # Wire JSONL log handler — Issue #503
    from minivess.observability.flow_logging import configure_flow_logging

    configure_flow_logging(logs_dir=Path(os.environ.get("LOGS_DIR", "/app/logs")))

    logger.info(
        "HPO flow started: study=%r, n_trials=%d, sampler=%s, strategy=%s (trigger: %s)",
        study_name,
        n_trials,
        sampler,
        strategy.value,
        trigger_source,
    )

    if search_space is None:
        search_space = _DEFAULT_SEARCH_SPACE
    if base_config is None:
        base_config = {}

    engine = HPOEngine(
        study_name=study_name,
        storage=os.environ.get("OPTUNA_STORAGE_URL"),
        pruner=pruner,
        sampler=sampler,
    )
    study = engine.create_study(direction="minimize")

    if strategy == AllocationStrategy.HYBRID:
        # HYBRID: set CUDA_VISIBLE_DEVICES from REPLICA_INDEX before optimizing
        replica_index = int(os.environ.get("REPLICA_INDEX", "0"))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(replica_index)
        logger.info("HYBRID strategy: CUDA_VISIBLE_DEVICES=%d", replica_index)

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
        "allocation_strategy": strategy.value,
    }


if __name__ == "__main__":
    hpo_flow()
