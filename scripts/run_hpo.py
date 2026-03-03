"""Run Optuna HPO study from experiment config YAML.

Usage::

    uv run python scripts/run_hpo.py --config configs/experiments/hpo_dynunet_example.yaml
    uv run python scripts/run_hpo.py --config configs/experiments/hpo_dynunet_example.yaml --dry-run

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running as script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import yaml  # noqa: E402

from minivess.optimization.hpo_engine import HPOEngine  # noqa: E402
from minivess.optimization.search_space import SearchSpace  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for HPO studies."""
    parser = argparse.ArgumentParser(description="Run Optuna HPO study")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to HPO experiment YAML config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print study config without running",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: in-memory)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    study_name = config.get("study_name", "hpo_study")
    direction = config.get("direction", "minimize")
    n_trials = config.get("n_trials", 10)
    sampler = config.get("sampler", "tpe")
    pruner = config.get("pruner")

    search_space = SearchSpace.from_dict(config["search_space"])

    logger.info(
        "HPO Config: study=%s, trials=%d, direction=%s", study_name, n_trials, direction
    )
    logger.info("Search space: %d params", len(search_space.params))

    if args.dry_run:
        logger.info("[DRY RUN] Would run %d trials with search space:", n_trials)
        for name, spec in search_space.params.items():
            logger.info("  %s: %s", name, spec)
        return

    engine = HPOEngine(
        study_name=study_name,
        storage=args.storage,
        pruner=pruner,
        sampler=sampler,
    )
    study = engine.create_study(direction=direction)

    def objective(trial):  # type: ignore[no-untyped-def]
        params = engine.suggest_params(trial, search_space.params)
        logger.info("Trial %d params: %s", trial.number, params)
        # Placeholder: in real usage, this calls training and returns val_loss
        return 0.0

    study.optimize(objective, n_trials=n_trials)

    logger.info("Best trial: %s", study.best_trial.params)
    logger.info("Best value: %f", study.best_value)


if __name__ == "__main__":
    main()
