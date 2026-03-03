"""Optuna-based HPO engine with ASHA (HyperbandPruner) support.

Wraps Optuna study creation, parameter suggestion, and pruner
configuration for hyperparameter optimization of segmentation models.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

logger = logging.getLogger(__name__)


class HPOEngine:
    """Optuna HPO engine with configurable samplers and pruners.

    Parameters
    ----------
    study_name:
        Name for the Optuna study.
    storage:
        Optuna storage URL (PostgreSQL, SQLite, or None for in-memory).
    pruner:
        Pruner type: ``"hyperband"`` for ASHA, ``None`` for no pruning.
    sampler:
        Sampler type: ``"tpe"`` (default), ``"cmaes"``, ``"grid"``.
    """

    def __init__(
        self,
        *,
        study_name: str,
        storage: str | None = None,
        pruner: str | None = None,
        sampler: str = "tpe",
    ) -> None:
        self.study_name = study_name
        self.storage = storage
        self._pruner = self._build_pruner(pruner)
        self._sampler = self._build_sampler(sampler)

    @staticmethod
    def _build_pruner(pruner: str | None) -> optuna.pruners.BasePruner | None:
        if pruner is None:
            return None
        if pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if pruner == "median":
            return optuna.pruners.MedianPruner()
        msg = f"Unknown pruner: {pruner!r}. Use 'hyperband' or 'median'."
        raise ValueError(msg)

    @staticmethod
    def _build_sampler(sampler: str) -> optuna.samplers.BaseSampler:
        if sampler == "tpe":
            return optuna.samplers.TPESampler()
        if sampler == "cmaes":
            return optuna.samplers.CmaEsSampler()
        if sampler == "grid":
            msg = "GridSampler requires search_space at init. Use TPE instead."
            raise ValueError(msg)
        msg = f"Unknown sampler: {sampler!r}. Use 'tpe' or 'cmaes'."
        raise ValueError(msg)

    def create_study(
        self,
        *,
        direction: str = "minimize",
    ) -> optuna.Study:
        """Create (or load) an Optuna study.

        Parameters
        ----------
        direction:
            Optimization direction: ``"minimize"`` or ``"maximize"``.

        Returns
        -------
        Optuna Study instance.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            pruner=self._pruner,
            sampler=self._sampler,
            load_if_exists=True,
        )
        logger.info(
            "Created Optuna study %r (direction=%s, pruner=%s)",
            self.study_name,
            direction,
            type(self._pruner).__name__ if self._pruner else "None",
        )
        return study

    def suggest_params(
        self,
        trial: optuna.Trial,
        search_space: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Suggest hyperparameters from a search space definition.

        Parameters
        ----------
        trial:
            Optuna trial instance.
        search_space:
            Dict mapping param names to their specs.
            Each spec has ``type`` (``"float"``, ``"int"``, ``"categorical"``)
            and type-specific keys (``low``, ``high``, ``log``, ``choices``).

        Returns
        -------
        Dict of suggested parameter values.
        """
        params: dict[str, Any] = {}
        for name, spec in search_space.items():
            param_type = spec["type"]
            if param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    low=spec["low"],
                    high=spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    low=spec["low"],
                    high=spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    choices=spec["choices"],
                )
            else:
                msg = f"Unknown param type {param_type!r} for {name!r}"
                raise ValueError(msg)
        return params
