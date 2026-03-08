"""Optuna-based HPO engine with ASHA (HyperbandPruner) support.

Wraps Optuna study creation, parameter suggestion, and pruner
configuration for hyperparameter optimization of segmentation models.
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Any

import optuna

logger = logging.getLogger(__name__)

_PG_PREFIXES = ("postgresql://", "postgresql+psycopg2://")
_POSTGRESQL_ERROR = (
    "PostgreSQL storage required for Optuna studies. "
    "Set OPTUNA_STORAGE_URL to a postgresql:// URL in .env.example. "
    "SQLite and in-memory are not supported in this project "
    "(CLAUDE.md MEMORY: PostgreSQL is ONLY Database 2026-03-08)."
)


def _validate_postgresql_url(url: str | None, *, env_fallback: bool = True) -> str:
    """Validate that *url* is a PostgreSQL URL. Falls back to OPTUNA_STORAGE_URL env var.

    Raises
    ------
    ValueError
        If the resolved URL is None or not a postgresql:// URL.
    """
    resolved = url
    if resolved is None and env_fallback:
        resolved = os.environ.get("OPTUNA_STORAGE_URL")
    if resolved is None or not any(resolved.startswith(p) for p in _PG_PREFIXES):
        raise ValueError(_POSTGRESQL_ERROR)
    return resolved


class AllocationStrategy(enum.Enum):
    """HPO trial allocation strategy.

    SEQUENTIAL — optimize in-process, one trial at a time (default).
    PARALLEL   — multiple workers via PostgreSQL-backed Optuna study.
    HYBRID     — multi-GPU on a single host via CUDA_VISIBLE_DEVICES.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class AllocationConfig:
    """Configuration for HPO trial allocation.

    All strategies require a postgresql:// storage URL (no SQLite, no in-memory).
    When *optuna_storage* is None, the OPTUNA_STORAGE_URL env var is read.

    Parameters
    ----------
    strategy:
        Allocation strategy to use.
    optuna_storage:
        PostgreSQL connection URL. If None, reads OPTUNA_STORAGE_URL env var.
    n_containers:
        Number of parallel worker containers (PARALLEL / HYBRID only).
    trials_per_container:
        Trials per container in HYBRID mode.
    """

    strategy: AllocationStrategy
    optuna_storage: str | None  # validated in __post_init__
    n_containers: int = 1
    trials_per_container: int = 1

    def __post_init__(self) -> None:
        # Resolve and validate storage -- raises ValueError for non-PostgreSQL
        self.optuna_storage = _validate_postgresql_url(self.optuna_storage)


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

    allocation: AllocationConfig | None = None  # set by from_config()

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

    @classmethod
    def from_config(cls, yaml_dict: dict[str, Any]) -> HPOEngine:
        """Construct HPOEngine from a YAML config dict.

        Parameters
        ----------
        yaml_dict:
            Dict with keys: study_name, sampler, pruner, allocation.
            allocation must have: strategy (str), optuna_storage (str|None).

        Returns
        -------
        HPOEngine instance with AllocationConfig attached.
        """
        alloc_raw = yaml_dict.get("allocation", {})
        strategy_str = alloc_raw.get("strategy", "sequential")
        strategy = AllocationStrategy(strategy_str.lower())
        allocation = AllocationConfig(
            strategy=strategy,
            optuna_storage=alloc_raw.get("optuna_storage"),
            n_containers=int(alloc_raw.get("n_containers", 1)),
            trials_per_container=int(alloc_raw.get("trials_per_container", 1)),
        )
        engine = cls(
            study_name=yaml_dict.get("study_name", "minivess_hpo"),
            storage=allocation.optuna_storage,
            pruner=yaml_dict.get("pruner"),
            sampler=yaml_dict.get("sampler", "tpe"),
        )
        engine.allocation = allocation
        return engine

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
