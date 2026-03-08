"""Tests for HPO allocation strategies -- Issue #504.

Validates AllocationStrategy enum and PostgreSQL-only enforcement.
ALL strategies require postgresql:// storage -- SQLite and in-memory are banned.

Plan: docs/planning/overnight-child-prefect-docker.xml Phase 4 (T-PD.4.1)
"""

from __future__ import annotations

import pytest


class TestAllocationStrategyParsing:
    def test_sequential_strategy_exists(self) -> None:
        from minivess.optimization.hpo_engine import AllocationStrategy

        assert AllocationStrategy.SEQUENTIAL is not None

    def test_parallel_strategy_exists(self) -> None:
        from minivess.optimization.hpo_engine import AllocationStrategy

        assert AllocationStrategy.PARALLEL is not None

    def test_hybrid_strategy_exists(self) -> None:
        from minivess.optimization.hpo_engine import AllocationStrategy

        assert AllocationStrategy.HYBRID is not None

    def test_allocation_config_stores_strategy(self) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        cfg = AllocationConfig(
            strategy=AllocationStrategy.SEQUENTIAL,
            optuna_storage="postgresql://user:pass@host/optuna",
        )
        assert cfg.strategy == AllocationStrategy.SEQUENTIAL

    def test_allocation_strategy_hybrid_sets_container_fields(self) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        cfg = AllocationConfig(
            strategy=AllocationStrategy.HYBRID,
            optuna_storage="postgresql://user:pass@host/optuna",
            n_containers=4,
            trials_per_container=3,
        )
        assert cfg.n_containers == 4
        assert cfg.trials_per_container == 3


class TestPostgreSQLEnforcement:
    """Every strategy must reject non-PostgreSQL storage."""

    VALID_PG_URL = "postgresql://user:pass@host/optuna"
    VALID_PG_PSYCOPG2_URL = "postgresql+psycopg2://minivess:secret@postgres:5432/optuna"

    @pytest.mark.parametrize(
        "strategy_name",
        ["SEQUENTIAL", "PARALLEL", "HYBRID"],
    )
    def test_all_strategies_reject_none_storage(self, strategy_name: str) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        strategy = AllocationStrategy[strategy_name]
        with pytest.raises(ValueError, match="[Pp]ostgreSQL"):
            AllocationConfig(strategy=strategy, optuna_storage=None)

    @pytest.mark.parametrize(
        "strategy_name",
        ["SEQUENTIAL", "PARALLEL", "HYBRID"],
    )
    def test_all_strategies_reject_sqlite_url(self, strategy_name: str) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        strategy = AllocationStrategy[strategy_name]
        with pytest.raises(ValueError, match="[Pp]ostgreSQL"):
            AllocationConfig(strategy=strategy, optuna_storage="sqlite:///optuna.db")

    @pytest.mark.parametrize(
        "strategy_name",
        ["SEQUENTIAL", "PARALLEL", "HYBRID"],
    )
    def test_all_strategies_accept_postgresql_url(self, strategy_name: str) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        strategy = AllocationStrategy[strategy_name]
        cfg = AllocationConfig(strategy=strategy, optuna_storage=self.VALID_PG_URL)
        assert cfg.optuna_storage == self.VALID_PG_URL

    @pytest.mark.parametrize(
        "strategy_name",
        ["SEQUENTIAL", "PARALLEL", "HYBRID"],
    )
    def test_all_strategies_accept_psycopg2_url(self, strategy_name: str) -> None:
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        strategy = AllocationStrategy[strategy_name]
        cfg = AllocationConfig(
            strategy=strategy, optuna_storage=self.VALID_PG_PSYCOPG2_URL
        )
        assert cfg.optuna_storage == self.VALID_PG_PSYCOPG2_URL

    def test_uses_env_var_when_storage_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When YAML optuna_storage is None, reads OPTUNA_STORAGE_URL env var."""
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        monkeypatch.setenv("OPTUNA_STORAGE_URL", self.VALID_PG_URL)
        cfg = AllocationConfig(
            strategy=AllocationStrategy.SEQUENTIAL,
            optuna_storage=None,  # should fall back to env var
        )
        assert cfg.optuna_storage == self.VALID_PG_URL

    def test_raises_when_neither_storage_nor_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No storage + no env var => ValueError."""
        from minivess.optimization.hpo_engine import (
            AllocationConfig,
            AllocationStrategy,
        )

        monkeypatch.delenv("OPTUNA_STORAGE_URL", raising=False)
        with pytest.raises(ValueError, match="[Pp]ostgreSQL"):
            AllocationConfig(
                strategy=AllocationStrategy.SEQUENTIAL,
                optuna_storage=None,
            )


class TestHPOEngineFromConfig:
    def test_hpo_engine_from_config_roundtrip(self) -> None:
        from minivess.optimization.hpo_engine import HPOEngine

        cfg = {
            "study_name": "test_study",
            "sampler": "tpe",
            "pruner": "hyperband",
            "allocation": {
                "strategy": "sequential",
                "optuna_storage": "postgresql://user:pass@host/optuna",
            },
        }
        engine = HPOEngine.from_config(cfg)
        assert engine.study_name == "test_study"
