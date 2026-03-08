"""Smoke tests for hpo_flow.py allocation strategy dispatch — Issue #504.

Tests that:
- SEQUENTIAL strategy runs in-process (no subprocess spawn)
- PARALLEL strategy raises NotImplementedError with instructions
- HYBRID strategy sets CUDA_VISIBLE_DEVICES from REPLICA_INDEX
- allocation_strategy is included in the return dict

All tests run without Docker (MINIVESS_ALLOW_HOST=1) and without PostgreSQL
(mocked storage). These are integration smoke tests, not unit tests.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def allow_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Bypass Docker gate and redirect LOGS_DIR to tmp_path for all tests."""
    monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
    monkeypatch.setenv("LOGS_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def disable_prefect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Prefect server requirement for unit-level smoke tests."""
    monkeypatch.setenv("PREFECT_DISABLED", "1")


class TestHpoFlowAllocationDispatch:
    """Verify hpo_flow dispatches correctly based on allocation_strategy."""

    def test_parallel_strategy_raises_not_implemented(self) -> None:
        """PARALLEL strategy must raise NotImplementedError with docker-compose hint."""
        from minivess.orchestration.flows.hpo_flow import hpo_flow

        with pytest.raises(NotImplementedError, match="hpo-worker"):
            hpo_flow(allocation_strategy="parallel")

    def test_parallel_strategy_error_mentions_scale(self) -> None:
        """PARALLEL error message must mention --scale flag."""
        from minivess.orchestration.flows.hpo_flow import hpo_flow

        with pytest.raises(NotImplementedError, match="--scale"):
            hpo_flow(allocation_strategy="parallel")

    def test_sequential_strategy_returns_result_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SEQUENTIAL strategy returns dict with expected keys."""
        from minivess.orchestration.flows.hpo_flow import hpo_flow

        # Mock HPOEngine.create_study to avoid PostgreSQL connection
        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.best_params = {}
        mock_study.best_value = float("inf")

        with (
            patch(
                "minivess.optimization.hpo_engine.HPOEngine.create_study",
                return_value=mock_study,
            ),
            patch(
                "minivess.optimization.hpo_engine._validate_postgresql_url",
                return_value="postgresql://mock/optuna",
            ),
        ):
            result = hpo_flow(
                n_trials=0,
                study_name="smoke_test",
                allocation_strategy="sequential",
            )

        assert result["allocation_strategy"] == "sequential"
        assert "best_params" in result
        assert "best_value" in result
        assert result["study_name"] == "smoke_test"

    def test_hybrid_strategy_sets_cuda_visible_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HYBRID strategy sets CUDA_VISIBLE_DEVICES from REPLICA_INDEX."""
        from minivess.orchestration.flows.hpo_flow import hpo_flow

        monkeypatch.setenv("REPLICA_INDEX", "2")

        mock_study = MagicMock()
        mock_study.trials = []

        with (
            patch(
                "minivess.optimization.hpo_engine.HPOEngine.create_study",
                return_value=mock_study,
            ),
            patch(
                "minivess.optimization.hpo_engine._validate_postgresql_url",
                return_value="postgresql://mock/optuna",
            ),
        ):
            hpo_flow(
                n_trials=0,
                study_name="smoke_hybrid",
                allocation_strategy="hybrid",
            )

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"

    def test_invalid_strategy_raises_value_error(self) -> None:
        """Unknown allocation_strategy raises ValueError."""
        from minivess.orchestration.flows.hpo_flow import hpo_flow

        with pytest.raises(ValueError):
            hpo_flow(allocation_strategy="invalid_strategy")
