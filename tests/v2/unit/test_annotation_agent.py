"""Tests for active learning annotation agent (T6.2).

Validates that:
- Volume disagreement scores are computed correctly from ensemble predictions
- Volumes are ranked by disagreement (highest first)
- Agent builds and runs with TestModel (no real LLM)

Staging tier: no model loading, no slow, no integration.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestEnsembleDisagreement:
    """Tests for disagreement computation from ensemble predictions."""

    def test_compute_disagreement_variance(self) -> None:
        """Disagreement score = mean voxel-wise variance across ensemble."""
        from minivess.agents.annotation.active_learning import (
            compute_ensemble_disagreement,
        )

        rng = np.random.default_rng(42)
        # 3 ensemble members, each predicts a (4, 4, 4) probability map
        predictions = [rng.random((4, 4, 4)).astype(np.float32) for _ in range(3)]
        score = compute_ensemble_disagreement(predictions)
        assert isinstance(score, float)
        assert score > 0.0

    def test_identical_predictions_zero_disagreement(self) -> None:
        """Identical ensemble predictions have zero disagreement."""
        from minivess.agents.annotation.active_learning import (
            compute_ensemble_disagreement,
        )

        pred = np.ones((4, 4, 4), dtype=np.float32) * 0.7
        score = compute_ensemble_disagreement([pred, pred.copy(), pred.copy()])
        assert score == pytest.approx(0.0, abs=1e-7)

    def test_disagreement_increases_with_variance(self) -> None:
        """Higher variance predictions produce higher disagreement."""
        from minivess.agents.annotation.active_learning import (
            compute_ensemble_disagreement,
        )

        # Low variance: all predictions near 0.5
        low_var = [
            np.full((4, 4, 4), 0.49, dtype=np.float32),
            np.full((4, 4, 4), 0.50, dtype=np.float32),
            np.full((4, 4, 4), 0.51, dtype=np.float32),
        ]
        # High variance: predictions spread widely
        high_var = [
            np.full((4, 4, 4), 0.1, dtype=np.float32),
            np.full((4, 4, 4), 0.5, dtype=np.float32),
            np.full((4, 4, 4), 0.9, dtype=np.float32),
        ]
        assert compute_ensemble_disagreement(high_var) > compute_ensemble_disagreement(
            low_var
        )


class TestVolumeRanking:
    """Tests for ranking volumes by disagreement."""

    def test_rank_volumes_by_disagreement(self) -> None:
        """Volumes with higher disagreement rank first."""
        from minivess.agents.annotation.active_learning import (
            rank_volumes_by_disagreement,
        )

        # Volume A: high disagreement (3 very different predictions)
        vol_a_preds = [np.full((4, 4, 4), v, dtype=np.float32) for v in [0.1, 0.5, 0.9]]
        # Volume B: low disagreement (3 similar predictions)
        vol_b_preds = [
            np.full((4, 4, 4), v, dtype=np.float32) for v in [0.49, 0.50, 0.51]
        ]
        # Volume C: medium disagreement
        vol_c_preds = [np.full((4, 4, 4), v, dtype=np.float32) for v in [0.3, 0.5, 0.7]]

        ensemble_predictions = {
            "vol_a": vol_a_preds,
            "vol_b": vol_b_preds,
            "vol_c": vol_c_preds,
        }
        ranked = rank_volumes_by_disagreement(ensemble_predictions)

        # vol_a has highest disagreement → rank first
        assert ranked[0][0] == "vol_a"
        # vol_b has lowest disagreement → rank last
        assert ranked[-1][0] == "vol_b"

    def test_rank_returns_scores(self) -> None:
        """Each ranked entry includes the disagreement score."""
        from minivess.agents.annotation.active_learning import (
            rank_volumes_by_disagreement,
        )

        preds = {
            "vol_x": [np.full((2, 2, 2), v, dtype=np.float32) for v in [0.2, 0.8]],
        }
        ranked = rank_volumes_by_disagreement(preds)
        assert len(ranked) == 1
        vol_id, score = ranked[0]
        assert vol_id == "vol_x"
        assert isinstance(score, float)
        assert score > 0.0

    def test_rank_empty_returns_empty(self) -> None:
        """Empty input returns empty list."""
        from minivess.agents.annotation.active_learning import (
            rank_volumes_by_disagreement,
        )

        assert rank_volumes_by_disagreement({}) == []


class TestAnnotationAgent:
    """Tests for the Pydantic AI annotation agent."""

    @pytest.fixture(autouse=True)
    def _skip_without_pydantic_ai(self) -> None:
        pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")

    def test_build_agent(self) -> None:
        """Agent builds without error."""
        from minivess.agents.annotation.active_learning import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "active-learning-annotation"

    def test_agent_has_tools(self) -> None:
        """Agent has the expected tools registered."""
        from minivess.agents.annotation.active_learning import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "get_disagreement_ranking" in tool_names
        assert "get_annotation_budget" in tool_names

    def test_create_agent_returns_prefect_agent(self) -> None:
        """Factory returns a PrefectAgent."""
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.annotation.active_learning import (
            create_annotation_agent,
        )

        pa = create_annotation_agent(model="test")
        assert isinstance(pa, PrefectAgent)
