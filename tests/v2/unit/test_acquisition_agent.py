"""Tests for conformal bandit acquisition agent (T6.1).

Validates that:
- Thompson sampling selects high-uncertainty volumes
- PCCP budget constraint limits acquisitions
- Agent builds and runs with TestModel (no real LLM)

Staging tier: no model loading, no slow, no integration.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestThompsonSampler:
    """Tests for the Thompson sampling component."""

    def test_sampler_explores_uncertain_arms(self) -> None:
        """Arms with few observations get explored (non-zero selection)."""
        from minivess.agents.acquisition.conformal_bandit import ThompsonSampler

        sampler = ThompsonSampler(n_arms=3, seed=42)
        # Arm 0: known bad (many failures)
        for _ in range(20):
            sampler.update(arm=0, reward=0.1)
        # Arm 1: uncertain (no observations) — default Beta(1,1)
        # Arm 2: known good (many successes)
        for _ in range(5):
            sampler.update(arm=2, reward=0.9)

        # Sample 200 times
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(200):
            arm = sampler.sample()
            counts[arm] += 1

        # Uncertain arm 1 should be explored more than bad arm 0
        assert counts[1] > counts[0]
        # Good arm 2 should be selected most often
        assert counts[2] > counts[0]

    def test_sampler_respects_seed(self) -> None:
        """Same seed produces same sequence."""
        from minivess.agents.acquisition.conformal_bandit import ThompsonSampler

        s1 = ThompsonSampler(n_arms=4, seed=123)
        s2 = ThompsonSampler(n_arms=4, seed=123)
        for _ in range(10):
            assert s1.sample() == s2.sample()

    def test_sampler_all_arms_valid(self) -> None:
        """All sampled arms are within valid range."""
        from minivess.agents.acquisition.conformal_bandit import ThompsonSampler

        sampler = ThompsonSampler(n_arms=5, seed=0)
        for _ in range(50):
            arm = sampler.sample()
            assert 0 <= arm < 5


class TestAcquisitionRanking:
    """Tests for uncertainty-based volume ranking."""

    def test_rank_volumes_by_uncertainty(self) -> None:
        """Volumes with higher uncertainty score rank first."""
        from minivess.agents.acquisition.conformal_bandit import (
            rank_volumes_by_uncertainty,
        )

        uncertainty_scores = {
            "vol_001": 0.3,
            "vol_002": 0.9,
            "vol_003": 0.1,
            "vol_004": 0.7,
        }
        ranked = rank_volumes_by_uncertainty(uncertainty_scores)
        assert ranked[0] == "vol_002"
        assert ranked[1] == "vol_004"
        assert ranked[-1] == "vol_003"

    def test_rank_empty_returns_empty(self) -> None:
        """Empty input returns empty list."""
        from minivess.agents.acquisition.conformal_bandit import (
            rank_volumes_by_uncertainty,
        )

        assert rank_volumes_by_uncertainty({}) == []


class TestConformalBanditWithPCCP:
    """Tests for the conformal bandit integrated with PCCP gate."""

    def test_acquisition_respects_pccp_budget(self) -> None:
        """Acquisition stops when PCCP budget is exhausted."""
        from minivess.agents.acquisition.conformal_bandit import (
            select_volumes_for_acquisition,
        )
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=3.0))
        uncertainty_scores = {
            f"vol_{i:03d}": float(np.random.default_rng(42).random()) for i in range(10)
        }

        selected = select_volumes_for_acquisition(
            uncertainty_scores=uncertainty_scores,
            gate=gate,
            confidence=0.9,
            cost_per_volume=1.0,
        )
        # Budget = 3.0, cost_per_volume = 1.0 → max 3 volumes
        assert len(selected) <= 3

    def test_acquisition_returns_highest_uncertainty_first(self) -> None:
        """Selected volumes are ordered by descending uncertainty."""
        from minivess.agents.acquisition.conformal_bandit import (
            select_volumes_for_acquisition,
        )
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=100.0))
        uncertainty_scores = {
            "vol_a": 0.2,
            "vol_b": 0.9,
            "vol_c": 0.5,
        }
        selected = select_volumes_for_acquisition(
            uncertainty_scores=uncertainty_scores,
            gate=gate,
            confidence=0.9,
            cost_per_volume=1.0,
        )
        assert selected[0] == "vol_b"

    def test_acquisition_rejects_low_confidence(self) -> None:
        """No volumes selected when confidence is below PCCP threshold."""
        from minivess.agents.acquisition.conformal_bandit import (
            select_volumes_for_acquisition,
        )
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.9, budget_total=100.0))
        uncertainty_scores = {"vol_a": 0.8, "vol_b": 0.7}
        selected = select_volumes_for_acquisition(
            uncertainty_scores=uncertainty_scores,
            gate=gate,
            confidence=0.3,  # below threshold
            cost_per_volume=1.0,
        )
        assert len(selected) == 0


class TestConformalBanditAgent:
    """Tests for the Pydantic AI acquisition agent."""

    @pytest.fixture(autouse=True)
    def _skip_without_pydantic_ai(self) -> None:
        pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")

    def test_build_agent(self) -> None:
        """Agent builds without error."""
        from minivess.agents.acquisition.conformal_bandit import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "conformal-bandit-acquisition"

    def test_agent_has_tools(self) -> None:
        """Agent has the expected tools registered."""
        from minivess.agents.acquisition.conformal_bandit import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "get_uncertainty_scores" in tool_names
        assert "get_budget_status" in tool_names

    def test_create_agent_returns_prefect_agent(self) -> None:
        """Factory returns a PrefectAgent."""
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.acquisition.conformal_bandit import (
            create_acquisition_agent,
        )

        pa = create_acquisition_agent(model="test")
        assert isinstance(pa, PrefectAgent)
