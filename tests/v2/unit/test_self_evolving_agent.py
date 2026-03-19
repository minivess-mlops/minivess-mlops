"""Tests for self-evolving segmentation agent (T6.3).

Validates that:
- Drift detection triggers retraining decision
- Retraining uses Prefect run_deployment() (not direct training)
- Human approval gate prevents unauthorized retraining
- PCCP confidence gate is enforced

Staging tier: no model loading, no slow, no integration.
"""

from __future__ import annotations

import pytest


class TestDriftAnalysis:
    """Tests for drift-based evolution decisions."""

    def test_high_drift_recommends_retrain(self) -> None:
        """Drift score above threshold recommends retraining."""
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            evaluate_drift_action,
        )

        ctx = EvolutionContext(
            drift_score=0.8,
            performance_history=[0.85, 0.84, 0.82, 0.79],
            retrain_cost_estimate=5.0,
        )
        action = evaluate_drift_action(ctx, drift_threshold=0.5)
        assert action == "retrain"

    def test_low_drift_recommends_monitor(self) -> None:
        """Drift score below threshold recommends monitoring."""
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            evaluate_drift_action,
        )

        ctx = EvolutionContext(
            drift_score=0.1,
            performance_history=[0.85, 0.85, 0.86, 0.85],
            retrain_cost_estimate=5.0,
        )
        action = evaluate_drift_action(ctx, drift_threshold=0.5)
        assert action == "monitor"

    def test_declining_performance_recommends_retrain(self) -> None:
        """Monotonically declining performance recommends retraining."""
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            evaluate_drift_action,
        )

        ctx = EvolutionContext(
            drift_score=0.3,  # below threshold
            performance_history=[0.90, 0.85, 0.78, 0.70],
            retrain_cost_estimate=5.0,
        )
        # Even with low drift, declining performance triggers retrain
        action = evaluate_drift_action(ctx, drift_threshold=0.5, decline_window=3)
        assert action == "retrain"


class TestRetrainingTrigger:
    """Tests for the Prefect deployment trigger."""

    def test_build_retraining_params(self) -> None:
        """Retraining params include deployment name and config overrides."""
        from minivess.agents.evolution.self_evolving import build_retraining_params

        params = build_retraining_params(
            deployment_name="training-flow/default",
            config_overrides={"learning_rate": 1e-4},
            reason="drift_detected",
        )
        assert params["deployment_name"] == "training-flow/default"
        assert params["parameters"]["learning_rate"] == 1e-4
        assert params["reason"] == "drift_detected"

    def test_retraining_requires_human_approval(self) -> None:
        """Retraining trigger includes human_approval_required flag."""
        from minivess.agents.evolution.self_evolving import build_retraining_params

        params = build_retraining_params(
            deployment_name="training-flow/default",
        )
        assert params["human_approval_required"] is True


class TestEvolutionWithPCCP:
    """Tests for PCCP-gated evolution decisions."""

    def test_retrain_blocked_by_low_confidence(self) -> None:
        """Retraining is blocked when PCCP confidence is too low."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            pccp_gated_evolution,
        )

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8, budget_total=100.0))
        ctx = EvolutionContext(
            drift_score=0.9,
            performance_history=[0.85, 0.80, 0.70],
            retrain_cost_estimate=10.0,
        )
        decision = pccp_gated_evolution(
            ctx=ctx,
            gate=gate,
            confidence=0.3,  # below threshold
            drift_threshold=0.5,
        )
        assert decision.action == "blocked"
        assert not decision.pccp_approved

    def test_retrain_approved_with_high_confidence(self) -> None:
        """Retraining proceeds when PCCP confidence is high enough."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            pccp_gated_evolution,
        )

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=100.0))
        ctx = EvolutionContext(
            drift_score=0.9,
            performance_history=[0.85, 0.80, 0.70],
            retrain_cost_estimate=10.0,
        )
        decision = pccp_gated_evolution(
            ctx=ctx,
            gate=gate,
            confidence=0.9,
            drift_threshold=0.5,
        )
        assert decision.action == "retrain"
        assert decision.pccp_approved

    def test_monitor_not_gated(self) -> None:
        """Monitor decisions bypass PCCP gate (no action taken)."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate
        from minivess.agents.evolution.self_evolving import (
            EvolutionContext,
            pccp_gated_evolution,
        )

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8, budget_total=100.0))
        ctx = EvolutionContext(
            drift_score=0.1,
            performance_history=[0.85, 0.86, 0.85],
            retrain_cost_estimate=10.0,
        )
        decision = pccp_gated_evolution(
            ctx=ctx,
            gate=gate,
            confidence=0.3,  # low confidence, but doesn't matter for monitor
            drift_threshold=0.5,
        )
        assert decision.action == "monitor"
        # PCCP not checked for monitor actions
        assert decision.pccp_approved is True


class TestSelfEvolvingAgent:
    """Tests for the Pydantic AI self-evolving agent."""

    @pytest.fixture(autouse=True)
    def _skip_without_pydantic_ai(self) -> None:
        pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")

    def test_build_agent(self) -> None:
        """Agent builds without error."""
        from minivess.agents.evolution.self_evolving import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "self-evolving-segmentation"

    def test_agent_has_tools(self) -> None:
        """Agent has the expected tools registered."""
        from minivess.agents.evolution.self_evolving import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "get_drift_metrics" in tool_names
        assert "get_performance_history" in tool_names
        assert "get_retraining_params" in tool_names

    def test_create_agent_returns_prefect_agent(self) -> None:
        """Factory returns a PrefectAgent."""
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.evolution.self_evolving import (
            create_self_evolving_agent,
        )

        pa = create_self_evolving_agent(model="test")
        assert isinstance(pa, PrefectAgent)
