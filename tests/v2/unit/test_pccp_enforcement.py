"""Tests for PCCP constraint enforcement layer (T6.4).

Validates that:
- Actions below confidence threshold are rejected
- Budget constraints are enforced
- Gate resets properly
- Edge cases (zero budget, boundary confidence) behave correctly

Staging tier: pure computation, no LLM, no model loading.
"""

from __future__ import annotations


class TestPCCPConfig:
    """Tests for PCCPConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Default config has sane values."""
        from minivess.agents.constraints.pccp import PCCPConfig

        config = PCCPConfig()
        assert 0.0 < config.alpha < 1.0
        assert 0.0 < config.min_confidence <= 1.0
        assert config.budget_total > 0.0

    def test_custom_config(self) -> None:
        """Custom config values are preserved."""
        from minivess.agents.constraints.pccp import PCCPConfig

        config = PCCPConfig(alpha=0.05, min_confidence=0.9, budget_total=50.0)
        assert config.alpha == 0.05
        assert config.min_confidence == 0.9
        assert config.budget_total == 50.0


class TestPCCPGate:
    """Tests for the PCCP gate enforcement."""

    def test_rejects_low_confidence(self) -> None:
        """Actions below min_confidence are rejected."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8))
        decision = gate.check(confidence=0.5, cost=1.0)
        assert not decision.approved
        assert "confidence" in decision.reason.lower()

    def test_approves_high_confidence(self) -> None:
        """Actions at or above min_confidence are approved."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8))
        decision = gate.check(confidence=0.9, cost=1.0)
        assert decision.approved

    def test_rejects_budget_exceeded(self) -> None:
        """Actions exceeding remaining budget are rejected."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=3.0))
        # Spend the budget
        gate.check(confidence=0.9, cost=1.0)
        gate.check(confidence=0.9, cost=1.0)
        gate.check(confidence=0.9, cost=1.0)
        # Next should be rejected
        decision = gate.check(confidence=0.9, cost=1.0)
        assert not decision.approved
        assert "budget" in decision.reason.lower()

    def test_budget_tracks_spending(self) -> None:
        """Budget spent increases with approved actions."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=10.0))
        gate.check(confidence=0.9, cost=3.0)
        assert gate.budget_spent == 3.0
        gate.check(confidence=0.9, cost=2.5)
        assert gate.budget_spent == 5.5

    def test_rejected_actions_dont_spend_budget(self) -> None:
        """Rejected actions do not consume budget."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8, budget_total=10.0))
        gate.check(confidence=0.3, cost=5.0)  # rejected: low confidence
        assert gate.budget_spent == 0.0

    def test_reset_budget(self) -> None:
        """Budget resets to zero."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=10.0))
        gate.check(confidence=0.9, cost=7.0)
        assert gate.budget_spent == 7.0
        gate.reset_budget()
        assert gate.budget_spent == 0.0

    def test_boundary_confidence_approved(self) -> None:
        """Confidence exactly at threshold is approved."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8))
        decision = gate.check(confidence=0.8, cost=1.0)
        assert decision.approved

    def test_zero_cost_always_passes_budget(self) -> None:
        """Zero-cost actions always pass budget check."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.5, budget_total=0.0))
        # budget_total=0 but cost=0 should still pass
        decision = gate.check(confidence=0.9, cost=0.0)
        assert decision.approved

    def test_decision_includes_metadata(self) -> None:
        """PCCPDecision includes confidence, threshold, and cost."""
        from minivess.agents.constraints.pccp import PCCPConfig, PCCPGate

        gate = PCCPGate(config=PCCPConfig(min_confidence=0.8, budget_total=100.0))
        decision = gate.check(confidence=0.95, cost=2.0)
        assert decision.confidence == 0.95
        assert decision.threshold == 0.8
        assert decision.cost == 2.0
        assert decision.budget_remaining == 98.0


class TestPCCPDecision:
    """Tests for PCCPDecision model."""

    def test_decision_model_fields(self) -> None:
        """PCCPDecision has all required fields."""
        from minivess.agents.constraints.pccp import PCCPDecision

        d = PCCPDecision(
            approved=True,
            confidence=0.9,
            threshold=0.8,
            cost=1.0,
            budget_remaining=99.0,
            reason="Passed all checks",
        )
        assert d.approved is True
        assert d.reason == "Passed all checks"
