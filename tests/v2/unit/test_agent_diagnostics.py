"""Tests for DiLLS-style agent diagnostics (Issue #16)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: AgentInteraction
# ---------------------------------------------------------------------------


class TestAgentInteraction:
    """Test agent interaction dataclass."""

    def test_construction(self) -> None:
        """AgentInteraction should capture a single agent step."""
        from minivess.observability.agent_diagnostics import AgentInteraction

        interaction = AgentInteraction(
            node_name="evaluate",
            input_summary="Run evaluation on test set",
            output_summary="Dice=0.85, HD95=3.2",
            latency_ms=1500.0,
            token_count=250,
        )
        assert interaction.node_name == "evaluate"
        assert interaction.latency_ms == 1500.0
        assert interaction.token_count == 250

    def test_defaults(self) -> None:
        """Optional fields should default to zero/empty."""
        from minivess.observability.agent_diagnostics import AgentInteraction

        interaction = AgentInteraction(
            node_name="plan",
            input_summary="Start planning",
            output_summary="Plan created",
        )
        assert interaction.latency_ms == 0.0
        assert interaction.token_count == 0
        assert interaction.metadata == {}

    def test_metadata(self) -> None:
        """Metadata should store arbitrary key-value pairs."""
        from minivess.observability.agent_diagnostics import AgentInteraction

        interaction = AgentInteraction(
            node_name="llm_call",
            input_summary="Prompt",
            output_summary="Response",
            metadata={"model": "claude-3.5-sonnet", "temperature": 0.0},
        )
        assert interaction.metadata["model"] == "claude-3.5-sonnet"


# ---------------------------------------------------------------------------
# T2: SessionSummary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    """Test session-level summary aggregation."""

    def test_construction(self) -> None:
        """SessionSummary should be constructible."""
        from minivess.observability.agent_diagnostics import SessionSummary

        summary = SessionSummary(
            session_id="sess-001",
            interactions=[],
        )
        assert summary.session_id == "sess-001"
        assert summary.total_steps == 0

    def test_aggregation(self) -> None:
        """SessionSummary should compute aggregate statistics."""
        from minivess.observability.agent_diagnostics import (
            AgentInteraction,
            SessionSummary,
        )

        interactions = [
            AgentInteraction(
                node_name="plan", input_summary="i", output_summary="o",
                latency_ms=100.0, token_count=50,
            ),
            AgentInteraction(
                node_name="execute", input_summary="i", output_summary="o",
                latency_ms=200.0, token_count=150,
            ),
            AgentInteraction(
                node_name="evaluate", input_summary="i", output_summary="o",
                latency_ms=300.0, token_count=100,
            ),
        ]
        summary = SessionSummary(session_id="sess-002", interactions=interactions)
        assert summary.total_steps == 3
        assert summary.total_latency_ms == 600.0
        assert summary.total_tokens == 300


# ---------------------------------------------------------------------------
# T3: AgentDiagnostics
# ---------------------------------------------------------------------------


class TestAgentDiagnostics:
    """Test DiLLS-style layered diagnostics."""

    def test_record_interaction(self) -> None:
        """record_interaction should store an AgentInteraction."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        diag.record_interaction(
            session_id="s1",
            node_name="plan",
            input_summary="Start",
            output_summary="Done",
            latency_ms=100.0,
        )
        assert len(diag.sessions) == 1
        assert len(diag.sessions["s1"]) == 1

    def test_multiple_sessions(self) -> None:
        """Interactions should be grouped by session_id."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        diag.record_interaction(session_id="s1", node_name="a", input_summary="", output_summary="")
        diag.record_interaction(session_id="s2", node_name="b", input_summary="", output_summary="")
        diag.record_interaction(session_id="s1", node_name="c", input_summary="", output_summary="")
        assert len(diag.sessions) == 2
        assert len(diag.sessions["s1"]) == 2
        assert len(diag.sessions["s2"]) == 1

    def test_summarize_session(self) -> None:
        """summarize_session should return a SessionSummary."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        diag.record_interaction(
            session_id="s1", node_name="plan",
            input_summary="i", output_summary="o",
            latency_ms=100.0, token_count=50,
        )
        diag.record_interaction(
            session_id="s1", node_name="execute",
            input_summary="i", output_summary="o",
            latency_ms=200.0, token_count=100,
        )
        summary = diag.summarize_session("s1")
        assert summary.total_steps == 2
        assert summary.total_latency_ms == 300.0
        assert summary.total_tokens == 150

    def test_summarize_aggregate(self) -> None:
        """summarize_aggregate should return cross-session statistics."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        diag.record_interaction(session_id="s1", node_name="a", input_summary="", output_summary="", latency_ms=100.0)
        diag.record_interaction(session_id="s1", node_name="b", input_summary="", output_summary="", latency_ms=200.0)
        diag.record_interaction(session_id="s2", node_name="c", input_summary="", output_summary="", latency_ms=300.0)

        agg = diag.summarize_aggregate()
        assert agg["total_sessions"] == 2
        assert agg["total_interactions"] == 3
        assert agg["total_latency_ms"] == 600.0

    def test_to_markdown(self) -> None:
        """to_markdown should produce a human-readable diagnostic report."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        diag.record_interaction(
            session_id="s1", node_name="plan",
            input_summary="Start planning", output_summary="Plan ready",
            latency_ms=150.0, token_count=80,
        )
        md = diag.to_markdown()
        assert "Agent Diagnostics" in md
        assert "s1" in md
        assert "plan" in md

    def test_empty_diagnostics(self) -> None:
        """to_markdown should handle empty diagnostics gracefully."""
        from minivess.observability.agent_diagnostics import AgentDiagnostics

        diag = AgentDiagnostics()
        md = diag.to_markdown()
        assert "Agent Diagnostics" in md
        assert "No sessions" in md or "0" in md
