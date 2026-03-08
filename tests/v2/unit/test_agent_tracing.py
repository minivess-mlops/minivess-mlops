"""Tests for agent tracing configuration (T-5.2)."""

from __future__ import annotations


class TestTracingModule:
    """Tests for the agents.tracing module."""

    def test_configure_returns_bool(self) -> None:
        """configure_agent_tracing returns a boolean."""
        from minivess.agents.tracing import configure_agent_tracing, reset_tracing

        reset_tracing()
        result = configure_agent_tracing()
        assert isinstance(result, bool)

    def test_get_status_returns_dict(self) -> None:
        """get_tracing_status returns status dict with expected keys."""
        from minivess.agents.tracing import get_tracing_status

        status = get_tracing_status()
        assert "enabled" in status
        assert "backend" in status
        assert "configured" in status

    def test_reset_clears_state(self) -> None:
        """reset_tracing clears the configured flag."""
        from minivess.agents.tracing import (
            get_tracing_status,
            reset_tracing,
        )

        reset_tracing()
        status = get_tracing_status()
        assert status["enabled"] is False

    def test_idempotent_configure(self) -> None:
        """Calling configure twice returns same result."""
        from minivess.agents.tracing import configure_agent_tracing, reset_tracing

        reset_tracing()
        first = configure_agent_tracing()
        second = configure_agent_tracing()
        # If first succeeded, second is also True (cached)
        # If first failed (no OTEL), second is also False
        assert first == second or second is True

    def test_graceful_without_langfuse(self) -> None:
        """Works without langfuse installed (returns False or falls back to OTEL)."""
        from minivess.agents.tracing import configure_agent_tracing, reset_tracing

        reset_tracing()
        # Should not raise regardless of installed packages
        result = configure_agent_tracing()
        assert isinstance(result, bool)
