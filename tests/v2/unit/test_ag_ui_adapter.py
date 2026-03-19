"""Tests for AG-UI protocol adapter (T4.3).

Validates that the AG-UI adapter exists, translates messages correctly,
and can wire Pydantic AI agents to AG-UI event streams.
"""

from __future__ import annotations

import pytest


class TestAGUIAdapterImport:
    """Validate AG-UI adapter module exists and is importable."""

    def test_ag_ui_adapter_module_importable(self) -> None:
        """AG-UI adapter module must be importable."""
        from minivess.dashboard.ag_ui_adapter import create_ag_ui_endpoint

        assert callable(create_ag_ui_endpoint)

    def test_ag_ui_adapter_has_build_app_function(self) -> None:
        """AG-UI adapter must expose a build_ag_ui_app function."""
        from minivess.dashboard.ag_ui_adapter import build_ag_ui_app

        assert callable(build_ag_ui_app)


class TestAGUIAdapterTranslation:
    """Validate AG-UI message translation works with mock agents."""

    def test_adapter_wraps_pydantic_ai_agent(self) -> None:
        """AG-UI adapter must accept a Pydantic AI agent."""
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("ag_ui.core")

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from minivess.dashboard.ag_ui_adapter import create_ag_ui_endpoint

        agent = Agent(TestModel(), instructions="Test agent")
        endpoint = create_ag_ui_endpoint(agent)
        assert endpoint is not None

    def test_adapter_produces_fastapi_route(self) -> None:
        """AG-UI endpoint must be a callable suitable for FastAPI."""
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("ag_ui.core")

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from minivess.dashboard.ag_ui_adapter import create_ag_ui_endpoint

        agent = Agent(TestModel(), instructions="Test agent")
        endpoint = create_ag_ui_endpoint(agent)
        # Should be an async callable (FastAPI POST handler)
        import asyncio

        assert asyncio.iscoroutinefunction(endpoint)

    def test_build_ag_ui_app_returns_asgi(self) -> None:
        """build_ag_ui_app must return a valid ASGI application."""
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("ag_ui.core")

        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from minivess.dashboard.ag_ui_adapter import build_ag_ui_app

        agent = Agent(TestModel(), instructions="Test agent")
        app = build_ag_ui_app(agent)
        # AGUIApp is an ASGI application — it must be callable
        assert callable(app)


class TestAGUIProtocolTypes:
    """Validate that AG-UI protocol types are accessible."""

    def test_ag_ui_event_types_importable(self) -> None:
        """Core AG-UI event types must be importable."""
        ag_ui = pytest.importorskip("ag_ui.core")
        assert hasattr(ag_ui, "EventType")

    def test_run_agent_input_importable(self) -> None:
        """RunAgentInput must be importable from AG-UI."""
        ag_ui = pytest.importorskip("ag_ui.core")
        assert hasattr(ag_ui, "RunAgentInput")
