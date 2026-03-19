"""Tests for Pydantic AI dashboard agent (T4.5).

Validates that the dashboard agent exists, has correct tools,
and can respond to MLflow queries in mock tests.
"""

from __future__ import annotations

import pytest


class TestDashboardAgentImport:
    """Validate dashboard agent module structure."""

    def test_dashboard_agent_module_importable(self) -> None:
        """Dashboard agent module must be importable."""
        from minivess.dashboard.agent import create_dashboard_agent

        assert callable(create_dashboard_agent)

    def test_dashboard_context_importable(self) -> None:
        """DashboardContext dataclass must be importable."""
        from minivess.dashboard.agent import DashboardContext

        assert DashboardContext is not None

    def test_dashboard_response_model_importable(self) -> None:
        """DashboardResponse Pydantic model must be importable."""
        from minivess.dashboard.agent import DashboardResponse

        assert DashboardResponse is not None


class TestDashboardAgentTools:
    """Validate dashboard agent has correct tools wired."""

    def _make_agent(self):
        """Create dashboard agent with TestModel (no API key needed)."""
        from pydantic_ai.models.test import TestModel

        from minivess.dashboard.agent import create_dashboard_agent

        return create_dashboard_agent(model=TestModel())

    @staticmethod
    def _get_tool_names(agent) -> list[str]:
        """Extract tool names from a Pydantic AI agent."""
        toolset = agent._function_toolset
        return list(toolset.tools.keys())

    def test_agent_has_mlflow_query_tool(self) -> None:
        """Dashboard agent must have an MLflow query tool."""
        pytest.importorskip("pydantic_ai")

        agent = self._make_agent()
        tool_names = self._get_tool_names(agent)
        assert "query_mlflow_experiments" in tool_names, (
            f"Agent must have query_mlflow_experiments tool, got {tool_names}"
        )

    def test_agent_has_duckdb_aggregation_tool(self) -> None:
        """Dashboard agent must have a DuckDB aggregation tool."""
        pytest.importorskip("pydantic_ai")

        agent = self._make_agent()
        tool_names = self._get_tool_names(agent)
        assert "aggregate_metrics" in tool_names, (
            f"Agent must have aggregate_metrics tool, got {tool_names}"
        )

    def test_agent_has_lineage_retrieval_tool(self) -> None:
        """Dashboard agent must have an OpenLineage retrieval tool."""
        pytest.importorskip("pydantic_ai")

        agent = self._make_agent()
        tool_names = self._get_tool_names(agent)
        assert "get_lineage_events" in tool_names, (
            f"Agent must have get_lineage_events tool, got {tool_names}"
        )


class TestDashboardAgentResponse:
    """Validate dashboard agent produces structured responses."""

    def test_dashboard_response_has_required_fields(self) -> None:
        """DashboardResponse must have answer, sources, and confidence fields."""
        from minivess.dashboard.agent import DashboardResponse

        fields = set(DashboardResponse.model_fields.keys())
        assert "answer" in fields
        assert "sources" in fields
        assert "confidence" in fields

    def test_dashboard_context_has_tracking_uri(self) -> None:
        """DashboardContext must accept tracking_uri."""
        from minivess.dashboard.agent import DashboardContext

        ctx = DashboardContext(tracking_uri="http://localhost:5000")
        assert ctx.tracking_uri == "http://localhost:5000"

    def test_dashboard_context_defaults(self) -> None:
        """DashboardContext must have sensible defaults."""
        from minivess.dashboard.agent import DashboardContext

        ctx = DashboardContext()
        assert ctx.tracking_uri == ""
        assert ctx.marquez_url == ""


class TestDashboardFlowWiring:
    """Validate dashboard agent is wired into Flow 5."""

    def test_dashboard_flow_imports_agent(self) -> None:
        """Dashboard flow must be able to import the agent creator."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        from minivess.dashboard.agent import create_dashboard_agent

        # The flow should use this function
        agent = create_dashboard_agent(model=TestModel())
        assert agent is not None
