"""Tests for agent Prefect UI observability (T-5.3).

Verifies that agent tasks are properly decorated with Prefect @task
and have descriptive names visible in the Prefect dashboard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")


class TestAgentTasksHavePrefectNames:
    """All agent decision point tasks have Prefect task names."""

    def test_summarize_experiment_is_prefect_task(self) -> None:
        """summarize_experiment is a Prefect @task with a name."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        assert hasattr(summarize_experiment, "fn")
        assert summarize_experiment.name == "summarize-experiment"

    def test_triage_drift_is_prefect_task(self) -> None:
        """triage_drift is a Prefect @task with a name."""
        from minivess.orchestration.flows.data_flow import triage_drift

        assert hasattr(triage_drift, "fn")
        assert triage_drift.name == "triage-drift"

    def test_narrate_figures_is_prefect_task(self) -> None:
        """narrate_figures is a Prefect @task with a name."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        assert hasattr(narrate_figures, "fn")
        assert narrate_figures.name == "narrate-figures"


class TestAgentConfigExposesModelForUI:
    """AgentConfig fields are serializable for Prefect artifact logging."""

    def test_agent_config_serializable(self) -> None:
        """AgentConfig can be serialized to dict for Prefect artifacts."""
        from minivess.agents.config import AgentConfig

        cfg = AgentConfig()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "model" in d
        assert "retries" in d

    def test_tracing_status_serializable(self) -> None:
        """Tracing status dict is JSON-serializable."""
        import json

        from minivess.agents.tracing import get_tracing_status

        status = get_tracing_status()
        serialized = json.dumps(status)
        assert isinstance(serialized, str)
