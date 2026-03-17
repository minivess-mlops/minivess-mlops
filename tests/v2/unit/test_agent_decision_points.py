"""Tests for Pydantic AI agent decision points (T-1.2, T-2.2, T-3.2).

All tests use pydantic_ai.models.test.TestModel — zero real LLM API calls.
"""

from __future__ import annotations

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai", reason="pydantic_ai not installed")
from pydantic_ai.models.test import TestModel  # noqa: E402

# ---------------------------------------------------------------------------
# T-1.2: Experiment summarizer agent
# ---------------------------------------------------------------------------


class TestExperimentSummarizer:
    """Tests for the experiment summarizer agent."""

    def test_build_agent(self):
        """Agent builds without error."""
        from minivess.agents.experiment_summarizer import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "experiment-summarizer"

    def test_agent_has_tool(self):
        """Agent has the fetch_run_metrics tool registered."""
        from minivess.agents.experiment_summarizer import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "fetch_run_metrics" in tool_names

    def test_create_summarizer_returns_prefect_agent(self):
        """Factory returns a PrefectAgent."""
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.experiment_summarizer import create_summarizer

        pa = create_summarizer(model="test")
        assert isinstance(pa, PrefectAgent)

    def test_run_with_test_model(self):
        """Agent runs to completion with TestModel."""
        from minivess.agents.experiment_summarizer import (
            AnalysisContext,
            _build_agent,
        )
        from minivess.agents.models import ExperimentSummary

        agent = _build_agent(model="test")
        ctx = AnalysisContext(
            experiment_name="test_exp",
            n_models=3,
            best_model="dynunet_dice_ce",
            best_metric_value=0.824,
        )
        test_output = {
            "narrative": "Model A outperformed B.",
            "best_model": "dynunet_dice_ce",
            "best_metric_value": 0.824,
            "key_findings": ["Finding 1"],
            "recommendations": [],
        }
        result = agent.run_sync(
            "Summarize this experiment.",
            deps=ctx,
            model=TestModel(custom_output_args=test_output, call_tools=[]),
        )
        assert isinstance(result.output, ExperimentSummary)
        assert len(result.output.key_findings) >= 1


# ---------------------------------------------------------------------------
# T-2.2: Drift triage agent
# ---------------------------------------------------------------------------


class TestDriftTriage:
    """Tests for the drift triage agent."""

    def test_build_agent(self):
        from minivess.agents.drift_triage import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "drift-triage"

    def test_agent_has_tools(self):
        from minivess.agents.drift_triage import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "get_drift_report" in tool_names
        assert "get_historical_drift" in tool_names
        assert "get_retraining_cost" in tool_names

    def test_create_drift_triage_returns_prefect_agent(self):
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.drift_triage import create_drift_triage_agent

        pa = create_drift_triage_agent(model="test")
        assert isinstance(pa, PrefectAgent)

    def test_run_with_test_model(self):
        from minivess.agents.drift_triage import DriftContext, _build_agent
        from minivess.agents.models import DriftTriageResult

        agent = _build_agent(model="test")
        ctx = DriftContext(drift_score=0.15, feature_drift_scores={"f1": 0.2})
        test_output = {
            "action": "monitor",
            "confidence": 0.85,
            "reasoning": "Low drift score",
            "affected_features": ["f1"],
            "severity": "low",
        }
        result = agent.run_sync(
            "Triage this drift.",
            deps=ctx,
            model=TestModel(custom_output_args=test_output, call_tools=[]),
        )
        assert isinstance(result.output, DriftTriageResult)
        assert result.output.action in {"monitor", "retrain", "investigate"}


# ---------------------------------------------------------------------------
# T-3.2: Figure narrator agent
# ---------------------------------------------------------------------------


class TestFigureNarrator:
    """Tests for the figure narration agent."""

    def test_build_agent(self):
        from minivess.agents.figure_narrator import _build_agent

        agent = _build_agent(model="test")
        assert agent.name == "figure-narrator"

    def test_agent_has_tools(self):
        from minivess.agents.figure_narrator import _build_agent

        agent = _build_agent(model="test")
        tool_names = list(agent._function_toolset.tools.keys())
        assert "get_figure_metadata" in tool_names
        assert "get_statistical_context" in tool_names

    def test_create_figure_narrator_returns_prefect_agent(self):
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        from minivess.agents.figure_narrator import create_figure_narrator

        pa = create_figure_narrator(model="test")
        assert isinstance(pa, PrefectAgent)

    def test_run_with_test_model(self):
        from minivess.agents.figure_narrator import FigureContext, _build_agent
        from minivess.agents.models import FigureCaption

        agent = _build_agent(model="test")
        ctx = FigureContext(
            figure_type="box_plot",
            n_conditions=4,
            primary_metric="Dice",
        )
        test_output = {
            "caption": "Comparison of 4 conditions on Dice metric.",
            "alt_text": "Box plot showing Dice scores",
            "statistical_note": None,
        }
        result = agent.run_sync(
            "Generate a caption for this figure.",
            deps=ctx,
            model=TestModel(custom_output_args=test_output, call_tools=[]),
        )
        assert isinstance(result.output, FigureCaption)
        assert len(result.output.caption) > 0
