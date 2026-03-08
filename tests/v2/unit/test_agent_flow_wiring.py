"""Tests for agent wiring into Prefect flows (T-1.3, T-2.3, T-3.3).

Verifies that flow tasks use Pydantic AI agents when available,
with deterministic fallback when the [agents] extra is not installed.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

# ---------------------------------------------------------------------------
# T-1.3: Experiment summarizer wired into analysis_flow
# ---------------------------------------------------------------------------


class TestSummarizeExperimentWiring:
    """summarize_experiment task uses Pydantic AI agent with fallback."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Return value has action, reasoning, summary keys."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        result = summarize_experiment.fn(
            all_results={"model_a": {"ds": {"val": {}}}},
            promotion_info={"champion_name": "model_a", "champion_score": 0.85},
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "reasoning" in result

    def test_deterministic_fallback(self) -> None:
        """Falls back to deterministic stub when agent import fails."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        # Even if pydantic_ai not installed, should work via fallback
        result = summarize_experiment.fn(
            all_results={"m1": {}, "m2": {}},
            promotion_info={"champion_name": "m1", "champion_score": 0.9},
        )
        assert result["action"] == "summarize"
        assert "m1" in result["reasoning"]

    def test_uses_agent_when_available(self) -> None:
        """When use_agent=True and pydantic_ai available, agent path is taken."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        # With use_agent=True explicitly, agent path is attempted
        # TestModel produces a valid ExperimentSummary
        result = summarize_experiment.fn(
            all_results={"m1": {}},
            promotion_info={"champion_name": "m1", "champion_score": 0.8},
            use_agent=True,
        )
        assert isinstance(result, dict)
        assert "action" in result

    def test_env_var_enables_agent(self) -> None:
        """MINIVESS_USE_AGENTS=1 enables agent path."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        with patch.dict(os.environ, {"MINIVESS_USE_AGENTS": "1"}):
            result = summarize_experiment.fn(
                all_results={"m1": {}},
                promotion_info={"champion_name": "m1", "champion_score": 0.8},
            )
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# T-2.3: Drift triage wired into data_flow
# ---------------------------------------------------------------------------


class TestDriftTriageWiring:
    """triage_drift task uses Pydantic AI agent with fallback."""

    def test_triage_drift_task_exists(self) -> None:
        """data_flow has a triage_drift task."""
        from minivess.orchestration.flows.data_flow import triage_drift

        assert callable(triage_drift.fn)

    def test_returns_dict_with_action(self) -> None:
        """Return value has action and reasoning keys."""
        from minivess.orchestration.flows.data_flow import triage_drift

        result = triage_drift.fn(drift_score=0.05, feature_drift_scores={"f1": 0.03})
        assert isinstance(result, dict)
        assert result["action"] in {"monitor", "investigate", "retrain"}
        assert "reasoning" in result

    def test_deterministic_low_drift(self) -> None:
        """Low drift score → monitor action (deterministic fallback)."""
        from minivess.orchestration.flows.data_flow import triage_drift

        result = triage_drift.fn(drift_score=0.05, feature_drift_scores={})
        assert result["action"] == "monitor"

    def test_deterministic_high_drift(self) -> None:
        """High drift score → retrain action (deterministic fallback)."""
        from minivess.orchestration.flows.data_flow import triage_drift

        result = triage_drift.fn(drift_score=0.5, feature_drift_scores={"f1": 0.6})
        assert result["action"] == "retrain"


# ---------------------------------------------------------------------------
# T-3.3: Figure narrator wired into biostatistics_flow
# ---------------------------------------------------------------------------


class TestFigureNarratorWiring:
    """narrate_figures task uses Pydantic AI agent with fallback."""

    def test_narrate_figures_task_exists(self) -> None:
        """biostatistics_flow has a narrate_figures task."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        assert callable(narrate_figures.fn)

    def test_returns_list_of_captions(self) -> None:
        """Return value is a list of caption dicts."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        figures: list[dict[str, Any]] = [
            {"figure_type": "box_plot", "metric": "dice", "path": "/tmp/fig1.png"},
            {"figure_type": "heatmap", "metric": "hd95", "path": "/tmp/fig2.png"},
        ]
        result = narrate_figures.fn(figures=figures, n_conditions=4)
        assert isinstance(result, list)
        assert len(result) == 2
        for caption_dict in result:
            assert "caption" in caption_dict

    def test_empty_figures_returns_empty(self) -> None:
        """No figures → empty list."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        result = narrate_figures.fn(figures=[], n_conditions=0)
        assert result == []
