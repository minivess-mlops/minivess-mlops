"""Integration tests: Pydantic AI agents inside Prefect tasks (T-5.1).

Exercises the full stack: Prefect @task → Pydantic AI Agent → TestModel.
Uses prefect_test_harness() (from conftest.py) for ephemeral Prefect server.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# T-5.1a: Experiment summarizer inside Prefect task context
# ---------------------------------------------------------------------------


class TestExperimentSummarizerInPrefect:
    """Experiment summarizer agent runs inside a Prefect task."""

    def test_summarize_task_runs_with_agent(self) -> None:
        """summarize_experiment task completes with use_agent=True."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        result = summarize_experiment.fn(
            all_results={"dice_ce_fold0": {"minivess": {"val": {}}}},
            promotion_info={"champion_name": "dice_ce_fold0", "champion_score": 0.82},
            use_agent=True,
        )
        assert isinstance(result, dict)
        assert result["action"] == "summarize"
        assert "reasoning" in result
        assert "summary" in result

    def test_summarize_task_runs_with_fallback(self) -> None:
        """summarize_experiment task completes with deterministic fallback."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        result = summarize_experiment.fn(
            all_results={"m1": {}, "m2": {}, "m3": {}},
            promotion_info={"champion_name": "m1", "champion_score": 0.9},
            use_agent=False,
        )
        assert result["action"] == "summarize"
        assert "3 models" in result["reasoning"]

    def test_agent_output_matches_stub_contract(self) -> None:
        """Agent output has same keys as deterministic stub."""
        from minivess.orchestration.flows.analysis_flow import summarize_experiment

        agent_result = summarize_experiment.fn(
            all_results={"m1": {}},
            promotion_info={"champion_name": "m1", "champion_score": 0.8},
            use_agent=True,
        )
        stub_result = summarize_experiment.fn(
            all_results={"m1": {}},
            promotion_info={"champion_name": "m1", "champion_score": 0.8},
            use_agent=False,
        )
        # Both must have action and reasoning
        assert set(agent_result.keys()) >= {"action", "reasoning"}
        assert set(stub_result.keys()) >= {"action", "reasoning"}


# ---------------------------------------------------------------------------
# T-5.1b: Drift triage inside Prefect task context
# ---------------------------------------------------------------------------


class TestDriftTriageInPrefect:
    """Drift triage agent runs inside a Prefect task."""

    def test_triage_task_runs_with_agent(self) -> None:
        """triage_drift task completes with use_agent=True."""
        from minivess.orchestration.flows.data_flow import triage_drift

        result = triage_drift.fn(
            drift_score=0.05,
            feature_drift_scores={"feature_a": 0.03},
            use_agent=True,
        )
        assert isinstance(result, dict)
        assert result["action"] == "monitor"

    def test_triage_high_drift_with_agent(self) -> None:
        """High drift score triggers retrain via agent."""
        from minivess.orchestration.flows.data_flow import triage_drift

        result = triage_drift.fn(
            drift_score=0.5,
            feature_drift_scores={"feature_a": 0.6, "feature_b": 0.4},
            use_agent=True,
        )
        assert result["action"] == "retrain"

    def test_triage_output_contract(self) -> None:
        """Agent and stub return compatible dicts."""
        from minivess.orchestration.flows.data_flow import triage_drift

        for use_agent in [True, False]:
            result = triage_drift.fn(
                drift_score=0.05,
                feature_drift_scores={},
                use_agent=use_agent,
            )
            assert "action" in result
            assert "reasoning" in result


# ---------------------------------------------------------------------------
# T-5.1c: Figure narrator inside Prefect task context
# ---------------------------------------------------------------------------


class TestFigureNarratorInPrefect:
    """Figure narrator agent runs inside a Prefect task."""

    def test_narrate_task_runs_with_agent(self) -> None:
        """narrate_figures task completes with use_agent=True."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        figures: list[dict[str, Any]] = [
            {"figure_type": "box_plot", "metric": "dice", "path": "/out/fig1.png"},
        ]
        result = narrate_figures.fn(figures=figures, n_conditions=4, use_agent=True)
        assert len(result) == 1
        assert "caption" in result[0]

    def test_narrate_multiple_figures(self) -> None:
        """Multiple figures produce multiple captions."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        figures: list[dict[str, Any]] = [
            {"figure_type": "box_plot", "metric": "dice", "path": "/out/fig1.png"},
            {"figure_type": "heatmap", "metric": "hd95", "path": "/out/fig2.png"},
            {"figure_type": "bar_chart", "metric": "clDice", "path": "/out/fig3.png"},
        ]
        result = narrate_figures.fn(figures=figures, n_conditions=3, use_agent=True)
        assert len(result) == 3

    def test_narrate_output_contract(self) -> None:
        """Agent and stub produce captions with same shape."""
        from minivess.orchestration.flows.biostatistics_flow import narrate_figures

        figures: list[dict[str, Any]] = [
            {"figure_type": "scatter", "metric": "dice", "path": "/out/f.png"},
        ]
        for use_agent in [True, False]:
            result = narrate_figures.fn(
                figures=figures, n_conditions=2, use_agent=use_agent
            )
            assert len(result) == 1
            assert "caption" in result[0]
            assert "figure_type" in result[0]
