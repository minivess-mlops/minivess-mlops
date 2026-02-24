"""Experiment comparison agent: query MLflow runs, summarise with LLM."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class ComparisonState(TypedDict):
    """State for the experiment comparison agent."""

    experiment_name: str
    query: str
    summary: str
    runs_data: list[dict[str, Any]]
    messages: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def fetch_runs_node(state: ComparisonState) -> dict[str, Any]:
    """Fetch experiment runs (stub — uses synthetic data for now)."""
    logger.info("Fetching runs for experiment=%s", state["experiment_name"])
    # In production, this would call RunAnalytics.load_experiment_runs()
    runs = state.get("runs_data", [])
    if not runs:
        runs = [
            {"run_id": "synthetic_1", "metric_val_dice": 0.82},
            {"run_id": "synthetic_2", "metric_val_dice": 0.78},
        ]
    return {
        "runs_data": runs,
        "messages": [*state["messages"], f"Fetched {len(runs)} runs"],
    }


def analyse_runs_node(state: ComparisonState) -> dict[str, Any]:
    """Analyse run metrics (deterministic stats)."""
    runs = state["runs_data"]
    if runs:
        dices = [r.get("metric_val_dice", 0.0) for r in runs]
        best = max(dices) if dices else 0.0
        analysis = f"Best Dice: {best:.3f} across {len(runs)} runs"
    else:
        analysis = "No runs found"
    return {
        "messages": [*state["messages"], analysis],
    }


def summarise_node(state: ComparisonState) -> dict[str, Any]:
    """Summarise comparison using LLM."""
    from minivess.agents.llm import call_llm

    runs_text = "\n".join(
        f"- Run {r.get('run_id', '?')}: Dice={r.get('metric_val_dice', '?')}"
        for r in state["runs_data"]
    )
    prompt = (
        f"Summarise these experiment runs for {state['experiment_name']}:\n"
        f"{runs_text}\n\nQuery: {state['query']}"
    )
    summary = call_llm(prompt)
    return {
        "summary": summary,
        "messages": [*state["messages"], "Summary generated"],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_comparison_graph() -> Any:
    """Build and compile the experiment comparison StateGraph.

    Returns
    -------
    Compiled LangGraph graph with .invoke() method.
    """
    graph = StateGraph(ComparisonState)

    graph.add_node("fetch_runs", fetch_runs_node)
    graph.add_node("analyse_runs", analyse_runs_node)
    graph.add_node("summarise", summarise_node)

    graph.set_entry_point("fetch_runs")
    graph.add_edge("fetch_runs", "analyse_runs")
    graph.add_edge("analyse_runs", "summarise")
    graph.add_edge("summarise", END)

    return graph.compile()
