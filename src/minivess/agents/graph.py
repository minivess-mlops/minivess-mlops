"""LangGraph agent definitions for ML pipeline orchestration.

Training pipeline graph: deterministic state machine for
data→train→evaluate→register→notify workflows.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class TrainingState(TypedDict):
    """State passed through the training pipeline graph."""

    model_name: str
    dataset: str
    status: str
    results: dict[str, Any]
    messages: list[str]
    metrics_pass: bool


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def prepare_data_node(state: TrainingState) -> dict[str, Any]:
    """Load and validate dataset."""
    logger.info("Preparing data for dataset=%s", state["dataset"])
    return {
        "messages": [*state["messages"], f"Data prepared: {state['dataset']}"],
        "status": "data_ready",
    }


def train_node(state: TrainingState) -> dict[str, Any]:
    """Run training loop (placeholder — real training wired externally)."""
    logger.info("Training model=%s on dataset=%s", state["model_name"], state["dataset"])
    results = {**state["results"], "train_loss": 0.3, "val_loss": 0.4}
    return {
        "messages": [*state["messages"], f"Training complete: {state['model_name']}"],
        "results": results,
        "status": "trained",
    }


def evaluate_node(state: TrainingState) -> dict[str, Any]:
    """Compute metrics on validation set."""
    logger.info("Evaluating model=%s", state["model_name"])
    results = {**state["results"], "val_dice": 0.75}
    return {
        "messages": [*state["messages"], "Evaluation complete"],
        "results": results,
        "status": "evaluated",
    }


def register_node(state: TrainingState) -> dict[str, Any]:
    """Register model in MLflow registry."""
    logger.info("Registering model=%s", state["model_name"])
    results = {**state["results"], "registered": True}
    return {
        "messages": [*state["messages"], f"Model registered: {state['model_name']}"],
        "results": results,
        "status": "completed",
    }


def notify_node(state: TrainingState) -> dict[str, Any]:
    """Log results and send notifications."""
    passed = state.get("metrics_pass", False)
    results = dict(state["results"])
    if not passed:
        results["skipped_registration"] = True
    return {
        "messages": [*state["messages"], "Notification sent"],
        "results": results,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: TrainingState) -> str:
    """Route to register or notify based on metrics."""
    if state.get("metrics_pass", False):
        return "register"
    return "notify"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_training_graph() -> Any:
    """Build and compile the training pipeline StateGraph.

    Returns
    -------
    Compiled LangGraph graph with .invoke() method.
    """
    graph = StateGraph(TrainingState)

    graph.add_node("prepare_data", prepare_data_node)
    graph.add_node("train_model", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("register", register_node)
    graph.add_node("notify", notify_node)

    graph.set_entry_point("prepare_data")
    graph.add_edge("prepare_data", "train_model")
    graph.add_edge("train_model", "evaluate")
    graph.add_conditional_edges("evaluate", _route_after_evaluate)
    graph.add_edge("register", "notify")
    graph.add_edge("notify", END)

    return graph.compile()
