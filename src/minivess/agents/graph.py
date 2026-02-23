"""LangGraph agent definitions for ML pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State object passed through the LangGraph pipeline."""

    task: str = ""
    model_name: str = ""
    dataset: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    status: str = "pending"


def build_training_graph() -> dict[str, Any]:
    """Build a LangGraph state graph for the training pipeline.

    Returns a graph configuration dict describing the nodes and edges.
    The actual LangGraph StateGraph is created lazily to avoid
    import-time dependency on langgraph.

    Nodes:
        prepare_data: Load and validate dataset
        train_model: Run training loop
        evaluate: Compute metrics on validation set
        register: Register model in MLflow if metrics pass
        notify: Send notification about results
    """
    return {
        "graph_name": "training_pipeline",
        "nodes": [
            {
                "name": "prepare_data",
                "description": "Load dataset via DVC, validate with Pandera/GE",
            },
            {
                "name": "train_model",
                "description": "Run SegmentationTrainer.fit()",
            },
            {
                "name": "evaluate",
                "description": "Compute metrics, run WeightWatcher",
            },
            {
                "name": "register",
                "description": "Register model in MLflow registry",
            },
            {
                "name": "notify",
                "description": "Log results, send notifications",
            },
        ],
        "edges": [
            {"from": "prepare_data", "to": "train_model"},
            {"from": "train_model", "to": "evaluate"},
            {"from": "evaluate", "to": "register", "condition": "metrics_pass"},
            {"from": "evaluate", "to": "notify", "condition": "metrics_fail"},
            {"from": "register", "to": "notify"},
        ],
        "entry_point": "prepare_data",
    }


def build_evaluation_graph() -> dict[str, Any]:
    """Build a LangGraph state graph for model evaluation.

    Nodes:
        load_model: Load model from registry
        run_inference: Run inference on test set
        compute_metrics: Calculate all metrics
        calibrate: Apply temperature scaling
        generate_report: Create model card + audit entry
    """
    return {
        "graph_name": "evaluation_pipeline",
        "nodes": [
            {
                "name": "load_model",
                "description": "Load from MLflow registry",
            },
            {
                "name": "run_inference",
                "description": "Run on test set with audit trail",
            },
            {
                "name": "compute_metrics",
                "description": "Dice, clDice, NSD, calibration",
            },
            {
                "name": "calibrate",
                "description": "Temperature scaling + conformal prediction",
            },
            {
                "name": "generate_report",
                "description": "Model card + data card + audit",
            },
        ],
        "edges": [
            {"from": "load_model", "to": "run_inference"},
            {"from": "run_inference", "to": "compute_metrics"},
            {"from": "compute_metrics", "to": "calibrate"},
            {"from": "calibrate", "to": "generate_report"},
        ],
        "entry_point": "load_model",
    }
