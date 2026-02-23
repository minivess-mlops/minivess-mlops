"""Braintrust evaluation framework for segmentation models and LLM agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from a single evaluation case."""

    input_id: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuite:
    """Evaluation suite configuration for Braintrust."""

    name: str
    description: str = ""
    scorers: list[str] = field(default_factory=list)

    def add_scorer(self, name: str) -> None:
        """Add a scorer to the suite if not already present."""
        if name not in self.scorers:
            self.scorers.append(name)

    def to_config(self) -> dict[str, Any]:
        """Serialize the suite to a configuration dict."""
        return {
            "name": self.name,
            "description": self.description,
            "scorers": list(self.scorers),
        }


def build_segmentation_eval_suite() -> EvalSuite:
    """Build a Braintrust evaluation suite for segmentation models."""
    suite = EvalSuite(
        name="minivess-segmentation-eval",
        description="Comprehensive evaluation of 3D vessel segmentation models",
    )
    suite.add_scorer("dice_score")
    suite.add_scorer("surface_dice")
    suite.add_scorer("hausdorff_95")
    suite.add_scorer("calibration_ece")
    suite.add_scorer("inference_time_ms")
    return suite


def build_agent_eval_suite() -> EvalSuite:
    """Build a Braintrust evaluation suite for LLM agents."""
    suite = EvalSuite(
        name="minivess-agent-eval",
        description="Evaluation of LLM agent pipeline orchestration quality",
    )
    suite.add_scorer("task_completion")
    suite.add_scorer("tool_usage_efficiency")
    suite.add_scorer("error_recovery")
    return suite
