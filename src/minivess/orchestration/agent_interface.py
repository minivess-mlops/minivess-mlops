"""Agent decision point interfaces for LLM-powered decisions inside Prefect tasks.

Defines the AgentDecisionPoint protocol and deterministic stub implementations.
These stubs will be replaced by Pydantic AI agents (via PrefectAgent) in a
follow-up PR (#341).

Architecture:
  - Prefect @flow handles macro-orchestration (scheduling, retries, Docker isolation)
  - Pydantic AI agents handle micro-decisions inside @task functions
  - AgentDecisionPoint is the interface both stubs and real agents implement

Two patterns available for real agent implementation:
  PATTERN A (recommended): Pydantic AI + PrefectAgent — each LLM call is a
    cached/retryable Prefect task with full observability
  PATTERN B: LangGraph StateGraph — for complex multi-step reasoning with
    its own checkpointer (opaque to Prefect)
"""

from __future__ import annotations

from typing import Any, Protocol


class AgentDecisionPoint(Protocol):
    """Interface for decision points inside Prefect @task functions.

    Implementations can be:
    - Deterministic stubs (current — rule-based, no LLM)
    - Pydantic AI agents (future — LLM-powered via PrefectAgent)
    - LangGraph graphs (future — complex multi-step reasoning)
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make a decision given context from the Prefect task.

        Parameters
        ----------
        context:
            Task-specific context (metrics, model metadata, etc.)

        Returns
        -------
        Decision dict with ``action`` and ``reasoning`` keys.
        """
        ...


# ---------------------------------------------------------------------------
# Stub implementations (deterministic, no LLM)
# ---------------------------------------------------------------------------


class DeterministicDriftTriage:
    """Stub for data_flow: drift triage decision.

    Rule: monitor if drift score < threshold, retrain otherwise.
    TODO(#341): Replace with Pydantic AI agent for nuanced triage.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold = threshold

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        drift_score = context.get("drift_score", 0.0)
        if drift_score < self._threshold:
            return {
                "action": "monitor",
                "reasoning": (
                    f"Drift score {drift_score:.3f} below threshold "
                    f"{self._threshold}. Continue monitoring."
                ),
            }
        return {
            "action": "retrain",
            "reasoning": (
                f"Drift score {drift_score:.3f} exceeds threshold "
                f"{self._threshold}. Retraining recommended."
            ),
        }


class DeterministicPromotionDecision:
    """Stub for deploy_flow: model promotion decision.

    Rule: promote if all metric thresholds are met.
    TODO(#341): Replace with Pydantic AI agent for risk-aware promotion.
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics = context.get("metrics", {})
        thresholds = context.get("thresholds", {"dsc": 0.5})

        failures = []
        for metric, threshold in thresholds.items():
            value = metrics.get(metric)
            if value is not None and value < threshold:
                failures.append(f"{metric}={value:.3f} < {threshold}")

        if failures:
            return {
                "action": "reject",
                "reasoning": f"Promotion rejected: {'; '.join(failures)}",
            }
        return {
            "action": "promote",
            "reasoning": "All metric thresholds met. Promotion approved.",
        }


class DeterministicQATriage:
    """Stub for qa_flow: QA anomaly triage.

    Rule: classify by number of failures.
    TODO(#341): Replace with Pydantic AI agent for severity classification.
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        n_failures = context.get("n_failures", 0)
        n_warnings = context.get("n_warnings", 0)

        if n_failures > 0:
            return {
                "action": "alert",
                "reasoning": (
                    f"{n_failures} failures found. Immediate attention required."
                ),
                "severity": "high",
            }
        if n_warnings > 0:
            return {
                "action": "review",
                "reasoning": (f"{n_warnings} warnings found. Review recommended."),
                "severity": "medium",
            }
        return {
            "action": "pass",
            "reasoning": "No issues found.",
            "severity": "low",
        }


class DeterministicExperimentSummary:
    """Stub for analysis_flow: experiment summarization.

    Rule: return structured summary from metrics dict.
    TODO(#341): Replace with Pydantic AI agent for natural-language summaries.
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        n_models = context.get("n_models", 0)
        best_model = context.get("best_model", "unknown")
        best_metric = context.get("best_metric_value", 0.0)

        return {
            "action": "summarize",
            "reasoning": (
                f"Evaluated {n_models} models. Best: {best_model} "
                f"(metric={best_metric:.4f})."
            ),
            "summary": {
                "n_models": n_models,
                "best_model": best_model,
                "best_metric_value": best_metric,
            },
        }


class DeterministicFigureNarration:
    """Stub for biostatistics_flow: figure caption generation.

    Rule: return template-based caption.
    TODO(#341): Replace with Pydantic AI agent for paper-quality captions.
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        figure_type = context.get("figure_type", "unknown")
        n_conditions = context.get("n_conditions", 0)
        primary_metric = context.get("primary_metric", "unknown")

        return {
            "action": "narrate",
            "reasoning": (
                f"Generated {figure_type} figure comparing {n_conditions} "
                f"conditions on {primary_metric}."
            ),
            "caption": (
                f"Comparison of {n_conditions} experimental conditions. "
                f"Primary metric: {primary_metric}."
            ),
        }
