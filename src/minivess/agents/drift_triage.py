"""Drift triage agent — Pydantic AI agent for data_flow.

Replaces DeterministicDriftTriage with LLM-powered nuanced drift analysis.
Falls back to deterministic stub when [agents] extra is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from minivess.agents.config import AgentConfig, load_agent_config
from minivess.agents.factory import make_prefect_agent
from minivess.agents.models import DriftTriageResult


@dataclass
class DriftContext:
    """Dependencies injected into the drift triage agent.

    Parameters
    ----------
    drift_score:
        Overall drift score from whylogs/Evidently.
    feature_drift_scores:
        Per-feature drift scores.
    historical_drift:
        Past drift trend data.
    retraining_cost_estimate:
        Estimated compute cost for retraining (optional).
    """

    drift_score: float = 0.0
    feature_drift_scores: dict[str, float] = field(default_factory=dict)
    historical_drift: list[dict[str, Any]] = field(default_factory=list)
    retraining_cost_estimate: float | None = None


_SYSTEM_PROMPT = """\
You are a data drift analyst for a biomedical image segmentation pipeline.
Analyze the provided drift metrics and recommend an action.

Consider:
- The overall drift score and which features are most affected
- Historical drift patterns (is this a trend or anomaly?)
- Retraining cost vs. drift severity trade-off
- False positive risk for low-confidence drift signals

Actions:
- "monitor": drift is within acceptable bounds, continue monitoring
- "investigate": drift is ambiguous, needs human review
- "retrain": drift is significant and retraining is justified
"""


def _build_agent(model: str | None = None) -> Agent[DriftContext, DriftTriageResult]:
    """Build the drift triage Agent (without PrefectAgent wrapper)."""
    config = load_agent_config()
    model_name = model or config.model

    agent = Agent(
        model_name,
        output_type=DriftTriageResult,
        deps_type=DriftContext,
        name="drift-triage",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def get_drift_report(ctx: RunContext[DriftContext]) -> dict[str, Any]:
        """Get the current drift report."""
        deps = ctx.deps
        return {
            "overall_drift_score": deps.drift_score,
            "feature_scores": deps.feature_drift_scores,
        }

    @agent.tool
    def get_historical_drift(ctx: RunContext[DriftContext]) -> list[dict[str, Any]]:
        """Get historical drift trends."""
        return deps.historical_drift if (deps := ctx.deps) else []

    @agent.tool
    def get_retraining_cost(ctx: RunContext[DriftContext]) -> dict[str, Any]:
        """Estimate the cost of retraining."""
        deps = ctx.deps
        return {
            "estimated_cost_usd": deps.retraining_cost_estimate,
            "available": deps.retraining_cost_estimate is not None,
        }

    return agent


def create_drift_triage_agent(
    model: str | None = None,
    config: AgentConfig | None = None,
) -> Any:
    """Create drift triage agent as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier.
    config:
        Override AgentConfig.

    Returns
    -------
    PrefectAgent wrapping the drift triage agent.
    """
    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
