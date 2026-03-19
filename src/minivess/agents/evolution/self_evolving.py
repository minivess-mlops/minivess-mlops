"""Self-evolving segmentation agent (TissueLab pattern).

Monitors model drift and performance trends, decides when to trigger
retraining via Prefect deployment. All retraining requires human approval.

Based on: Li et al. (2025), "A co-evolving agentic AI system for medical
imaging analysis" (TissueLab, arXiv:2509.20279).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from minivess.agents.constraints.pccp import PCCPGate

with contextlib.suppress(ImportError):
    from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evolution decision model
# ---------------------------------------------------------------------------


class EvolutionDecision(BaseModel):
    """Result of a self-evolving agent decision.

    Parameters
    ----------
    action:
        Recommended action: monitor, retrain, or blocked.
    pccp_approved:
        Whether the PCCP gate approved this action.
    confidence:
        Confidence in the decision.
    reason:
        Explanation of the decision.
    deployment_params:
        Parameters for Prefect run_deployment() if action == retrain.
    """

    action: Literal["monitor", "retrain", "blocked"] = Field(
        description="Recommended action"
    )
    pccp_approved: bool = Field(description="Whether PCCP gate approved this action")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    reason: str = Field(description="Explanation of the decision")
    deployment_params: dict[str, Any] | None = Field(
        default=None,
        description="Parameters for Prefect run_deployment() if retraining",
    )


# ---------------------------------------------------------------------------
# Context and pure computation
# ---------------------------------------------------------------------------


@dataclass
class EvolutionContext:
    """Dependencies for the self-evolving agent.

    Parameters
    ----------
    drift_score:
        Overall drift score from drift detection (0-1).
    performance_history:
        Recent model performance values (e.g., Dice scores).
    retrain_cost_estimate:
        Estimated GPU cost for retraining in USD.
    feature_drift_scores:
        Per-feature drift scores.
    """

    drift_score: float = 0.0
    performance_history: list[float] = field(default_factory=list)
    retrain_cost_estimate: float = 0.0
    feature_drift_scores: dict[str, float] = field(default_factory=dict)


def _is_declining(values: list[float], window: int = 3) -> bool:
    """Check if the last `window` values show monotonic decline.

    Parameters
    ----------
    values:
        Performance history values.
    window:
        Number of recent values to check.

    Returns
    -------
    True if the last `window` values are monotonically declining.
    """
    if len(values) < window:
        return False
    recent = values[-window:]
    return all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))


def evaluate_drift_action(
    ctx: EvolutionContext,
    drift_threshold: float = 0.5,
    decline_window: int = 3,
) -> Literal["monitor", "retrain"]:
    """Evaluate whether drift warrants retraining.

    Parameters
    ----------
    ctx:
        Evolution context with drift and performance data.
    drift_threshold:
        Drift score above which retraining is recommended.
    decline_window:
        Number of recent values to check for performance decline.

    Returns
    -------
    Recommended action: "monitor" or "retrain".
    """
    if ctx.drift_score >= drift_threshold:
        return "retrain"
    if _is_declining(ctx.performance_history, window=decline_window):
        return "retrain"
    return "monitor"


def build_retraining_params(
    deployment_name: str = "training-flow/default",
    config_overrides: dict[str, Any] | None = None,
    reason: str = "drift_detected",
) -> dict[str, Any]:
    """Build parameters for Prefect run_deployment() trigger.

    Parameters
    ----------
    deployment_name:
        Prefect deployment name to trigger.
    config_overrides:
        Config overrides to pass to the training flow.
    reason:
        Reason for triggering retraining.

    Returns
    -------
    Dict suitable for Prefect run_deployment() call.
    """
    return {
        "deployment_name": deployment_name,
        "parameters": config_overrides or {},
        "reason": reason,
        "human_approval_required": True,
    }


def pccp_gated_evolution(
    ctx: EvolutionContext,
    gate: PCCPGate,
    confidence: float,
    drift_threshold: float = 0.5,
    retrain_cost: float = 10.0,
    decline_window: int = 3,
) -> EvolutionDecision:
    """Make a PCCP-gated evolution decision.

    Monitor decisions bypass the gate (no action is taken).
    Retrain decisions must pass the PCCP confidence and budget gates.

    Parameters
    ----------
    ctx:
        Evolution context.
    gate:
        PCCP gate for constraint enforcement.
    confidence:
        Confidence level for the evolution decision.
    drift_threshold:
        Drift score threshold for retraining.
    retrain_cost:
        Budget cost of a retraining run.
    decline_window:
        Window for performance decline detection.

    Returns
    -------
    EvolutionDecision with action and PCCP status.
    """
    raw_action = evaluate_drift_action(
        ctx, drift_threshold=drift_threshold, decline_window=decline_window
    )

    if raw_action == "monitor":
        return EvolutionDecision(
            action="monitor",
            pccp_approved=True,
            confidence=confidence,
            reason="Drift within acceptable bounds, continuing to monitor",
        )

    # Retrain requires PCCP gate approval
    pccp_decision = gate.check(confidence=confidence, cost=retrain_cost)

    if not pccp_decision.approved:
        return EvolutionDecision(
            action="blocked",
            pccp_approved=False,
            confidence=confidence,
            reason=f"Retraining blocked by PCCP gate: {pccp_decision.reason}",
        )

    params = build_retraining_params(reason="drift_detected")
    return EvolutionDecision(
        action="retrain",
        pccp_approved=True,
        confidence=confidence,
        reason=(
            f"Drift score {ctx.drift_score:.3f} exceeds threshold "
            f"{drift_threshold:.3f}, retraining recommended"
        ),
        deployment_params=params,
    )


# ---------------------------------------------------------------------------
# Pydantic AI Agent
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are a self-evolving segmentation pipeline monitor. Your role is to analyze
model drift and performance trends, then decide whether to trigger retraining.

Consider:
- Current drift score relative to historical baselines
- Performance trajectory (stable, declining, improving)
- Retraining cost vs. expected performance improvement
- All retraining requires human approval — never auto-retrain

Actions:
- "monitor": drift is within bounds, continue watching
- "retrain": drift or performance decline warrants retraining
  (triggers Prefect run_deployment() with human_approval_required=True)
"""


def _build_agent(model: str | None = None) -> Any:
    """Build the self-evolving Agent (without PrefectAgent wrapper)."""
    from minivess.agents.config import load_agent_config

    config = load_agent_config()
    model_name = model or config.model

    agent: Agent[EvolutionContext, EvolutionDecision] = Agent(
        model_name,
        output_type=EvolutionDecision,
        deps_type=EvolutionContext,
        name="self-evolving-segmentation",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def get_drift_metrics(ctx: RunContext[EvolutionContext]) -> dict[str, Any]:
        """Get current drift metrics."""
        deps = ctx.deps
        return {
            "overall_drift_score": deps.drift_score,
            "feature_drift_scores": deps.feature_drift_scores,
        }

    @agent.tool
    def get_performance_history(
        ctx: RunContext[EvolutionContext],
    ) -> dict[str, Any]:
        """Get recent model performance history."""
        deps = ctx.deps
        history = deps.performance_history
        return {
            "values": history,
            "n_entries": len(history),
            "latest": history[-1] if history else None,
            "trend": "declining" if _is_declining(history) else "stable",
        }

    @agent.tool
    def get_retraining_params(
        ctx: RunContext[EvolutionContext],
    ) -> dict[str, Any]:
        """Get estimated retraining cost and deployment params."""
        return {
            "estimated_cost_usd": ctx.deps.retrain_cost_estimate,
            "deployment_name": "training-flow/default",
            "human_approval_required": True,
        }

    return agent


def create_self_evolving_agent(
    model: str | None = None,
    config: Any | None = None,
) -> Any:
    """Create self-evolving agent as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier.
    config:
        Override AgentConfig.

    Returns
    -------
    PrefectAgent wrapping the self-evolving agent.
    """
    from minivess.agents.factory import make_prefect_agent

    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
