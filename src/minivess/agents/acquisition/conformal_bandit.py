"""Conformal bandit acquisition agent (Flow 0b).

Thompson sampling with PCCP budget constraint for uncertainty-driven
data acquisition. Selects which volumes to acquire next based on
conformal prediction uncertainty scores.

Based on: Zhao et al. (2025), "ConfAgents: A Conformal-Guided Multi-Agent
Framework for Cost-Efficient Medical Diagnosis."
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from minivess.agents.constraints.pccp import PCCPGate

with contextlib.suppress(ImportError):
    from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------


class ThompsonSampler:
    """Thompson sampling with Beta priors for multi-armed bandits.

    Each arm maintains a Beta(alpha, beta) distribution. Sampling draws
    from each arm's posterior and selects the arm with the highest sample.

    Parameters
    ----------
    n_arms:
        Number of bandit arms.
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, n_arms: int, seed: int = 42) -> None:
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)
        # Beta priors: (alpha, beta) starting at (1, 1) = uniform
        self._alphas = np.ones(n_arms, dtype=np.float64)
        self._betas = np.ones(n_arms, dtype=np.float64)

    def sample(self) -> int:
        """Sample from each arm's posterior and return the best arm.

        Returns
        -------
        Index of the selected arm.
        """
        samples = self._rng.beta(self._alphas, self._betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update the posterior for an arm after observing a reward.

        Parameters
        ----------
        arm:
            Index of the arm that was played.
        reward:
            Observed reward in [0, 1].
        """
        self._alphas[arm] += reward
        self._betas[arm] += 1.0 - reward


# ---------------------------------------------------------------------------
# Volume ranking utilities
# ---------------------------------------------------------------------------


def rank_volumes_by_uncertainty(
    uncertainty_scores: dict[str, float],
) -> list[str]:
    """Rank volumes by uncertainty score (highest first).

    Parameters
    ----------
    uncertainty_scores:
        Mapping of volume_id → uncertainty score.

    Returns
    -------
    List of volume IDs sorted by descending uncertainty.
    """
    if not uncertainty_scores:
        return []
    return sorted(uncertainty_scores, key=uncertainty_scores.__getitem__, reverse=True)


def select_volumes_for_acquisition(
    uncertainty_scores: dict[str, float],
    gate: PCCPGate,
    confidence: float,
    cost_per_volume: float = 1.0,
) -> list[str]:
    """Select volumes for acquisition with PCCP budget enforcement.

    Iterates through volumes ranked by uncertainty (highest first).
    Each acquisition attempt is checked against the PCCP gate.
    Stops when the gate rejects (budget exhausted or confidence too low).

    Parameters
    ----------
    uncertainty_scores:
        Mapping of volume_id → uncertainty score.
    gate:
        PCCP gate for constraint enforcement.
    confidence:
        Confidence level for the acquisition decision.
    cost_per_volume:
        Cost per volume acquisition.

    Returns
    -------
    List of volume IDs selected for acquisition.
    """
    ranked = rank_volumes_by_uncertainty(uncertainty_scores)
    selected: list[str] = []

    for vol_id in ranked:
        decision = gate.check(confidence=confidence, cost=cost_per_volume)
        if not decision.approved:
            logger.info("Acquisition stopped at volume %s: %s", vol_id, decision.reason)
            break
        selected.append(vol_id)

    logger.info("Selected %d/%d volumes for acquisition", len(selected), len(ranked))
    return selected


# ---------------------------------------------------------------------------
# Pydantic AI Agent
# ---------------------------------------------------------------------------


@dataclass
class AcquisitionContext:
    """Dependencies injected into the acquisition agent.

    Parameters
    ----------
    uncertainty_scores:
        Per-volume uncertainty scores from conformal prediction.
    budget_total:
        Total acquisition budget (number of volumes).
    budget_spent:
        Budget already consumed.
    acquisition_cost_per_volume:
        Cost per volume in budget units.
    """

    uncertainty_scores: dict[str, float] = field(default_factory=dict)
    budget_total: float = 10.0
    budget_spent: float = 0.0
    acquisition_cost_per_volume: float = 1.0


_SYSTEM_PROMPT = """\
You are a data acquisition analyst for a biomedical image segmentation pipeline.
Your task is to decide which unlabeled volumes should be acquired for annotation
based on model uncertainty scores.

Consider:
- Volumes with high uncertainty are most informative (active learning principle)
- The acquisition budget constrains how many volumes can be acquired
- Diversify across morphology classes to avoid redundant acquisitions
- Thompson sampling balances exploration vs exploitation of known informative regions

Actions:
- Recommend specific volumes for acquisition
- Justify choices based on uncertainty and budget
- Flag if remaining budget is insufficient for meaningful acquisition
"""


def _build_agent(model: str | None = None) -> Any:
    """Build the acquisition Agent (without PrefectAgent wrapper)."""
    from minivess.agents.config import load_agent_config
    from minivess.agents.models import AcquisitionDecision

    config = load_agent_config()
    model_name = model or config.model

    agent: Agent[AcquisitionContext, AcquisitionDecision] = Agent(
        model_name,
        output_type=AcquisitionDecision,
        deps_type=AcquisitionContext,
        name="conformal-bandit-acquisition",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def get_uncertainty_scores(
        ctx: RunContext[AcquisitionContext],
    ) -> dict[str, float]:
        """Get per-volume uncertainty scores from conformal prediction."""
        return ctx.deps.uncertainty_scores

    @agent.tool
    def get_budget_status(ctx: RunContext[AcquisitionContext]) -> dict[str, Any]:
        """Get current acquisition budget status."""
        deps = ctx.deps
        return {
            "budget_total": deps.budget_total,
            "budget_spent": deps.budget_spent,
            "budget_remaining": deps.budget_total - deps.budget_spent,
            "cost_per_volume": deps.acquisition_cost_per_volume,
            "max_acquirable": int(
                (deps.budget_total - deps.budget_spent)
                / deps.acquisition_cost_per_volume
            )
            if deps.acquisition_cost_per_volume > 0
            else 0,
        }

    return agent


def create_acquisition_agent(
    model: str | None = None,
    config: Any | None = None,
) -> Any:
    """Create acquisition agent as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier.
    config:
        Override AgentConfig.

    Returns
    -------
    PrefectAgent wrapping the acquisition agent.
    """
    from minivess.agents.factory import make_prefect_agent

    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
