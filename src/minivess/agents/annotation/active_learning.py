"""Active learning annotation agent.

Ranks unlabeled volumes by model disagreement to prioritize annotation
effort on the most informative samples. Uses ensemble prediction variance
as the disagreement metric.

Based on: MONAI Label active learning loop and
Guo et al. (2025), "K-Prism" proofreading paradigm.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ensemble disagreement computation
# ---------------------------------------------------------------------------


def compute_ensemble_disagreement(
    predictions: list[NDArray[np.float32]],
) -> float:
    """Compute ensemble disagreement as mean voxel-wise variance.

    Parameters
    ----------
    predictions:
        List of prediction probability maps from ensemble members.
        Each array has shape (D, H, W) with values in [0, 1].

    Returns
    -------
    Mean voxel-wise variance across ensemble members.
    """
    stacked = np.stack(predictions, axis=0)  # (M, D, H, W)
    voxel_variance = np.var(stacked, axis=0)  # (D, H, W)
    return float(np.mean(voxel_variance))


def rank_volumes_by_disagreement(
    ensemble_predictions: dict[str, list[NDArray[np.float32]]],
) -> list[tuple[str, float]]:
    """Rank volumes by ensemble disagreement (highest first).

    Parameters
    ----------
    ensemble_predictions:
        Mapping of volume_id → list of prediction maps from ensemble members.

    Returns
    -------
    List of (volume_id, disagreement_score) tuples sorted by descending score.
    """
    if not ensemble_predictions:
        return []

    scores: list[tuple[str, float]] = []
    for vol_id, preds in ensemble_predictions.items():
        score = compute_ensemble_disagreement(preds)
        scores.append((vol_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Pydantic AI Agent
# ---------------------------------------------------------------------------


@dataclass
class AnnotationContext:
    """Dependencies injected into the annotation agent.

    Parameters
    ----------
    disagreement_ranking:
        Pre-computed (volume_id, score) tuples sorted by disagreement.
    total_unlabeled:
        Total number of unlabeled volumes.
    annotation_budget:
        Maximum number of volumes to annotate in this round.
    current_dataset_size:
        Number of labeled volumes in the training set.
    """

    disagreement_ranking: list[tuple[str, float]] = field(default_factory=list)
    total_unlabeled: int = 0
    annotation_budget: int = 5
    current_dataset_size: int = 0


_SYSTEM_PROMPT = """\
You are an active learning annotation advisor for a biomedical image segmentation
pipeline. Your task is to recommend which unlabeled volumes should be annotated
next based on model disagreement analysis.

Consider:
- Volumes with highest ensemble disagreement are most informative
- Balance between informativeness and diversity of morphology classes
- The annotation budget constrains how many volumes can be sent for labeling
- Consider the current dataset size — early rounds benefit from diverse samples,
  later rounds benefit from targeted hard examples

Actions:
- Recommend the top-K volumes for annotation based on disagreement ranking
- Explain why each volume is informative
- Estimate the expected improvement in model performance
"""


def _build_agent(model: str | None = None) -> Any:
    """Build the annotation Agent (without PrefectAgent wrapper)."""
    from minivess.agents.config import load_agent_config
    from minivess.agents.models import AnnotationPriority

    config = load_agent_config()
    model_name = model or config.model

    agent: Agent[AnnotationContext, AnnotationPriority] = Agent(
        model_name,
        output_type=AnnotationPriority,
        deps_type=AnnotationContext,
        name="active-learning-annotation",
        system_prompt=_SYSTEM_PROMPT,
    )

    @agent.tool
    def get_disagreement_ranking(
        ctx: RunContext[AnnotationContext],
    ) -> list[dict[str, Any]]:
        """Get volumes ranked by ensemble disagreement."""
        return [
            {"volume_id": vol_id, "disagreement_score": score}
            for vol_id, score in ctx.deps.disagreement_ranking
        ]

    @agent.tool
    def get_annotation_budget(
        ctx: RunContext[AnnotationContext],
    ) -> dict[str, Any]:
        """Get annotation budget and dataset statistics."""
        deps = ctx.deps
        return {
            "annotation_budget": deps.annotation_budget,
            "total_unlabeled": deps.total_unlabeled,
            "current_dataset_size": deps.current_dataset_size,
        }

    return agent


def create_annotation_agent(
    model: str | None = None,
    config: Any | None = None,
) -> Any:
    """Create annotation agent as PrefectAgent for durable execution.

    Parameters
    ----------
    model:
        Override model identifier.
    config:
        Override AgentConfig.

    Returns
    -------
    PrefectAgent wrapping the annotation agent.
    """
    from minivess.agents.factory import make_prefect_agent

    agent = _build_agent(model=model)
    return make_prefect_agent(agent, config=config)
