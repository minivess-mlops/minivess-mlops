"""Registry promotion integration for the deploy flow.

Promotes champion models through DEVELOPMENT -> STAGING -> PRODUCTION
stages and creates audit trails for IEC 62304 compliance.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.observability.model_registry import (
        ModelRegistry,
        PromotionCriteria,
        PromotionResult,
    )
    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


def promote_champion_for_deploy(
    champion: ChampionModel,
    registry: ModelRegistry,
    criteria: PromotionCriteria,
) -> PromotionResult:
    """Promote a champion model through registry stages for deployment.

    Attempts to promote from DEVELOPMENT -> STAGING -> PRODUCTION.

    Parameters
    ----------
    champion:
        Champion model to promote.
    registry:
        Model registry instance.
    criteria:
        Promotion criteria (min/max thresholds).

    Returns
    -------
    :class:`PromotionResult` indicating whether promotion was approved.
    """
    from minivess.observability.model_registry import ModelStage

    model_name = f"minivess-{champion.category}"
    version = "1.0.0"

    # Get or create the model version
    try:
        mv = registry.get_version(model_name, version)
    except KeyError:
        mv = registry.register_version(model_name, version, champion.metrics)

    # Promote DEVELOPMENT -> STAGING -> PRODUCTION
    if mv.stage == ModelStage.DEVELOPMENT:
        result = registry.promote(model_name, version, ModelStage.STAGING, criteria)
        if not result.approved:
            logger.warning(
                "Promotion to STAGING rejected for %s: %s",
                model_name,
                result.reason,
            )
            return result

    mv = registry.get_version(model_name, version)
    if mv.stage == ModelStage.STAGING:
        result = registry.promote(model_name, version, ModelStage.PRODUCTION, criteria)
        if not result.approved:
            logger.warning(
                "Promotion to PRODUCTION rejected for %s: %s",
                model_name,
                result.reason,
            )
        return result

    # Already in PRODUCTION or beyond
    from minivess.observability.model_registry import PromotionResult as PR

    return PR(
        approved=True,
        reason=f"Model already in {mv.stage.value}",
        metrics=champion.metrics,
    )


def create_deployment_audit_trail(
    *,
    champion: ChampionModel,
    onnx_path: Path,
    bento_tag: str,
    promotion_approved: bool,
) -> dict[str, Any]:
    """Create an audit trail record for a deployment action.

    Parameters
    ----------
    champion:
        Champion model being deployed.
    onnx_path:
        Path to the exported ONNX model.
    bento_tag:
        BentoML model tag.
    promotion_approved:
        Whether the promotion was approved.

    Returns
    -------
    Audit trail dict with timestamps and deployment details.
    """
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": champion.run_id,
        "experiment_id": champion.experiment_id,
        "category": champion.category,
        "metrics": champion.metrics,
        "onnx_path": str(onnx_path),
        "bento_tag": bento_tag,
        "promotion_approved": promotion_approved,
    }
