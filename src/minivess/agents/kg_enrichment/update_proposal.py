"""KG update proposals with human review gate.

Proposals are NEVER auto-applied. They are saved as YAML files in a
proposals directory for manual review and approval.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from minivess.agents.kg_enrichment.contradiction_detector import (
    Contradiction,  # noqa: TC001
)

logger = logging.getLogger(__name__)


class UpdateProposal(BaseModel):
    """A proposed update to a KG decision node.

    All proposals start with status='pending_review'. They are NEVER
    automatically applied to the knowledge graph.
    """

    node_id: str = Field(description="Target KG decision node ID")
    proposed_changes: dict[str, Any] = Field(
        description="Dict of proposed changes to the node"
    )
    rationale: str = Field(description="Reason for the proposed change")
    source_pmid: str = Field(default="", description="PubMed ID of the source article")
    status: str = Field(
        default="pending_review",
        description="Proposal status (always 'pending_review' — never auto-applied)",
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Detected contradictions with existing KG constraints",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 timestamp of proposal creation",
    )


def save_proposal(
    proposal: UpdateProposal,
    *,
    output_dir: Path,
) -> Path:
    """Save a KG update proposal as a YAML file for human review.

    Parameters
    ----------
    proposal:
        The UpdateProposal to persist.
    output_dir:
        Directory to write the proposal YAML file.

    Returns
    -------
    Path to the saved YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from node_id and timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"proposal_{proposal.node_id}_{timestamp}.yaml"
    output_path = output_dir / filename

    # Serialize to dict, converting Contradiction models to dicts
    data: dict[str, Any] = {
        "node_id": proposal.node_id,
        "proposed_changes": proposal.proposed_changes,
        "rationale": proposal.rationale,
        "source_pmid": proposal.source_pmid,
        "status": proposal.status,
        "created_at": proposal.created_at,
    }

    if proposal.contradictions:
        data["contradictions"] = [
            {
                "description": c.description,
                "severity": c.severity,
                "source_node": c.source_node,
                "conflicting_constraint": c.conflicting_constraint,
            }
            for c in proposal.contradictions
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    logger.info("Saved KG update proposal to %s", output_path)
    return output_path
