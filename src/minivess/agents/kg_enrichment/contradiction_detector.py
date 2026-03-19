"""Contradiction detection for KG update proposals.

Detects conflicts between proposed KG updates and existing constraints.
Uses structured comparison (NOT regex — Rule #16).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Contradiction(BaseModel):
    """A detected contradiction between a proposal and an existing KG constraint."""

    description: str = Field(
        description="Human-readable description of the contradiction"
    )
    severity: Literal["hard", "soft", "warning"] = Field(
        description=(
            "hard: proposal violates a non-negotiable constraint; "
            "soft: proposal conflicts with a preference; "
            "warning: potential issue worth reviewing"
        )
    )
    source_node: str = Field(description="KG node ID where the constraint originates")
    conflicting_constraint: str = Field(
        description="The specific constraint text that is violated"
    )


def detect_contradictions(
    proposal: dict[str, Any],
    kg_constraints: list[dict[str, Any]],
) -> list[Contradiction]:
    """Detect contradictions between a proposal and existing KG constraints.

    Parameters
    ----------
    proposal:
        Dict with keys: node_id, proposed_value, context.
    kg_constraints:
        List of constraint dicts, each with: node_id, constraint, applies_to,
        banned_values.

    Returns
    -------
    List of Contradiction objects. Empty if no conflicts found.
    """
    contradictions: list[Contradiction] = []

    proposed_node = proposal.get("node_id", "")
    proposed_value = proposal.get("proposed_value", "")
    context = proposal.get("context", "")

    for constraint in kg_constraints:
        constraint_node = constraint.get("node_id", "")
        banned_values = constraint.get("banned_values", [])
        applies_to = constraint.get("applies_to", [])
        constraint_text = constraint.get("constraint", "")

        # Check if the proposal targets the same node
        if constraint_node != proposed_node:
            continue

        # Check if the proposed value is in the banned list
        if proposed_value in banned_values:
            # Check if the context matches any applies_to scope
            context_matches = not applies_to  # If no scope, applies globally
            for scope in applies_to:
                if scope.lower() in context.lower():
                    context_matches = True
                    break

            if context_matches:
                contradictions.append(
                    Contradiction(
                        description=(
                            f"Proposed value '{proposed_value}' for node "
                            f"'{proposed_node}' is banned: {constraint_text}"
                        ),
                        severity="hard",
                        source_node=constraint_node,
                        conflicting_constraint=constraint_text,
                    )
                )

    logger.info(
        "Contradiction check: %d conflicts found for proposal on node '%s'",
        len(contradictions),
        proposed_node,
    )
    return contradictions
