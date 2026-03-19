"""PCCP constraint enforcement layer for agent actions.

Provides a confidence gate that all agent actions must pass before execution.
PCCP = Posterior Conformal Prediction Constraint — ensures agent decisions
are backed by sufficient statistical confidence from conformal prediction.

Based on: Kwon & Kim (2026), "Conformal selective prediction with
cost-aware deferral for safe clinical triage under distribution shift."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PCCPDecision(BaseModel):
    """Result of a PCCP gate check.

    Parameters
    ----------
    approved:
        Whether the action passed all PCCP checks.
    confidence:
        The confidence level of the proposed action.
    threshold:
        The minimum confidence threshold required.
    cost:
        The cost of the proposed action.
    budget_remaining:
        Remaining budget after this decision.
    reason:
        Explanation of the decision.
    """

    approved: bool = Field(description="Whether the action passed all PCCP checks")
    confidence: float = Field(description="Confidence level of the proposed action")
    threshold: float = Field(description="Minimum confidence threshold required")
    cost: float = Field(description="Cost of the proposed action")
    budget_remaining: float = Field(description="Remaining budget after decision")
    reason: str = Field(description="Explanation of the decision")


@dataclass
class PCCPConfig:
    """Configuration for PCCP constraint enforcement.

    Parameters
    ----------
    alpha:
        Significance level for conformal prediction (e.g., 0.1 for 90% coverage).
    min_confidence:
        Minimum confidence level required for agent actions.
    budget_total:
        Total budget units available for agent actions.
    """

    alpha: float = 0.1
    min_confidence: float = 0.8
    budget_total: float = 100.0


@dataclass
class PCCPGate:
    """PCCP constraint enforcement gate for agent actions.

    Validates that agent actions meet confidence and budget requirements
    before allowing execution. Tracks cumulative budget spending.

    Parameters
    ----------
    config:
        PCCP configuration.
    """

    config: PCCPConfig = field(default_factory=PCCPConfig)
    budget_spent: float = 0.0

    def check(self, confidence: float, cost: float = 1.0) -> PCCPDecision:
        """Check if an action passes the PCCP gate.

        Parameters
        ----------
        confidence:
            Confidence level of the proposed action (0-1).
        cost:
            Cost units for this action.

        Returns
        -------
        PCCPDecision with approval status and reasoning.
        """
        budget_remaining = self.config.budget_total - self.budget_spent

        # Check 1: confidence threshold
        if confidence < self.config.min_confidence:
            logger.info(
                "PCCP gate REJECTED: confidence %.3f < threshold %.3f",
                confidence,
                self.config.min_confidence,
            )
            return PCCPDecision(
                approved=False,
                confidence=confidence,
                threshold=self.config.min_confidence,
                cost=cost,
                budget_remaining=budget_remaining,
                reason=(
                    f"Confidence {confidence:.3f} below minimum threshold "
                    f"{self.config.min_confidence:.3f}"
                ),
            )

        # Check 2: budget constraint
        if cost > 0 and self.budget_spent + cost > self.config.budget_total:
            logger.info(
                "PCCP gate REJECTED: cost %.1f exceeds remaining budget %.1f",
                cost,
                budget_remaining,
            )
            return PCCPDecision(
                approved=False,
                confidence=confidence,
                threshold=self.config.min_confidence,
                cost=cost,
                budget_remaining=budget_remaining,
                reason=(
                    f"Budget exceeded: cost {cost:.1f} > remaining "
                    f"{budget_remaining:.1f}"
                ),
            )

        # All checks pass — spend budget
        self.budget_spent += cost
        budget_remaining = self.config.budget_total - self.budget_spent

        logger.info(
            "PCCP gate APPROVED: confidence %.3f, cost %.1f, budget remaining %.1f",
            confidence,
            cost,
            budget_remaining,
        )
        return PCCPDecision(
            approved=True,
            confidence=confidence,
            threshold=self.config.min_confidence,
            cost=cost,
            budget_remaining=budget_remaining,
            reason="Passed all PCCP checks",
        )

    def reset_budget(self) -> None:
        """Reset the budget tracker to zero."""
        self.budget_spent = 0.0
        logger.info("PCCP budget reset")
