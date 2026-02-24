"""Interactive annotation session management.

Tracks user interactions during interactive annotation workflows,
records prompt-result pairs, and computes agreement metrics against
reference annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class InteractionRecord:
    """Record of a single annotation interaction.

    Parameters
    ----------
    prompt_description:
        Human-readable description of the prompt.
    predicted_mask:
        Binary segmentation mask produced.
    timestamp:
        When the interaction occurred.
    """

    prompt_description: str
    predicted_mask: NDArray
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )


class AnnotationSession:
    """Manages an interactive annotation session for a volume.

    Parameters
    ----------
    volume_id:
        Identifier for the volume being annotated.
    """

    def __init__(self, volume_id: str) -> None:
        self.volume_id = volume_id
        self.interactions: list[InteractionRecord] = []
        self._started = datetime.now(UTC)

    def add_interaction(
        self,
        prompt_description: str,
        predicted_mask: NDArray,
    ) -> None:
        """Record a prompt-result interaction.

        Parameters
        ----------
        prompt_description:
            Description of the prompt used.
        predicted_mask:
            Resulting segmentation mask.
        """
        self.interactions.append(
            InteractionRecord(
                prompt_description=prompt_description,
                predicted_mask=predicted_mask,
            ),
        )

    def compute_agreement(self, reference: NDArray) -> float:
        """Compute Dice agreement between latest prediction and reference.

        Parameters
        ----------
        reference:
            Ground-truth binary mask.

        Returns
        -------
        Dice similarity coefficient.
        """
        if not self.interactions:
            return 0.0

        pred = self.interactions[-1].predicted_mask.astype(bool)
        ref = reference.astype(bool)

        intersection = np.sum(pred & ref)
        total = np.sum(pred) + np.sum(ref)

        if total == 0:
            return 1.0
        return float(2.0 * intersection / total)

    def to_markdown(self) -> str:
        """Generate a session report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Interactive Annotation Session",
            "",
            f"**Generated:** {now}",
            f"**Volume:** {self.volume_id}",
            f"**Interactions:** {len(self.interactions)}",
            "",
        ]

        if not self.interactions:
            sections.append("No interactions recorded.")
            sections.append("")
            return "\n".join(sections)

        sections.extend([
            "## Interaction Log",
            "",
            "| # | Prompt | Mask Voxels | Timestamp |",
            "|---|--------|-------------|-----------|",
        ])

        for i, rec in enumerate(self.interactions, 1):
            voxels = int(rec.predicted_mask.sum())
            sections.append(
                f"| {i} | {rec.prompt_description} "
                f"| {voxels:,} | {rec.timestamp} |"
            )

        sections.append("")
        return "\n".join(sections)
