"""MedSAM3 interactive annotation adapter (Liu et al., 2025).

Medical concept-aware prompting for interactive 3D segmentation
annotation. Supports point, box, mask, and concept-based prompts
for efficient annotation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PromptType(StrEnum):
    """Types of interactive segmentation prompts."""

    POINT = "point"
    BOX = "box"
    MASK = "mask"
    CONCEPT = "concept"


class MedicalConcept(StrEnum):
    """Medical anatomical concepts for concept-aware prompting."""

    VESSEL = "vessel"
    TUMOR = "tumor"
    ORGAN = "organ"
    TISSUE = "tissue"
    LESION = "lesion"


@dataclass
class AnnotationPrompt:
    """Interactive annotation prompt.

    Parameters
    ----------
    prompt_type:
        Type of prompt (point, box, mask, concept).
    coordinates:
        Spatial coordinates. For point: (x, y, z).
        For box: (x1, y1, z1, x2, y2, z2).
    label:
        Foreground (1) or background (0) label.
    concept:
        Medical concept for concept-aware prompting.
    """

    prompt_type: str
    coordinates: tuple[int, ...] | None = None
    label: int = 1
    concept: str | None = None


@dataclass
class MedSAM3Config:
    """Configuration for MedSAM3 interactive predictor.

    Parameters
    ----------
    model_name:
        Model identifier.
    concept_vocabulary:
        List of supported medical concepts.
    spatial_dims:
        Number of spatial dimensions.
    """

    model_name: str = "medsam3_base"
    concept_vocabulary: list[str] = field(
        default_factory=lambda: [c.value for c in MedicalConcept],
    )
    spatial_dims: int = 3


class MedSAM3Predictor:
    """Interactive annotation predictor with concept-aware prompting.

    Accumulates user prompts (points, boxes, concepts) and produces
    segmentation predictions. Designed for interactive annotation
    workflows with Label Studio or standalone use.

    Parameters
    ----------
    config:
        MedSAM3 configuration.
    """

    def __init__(self, config: MedSAM3Config) -> None:
        self.config = config
        self.prompts: list[AnnotationPrompt] = []

    def add_prompt(self, prompt: AnnotationPrompt) -> None:
        """Register an interactive prompt.

        Parameters
        ----------
        prompt:
            Annotation prompt to add.
        """
        self.prompts.append(prompt)

    def reset(self) -> None:
        """Clear all accumulated prompts."""
        self.prompts.clear()

    def to_annotation_record(self) -> dict[str, Any]:
        """Export current state as annotation metadata.

        Returns
        -------
        Dictionary with model info and prompt history.
        """
        return {
            "model": self.config.model_name,
            "spatial_dims": self.config.spatial_dims,
            "prompts": [
                {
                    "type": p.prompt_type,
                    "coordinates": p.coordinates,
                    "label": p.label,
                    "concept": p.concept,
                }
                for p in self.prompts
            ],
            "num_prompts": len(self.prompts),
        }
