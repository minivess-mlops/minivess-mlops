"""API models for the enhanced deploy server.

Defines request/response dataclasses and enums for multi-model
segmentation inference with configurable output modes and UQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np  # noqa: TC002 — used at runtime in dataclass annotations
from numpy.typing import (
    NDArray,  # noqa: TC002 — used at runtime in dataclass annotations
)


class OutputMode(StrEnum):
    """Output modes for segmentation inference."""

    BINARY = "binary"
    PROBABILITIES = "probabilities"
    FULL = "full"
    UQ = "uq"


class ModelName(StrEnum):
    """Available champion model categories."""

    BALANCED = "balanced"
    TOPOLOGY = "topology"
    OVERLAP = "overlap"


class UQMethod(StrEnum):
    """Uncertainty quantification methods."""

    MORPHOLOGICAL = "morphological"
    DISTANCE_TRANSFORM = "distance_transform"
    RISK_CONTROL = "risk_control"


@dataclass
class SegmentationRequest:
    """Request for segmentation inference.

    Parameters
    ----------
    volume:
        Input volume, shape (D, H, W) or (C, D, H, W).
    model_name:
        Champion category to use.
    output_mode:
        What to include in the response.
    ensemble_strategy:
        Optional ensemble strategy name.
    uq_methods:
        Optional UQ methods to apply.
    confidence_level:
        Confidence level for UQ prediction sets.
    """

    volume: NDArray[np.float32]
    model_name: str = "balanced"
    output_mode: str = "binary"
    ensemble_strategy: str | None = None
    uq_methods: list[str] | None = None
    confidence_level: float = 0.95

    def validate(self) -> list[str]:
        """Validate the request fields.

        Returns
        -------
        List of error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate output_mode
        valid_modes = {m.value for m in OutputMode}
        if self.output_mode not in valid_modes:
            errors.append(
                f"Invalid output_mode '{self.output_mode}'. "
                f"Must be one of: {sorted(valid_modes)}"
            )

        # Validate volume dimensions
        if self.volume.ndim < 3:  # noqa: PLR2004
            errors.append(
                f"Volume must be at least 3D (D, H, W), got {self.volume.ndim} dimensions"
            )

        # Validate confidence_level
        if not 0.0 < self.confidence_level <= 1.0:
            errors.append(
                f"confidence_level must be in (0, 1], got {self.confidence_level}"
            )

        return errors


@dataclass
class SegmentationResponse:
    """Response from segmentation inference.

    Parameters
    ----------
    segmentation:
        Binary segmentation mask.
    shape:
        Shape of the segmentation.
    model_name:
        Which model produced the prediction.
    inference_time_ms:
        Inference time in milliseconds.
    probabilities:
        Optional probability map.
    uncertainty:
        Optional UQ results.
    ensemble_member_count:
        Number of ensemble members used.
    """

    segmentation: NDArray[np.int64]
    shape: list[int]
    model_name: str
    inference_time_ms: float
    probabilities: NDArray[np.float32] | None = None
    uncertainty: list[dict[str, Any]] | None = None
    ensemble_member_count: int = 0
    sdc_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Converts numpy arrays to nested lists.
        """
        result: dict[str, Any] = {
            "segmentation": self.segmentation.tolist(),
            "shape": self.shape,
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
            "probabilities": (
                self.probabilities.tolist() if self.probabilities is not None else None
            ),
            "uncertainty": self.uncertainty,
            "ensemble_member_count": self.ensemble_member_count,
            "sdc_confidence": self.sdc_confidence,
        }
        return result
