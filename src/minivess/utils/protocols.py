"""Structural Protocol types for cross-package decoupling.

These Protocols enable type-safe dependency inversion: modules can depend
on a Protocol rather than a concrete ABC, reducing import coupling.

Usage::

    from minivess.utils.protocols import Predictor


    def evaluate(model: Predictor, data: Tensor) -> float:
        output = model(data)
        return output.prediction.mean().item()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from minivess.adapters.base import SegmentationOutput


@runtime_checkable
class Predictor(Protocol):
    """Protocol for any model that can produce segmentation predictions."""

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run inference on a batch of 3D volumes."""
        ...

    def __call__(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Call forward (nn.Module convention)."""
        ...


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for models that can save/load weights."""

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights."""
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights."""
        ...


@runtime_checkable
class MetricComputer(Protocol):
    """Protocol for metric collection (update → compute → reset cycle)."""

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """Update metrics with a batch."""
        ...

    def compute(self) -> Any:
        """Compute aggregated metrics."""
        ...

    def reset(self) -> None:
        """Reset for next epoch."""
        ...
