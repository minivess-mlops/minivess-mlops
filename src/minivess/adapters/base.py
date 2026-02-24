from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor, nn


@dataclass
class SegmentationOutput:
    """Standardized output from any segmentation model adapter."""

    prediction: Tensor  # (B, C, D, H, W) class probabilities
    logits: Tensor  # (B, C, D, H, W) raw logits before softmax
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC, nn.Module):
    """Abstract base class for pluggable segmentation model adapters.

    All segmentation models in the pipeline must implement this interface.
    This enables model-agnostic training, evaluation, and serving.
    """

    @abstractmethod
    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run inference on a batch of 3D volumes.

        Args:
            images: Input tensor of shape (B, C, D, H, W).
            **kwargs: Model-specific parameters (e.g., prompts for SAMv3).

        Returns:
            SegmentationOutput with predictions and raw logits.
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return model configuration as a serializable dict."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file."""
        ...

    @abstractmethod
    def trainable_parameters(self) -> int:
        """Return the count of trainable parameters."""
        ...

    @abstractmethod
    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export the model to ONNX format."""
        ...
