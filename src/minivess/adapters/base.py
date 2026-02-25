from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SegmentationOutput:
    """Standardized output from any segmentation model adapter."""

    prediction: Tensor  # (B, C, D, H, W) class probabilities
    logits: Tensor  # (B, C, D, H, W) raw logits before softmax
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfigInfo:
    """Typed configuration snapshot returned by ModelAdapter.get_config().

    Common fields are typed attributes; adapter-specific fields go in ``extras``.
    """

    family: str
    name: str
    in_channels: int | None = None
    out_channels: int | None = None
    trainable_params: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a serializable dict (common fields + extras merged)."""
        d: dict[str, Any] = {
            "family": self.family,
            "name": self.name,
        }
        if self.in_channels is not None:
            d["in_channels"] = self.in_channels
        if self.out_channels is not None:
            d["out_channels"] = self.out_channels
        if self.trainable_params is not None:
            d["trainable_params"] = self.trainable_params
        d.update(self.extras)
        return d


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
    def get_config(self) -> AdapterConfigInfo:
        """Return model configuration as a typed dataclass."""
        ...

    def _build_output(self, logits: Tensor, architecture: str) -> SegmentationOutput:
        """Build a standardized SegmentationOutput from raw logits.

        Applies softmax along dim=1 and wraps the result with metadata.
        Subclasses with standard forward logic can call this instead of
        manually constructing SegmentationOutput.

        Args:
            logits: Raw model output tensor (B, C, D, H, W).
            architecture: Architecture name for metadata.

        Returns:
            SegmentationOutput with softmax predictions and raw logits.
        """
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": architecture},
        )

    def _build_config(self, **extras: Any) -> AdapterConfigInfo:
        """Build an AdapterConfigInfo from self.config and extras.

        Auto-populates family, name, in_channels, out_channels, and
        trainable_params from ``self.config`` and ``self.trainable_parameters()``.
        Any keyword arguments are placed in the ``extras`` dict.

        Args:
            **extras: Adapter-specific configuration fields.

        Returns:
            AdapterConfigInfo with common fields auto-populated.
        """
        return AdapterConfigInfo(
            family=self.config.family.value,
            name=self.config.name,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            trainable_params=self.trainable_parameters(),
            extras=dict(extras),
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Default implementation loads into ``self.net``. Override for adapters
        that manage weights differently (e.g., LoRA, CommaAdapter).
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file.

        Default implementation saves ``self.net`` state dict.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def trainable_parameters(self) -> int:
        """Return the count of trainable parameters.

        Default implementation counts ``self.net`` parameters with
        ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export the model to ONNX format.

        Uses the dynamo-based exporter (PyTorch 2.5+) with fallback
        to the legacy TorchScript exporter for compatibility.
        Override for adapters that need special export logic.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                onnx_program = torch.onnx.export(
                    self.net,
                    example_input,
                    dynamo=True,
                )
                onnx_program.save(str(path))
            except Exception:
                torch.onnx.export(
                    self.net,
                    example_input,
                    str(path),
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
