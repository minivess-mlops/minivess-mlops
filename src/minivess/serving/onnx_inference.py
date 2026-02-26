"""ONNX Runtime inference engine for segmentation models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class OnnxTensorSpec:
    """Specification for an ONNX model input or output tensor."""

    name: str
    shape: list[int | str | None]
    type: str


@dataclass
class OnnxPrediction:
    """Typed result from ONNX inference."""

    segmentation: NDArray[np.int64]
    probabilities: NDArray[np.float32]
    shape: list[int]


@dataclass
class OnnxModelMetadata:
    """Typed metadata for an ONNX model."""

    inputs: list[OnnxTensorSpec] = field(default_factory=list)
    outputs: list[OnnxTensorSpec] = field(default_factory=list)


class OnnxSegmentationInference:
    """ONNX Runtime inference engine for segmentation models."""

    def __init__(self, model_path: Path, *, use_gpu: bool = False) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            "ONNX model loaded from %s (providers: %s)",
            model_path,
            providers,
        )

    def predict(self, volume: NDArray[np.float32]) -> OnnxPrediction:
        """Run inference on a 3D volume.

        Args:
            volume: Input, shape (B, C, D, H, W) float32.

        Returns:
            OnnxPrediction with segmentation labels and probabilities.
        """
        logits = self.session.run(
            [self.output_name],
            {self.input_name: volume},
        )[0]

        # Numerically stable softmax along class axis
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        labels = probs.argmax(axis=1)  # (B, D, H, W)

        return OnnxPrediction(
            segmentation=labels,
            probabilities=probs,
            shape=list(labels.shape),
        )

    def get_metadata(self) -> OnnxModelMetadata:
        """Return model metadata including input/output specs."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        return OnnxModelMetadata(
            inputs=[
                OnnxTensorSpec(name=i.name, shape=i.shape, type=i.type) for i in inputs
            ],
            outputs=[
                OnnxTensorSpec(name=o.name, shape=o.shape, type=o.type) for o in outputs
            ],
        )
