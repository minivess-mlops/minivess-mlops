"""ONNX Runtime inference engine for segmentation models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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

    def predict(self, volume: NDArray[np.float32]) -> dict[str, Any]:
        """Run inference on a 3D volume.

        Args:
            volume: Input, shape (B, C, D, H, W) float32.

        Returns:
            Dict with segmentation labels and probabilities.
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

        return {
            "segmentation": labels,
            "probabilities": probs,
            "shape": list(labels.shape),
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata including input/output specs."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        return {
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type} for i in inputs
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type} for o in outputs
            ],
        }
