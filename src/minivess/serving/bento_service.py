"""BentoML services for 3D vessel segmentation inference.

Provides two service classes:

- ``SegmentationService`` — PyTorch-backed (legacy, for direct model loading)
- ``OnnxSegmentationService`` — ONNX Runtime-backed (primary for deployment)
"""

from __future__ import annotations

import logging
from typing import Any

import bentoml
import numpy as np  # noqa: TCH002 - BentoML inspects annotations at runtime
from numpy.typing import NDArray  # noqa: TCH002

from minivess.config.defaults import BENTO_MODEL_TAG

logger = logging.getLogger(__name__)


@bentoml.service(
    name="minivess-segmentation",
    traffic={"timeout": 120},
    resources={"cpu": "2", "memory": "4Gi"},
)
class SegmentationService:
    """BentoML service for 3D vessel segmentation inference (PyTorch)."""

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = bentoml.pytorch.load_model(BENTO_MODEL_TAG)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded on %s", self.device)

    @bentoml.api()
    def predict(self, volume: NDArray[np.float32]) -> dict[str, Any]:
        """Run segmentation on a 3D volume.

        Parameters
        ----------
        volume:
            Input volume, shape (D, H, W) or (C, D, H, W).

        Returns
        -------
        Dict with 'segmentation' (integer labels) and 'probabilities'.
        """
        import torch

        # Ensure (B, C, D, H, W) shape
        if volume.ndim == 3:  # noqa: PLR2004
            tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:  # noqa: PLR2004
            tensor = torch.from_numpy(volume).unsqueeze(0)
        else:
            msg = f"Expected 3D or 4D volume, got shape {volume.shape}"
            raise ValueError(msg)

        tensor = tensor.float().to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        probs = output.prediction.cpu().numpy()[0]  # (C, D, H, W)
        labels = probs.argmax(axis=0)  # (D, H, W)

        return {
            "segmentation": labels.tolist(),
            "probabilities": probs.tolist(),
            "shape": list(labels.shape),
        }

    @bentoml.api()
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "model": BENTO_MODEL_TAG}


class OnnxSegmentationService:
    """ONNX Runtime-backed segmentation service for deployment.

    Primary service for Flow 4 deployment — uses ONNX Runtime for
    hardware-agnostic, lightweight inference without PyTorch dependency.
    """

    def __init__(self, model_tag: str | None = None) -> None:
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        self._model_tag = model_tag or BENTO_MODEL_TAG
        model_ref = bentoml.models.get(self._model_tag)
        model_path = model_ref.path_of("saved_model.onnx")
        self._engine = OnnxSegmentationInference(model_path)
        logger.info("ONNX model loaded: %s", self._model_tag)

    def predict(self, volume: NDArray[np.float32]) -> dict[str, Any]:
        """Run ONNX segmentation on a volume.

        Parameters
        ----------
        volume:
            Input volume, shape (B, C, D, H, W).

        Returns
        -------
        Dict with 'segmentation', 'probabilities', and 'shape'.

        Raises
        ------
        ValueError
            If volume has fewer than 3 dimensions.
        """
        if volume.ndim < 3:  # noqa: PLR2004
            msg = f"Expected 3D+ volume, got shape {volume.shape}"
            raise ValueError(msg)

        # Ensure (B, C, D, H, W) shape
        if volume.ndim == 3:  # noqa: PLR2004
            volume = volume[np.newaxis, np.newaxis, ...]
        elif volume.ndim == 4:  # noqa: PLR2004
            volume = volume[np.newaxis, ...]

        result = self._engine.predict(volume)
        return {
            "segmentation": result.segmentation.tolist(),
            "probabilities": result.probabilities.tolist(),
            "shape": result.shape,
        }

    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "model": self._model_tag}
