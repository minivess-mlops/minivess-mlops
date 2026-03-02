"""Model registry server — lazy-loading multi-model inference router.

Routes inference requests to the appropriate champion model based on
the model_name field in SegmentationRequest. Models are loaded lazily
on first use to minimize startup time and memory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.serving.api_models import SegmentationRequest, SegmentationResponse

logger = logging.getLogger(__name__)


class ModelRegistryServer:
    """Multi-model inference server with lazy loading.

    Parameters
    ----------
    model_paths:
        Mapping from model name to ONNX model file path.
    """

    def __init__(self, model_paths: dict[str, Path]) -> None:
        self._model_paths = model_paths
        self._loaded_models: dict[str, Any] = {}
        logger.info(
            "ModelRegistryServer initialized with %d models: %s",
            len(model_paths),
            list(model_paths.keys()),
        )

    def _load_model(self, name: str) -> Any:
        """Lazy-load an ONNX model by name."""
        if name not in self._model_paths:
            msg = (
                f"Model '{name}' not found. Available: {list(self._model_paths.keys())}"
            )
            raise KeyError(msg)

        if name not in self._loaded_models:
            import onnxruntime as ort

            path = self._model_paths[name]
            session = ort.InferenceSession(str(path))
            self._loaded_models[name] = session
            logger.info("Loaded model '%s' from %s", name, path)

        return self._loaded_models[name]

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Run inference using the requested model.

        Parameters
        ----------
        request:
            Segmentation request with volume and model selection.

        Returns
        -------
        SegmentationResponse

        Raises
        ------
        KeyError
            If the requested model is not registered.
        """
        import time

        import numpy as np

        from minivess.serving.api_models import SegmentationResponse

        model_name = request.model_name

        # Validate model exists before loading
        if model_name not in self._model_paths:
            msg = (
                f"Model '{model_name}' not found. "
                f"Available: {list(self._model_paths.keys())}"
            )
            raise KeyError(msg)

        session = self._load_model(model_name)

        # Ensure (B, C, D, H, W) shape
        volume = request.volume
        if volume.ndim == 3:  # noqa: PLR2004
            volume = volume[np.newaxis, np.newaxis, ...]
        elif volume.ndim == 4:  # noqa: PLR2004
            volume = volume[np.newaxis, ...]

        start = time.monotonic()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: volume})
        elapsed_ms = (time.monotonic() - start) * 1000

        probs = outputs[0]  # (B, C, D, H, W)
        labels = np.argmax(probs[0], axis=0).astype(np.int64)  # (D, H, W)

        return SegmentationResponse(
            segmentation=labels,
            shape=list(labels.shape),
            model_name=model_name,
            inference_time_ms=elapsed_ms,
            probabilities=(
                probs[0].astype(np.float32) if request.output_mode != "binary" else None
            ),
        )

    def health(self) -> dict[str, Any]:
        """Return health status with available and loaded models."""
        return {
            "status": "healthy",
            "available_models": list(self._model_paths.keys()),
            "loaded_models": list(self._loaded_models.keys()),
        }
