"""BentoML service for 3D vessel segmentation inference."""

from __future__ import annotations

import logging
from typing import Any

import bentoml
import numpy as np  # noqa: TCH002 - BentoML inspects annotations at runtime
from numpy.typing import NDArray  # noqa: TCH002

logger = logging.getLogger(__name__)

BENTO_MODEL_TAG = "minivess-segmentor"


@bentoml.service(
    name="minivess-segmentation",
    traffic={"timeout": 120},
    resources={"cpu": "2", "memory": "4Gi"},
)
class SegmentationService:
    """BentoML service for 3D vessel segmentation inference."""

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

        Args:
            volume: Input volume, shape (D, H, W) or (C, D, H, W).

        Returns:
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
