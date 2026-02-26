"""Sliding window inference for full-volume 3D segmentation.

Wraps MONAI's sliding_window_inference to produce per-volume
argmax predictions from trained segmentation models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference  # type: ignore[attr-defined]

from minivess.adapters.base import SegmentationOutput

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SlidingWindowInferenceRunner:
    """Run sliding window inference on full 3D volumes.

    Parameters
    ----------
    roi_size:
        Patch size (D, H, W) for the sliding window.
    num_classes:
        Number of segmentation classes (including background).
    overlap:
        Fraction of overlap between adjacent patches (0.0–1.0).
    sw_batch_size:
        Batch size for the sliding window (number of patches
        processed simultaneously).
    mode:
        Aggregation mode for overlapping regions.
    """

    def __init__(
        self,
        roi_size: tuple[int, int, int],
        num_classes: int = 2,
        overlap: float = 0.25,
        sw_batch_size: int = 4,
        mode: str = "gaussian",
    ) -> None:
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.mode = mode

    def predict_volume(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
    ) -> NDArray[np.integer]:
        """Run inference on a single volume and return argmax prediction.

        Parameters
        ----------
        model:
            Trained segmentation model. Can return either a raw Tensor
            or a SegmentationOutput (ModelAdapter).
        image:
            Input tensor of shape ``(1, C, D, H, W)``.
        device:
            Device to run inference on.

        Returns
        -------
        NDArray
            Integer class predictions of shape ``(D, H, W)``.
        """
        device = torch.device(device)
        model.eval()
        model.to(device)
        image = image.to(device)

        def _forward(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                output = model(x)
            if isinstance(output, SegmentationOutput):
                return output.logits  # type: ignore[no-any-return]
            return output  # type: ignore[no-any-return]

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=_forward,
                overlap=self.overlap,
                mode=self.mode,
            )

        # argmax → (1, D, H, W) → (D, H, W) numpy
        assert isinstance(logits, torch.Tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        return pred

    def infer_dataset(
        self,
        model: torch.nn.Module,
        loader: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[list[NDArray[np.integer]], list[NDArray[np.integer]]]:
        """Run inference on all volumes in a DataLoader.

        Parameters
        ----------
        model:
            Trained segmentation model.
        loader:
            Validation DataLoader yielding batches with ``"image"``
            and ``"label"`` keys. Batch size should be 1 for
            full-volume inference.
        device:
            Device to run inference on.

        Returns
        -------
        tuple
            (predictions, labels) — parallel lists of numpy arrays,
            each of shape ``(D, H, W)``.
        """
        predictions: list[NDArray[np.integer]] = []
        labels: list[NDArray[np.integer]] = []

        for batch in loader:
            image = batch["image"]
            label = batch["label"]

            # Ensure batch dim
            if image.ndim == 4:
                image = image.unsqueeze(0)

            pred = self.predict_volume(model, image, device=device)
            predictions.append(pred)

            # Extract label as numpy: squeeze batch + channel dims
            label_np = label.squeeze().cpu().numpy().astype(np.int64)
            labels.append(label_np)

            logger.debug(
                "Inferred volume: pred shape=%s, label shape=%s",
                pred.shape,
                label_np.shape,
            )

        return predictions, labels
