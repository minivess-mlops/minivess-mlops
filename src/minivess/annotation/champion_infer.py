"""Champion pre-segmentation bridge for MONAI Label.

Wraps the BentoML champion model endpoint so MONAI Label can use it
as an initial pre-segmentation source. When an annotator opens a volume
in 3D Slicer, MONAI Label calls this infer class to get a starting mask.

If BentoML is unavailable, falls back to a zero mask (graceful degradation
for annotation — the annotator can still segment from scratch).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 — used in function signatures

logger = logging.getLogger(__name__)


class ChampionInfer:
    """Pre-segmentation via BentoML champion model.

    Parameters
    ----------
    bentoml_url:
        Base URL of the BentoML serving endpoint (e.g. ``http://localhost:3333``).
    timeout_s:
        HTTP request timeout in seconds.
    """

    def __init__(self, bentoml_url: str, *, timeout_s: float = 120.0) -> None:
        self._bentoml_url = bentoml_url.rstrip("/")
        self._timeout_s = timeout_s

    def run_inference(self, volume: NDArray[np.float32]) -> dict[str, Any]:
        """Run champion inference on a volume.

        Returns
        -------
        dict with key ``"pred"`` containing the binary mask as NDArray.
        If BentoML is unreachable, returns a zero mask matching the input shape.
        """
        try:
            mask = self._call_bentoml(volume)
        except Exception:
            logger.warning(
                "BentoML unreachable at %s — returning zero mask for annotation",
                self._bentoml_url,
                exc_info=True,
            )
            mask = np.zeros(volume.shape, dtype=np.int64)

        # Ensure binary
        mask = (mask > 0).astype(np.int64)
        return {"pred": mask}

    def _call_bentoml(self, volume: NDArray[np.float32]) -> NDArray[np.int64]:
        """Call BentoML REST endpoint for segmentation.

        Sends the volume as JSON to the BentoML predict endpoint and
        returns the segmentation mask.
        """
        url = f"{self._bentoml_url}/predict"
        payload = json.dumps(
            {"volume": volume.tolist()},
        ).encode("utf-8")

        req = urllib.request.Request(  # noqa: S310
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
            result = json.loads(resp.read().decode("utf-8"))

        seg = np.array(result["segmentation"], dtype=np.int64)
        return seg
