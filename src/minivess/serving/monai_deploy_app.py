"""MONAI Deploy MAP application for clinical deployment.

Provides MAP (MONAI Application Package) compatible classes for
packaging the MinIVess segmentation model for clinical environments.

The MAP application can be packaged with::

    monai-deploy package src/minivess/serving/monai_deploy_app.py \\
        --model model.onnx --output minivess-map.tgz

Note: MONAI Deploy SDK is optional — this module uses duck-typing
to work without the SDK installed (for testing and non-clinical use).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MiniVessInferenceOperator:
    """ONNX-based inference operator for MONAI Deploy MAP.

    Wraps :class:`OnnxSegmentationInference` in a MAP-compatible
    operator interface with ``process()`` method.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._engine: Any = None

    def _ensure_engine(self) -> None:
        """Lazy-load the ONNX engine on first use."""
        if self._engine is None:
            from minivess.serving.onnx_inference import OnnxSegmentationInference

            self._engine = OnnxSegmentationInference(self.model_path)

    def process(self, volume: NDArray[np.float32]) -> dict[str, Any]:
        """Process an input volume and return segmentation results.

        Parameters
        ----------
        volume:
            Input volume, shape (B, C, D, H, W) float32.

        Returns
        -------
        Dict with 'segmentation' (ndarray) and 'probabilities' (ndarray).
        """
        self._ensure_engine()
        result = self._engine.predict(volume)
        return {
            "segmentation": result.segmentation,
            "probabilities": result.probabilities,
            "shape": result.shape,
        }


class MiniVessSegApp:
    """MONAI Deploy MAP application for 3D vessel segmentation.

    Provides a ``compose()`` method that returns the operator pipeline.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def compose(self) -> list[MiniVessInferenceOperator]:
        """Compose the operator pipeline.

        Returns
        -------
        List of operators in execution order.
        """
        inference_op = MiniVessInferenceOperator(model_path=self.model_path)
        return [inference_op]


def generate_map_manifest(
    app_name: str,
    version: str,
    model_name: str,
) -> dict[str, Any]:
    """Generate a MONAI Deploy MAP manifest.

    Parameters
    ----------
    app_name:
        Application name.
    version:
        Application version string.
    model_name:
        Name of the model being packaged.

    Returns
    -------
    MAP manifest dict conforming to MAP specification.
    """
    return {
        "api-version": "1.0",
        "application": {
            "name": app_name,
            "version": version,
            "model_name": model_name,
            "description": "MinIVess 3D vessel segmentation MAP application",
        },
        "resources": {
            "cpu": 2,
            "memory": "4Gi",
            "gpu": 0,
        },
        "input": {
            "formats": ["nifti"],
            "description": "3D microscopy volume (NIfTI format)",
        },
        "output": {
            "formats": ["nifti"],
            "description": "Binary vessel segmentation mask",
        },
    }


def generate_map_main_py() -> str:
    """Generate the ``__main__.py`` entry point for MAP packaging.

    Returns
    -------
    Python source code string for the entry point.
    """
    return '''\
"""MAP entry point for MinIVess segmentation application."""

from __future__ import annotations

from pathlib import Path

from minivess.serving.monai_deploy_app import MiniVessSegApp


def main() -> None:
    """Run the MAP application."""
    model_path = Path("/opt/models/model.onnx")
    app = MiniVessSegApp(model_path=model_path)
    operators = app.compose()
    print(f"MiniVessSegApp ready with {len(operators)} operator(s)")


if __name__ == "__main__":
    main()
'''
