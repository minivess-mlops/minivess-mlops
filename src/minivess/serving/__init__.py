"""Serving -- BentoML service definitions and ONNX Runtime inference."""

from __future__ import annotations

from minivess.serving.bento_service import BENTO_MODEL_TAG, SegmentationService
from minivess.serving.onnx_inference import OnnxSegmentationInference

__all__ = [
    "BENTO_MODEL_TAG",
    "OnnxSegmentationInference",
    "SegmentationService",
]
