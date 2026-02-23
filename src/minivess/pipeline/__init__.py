"""Pipeline — Training, inference, and evaluation orchestration."""

from __future__ import annotations

from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.metrics import MetricResult, SegmentationMetrics
from minivess.pipeline.trainer import EpochResult, SegmentationTrainer

__all__ = [
    "EpochResult",
    "MetricResult",
    "SegmentationMetrics",
    "SegmentationTrainer",
    "build_loss_function",
]
