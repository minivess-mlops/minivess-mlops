"""Pipeline — Training, inference, and evaluation orchestration."""

from __future__ import annotations

from minivess.pipeline.ablation import (
    DYNUNET_WIDTH_PRESETS,
    build_ablation_grid,
)
from minivess.pipeline.hpo import (
    SearchSpace,
    build_trial_config,
    create_study,
    make_objective,
    run_hpo,
)
from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.metrics import MetricResult, SegmentationMetrics
from minivess.pipeline.trainer import EpochResult, SegmentationTrainer

__all__ = [
    "DYNUNET_WIDTH_PRESETS",
    "EpochResult",
    "MetricResult",
    "SearchSpace",
    "SegmentationMetrics",
    "SegmentationTrainer",
    "build_ablation_grid",
    "build_loss_function",
    "build_trial_config",
    "create_study",
    "make_objective",
    "run_hpo",
]
