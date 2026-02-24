"""Pipeline — Training, inference, and evaluation orchestration."""

from __future__ import annotations

from minivess.pipeline.ablation import (
    DYNUNET_WIDTH_PRESETS,
    build_ablation_grid,
)
from minivess.pipeline.ci import (
    ConfidenceInterval,
    bca_bootstrap_ci,
    bootstrap_ci,
    compute_metrics_with_ci,
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
    "ConfidenceInterval",
    "DYNUNET_WIDTH_PRESETS",
    "EpochResult",
    "MetricResult",
    "SearchSpace",
    "SegmentationMetrics",
    "SegmentationTrainer",
    "bca_bootstrap_ci",
    "bootstrap_ci",
    "build_ablation_grid",
    "build_loss_function",
    "build_trial_config",
    "compute_metrics_with_ci",
    "create_study",
    "make_objective",
    "run_hpo",
]
