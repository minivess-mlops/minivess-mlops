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
from minivess.pipeline.federated import (
    DPConfig,
    FederatedAveraging,
    FLClientConfig,
    FLRoundResult,
    FLServerConfig,
    FLSimulator,
    FLStrategy,
)
from minivess.pipeline.hpo import (
    SearchSpace,
    build_objective,
    build_study,
    build_trial_config,
    run_hpo,
)
from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.metrics import MetricResult, SegmentationMetrics
from minivess.pipeline.preflight import (
    CheckStatus,
    PreflightCheck,
    PreflightResult,
    check_data_exists,
    check_disk_space,
    check_gpu,
    check_ram,
    check_swap,
    detect_environment,
    run_preflight,
)
from minivess.pipeline.segmentation_qc import (
    QCFlag,
    QCResult,
    SegmentationQC,
    evaluate_segmentation_quality,
)
from minivess.pipeline.trainer import EpochResult, SegmentationTrainer

__all__ = [
    "ConfidenceInterval",
    "DPConfig",
    "DYNUNET_WIDTH_PRESETS",
    "EpochResult",
    "FLClientConfig",
    "FLRoundResult",
    "FLServerConfig",
    "FLSimulator",
    "FLStrategy",
    "FederatedAveraging",
    "MetricResult",
    "QCFlag",
    "QCResult",
    "SearchSpace",
    "SegmentationMetrics",
    "SegmentationQC",
    "SegmentationTrainer",
    "CheckStatus",
    "PreflightCheck",
    "PreflightResult",
    "bca_bootstrap_ci",
    "bootstrap_ci",
    "build_ablation_grid",
    "build_loss_function",
    "build_trial_config",
    "build_study",
    "build_objective",
    "check_data_exists",
    "check_disk_space",
    "check_gpu",
    "check_ram",
    "check_swap",
    "compute_metrics_with_ci",
    "detect_environment",
    "evaluate_segmentation_quality",
    "run_hpo",
    "run_preflight",
]
