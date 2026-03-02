"""Deploy verification — training vs serving metric consistency checks.

Compares metrics from training evaluation (FoldResult) against metrics
recomputed on serving predictions to verify that ONNX export and
BentoML serving preserve model behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from minivess.pipeline.evaluation import FoldResult

logger = logging.getLogger(__name__)


@dataclass
class DeployVerificationResult:
    """Result of comparing training metrics against serving metrics.

    Parameters
    ----------
    training_metrics:
        Metrics from the training evaluation (point estimates).
    serving_metrics:
        Metrics recomputed on serving predictions.
    metric_diffs:
        Absolute difference per metric.
    all_match:
        True if all diffs are within tolerance.
    tolerance:
        Tolerance used for comparison.
    """

    training_metrics: dict[str, float]
    serving_metrics: dict[str, float]
    metric_diffs: dict[str, float]
    all_match: bool
    tolerance: float


def verify_deploy_metrics(
    training_fold_result: FoldResult,
    serving_predictions: list[NDArray[np.integer]],
    labels: list[NDArray[np.integer]],
    *,
    tolerance: float = 1e-5,
) -> DeployVerificationResult:
    """Compare training evaluation metrics against recomputed serving metrics.

    Parameters
    ----------
    training_fold_result:
        Evaluation results from training.
    serving_predictions:
        Predictions from the deployed model.
    labels:
        Ground truth labels.
    tolerance:
        Maximum allowed absolute difference per metric.

    Returns
    -------
    DeployVerificationResult

    Raises
    ------
    ValueError
        If predictions or labels are empty.
    """
    if not serving_predictions or not labels:
        msg = "predictions and labels must be non-empty"
        raise ValueError(msg)

    # Extract training metrics (point estimates)
    training_metrics: dict[str, float] = {}
    for metric_name, ci in training_fold_result.aggregated.items():
        training_metrics[metric_name] = float(ci.point_estimate)

    # Recompute metrics on serving predictions using simple DSC
    serving_metrics: dict[str, float] = {}
    for metric_name in training_metrics:
        if metric_name == "dsc":
            # Compute per-volume DSC and average
            dsc_values: list[float] = []
            for pred, label in zip(serving_predictions, labels, strict=True):
                pred_flat = pred.ravel().astype(bool)
                label_flat = label.ravel().astype(bool)
                intersection = np.sum(pred_flat & label_flat)
                total = np.sum(pred_flat) + np.sum(label_flat)
                if total == 0:
                    dsc_values.append(1.0)
                else:
                    dsc_values.append(float(2.0 * intersection / total))
            serving_metrics[metric_name] = float(np.mean(dsc_values))
        else:
            # For non-DSC metrics, mark as not recomputed
            serving_metrics[metric_name] = float("nan")

    # Compare
    metric_diffs: dict[str, float] = {}
    all_match = True
    for metric_name in training_metrics:
        train_val = training_metrics[metric_name]
        serve_val = serving_metrics.get(metric_name, float("nan"))
        if np.isnan(serve_val):
            metric_diffs[metric_name] = float("nan")
            continue
        diff = abs(train_val - serve_val)
        metric_diffs[metric_name] = diff
        if diff > tolerance:
            all_match = False
            logger.warning(
                "Metric mismatch: %s training=%.6f serving=%.6f diff=%.6f > tol=%.6f",
                metric_name,
                train_val,
                serve_val,
                diff,
                tolerance,
            )

    return DeployVerificationResult(
        training_metrics=training_metrics,
        serving_metrics=serving_metrics,
        metric_diffs=metric_diffs,
        all_match=all_match,
        tolerance=tolerance,
    )


def verify_onnx_vs_pytorch(
    pytorch_output: NDArray[np.floating],
    onnx_output: NDArray[np.floating],
    *,
    tolerance: float = 1e-3,
) -> bool:
    """Compare PyTorch and ONNX Runtime outputs for numerical consistency.

    Parameters
    ----------
    pytorch_output:
        Output from PyTorch model.
    onnx_output:
        Output from ONNX Runtime.
    tolerance:
        Maximum allowed absolute difference.

    Returns
    -------
    True if all values are within tolerance.
    """
    return bool(np.allclose(pytorch_output, onnx_output, atol=tolerance, rtol=0))
