"""Generic per-head validation metrics for multi-task models.

Computes metrics for ANY auxiliary heads defined in config:
- Regression heads: MAE, RMSE
- Classification heads: accuracy, F1 (macro)
- Segmentation heads: Dice coefficient

Task-agnostic — metric type is determined by head_type field,
NOT by head name.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from minivess.adapters.base import SegmentationOutput  # noqa: TC001
from minivess.adapters.multitask_adapter import AuxHeadConfig  # noqa: TC001

logger = logging.getLogger(__name__)


def compute_per_head_metrics(
    output: SegmentationOutput,
    batch: dict[str, Tensor],
    head_configs: list[AuxHeadConfig],
) -> dict[str, float]:
    """Compute validation metrics for each auxiliary head.

    Args:
        output: SegmentationOutput with aux predictions in metadata.
        batch: Dict with ground truth keys matching head configs.
        head_configs: List of AuxHeadConfig defining heads and types.

    Returns:
        Dict mapping "{head_name}/{metric_name}" to float values.
    """
    metrics: dict[str, float] = {}

    for config in head_configs:
        pred = output.metadata.get(config.name)
        if pred is None:
            logger.warning(
                "Head '%s' not found in output metadata, skipping metrics", config.name
            )
            continue

        gt_key = config.gt_key
        gt = batch.get(gt_key)
        if gt is None:
            logger.warning(
                "GT key '%s' not found in batch, skipping metrics for '%s'",
                gt_key,
                config.name,
            )
            continue

        if config.head_type == "regression":
            _compute_regression_metrics(metrics, config.name, pred, gt)
        elif config.head_type == "classification":
            _compute_classification_metrics(metrics, config.name, pred, gt)
        elif config.head_type == "segmentation":
            _compute_segmentation_metrics(metrics, config.name, pred, gt)
        else:
            logger.warning(
                "Unknown head type '%s' for '%s', skipping",
                config.head_type,
                config.name,
            )

    return metrics


def _compute_regression_metrics(
    metrics: dict[str, float],
    name: str,
    pred: Tensor,
    gt: Tensor,
) -> None:
    """Compute MAE and RMSE for a regression head."""
    with torch.no_grad():
        diff = (pred.float() - gt.float()).abs()
        metrics[f"{name}/mae"] = diff.mean().item()
        metrics[f"{name}/rmse"] = diff.pow(2).mean().sqrt().item()


def _compute_classification_metrics(
    metrics: dict[str, float],
    name: str,
    pred: Tensor,
    gt: Tensor,
) -> None:
    """Compute accuracy and macro F1 for a classification head."""
    with torch.no_grad():
        # pred: [B, C, ...], gt: [B, ...]
        pred_classes = pred.argmax(dim=1)
        gt_flat = gt.long().flatten()
        pred_flat = pred_classes.flatten()

        # Accuracy
        correct = (pred_flat == gt_flat).float().mean()
        metrics[f"{name}/accuracy"] = correct.item()

        # Macro F1
        n_classes = pred.shape[1]
        f1_sum = 0.0
        for c in range(n_classes):
            tp = ((pred_flat == c) & (gt_flat == c)).float().sum()
            fp = ((pred_flat == c) & (gt_flat != c)).float().sum()
            fn = ((pred_flat != c) & (gt_flat == c)).float().sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_sum += f1.item()
        metrics[f"{name}/f1"] = f1_sum / max(n_classes, 1)


def _compute_segmentation_metrics(
    metrics: dict[str, float],
    name: str,
    pred: Tensor,
    gt: Tensor,
) -> None:
    """Compute Dice coefficient for a segmentation head."""
    with torch.no_grad():
        # Binarize prediction
        if pred.shape[1] > 1:
            pred_bin = pred.argmax(dim=1).float()
        else:
            pred_bin = (pred.squeeze(1) > 0.5).float()

        gt_bin = gt.float()
        if gt_bin.dim() == pred_bin.dim() + 1:
            gt_bin = gt_bin.squeeze(1)

        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum()
        dice = (2.0 * intersection / (union + 1e-8)).item()
        metrics[f"{name}/dice"] = dice
