from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score
from torchmetrics.segmentation import DiceScore

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Aggregated metrics from evaluation."""

    values: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


class SegmentationMetrics:
    """GPU-accelerated segmentation metrics using TorchMetrics.

    Tracks Dice score and F1 score (for binary segmentation).
    Additionally accumulates foreground probabilities for Tier 1 calibration
    metrics (ECE, Brier, NLL, etc.) when soft predictions are provided.
    """

    def __init__(
        self,
        num_classes: int = 2,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_classes = num_classes

        # Build metric collection
        metrics: dict[str, Any] = {}

        # Dice score per class
        metrics["dice"] = DiceScore(
            num_classes=num_classes,
            average="macro",
            input_format="index",
        )

        # Binary F1 for foreground class (class 1)
        if num_classes == 2:
            metrics["f1_foreground"] = BinaryF1Score()

        self.metrics = MetricCollection(metrics).to(self.device)

        # Calibration accumulators — probabilities + labels for Tier 1 metrics.
        # Only populated when soft predictions (B, C, D, H, W) are provided.
        self._cal_probs: list[Tensor] = []
        self._cal_labels: list[Tensor] = []

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """Update metrics with a batch of predictions and targets.

        Parameters
        ----------
        prediction:
            ``(B, C, D, H, W)`` class probabilities or ``(B, D, H, W)``
            argmax indices.
        target:
            ``(B, D, H, W)`` integer class labels or ``(B, 1, D, H, W)``.
        """
        # Squeeze channel dim from target if present
        if target.ndim == 5:
            target = target.squeeze(1)
        target = target.long()

        # Extract foreground probabilities BEFORE argmax (for calibration)
        if prediction.ndim == 5 and prediction.shape[1] > 1:
            # (B, C, D, H, W) with C >= 2: foreground = channel 1
            fg_probs = prediction[:, 1, ...].detach().cpu()
            self._cal_probs.append(fg_probs)
            self._cal_labels.append(target.detach().cpu())
            pred_indices = prediction.argmax(dim=1)
        elif prediction.ndim == 5 and prediction.shape[1] == 1:
            # (B, 1, D, H, W): single-channel sigmoid output
            fg_probs = prediction.squeeze(1).detach().cpu()
            self._cal_probs.append(fg_probs)
            self._cal_labels.append(target.detach().cpu())
            pred_indices = (prediction.squeeze(1) > 0.5).long()
        else:
            # (B, D, H, W) hard indices — no probabilities available
            pred_indices = prediction

        pred_indices = pred_indices.to(self.device)
        target = target.to(self.device)

        # Update dice with integer predictions and targets
        self.metrics["dice"].update(pred_indices, target)

        # Update binary F1 for foreground
        if "f1_foreground" in self.metrics:
            fg_pred = (pred_indices == 1).float().flatten()
            fg_target = (target == 1).long().flatten()
            self.metrics["f1_foreground"].update(fg_pred, fg_target)

    def compute(self) -> MetricResult:
        """Compute aggregated metrics including Tier 1 calibration metrics."""
        computed = self.metrics.compute()
        values: dict[str, float] = {
            k: v.item() if isinstance(v, Tensor) else v for k, v in computed.items()
        }

        # Tier 1 calibration metrics from accumulated probabilities
        if self._cal_probs:
            try:
                from minivess.pipeline.calibration_metrics import (
                    compute_all_calibration_metrics,
                )

                all_probs = torch.cat(self._cal_probs).numpy().ravel()
                all_labels = torch.cat(self._cal_labels).numpy().ravel()

                cal_metrics = compute_all_calibration_metrics(
                    all_probs, all_labels, tier="fast"
                )
                for name, val in cal_metrics.items():
                    values[f"val_{name}"] = val
            except Exception:
                logger.warning("Failed to compute calibration metrics", exc_info=True)

        return MetricResult(values=values)

    def reset(self) -> None:
        """Reset all metrics for next epoch."""
        self.metrics.reset()
        self._cal_probs.clear()
        self._cal_labels.clear()
