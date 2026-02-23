from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score
from torchmetrics.segmentation import DiceScore


@dataclass
class MetricResult:
    """Aggregated metrics from evaluation."""

    values: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


class SegmentationMetrics:
    """GPU-accelerated segmentation metrics using TorchMetrics.

    Tracks Dice score and F1 score (for binary segmentation).
    Additional metrics (clDice, NSD) can be added as custom TorchMetrics.
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
        # Convert probabilities to class indices if needed
        if prediction.ndim == 5 and prediction.shape[1] > 1:
            pred_indices = prediction.argmax(dim=1)
        elif prediction.ndim == 5 and prediction.shape[1] == 1:
            pred_indices = (prediction.squeeze(1) > 0.5).long()
        else:
            pred_indices = prediction

        # Squeeze channel dim from target if present
        if target.ndim == 5:
            target = target.squeeze(1)
        target = target.long()

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
        """Compute aggregated metrics."""
        computed = self.metrics.compute()
        values = {
            k: v.item() if isinstance(v, Tensor) else v for k, v in computed.items()
        }
        return MetricResult(values=values)

    def reset(self) -> None:
        """Reset all metrics for next epoch."""
        self.metrics.reset()
