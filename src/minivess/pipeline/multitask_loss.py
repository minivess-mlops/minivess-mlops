"""Generic multi-task loss composer with upgraded criterion interface.

Supports ANY number of auxiliary heads defined in config. Task-agnostic —
it knows about loss TYPES (smooth_l1, mse, bce, cross_entropy) but NOT
about specific TASKS (SDF, centerline, etc.). Tasks are config.

NOT registered in build_loss_function() (different signature).
Constructed by model builder from config. Trainer upgraded (T9a) to
detect and call appropriately via isinstance check.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from minivess.adapters.base import SegmentationOutput  # noqa: TC001 — used at runtime


@dataclasses.dataclass
class AuxHeadLossConfig:
    """Configuration for one auxiliary head's loss.

    Args:
        name: Head name (matches key in SegmentationOutput.metadata).
        loss_type: One of "smooth_l1", "mse", "bce", "cross_entropy".
        weight: Loss weight multiplier.
        gt_key: Key in batch dict for this head's ground truth.
        mask_to_foreground: If True, only compute loss inside foreground mask.
    """

    name: str
    loss_type: str
    weight: float
    gt_key: str
    mask_to_foreground: bool = False


def _build_aux_loss(loss_type: str) -> nn.Module:
    """Build an auxiliary loss function from type string."""
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction="mean")
    if loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(reduction="mean")
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction="mean")
    msg = f"Unknown aux loss type: {loss_type}. Supported: smooth_l1, mse, bce, cross_entropy"
    raise ValueError(msg)


class MultiTaskLoss(nn.Module):
    """Generic multi-task loss composer.

    forward(output: SegmentationOutput, batch: dict) -> Tensor

    L_total = w_seg * seg_criterion(output.logits, batch["label"])
            + sum(w_i * L_i(output.metadata[name_i], batch[gt_key_i]))

    Stores per-component losses in self.component_losses dict.

    Args:
        seg_criterion: Loss function for primary segmentation task.
        aux_head_configs: List of AuxHeadLossConfig for auxiliary heads.
        seg_weight: Weight for segmentation loss (default 1.0).
    """

    def __init__(
        self,
        seg_criterion: nn.Module,
        aux_head_configs: list[AuxHeadLossConfig] | None = None,
        seg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.seg_criterion = seg_criterion
        self.aux_head_configs = aux_head_configs or []
        self.seg_weight = seg_weight
        self.component_losses: dict[str, float] = {}

        # Build aux loss functions
        self._aux_losses: dict[str, nn.Module] = {}
        for config in self.aux_head_configs:
            self._aux_losses[config.name] = _build_aux_loss(config.loss_type)

    def forward(self, output: SegmentationOutput, batch: dict[str, Tensor]) -> Tensor:
        """Compute total multi-task loss.

        Args:
            output: SegmentationOutput with logits and metadata containing aux predictions.
            batch: Dict with "label" and aux GT keys.

        Returns:
            Scalar total loss tensor.
        """
        self.component_losses = {}

        # Primary segmentation loss
        label = batch["label"]
        # CrossEntropyLoss expects [B, D, H, W] long target; squeeze channel dim if present
        if label.dim() == output.logits.dim() and label.shape[1] == 1:
            label = label.squeeze(1)
        if label.is_floating_point():
            label = label.long()
        seg_loss = self.seg_criterion(output.logits, label)
        self.component_losses["loss/seg"] = seg_loss.item()
        total: Tensor = self.seg_weight * seg_loss

        # Auxiliary head losses
        for config in self.aux_head_configs:
            pred = output.metadata[config.name]
            gt = batch[config.gt_key]
            loss_fn = self._aux_losses[config.name]

            if config.mask_to_foreground:
                fg_mask = (batch["label"] > 0).float()
                # If no foreground, aux loss is zero
                if fg_mask.sum() == 0:
                    aux_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
                else:
                    aux_loss = loss_fn(pred * fg_mask, gt * fg_mask)
            else:
                if config.loss_type == "bce":
                    aux_loss = F.binary_cross_entropy_with_logits(pred, gt)
                else:
                    aux_loss = loss_fn(pred, gt)

            self.component_losses[f"loss/{config.name}"] = aux_loss.item()
            total = total + config.weight * aux_loss

        return total
