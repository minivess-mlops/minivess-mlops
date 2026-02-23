from __future__ import annotations

from typing import TYPE_CHECKING

from monai.losses import DiceCELoss, DiceLoss, FocalLoss

if TYPE_CHECKING:
    from torch import nn


def build_loss_function(
    loss_name: str = "dice_ce",
    *,
    num_classes: int = 2,
    softmax: bool = True,
    to_onehot_y: bool = True,
) -> nn.Module:
    """Factory for segmentation loss functions.

    Parameters
    ----------
    loss_name:
        Loss function identifier. One of ``"dice_ce"``, ``"dice"``, ``"focal"``.
    num_classes:
        Number of segmentation classes (including background).
    softmax:
        Whether to apply softmax to model outputs (used by Dice-based losses).
    to_onehot_y:
        Whether to convert integer labels to one-hot encoding.

    Returns
    -------
    nn.Module
        Configured loss function ready for ``loss(logits, labels)`` calls.
    """
    if loss_name == "dice_ce":
        return DiceCELoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_ce=0.5,
            lambda_dice=0.5,
        )
    if loss_name == "dice":
        return DiceLoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
        )
    if loss_name == "focal":
        return FocalLoss(
            gamma=2.0,
            to_onehot_y=to_onehot_y,
        )
    msg = f"Unknown loss function: {loss_name}"
    raise ValueError(msg)
