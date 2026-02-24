from __future__ import annotations

import torch
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.losses.cldice import SoftclDiceLoss
from torch import nn


class VesselCompoundLoss(nn.Module):
    """Compound loss combining DiceCE + SoftclDice for vessel segmentation.

    Parameters
    ----------
    lambda_dice_ce:
        Weight for DiceCE component.
    lambda_cldice:
        Weight for SoftclDice component.
    softmax:
        Apply softmax to logits.
    to_onehot_y:
        Convert labels to one-hot.
    """

    def __init__(
        self,
        lambda_dice_ce: float = 0.5,
        lambda_cldice: float = 0.5,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_dice_ce = lambda_dice_ce
        self.lambda_cldice = lambda_cldice
        self.dice_ce = DiceCELoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_ce=0.5,
            lambda_dice=0.5,
        )
        self.cldice = SoftclDiceLoss(
            smooth=1e-5,
            iter_=3,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute compound loss.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth (B, 1, D, H, W) integer labels.
        """
        dice_ce_loss = self.dice_ce(logits, labels)

        # SoftclDice expects softmax probabilities and one-hot labels
        probs = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]
        labels_onehot = torch.zeros_like(logits)
        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        for c in range(n_classes):
            labels_onehot[:, c] = (labels_squeeze == c).float()

        cldice_loss = self.cldice(probs, labels_onehot)

        return self.lambda_dice_ce * dice_ce_loss + self.lambda_cldice * cldice_loss


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
        Loss function identifier. One of ``"dice_ce"``, ``"dice"``,
        ``"focal"``, ``"cldice"``, ``"dice_ce_cldice"``.
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
    if loss_name == "cldice":
        return SoftclDiceLoss(smooth=1e-5, iter_=3)
    if loss_name == "dice_ce_cldice":
        return VesselCompoundLoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
        )
    msg = f"Unknown loss function: {loss_name}"
    raise ValueError(msg)
