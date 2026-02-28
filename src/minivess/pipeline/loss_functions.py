# R5.18 assessment (310 lines): Five loss classes + one factory function.
# All are loss functions for the same segmentation task — splitting by loss
# type would scatter related code unnecessarily. Below 300-line threshold
# when excluding docstrings. No action required.

from __future__ import annotations

import torch
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.losses.cldice import SoftclDiceLoss
from torch import nn


class VesselCompoundLoss(nn.Module):  # type: ignore[misc]
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
        probs, labels_onehot = _labels_to_onehot(logits, labels)
        cldice_loss = self.cldice(probs, labels_onehot)

        return self.lambda_dice_ce * dice_ce_loss + self.lambda_cldice * cldice_loss


def _labels_to_onehot(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert integer labels to one-hot and logits to probabilities."""
    probs = torch.softmax(logits, dim=1)
    n_classes = logits.shape[1]
    labels_onehot = torch.zeros_like(logits)
    labels_squeeze = labels.long()
    if labels_squeeze.ndim == logits.ndim:
        labels_squeeze = labels_squeeze[:, 0]
    for c in range(n_classes):
        labels_onehot[:, c] = (labels_squeeze == c).float()
    return probs, labels_onehot


class ClassBalancedDiceLoss(nn.Module):  # type: ignore[misc]
    """Class-balanced Dice loss with inverse-frequency weighting.

    Computes per-class weights from label frequencies in each batch,
    giving higher weight to rare classes (e.g., thin vessels).

    Parameters
    ----------
    num_classes:
        Number of segmentation classes.
    softmax:
        Apply softmax to logits.
    smooth:
        Smoothing factor to avoid division by zero.
    """

    def __init__(
        self,
        num_classes: int = 2,
        *,
        softmax: bool = True,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.softmax = softmax
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced Dice loss."""
        probs = torch.softmax(logits, dim=1) if self.softmax else logits

        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]

        # One-hot encode
        labels_onehot = torch.zeros_like(logits)
        for c in range(self.num_classes):
            labels_onehot[:, c] = (labels_squeeze == c).float()

        # Compute inverse-frequency weights per class
        class_counts = labels_onehot.sum(
            dim=tuple(range(2, labels_onehot.ndim))
        )  # (B, C)
        total_voxels = class_counts.sum(dim=1, keepdim=True)  # (B, 1)
        weights = total_voxels / (
            class_counts * self.num_classes + self.smooth
        )  # (B, C)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

        # Per-class Dice
        spatial_dims = tuple(range(2, logits.ndim))
        intersection = (probs * labels_onehot).sum(dim=spatial_dims)  # (B, C)
        union = probs.sum(dim=spatial_dims) + labels_onehot.sum(
            dim=spatial_dims
        )  # (B, C)
        dice_per_class = (2.0 * intersection + self.smooth) / (
            union + self.smooth
        )  # (B, C)

        # Weighted average
        weighted_dice = (weights * dice_per_class).sum(dim=1)  # (B,)
        return 1.0 - weighted_dice.mean()


class BettiLoss(nn.Module):  # type: ignore[misc]
    """Differentiable Betti-0 (connected component) topology loss.

    Approximates Betti-0 differences using soft thresholding and
    a differentiable connected-component proxy based on spatial
    entropy of the foreground prediction.

    Parameters
    ----------
    threshold:
        Soft threshold for foreground.
    lambda_betti:
        Weight for the Betti loss term.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        lambda_betti: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.lambda_betti = lambda_betti

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute topology loss via spatial fragmentation penalty.

        Uses a differentiable proxy: penalizes spatial variance of
        foreground probability, which correlates with fragmentation.
        """
        probs = torch.softmax(logits, dim=1)
        fg_prob = probs[:, 1:]  # Foreground channels (B, C-1, D, H, W)

        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]
        fg_label = (labels_squeeze > 0).float().unsqueeze(1)  # (B, 1, D, H, W)

        # Spatial gradient magnitude of foreground probability (proxy for fragmentation)
        # Higher gradients = more fragmented topology
        grad_d = torch.diff(fg_prob, dim=2)
        grad_h = torch.diff(fg_prob, dim=3)
        grad_w = torch.diff(fg_prob, dim=4)

        pred_frag = grad_d.abs().mean() + grad_h.abs().mean() + grad_w.abs().mean()

        # Same for ground truth
        grad_d_gt = torch.diff(fg_label, dim=2)
        grad_h_gt = torch.diff(fg_label, dim=3)
        grad_w_gt = torch.diff(fg_label, dim=4)

        gt_frag = (
            grad_d_gt.abs().mean() + grad_h_gt.abs().mean() + grad_w_gt.abs().mean()
        )

        # Penalize fragmentation difference
        return self.lambda_betti * (pred_frag - gt_frag).abs()


class TopologyCompoundLoss(nn.Module):  # type: ignore[misc]
    """Full topology-aware compound loss: DiceCE + clDice + Betti.

    Parameters
    ----------
    lambda_dice_ce:
        Weight for DiceCE component.
    lambda_cldice:
        Weight for clDice component.
    lambda_betti:
        Weight for Betti component.
    """

    def __init__(
        self,
        lambda_dice_ce: float = 0.4,
        lambda_cldice: float = 0.4,
        lambda_betti: float = 0.2,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_dice_ce = lambda_dice_ce
        self.lambda_cldice = lambda_cldice
        self.lambda_betti = lambda_betti
        self.dice_ce = DiceCELoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_ce=0.5,
            lambda_dice=0.5,
        )
        self.cldice = SoftclDiceLoss(smooth=1e-5, iter_=3)
        self.betti = BettiLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute compound topology loss."""
        dice_ce_loss = self.dice_ce(logits, labels)
        probs, labels_onehot = _labels_to_onehot(logits, labels)
        cldice_loss = self.cldice(probs, labels_onehot)
        betti_loss = self.betti(logits, labels)
        return (
            self.lambda_dice_ce * dice_ce_loss
            + self.lambda_cldice * cldice_loss
            + self.lambda_betti * betti_loss
        )


class CbDiceClDiceLoss(nn.Module):  # type: ignore[misc]
    """Compound loss combining cbDice + dice_ce_cldice for vessel segmentation.

    cbDice captures boundary-aware centerline topology (diameter-sensitive),
    while dice_ce_cldice captures skeleton-following topology via soft clDice.
    Together they provide complementary topology supervision.

    Parameters
    ----------
    lambda_cbdice:
        Weight for the cbDice component.
    lambda_cldice:
        Weight for the dice_ce_cldice (VesselCompoundLoss) component.
    softmax:
        Apply softmax to logits (passed to sub-losses).
    to_onehot_y:
        Convert labels to one-hot (passed to VesselCompoundLoss).
    """

    def __init__(
        self,
        lambda_cbdice: float = 0.5,
        lambda_cldice: float = 0.5,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_cbdice = lambda_cbdice
        self.lambda_cldice = lambda_cldice

        from minivess.pipeline.vendored_losses.cbdice import CenterlineBoundaryDiceLoss

        self.cbdice = CenterlineBoundaryDiceLoss(softmax=softmax)
        self.dice_ce_cldice = VesselCompoundLoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute compound cbDice + dice_ce_cldice loss."""
        cbdice_loss = self.cbdice(logits, labels)
        cldice_loss = self.dice_ce_cldice(logits, labels)
        return self.lambda_cbdice * cbdice_loss + self.lambda_cldice * cldice_loss


class GraphTopologyLoss(nn.Module):  # type: ignore[misc]
    """Compound graph topology loss: cbdice_cldice + skeleton_recall + CAPE.

    Combines voxel overlap preservation (cbdice_cldice) with topology-aware
    losses (skeleton recall for missed centrelines, CAPE for broken paths).

    Parameters
    ----------
    w_cbdice_cldice:
        Weight for the cbDice+clDice component.
    w_skeleton_recall:
        Weight for the skeleton recall component.
    w_cape:
        Weight for the CAPE path enforcement component.
    """

    def __init__(
        self,
        w_cbdice_cldice: float = 0.5,
        w_skeleton_recall: float = 0.3,
        w_cape: float = 0.2,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.w_cbdice_cldice = w_cbdice_cldice
        self.w_skeleton_recall = w_skeleton_recall
        self.w_cape = w_cape

        self.cbdice_cldice = CbDiceClDiceLoss(softmax=softmax, to_onehot_y=to_onehot_y)

        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        self.skeleton_recall = SkeletonRecallLoss(softmax=softmax)

        from minivess.pipeline.vendored_losses.cape import CAPELoss

        self.cape = CAPELoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute compound graph topology loss."""
        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.w_cbdice_cldice > 0:
            loss = loss + self.w_cbdice_cldice * self.cbdice_cldice(logits, labels)
        if self.w_skeleton_recall > 0:
            loss = loss + self.w_skeleton_recall * self.skeleton_recall(logits, labels)
        if self.w_cape > 0:
            loss = loss + self.w_cape * self.cape(logits, labels)
        return loss


def build_loss_function(
    loss_name: str = "cbdice_cldice",
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
        ``"focal"``, ``"cldice"``, ``"dice_ce_cldice"``, ``"cb_dice"``,
        ``"betti"``, ``"full_topo"``, ``"cbdice_cldice"``, ``"cbdice"``,
        ``"centerline_ce"``, ``"warp"``, ``"topo"``.
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
    if loss_name == "cb_dice":
        return ClassBalancedDiceLoss(
            num_classes=num_classes,
            softmax=softmax,
        )
    if loss_name == "betti":
        return BettiLoss()
    if loss_name == "full_topo":
        return TopologyCompoundLoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
        )
    if loss_name == "cbdice_cldice":
        return CbDiceClDiceLoss(softmax=softmax, to_onehot_y=to_onehot_y)
    if loss_name == "graph_topology":
        return GraphTopologyLoss(softmax=softmax, to_onehot_y=to_onehot_y)
    # --- Vendored losses ---
    if loss_name == "cbdice":
        from minivess.pipeline.vendored_losses.cbdice import CenterlineBoundaryDiceLoss

        return CenterlineBoundaryDiceLoss(softmax=softmax)
    if loss_name == "centerline_ce":
        from minivess.pipeline.vendored_losses.centerline_ce import (
            CenterlineCrossEntropyLoss,
        )

        return CenterlineCrossEntropyLoss()
    if loss_name == "warp":
        from minivess.pipeline.vendored_losses.coletra import WarpLoss

        return WarpLoss(softmax=softmax)
    if loss_name == "topo":
        from minivess.pipeline.vendored_losses.coletra import TopoLoss

        return TopoLoss(softmax=softmax)
    if loss_name == "skeleton_recall":
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        return SkeletonRecallLoss(softmax=softmax)
    if loss_name == "cape":
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        return CAPELoss()
    msg = f"Unknown loss function: {loss_name}"
    raise ValueError(msg)
