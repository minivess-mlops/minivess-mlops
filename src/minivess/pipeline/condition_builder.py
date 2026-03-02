"""Condition-to-model-and-loss factory for topology experiments.

Bridges between YAML condition configs and the adapter/loss infrastructure.
Given a base model and condition config, produces:
  - A (possibly wrapped) model with TFFM and/or multi-task heads
  - A (possibly multi-task) loss criterion

Uses existing apply_wrappers() from model_builder.py for model wrapping.
Uses MultiTaskLoss from multitask_loss.py for multi-task loss composition.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn  # noqa: TC002 — used at runtime in function signature

logger = logging.getLogger(__name__)


def parse_aux_head_configs(
    heads: list[dict[str, Any]],
) -> list[Any]:
    """Parse YAML auxiliary head dicts into AuxHeadConfig objects.

    Args:
        heads: List of dicts from YAML, each with name, type, out_channels,
            and optionally gt_key.

    Returns:
        List of AuxHeadConfig instances.
    """
    from minivess.adapters.multitask_adapter import AuxHeadConfig

    return [
        AuxHeadConfig(
            name=h["name"],
            head_type=h["type"],
            out_channels=h.get("out_channels", 1),
            gt_key=h.get("gt_key", h["name"]),
        )
        for h in heads
    ]


def build_condition_model(
    base_model: nn.Module,
    condition: dict[str, Any],
) -> nn.Module:
    """Build a (possibly wrapped) model from a condition config.

    If the condition has wrappers, applies them via apply_wrappers().
    If no wrappers, returns the base model unchanged.

    Args:
        base_model: Base model (e.g., DynUNetAdapter instance).
        condition: Condition config dict with 'wrappers' key.

    Returns:
        Model with wrappers applied (or original if no wrappers).
    """
    wrappers: list[dict[str, Any]] = condition.get("wrappers", [])

    if not wrappers:
        return base_model

    from minivess.adapters.model_builder import apply_wrappers

    return apply_wrappers(base_model, wrappers)


def build_condition_loss(
    loss_name: str,
    condition: dict[str, Any],
    *,
    num_classes: int = 2,
) -> nn.Module:
    """Build a (possibly multi-task) loss from a condition config.

    If the condition has multitask wrappers, wraps the base criterion
    in MultiTaskLoss with AuxHeadLossConfig for each auxiliary head.
    Otherwise returns the standard criterion.

    Args:
        loss_name: Base loss function name (e.g., "cbdice_cldice").
        condition: Condition config dict with 'wrappers' key.
        num_classes: Number of segmentation classes.

    Returns:
        Loss criterion (nn.Module).
    """
    from minivess.pipeline.loss_functions import build_loss_function

    base_criterion = build_loss_function(loss_name, num_classes=num_classes)

    # Check if any wrapper is multitask
    wrappers: list[dict[str, Any]] = condition.get("wrappers", [])
    multitask_wrapper = None
    for w in wrappers:
        if w.get("type") == "multitask":
            multitask_wrapper = w
            break

    if multitask_wrapper is None:
        return base_criterion

    # Build MultiTaskLoss
    from minivess.pipeline.multitask_loss import AuxHeadLossConfig, MultiTaskLoss

    aux_heads = multitask_wrapper.get("auxiliary_heads", [])
    aux_configs = [
        AuxHeadLossConfig(
            name=h["name"],
            loss_type=h.get("loss", "smooth_l1"),
            weight=h.get("weight", 0.25),
            gt_key=h.get("gt_key", h["name"]),
        )
        for h in aux_heads
    ]

    return MultiTaskLoss(
        seg_criterion=base_criterion,
        aux_head_configs=aux_configs,
    )
