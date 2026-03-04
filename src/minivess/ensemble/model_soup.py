"""Stochastic Weight Averaging (SWA) utilities for model soup.

Uniform SWA averages state dicts across fold checkpoints to produce
a single model with lower generalization error and zero inference overhead.

References:
    - Izmailov et al. (2018), "Averaging Weights Leads to Wider Optima
      and Better Generalization"
    - Wortsman et al. (2022), "Model Soups"
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def uniform_swa(state_dicts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Uniform (equal-weight) averaging of multiple state dicts.

    Parameters
    ----------
    state_dicts:
        List of state dicts from the same architecture. All must have
        identical keys and tensor shapes.

    Returns
    -------
    Averaged state dict.

    Raises
    ------
    ValueError
        If the list is empty.
    """
    if not state_dicts:
        msg = "Need at least one state dict for SWA"
        raise ValueError(msg)

    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    n = len(state_dicts)
    averaged: dict[str, Tensor] = {}

    for key in state_dicts[0]:
        ref = state_dicts[0][key]
        if ref.is_floating_point():
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            averaged[key] = stacked.mean(dim=0).to(ref.dtype)
        else:
            # Non-floating (e.g., BatchNorm num_batches_tracked): take from first
            averaged[key] = ref.clone()

    logger.info("SWA: averaged %d state dicts (%d parameters)", n, len(averaged))
    return averaged


def swa_from_checkpoints(
    checkpoints: list[dict[str, Any]],
    *,
    state_dict_key: str = "state_dict",
) -> dict[str, Tensor]:
    """Extract state dicts from checkpoint dicts and average them.

    Parameters
    ----------
    checkpoints:
        List of checkpoint dicts, each containing a state_dict under
        the specified key.
    state_dict_key:
        Key name for the state dict within each checkpoint.

    Returns
    -------
    Averaged state dict.
    """
    state_dicts = [ckpt[state_dict_key] for ckpt in checkpoints]
    return uniform_swa(state_dicts)
