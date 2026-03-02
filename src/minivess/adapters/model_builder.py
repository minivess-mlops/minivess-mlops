"""Generic wrapper composition for model building.

Applies config-driven wrappers (TFFM, multi-task, etc.) to a base model
sequentially. Researchers add new wrapper types via config YAML only.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn  # noqa: TC002 — used at runtime in function signature

logger = logging.getLogger(__name__)


def apply_wrappers(
    model: nn.Module,
    wrappers: list[dict[str, Any]],
) -> nn.Module:
    """Apply a sequence of config-driven wrappers to a base model.

    Each wrapper dict must have a "type" key. Supported types:
    - "tffm": TFFMWrapper (grid_size, hidden_dim, n_heads, k_neighbors)
    - "multitask": MultiTaskAdapter (auxiliary_heads list)

    Args:
        model: Base model to wrap.
        wrappers: List of wrapper config dicts, applied in order.

    Returns:
        Wrapped model (or original if no wrappers).
    """
    for wrapper_cfg in wrappers:
        wrapper_type = wrapper_cfg["type"]

        if wrapper_type == "tffm":
            from minivess.adapters.tffm_wrapper import TFFMWrapper

            model = TFFMWrapper(
                base_model=model,
                grid_size=wrapper_cfg.get("grid_size", 8),
                hidden_dim=wrapper_cfg.get("hidden_dim", 32),
                n_heads=wrapper_cfg.get("n_heads", 4),
                k_neighbors=wrapper_cfg.get("k_neighbors", 8),
            )
            logger.info(
                "Applied TFFMWrapper (grid=%d)", wrapper_cfg.get("grid_size", 8)
            )

        elif wrapper_type == "multitask":
            from minivess.adapters.multitask_adapter import (
                AuxHeadConfig,
                MultiTaskAdapter,
            )

            aux_heads = wrapper_cfg.get("auxiliary_heads", [])
            aux_configs = [
                AuxHeadConfig(
                    name=h["name"],
                    head_type=h["type"],
                    out_channels=h.get("out_channels", 1),
                )
                for h in aux_heads
            ]
            model = MultiTaskAdapter(base_model=model, aux_head_configs=aux_configs)
            logger.info("Applied MultiTaskAdapter (%d aux heads)", len(aux_configs))

        else:
            msg = f"Unknown wrapper type: {wrapper_type!r}. Supported: tffm, multitask"
            raise ValueError(msg)

    return model
