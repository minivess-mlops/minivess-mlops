"""Checkpoint metadata tracking for MLflow logging.

Computes checkpoint file size, format, and parameter count for
cost appendix and experiment tracking.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime
from typing import Any

import torch


def compute_checkpoint_metrics(checkpoint_path: Path) -> dict[str, Any]:
    """Compute checkpoint metadata metrics.

    Parameters
    ----------
    checkpoint_path:
        Path to a PyTorch checkpoint file (``.pt`` or ``.pth``).

    Returns
    -------
    Dict with keys:
        - ``checkpoint/size_mb``: File size in megabytes.
        - ``checkpoint/format``: File extension without dot.
        - ``checkpoint/n_params``: Total number of parameters in model_state_dict.
    """
    metrics: dict[str, Any] = {}

    # File size
    if checkpoint_path.exists():
        size_bytes = checkpoint_path.stat().st_size
        metrics["checkpoint/size_mb"] = round(size_bytes / (1024 * 1024), 4)
    else:
        metrics["checkpoint/size_mb"] = 0.0

    # Format
    metrics["checkpoint/format"] = checkpoint_path.suffix.lstrip(".")

    # Parameter count
    loaded = torch.load(checkpoint_path, weights_only=True)
    state_dict = loaded.get("model_state_dict", loaded)

    n_params = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            n_params += value.numel()
    metrics["checkpoint/n_params"] = n_params

    return metrics
