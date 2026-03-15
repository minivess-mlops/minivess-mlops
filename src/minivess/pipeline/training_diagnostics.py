"""Lightweight training diagnostics for model convergence debugging.

Provides per-epoch gradient norms, prediction statistics, and NaN/Inf
detection. Used by trainer.py to diagnose convergence issues (H2-H5).

Overhead: <1% of training time. Always enabled.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total L2 norm of all parameter gradients.

    Returns 0.0 if no parameters have gradients (e.g., before first backward).
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.float().norm(2).item() ** 2
    return float(total_norm_sq**0.5)


def compute_prediction_stats(predictions: Tensor) -> dict[str, float]:
    """Compute summary statistics of model predictions.

    Parameters
    ----------
    predictions:
        Model output logits or probabilities, any shape.

    Returns
    -------
    Dict with keys: pred_min, pred_max, pred_mean, pred_std.
    """
    flat = predictions.detach().float()
    return {
        "pred_min": flat.min().item(),
        "pred_max": flat.max().item(),
        "pred_mean": flat.mean().item(),
        "pred_std": flat.std().item(),
    }


def has_nan_or_inf(tensor: Tensor) -> bool:
    """Check if tensor contains any NaN or Inf values."""
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


def log_epoch_diagnostics(
    model: nn.Module,
    predictions: Tensor,
    epoch: int,
    phase: str = "train",
) -> dict[str, float]:
    """Log diagnostic metrics for one epoch.

    Parameters
    ----------
    model:
        The model (for gradient norms).
    predictions:
        Model output logits from last batch.
    epoch:
        Current epoch number.
    phase:
        "train" or "val".

    Returns
    -------
    Dict of diagnostic metrics for MLflow logging.
    """
    grad_norm = compute_gradient_norm(model) if phase == "train" else 0.0
    pred_stats = compute_prediction_stats(predictions)

    diagnostics = {
        f"{phase}_grad_norm": grad_norm,
        **{f"{phase}_{k}": v for k, v in pred_stats.items()},
    }

    if has_nan_or_inf(predictions):
        logger.warning(
            "Epoch %d %s: NaN/Inf detected in predictions! "
            "Stats: min=%.4f, max=%.4f, mean=%.4f",
            epoch,
            phase,
            pred_stats["pred_min"],
            pred_stats["pred_max"],
            pred_stats["pred_mean"],
        )
        diagnostics[f"{phase}_has_nan_inf"] = 1.0
    else:
        diagnostics[f"{phase}_has_nan_inf"] = 0.0

    if grad_norm > 0 and phase == "train":
        logger.debug(
            "Epoch %d diagnostics: grad_norm=%.4f, pred_range=[%.4f, %.4f]",
            epoch,
            grad_norm,
            pred_stats["pred_min"],
            pred_stats["pred_max"],
        )

    return diagnostics
