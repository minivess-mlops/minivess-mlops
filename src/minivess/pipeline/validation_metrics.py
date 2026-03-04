from __future__ import annotations

import math

import numpy as np
import torch


def compute_sdc(
    softmax_probs: torch.Tensor | np.ndarray,
    hard_predictions: torch.Tensor | np.ndarray,
    *,
    foreground_class: int = 1,
    eps: float = 1e-7,
) -> float:
    """Compute Soft Dice Confidence (SDC) for selective prediction.

    SDC = 2 * sum(p * y_hat) / sum(p + y_hat), where p is the softmax
    probability of the predicted class and y_hat is the binary hard prediction.
    Near-optimal confidence estimator per Borges et al. (2025, arXiv:2402.10665v4).

    Parameters
    ----------
    softmax_probs:
        Softmax probabilities, shape (B, C, ...) where C >= 2.
    hard_predictions:
        Hard predictions (class indices), shape (B, 1, ...) or (B, ...).
    foreground_class:
        Which class index to compute SDC for (default: 1 = foreground).
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    float
        SDC value in [0, 1]. Higher = more confident. Returns 0.0 when
        there is no foreground prediction and no foreground probability.
    """
    if isinstance(softmax_probs, np.ndarray):
        softmax_probs = torch.from_numpy(softmax_probs)
    if isinstance(hard_predictions, np.ndarray):
        hard_predictions = torch.from_numpy(hard_predictions)

    # Extract foreground probability: shape (B, ...)
    p = softmax_probs[:, foreground_class]

    # Binary mask of foreground predictions: shape (B, ...)
    if hard_predictions.ndim == softmax_probs.ndim:
        # shape (B, 1, ...) → squeeze channel dim
        y_hat = (hard_predictions[:, 0] == foreground_class).float()
    else:
        # shape (B, ...) already
        y_hat = (hard_predictions == foreground_class).float()

    numerator = 2.0 * (p * y_hat).sum()
    denominator = p.sum() + y_hat.sum()

    if denominator < eps:
        return 0.0

    sdc: float = (numerator / denominator).item()
    return max(0.0, min(1.0, sdc))


def normalize_masd(masd: float, *, max_masd: float = 50.0) -> float:
    """Normalize MASD from [0, inf) to [0, 1] score (higher is better).

    Parameters
    ----------
    masd:
        Mean Average Surface Distance in voxel units. Lower is better.
    max_masd:
        MASD value that maps to 0.0 (worst). Values above are clamped.

    Returns
    -------
    float
        Normalized score in [0, 1]. 1.0 = perfect, 0.0 = worst.
    """
    if math.isnan(masd):
        return 0.0
    return max(0.0, min(1.0, 1.0 - masd / max_masd))


def compute_compound_masd_cldice(
    *,
    masd: float,
    cldice: float,
    w_masd: float = 0.5,
    w_cldice: float = 0.5,
    max_masd: float = 50.0,
) -> float:
    """Compute compound metric: w_masd * normalize_masd(masd) + w_cldice * cldice.

    Parameters
    ----------
    masd:
        Mean Average Surface Distance (lower is better).
    cldice:
        Centre Line Dice coefficient in [0, 1] (higher is better).
    w_masd:
        Weight for the normalized MASD component.
    w_cldice:
        Weight for the clDice component.
    max_masd:
        Maximum MASD for normalization.

    Returns
    -------
    float
        Compound score in [0, 1]. Higher is better.
    """
    if math.isnan(cldice):
        return 0.0
    norm = normalize_masd(masd, max_masd=max_masd)
    result = w_masd * norm + w_cldice * cldice
    return max(0.0, min(1.0, result))


def compute_compound_nsd_cldice(
    *,
    nsd: float,
    cldice: float,
    w_nsd: float = 0.5,
    w_cldice: float = 0.5,
) -> float:
    """Compute compound metric: w_nsd * NSD + w_cldice * clDice.

    Both NSD and clDice are in [0, 1] with consistent scale, avoiding
    the range-collapse issue of the MASD+clDice compound metric.

    Parameters
    ----------
    nsd:
        Normalized Surface Dice in [0, 1] (higher is better).
    cldice:
        Centre Line Dice coefficient in [0, 1] (higher is better).
    w_nsd:
        Weight for the NSD component.
    w_cldice:
        Weight for the clDice component.

    Returns
    -------
    float
        Compound score in [0, 1]. Higher is better.
    """
    if math.isnan(nsd):
        nsd = 0.0
    if math.isnan(cldice):
        cldice = 0.0
    result = w_nsd * nsd + w_cldice * cldice
    return max(0.0, min(1.0, result))
