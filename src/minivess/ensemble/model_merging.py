"""Training-free model merging utilities.

Provides linear interpolation, SLERP (Spherical Linear Interpolation),
and layer-wise merging of state dicts. All methods operate on state_dict
level and are model-agnostic.

References:
    - Wortsman et al. (2022), "Model Soups"
    - Yang et al. (2025), "MedSAMix"
    - Goddard et al. (2024), "Arcee's MergeKit" (SLERP for LLMs)
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def linear_merge(
    sd1: dict[str, Tensor],
    sd2: dict[str, Tensor],
    *,
    t: float = 0.5,
) -> dict[str, Tensor]:
    """Linear (weighted average) merge of two state dicts.

    merged[key] = (1 - t) * sd1[key] + t * sd2[key]

    Parameters
    ----------
    sd1, sd2:
        State dicts with identical keys and shapes.
    t:
        Interpolation weight in [0, 1]. t=0 → sd1, t=1 → sd2.

    Returns
    -------
    Merged state dict.
    """
    merged: dict[str, Tensor] = {}
    for key in sd1:
        if sd1[key].is_floating_point():
            merged[key] = (1.0 - t) * sd1[key] + t * sd2[key]
        else:
            merged[key] = sd1[key].clone()
    return merged


def _slerp_tensor(v1: Tensor, v2: Tensor, t: float) -> Tensor:
    """SLERP interpolation between two flattened tensors.

    Falls back to linear interpolation when vectors are (near-)parallel.
    """
    v1_flat = v1.float().flatten()
    v2_flat = v2.float().flatten()

    # Normalize
    norm1 = v1_flat.norm()
    norm2 = v2_flat.norm()

    if norm1 < 1e-8 or norm2 < 1e-8:
        # Degenerate: fall back to linear
        return ((1.0 - t) * v1 + t * v2).to(v1.dtype)

    v1_unit = v1_flat / norm1
    v2_unit = v2_flat / norm2

    # Cosine of angle between vectors
    cos_omega = torch.clamp(torch.dot(v1_unit, v2_unit), -1.0, 1.0).item()
    omega = math.acos(cos_omega)

    if abs(omega) < 1e-6:
        # Nearly parallel: linear interpolation
        return ((1.0 - t) * v1 + t * v2).to(v1.dtype)

    sin_omega = math.sin(omega)
    coeff1 = math.sin((1.0 - t) * omega) / sin_omega
    coeff2 = math.sin(t * omega) / sin_omega

    # Interpolate in original (non-normalized) space, preserving magnitude
    magnitude = (1.0 - t) * norm1.item() + t * norm2.item()
    result_flat = coeff1 * v1_unit + coeff2 * v2_unit
    result_flat = result_flat * magnitude

    merged: Tensor = result_flat.reshape(v1.shape).to(v1.dtype)
    return merged


def slerp_merge(
    sd1: dict[str, Tensor],
    sd2: dict[str, Tensor],
    *,
    t: float = 0.5,
) -> dict[str, Tensor]:
    """SLERP (Spherical Linear Interpolation) merge of two state dicts.

    Parameters
    ----------
    sd1, sd2:
        State dicts with identical keys and shapes.
    t:
        Interpolation weight in [0, 1]. t=0 → sd1, t=1 → sd2.

    Returns
    -------
    Merged state dict.
    """
    merged: dict[str, Tensor] = {}
    for key in sd1:
        if sd1[key].is_floating_point():
            merged[key] = _slerp_tensor(sd1[key], sd2[key], t)
        else:
            merged[key] = sd1[key].clone()
    return merged


def layer_wise_merge(
    sd1: dict[str, Tensor],
    sd2: dict[str, Tensor],
    *,
    layer_weights: dict[str, float],
    method: str = "linear",
) -> dict[str, Tensor]:
    """Layer-wise merge with per-layer interpolation weights.

    Parameters
    ----------
    sd1, sd2:
        State dicts with identical keys and shapes.
    layer_weights:
        Dict mapping parameter names to interpolation weights t in [0, 1].
        Missing keys default to 0.5.
    method:
        Interpolation method: "linear" or "slerp".

    Returns
    -------
    Merged state dict.
    """
    merge_fn = _slerp_tensor if method == "slerp" else None
    merged: dict[str, Tensor] = {}

    for key in sd1:
        if not sd1[key].is_floating_point():
            merged[key] = sd1[key].clone()
            continue

        t = layer_weights.get(key, 0.5)
        if merge_fn is not None:
            merged[key] = merge_fn(sd1[key], sd2[key], t)
        else:
            merged[key] = (1.0 - t) * sd1[key] + t * sd2[key]

    return merged
