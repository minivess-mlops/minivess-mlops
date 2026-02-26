from __future__ import annotations

import math


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
