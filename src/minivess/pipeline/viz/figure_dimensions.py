"""Preset-based figure dimension management.

No hardcoded figsizes anywhere in viz code — all presets come from here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Presets: (width_inches, height_inches)
# ---------------------------------------------------------------------------

FIGURE_DIMENSIONS: dict[str, tuple[float, float]] = {
    "single": (8.0, 6.0),
    "double": (14.0, 6.0),
    "triple": (18.0, 6.0),
    "matrix": (10.0, 8.0),
    "forest": (10.0, 12.0),
    "specification_curve": (16.0, 12.0),
}


def get_figsize(preset: str) -> tuple[float, float]:
    """Return figure dimensions for a named preset.

    Parameters
    ----------
    preset:
        Preset name (e.g. ``"single"``, ``"double"``, ``"matrix"``).

    Returns
    -------
    ``(width, height)`` in inches.

    Raises
    ------
    KeyError:
        If the preset name is not found.
    """
    return FIGURE_DIMENSIONS[preset]
