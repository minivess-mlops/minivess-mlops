"""Multi-format figure export with optional JSON data.

Exports PNG (300 DPI), SVG (vector), EPS (LaTeX) with optional
reproducibility JSON alongside each figure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

DEFAULT_FORMATS = ("png",)


def save_figure(
    fig: Figure,
    name: str,
    output_dir: Path | None = None,
    formats: list[str] | None = None,
    data: dict[str, Any] | None = None,
) -> Path | None:
    """Save figure in multiple formats with optional data export.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    name:
        Base filename (without extension).
    output_dir:
        Output directory. Defaults to current working directory.
    formats:
        List of formats (``"png"``, ``"svg"``, ``"eps"``).
        Defaults to ``["png"]``.
    data:
        Optional reproducibility data to save as JSON alongside the figure.

    Returns
    -------
    Path to the first saved file, or ``None`` if nothing was saved.
    """
    if output_dir is None:
        from pathlib import Path as _Path

        output_dir = _Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    fmt_list = formats if formats else list(DEFAULT_FORMATS)

    first_path: Path | None = None

    for fmt in fmt_list:
        out_path = output_dir / f"{name}.{fmt}"
        fig.savefig(str(out_path), format=fmt, dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", out_path)
        if first_path is None:
            first_path = out_path

    if data is not None:
        json_path = output_dir / f"{name}.json"
        json_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved figure data: %s", json_path)

    return first_path
