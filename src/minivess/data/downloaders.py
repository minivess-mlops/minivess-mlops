"""Automated dataset downloaders.

Only VesselNN supports fully automated download (git clone).
Other datasets require manual download — see ``acquisition_registry.py``
for human-readable instructions.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

_VESSELNN_URL = "https://github.com/petteriTeikari/vesselNN"


def download_vesselnn(
    target_dir: Path,
    *,
    skip_existing: bool = True,
) -> Path:
    """Download VesselNN dataset via git clone.

    Parameters
    ----------
    target_dir:
        Directory to clone into.
    skip_existing:
        If True and target_dir contains a ``.git`` directory, skip.

    Returns
    -------
    Path to the cloned directory.

    Raises
    ------
    RuntimeError
        If git clone fails.
    """
    if skip_existing and (target_dir / ".git").is_dir():
        logger.info("VesselNN already cloned at %s, skipping", target_dir)
        return target_dir

    logger.info("Cloning VesselNN to %s", target_dir)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _VESSELNN_URL, str(target_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        msg = f"git clone failed (exit {result.returncode}): {result.stderr}"
        raise RuntimeError(msg)

    logger.info("VesselNN cloned successfully to %s", target_dir)
    return target_dir


# ---------------------------------------------------------------------------
# Downloader dispatch
# ---------------------------------------------------------------------------


_DOWNLOADERS: dict[str, Callable[..., Path]] = {
    "vesselnn": download_vesselnn,
}


def get_downloader(dataset_name: str) -> Callable[..., Path] | None:
    """Return the automated downloader for a dataset, or None if manual.

    Parameters
    ----------
    dataset_name:
        Dataset identifier.

    Returns
    -------
    Callable that downloads the dataset, or ``None`` if manual download required.
    """
    return _DOWNLOADERS.get(dataset_name)
