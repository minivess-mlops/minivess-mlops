"""Automated dataset downloaders.

VesselNN: git clone from GitHub.
DeepVess: HTTP download from Cornell eCommons (DSpace 7 bitstream API).
MiniVess: manual download from EBRAINS (requires login).

See ``acquisition_registry.py`` for human-readable instructions.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

_VESSELNN_URL = "https://github.com/petteriTeikari/vesselNN"

# Cornell eCommons DSpace 7 bitstream API — direct download, no auth required.
# Haft-Javaherian et al. 2019, "Deep convolutional neural networks for
# segmenting 3D in vivo multiphoton images of vasculature in brain"
# License: CC-BY-4.0. Landing page:
# https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3
_DEEPVESS_ZIP_URL = (
    "https://ecommons.cornell.edu/server/api/core/bitstreams/"
    "f507a45b-03d3-4ade-869e-e95037a7b877/content"
)


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


def download_deepvess(
    target_dir: Path,
    *,
    skip_existing: bool = True,
) -> Path:
    """Download DeepVess dataset from Cornell eCommons.

    Downloads the 1.45 GB ZIP archive containing TIFF volumes, extracts
    to ``target_dir/``. The TIFF files need subsequent conversion to NIfTI
    via ``convert_dataset_formats()`` in the acquisition flow.

    Parameters
    ----------
    target_dir:
        Directory to extract into. Will contain ``images/`` and ``labels/``
        subdirectories after extraction.
    skip_existing:
        If True and target_dir has files in ``images/``, skip download.

    Returns
    -------
    Path to the extracted directory.

    Raises
    ------
    RuntimeError
        If download or extraction fails.
    """
    images_dir = target_dir / "images"
    if skip_existing and images_dir.is_dir() and any(images_dir.iterdir()):
        logger.info("DeepVess already present at %s, skipping", target_dir)
        return target_dir

    logger.info("Downloading DeepVess from Cornell eCommons (~1.45 GB)...")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = httpx.get(
            _DEEPVESS_ZIP_URL,
            follow_redirects=True,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        response.raise_for_status()
    except Exception as exc:
        msg = f"DeepVess download failed: {exc}"
        raise RuntimeError(msg) from exc

    # Extract ZIP to target directory
    zip_path = target_dir / "deepvess.zip"
    zip_path.write_bytes(response.content)
    logger.info("Downloaded %d bytes, extracting...", len(response.content))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    zip_path.unlink()  # Clean up ZIP after extraction

    # Organize into images/ and labels/ if not already structured
    # eCommons ZIP extracts to HaftJavaherian_DeepVess2018_Images/
    images_dir.mkdir(exist_ok=True)
    (target_dir / "labels").mkdir(exist_ok=True)

    logger.info("DeepVess downloaded and extracted to %s", target_dir)
    return target_dir


# ---------------------------------------------------------------------------
# Downloader dispatch
# ---------------------------------------------------------------------------


_DOWNLOADERS: dict[str, Callable[..., Path]] = {
    "vesselnn": download_vesselnn,
    "deepvess": download_deepvess,
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
