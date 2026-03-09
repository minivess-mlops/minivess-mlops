"""MiniVess dataset download and organisation — importable module.

Extracted from ``scripts/download_minivess.py`` to be importable by
the acquisition flow (Flow 0). The script becomes a thin CLI wrapper.

The MiniVess dataset (Poon et al., 2023) contains 70 two-photon fluorescence
microscopy volumes of rodent cerebrovasculature.

Three data sources (checked in order):
1. Already-organised data in output_dir → skip (idempotent)
2. Local ZIP in dataset_local/ → extract + reorganise
3. EBRAINS data-proxy API → download + reorganise (requires httpx)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

EBRAINS_DATASET_ID = "bf268b89-1420-476b-b428-b85a913eb523"
EBRAINS_API_BASE = f"https://data-proxy.ebrains.eu/api/v1/datasets/{EBRAINS_DATASET_ID}"
EXPECTED_VOLUMES = 70


def is_dataset_ready(data_dir: Path, expected_volumes: int = EXPECTED_VOLUMES) -> bool:
    """Check if the dataset is already organised and complete.

    Checks for either imagesTr/labelsTr or raw/seg layouts.

    Parameters
    ----------
    data_dir:
        Directory containing the dataset.
    expected_volumes:
        Minimum number of volumes required (default: 70).

    Returns
    -------
    True if dataset is complete and ready.
    """
    # Check imagesTr/labelsTr layout
    img_dir = data_dir / "imagesTr"
    lbl_dir = data_dir / "labelsTr"
    if img_dir.exists() and lbl_dir.exists():
        n_images = len(list(img_dir.glob("*.nii.gz")))
        n_labels = len(list(lbl_dir.glob("*.nii.gz")))
        if n_images >= expected_volumes and n_labels >= expected_volumes:
            return True

    # Check raw/seg layout
    raw_dir = data_dir / "raw"
    seg_dir = data_dir / "seg"
    if raw_dir.exists() and seg_dir.exists():
        n_raw = len(list(raw_dir.glob("*.nii.gz")))
        n_seg = len(list(seg_dir.glob("*.nii.gz")))
        if n_raw >= expected_volumes and n_seg >= expected_volumes:
            return True

    return False


def reorganise_ebrains_to_loader(
    ebrains_dir: Path,
    output_dir: Path,
) -> None:
    """Reorganise EBRAINS raw/seg layout into imagesTr/labelsTr for MONAI.

    EBRAINS layout::

        raw / mv01.nii.gz, seg / mv01_y.nii.gz, json / mv01.json

    Loader layout::

        imagesTr / mv01.nii.gz, labelsTr / mv01.nii.gz, metadata / mv01.json

    Labels are renamed: ``mv01_y.nii.gz → mv01.nii.gz`` (strip _y suffix).

    Parameters
    ----------
    ebrains_dir:
        Directory containing raw/, seg/, and optionally json/ subdirectories.
    output_dir:
        Target directory for imagesTr/, labelsTr/, metadata/.

    Raises
    ------
    FileNotFoundError
        If raw/ or seg/ subdirectories are missing.
    """
    raw_dir = ebrains_dir / "raw"
    seg_dir = ebrains_dir / "seg"

    if not raw_dir.exists():
        msg = f"raw/ directory not found in {ebrains_dir}"
        raise FileNotFoundError(msg)
    if not seg_dir.exists():
        msg = f"seg/ directory not found in {ebrains_dir}"
        raise FileNotFoundError(msg)

    img_out = output_dir / "imagesTr"
    lbl_out = output_dir / "labelsTr"
    meta_out = output_dir / "metadata"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    # Copy images (raw/mvXX.nii.gz → imagesTr/mvXX.nii.gz)
    for raw_file in sorted(raw_dir.glob("*.nii.gz")):
        shutil.copy2(raw_file, img_out / raw_file.name)

    # Copy and rename labels (seg/mvXX_y.nii.gz → labelsTr/mvXX.nii.gz)
    for seg_file in sorted(seg_dir.glob("*.nii.gz")):
        # Strip _y suffix: mv01_y.nii.gz → mv01.nii.gz
        new_name = seg_file.name.replace("_y.nii.gz", ".nii.gz")
        shutil.copy2(seg_file, lbl_out / new_name)

    # Copy metadata if present
    json_dir = ebrains_dir / "json"
    if json_dir.exists():
        for json_file in sorted(json_dir.glob("*.json")):
            shutil.copy2(json_file, meta_out / json_file.name)

    n_images = len(list(img_out.glob("*.nii.gz")))
    n_labels = len(list(lbl_out.glob("*.nii.gz")))
    logger.info(
        "Reorganised: %d images, %d labels in %s", n_images, n_labels, output_dir
    )


def extract_and_reorganise(
    zip_path: Path,
    output_dir: Path,
) -> None:
    """Extract EBRAINS ZIP and reorganise into loader-expected structure.

    Parameters
    ----------
    zip_path:
        Path to the downloaded EBRAINS ZIP file.
    output_dir:
        Target directory for imagesTr/, labelsTr/, metadata/.

    Raises
    ------
    FileNotFoundError
        If zip_path does not exist.
    """
    if not zip_path.exists():
        msg = f"ZIP file not found: {zip_path}"
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger.info("Extracting %s to temporary directory...", zip_path.name)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        reorganise_ebrains_to_loader(tmp_path, output_dir)


def download_from_ebrains(output_dir: Path) -> None:
    """Download individual files from the EBRAINS data-proxy API.

    Falls back to this when no local ZIP is available.
    Uses httpx for downloads with progress reporting.

    Parameters
    ----------
    output_dir:
        Target directory for reorganised NIfTI files.

    Raises
    ------
    ImportError
        If httpx is not installed.
    """
    try:
        import httpx
    except ImportError:
        msg = (
            "httpx is required for downloading. Install with: uv add httpx\n"
            "Alternatively, download the ZIP manually from EBRAINS and place "
            "it in dataset_local/"
        )
        raise ImportError(msg) from None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        seg_dir = tmp_path / "seg"
        json_dir = tmp_path / "json"
        raw_dir.mkdir()
        seg_dir.mkdir()
        json_dir.mkdir()

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            for i in range(1, EXPECTED_VOLUMES + 1):
                vol_id = f"mv{i:02d}"

                # Download raw image
                _download_file(
                    client,
                    f"{EBRAINS_API_BASE}/raw/{vol_id}.nii.gz",
                    raw_dir / f"{vol_id}.nii.gz",
                )

                # Download segmentation
                _download_file(
                    client,
                    f"{EBRAINS_API_BASE}/seg/{vol_id}_y.nii.gz",
                    seg_dir / f"{vol_id}_y.nii.gz",
                )

                # Download metadata
                _download_file(
                    client,
                    f"{EBRAINS_API_BASE}/json/{vol_id}.json",
                    json_dir / f"{vol_id}.json",
                )

                if i % 10 == 0:
                    logger.info("Downloaded %d/%d volumes", i, EXPECTED_VOLUMES)

        reorganise_ebrains_to_loader(tmp_path, output_dir)


def _download_file(client: object, url: str, output_path: Path) -> None:
    """Download a single file with httpx.

    Parameters
    ----------
    client:
        httpx.Client instance.
    url:
        URL to download from.
    output_path:
        Local path to write the downloaded content to.
    """
    import httpx

    assert isinstance(client, httpx.Client)
    response = client.get(url)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def download_minivess(
    output_dir: Path,
    zip_path: Path | None = None,
) -> bool:
    """Download and organise MiniVess dataset.

    Checks three sources in order:
    1. Already organised → skip (returns True)
    2. Local ZIP → extract + reorganise
    3. EBRAINS API → download + reorganise

    Parameters
    ----------
    output_dir:
        Target directory for the organised dataset.
    zip_path:
        Optional path to a local EBRAINS ZIP file.

    Returns
    -------
    True if dataset is ready after this call.
    """
    if is_dataset_ready(output_dir):
        logger.info("Dataset already ready at %s — skipping", output_dir)
        return True

    if zip_path is not None and zip_path.exists():
        logger.info("Found local ZIP: %s — extracting", zip_path)
        extract_and_reorganise(zip_path, output_dir)
        return is_dataset_ready(output_dir)

    logger.info("No local data found — downloading from EBRAINS API...")
    download_from_ebrains(output_dir)

    ready = is_dataset_ready(output_dir)
    if ready:
        logger.info("Dataset ready at %s (%d volumes)", output_dir, EXPECTED_VOLUMES)
    else:
        logger.error("Dataset incomplete after download — check logs for errors")
    return ready
