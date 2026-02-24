"""Download and organise the MiniVess dataset from EBRAINS.

The MiniVess dataset (Poon et al., 2023) contains 70 two-photon fluorescence
microscopy volumes of rodent cerebrovasculature. Public access, no auth required.

Usage:
    uv run python scripts/download_minivess.py [--output-dir data/raw/minivess]

Three data sources (checked in order):
1. Already-organised data in output_dir → skip (idempotent)
2. Local ZIP in dataset_local/ → extract + reorganise
3. EBRAINS data-proxy API → download + reorganise
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"
DEFAULT_ZIP_PATH = (
    PROJECT_ROOT / "dataset_local" / f"d-{EBRAINS_DATASET_ID}.zip"
)


def is_dataset_ready(data_dir: Path, expected_volumes: int = EXPECTED_VOLUMES) -> bool:
    """Check if the dataset is already organised and complete.

    Checks for either imagesTr/labelsTr or raw/seg layouts.
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

    EBRAINS layout:
        raw/mv01.nii.gz, seg/mv01_y.nii.gz, json/mv01.json

    Loader layout:
        imagesTr/mv01.nii.gz, labelsTr/mv01.nii.gz, metadata/mv01.json

    Labels are renamed: mv01_y.nii.gz → mv01.nii.gz (strip _y suffix).

    Parameters
    ----------
    ebrains_dir:
        Directory containing raw/, seg/, and optionally json/ subdirectories.
    output_dir:
        Target directory for imagesTr/, labelsTr/, metadata/.
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
    Uses httpx for async downloads with progress reporting.
    """
    try:
        import httpx
    except ImportError:
        msg = (
            "httpx is required for downloading. Install with: uv add httpx\n"
            "Alternatively, download the ZIP manually from EBRAINS and place it at:\n"
            f"  {DEFAULT_ZIP_PATH}"
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
    """Download a single file with progress reporting."""
    import httpx

    assert isinstance(client, httpx.Client)
    response = client.get(url)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main(output_dir: Path | None = None, zip_path: Path | None = None) -> None:
    """Main entry point: download and organise MiniVess dataset.

    Checks three sources in order:
    1. Already organised → skip
    2. Local ZIP → extract
    3. EBRAINS API → download
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    zip_path = zip_path or DEFAULT_ZIP_PATH

    if is_dataset_ready(output_dir):
        logger.info("Dataset already ready at %s — skipping", output_dir)
        return

    if zip_path.exists():
        logger.info("Found local ZIP: %s — extracting", zip_path)
        extract_and_reorganise(zip_path, output_dir)
        return

    logger.info("No local data found — downloading from EBRAINS API...")
    download_from_ebrains(output_dir)

    if is_dataset_ready(output_dir):
        logger.info("Dataset ready at %s (%d volumes)", output_dir, EXPECTED_VOLUMES)
    else:
        logger.error("Dataset incomplete after download — check logs for errors")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download MiniVess dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for organised dataset",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help="Path to local EBRAINS ZIP file",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, zip_path=args.zip_path)
