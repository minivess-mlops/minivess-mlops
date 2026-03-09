"""Download and organise the MiniVess dataset from EBRAINS.

Thin CLI wrapper around ``minivess.data.downloader``.

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

import argparse
import logging
from pathlib import Path

from minivess.data.downloader import (
    EBRAINS_DATASET_ID,
    download_minivess,
    extract_and_reorganise,
    is_dataset_ready,
    reorganise_ebrains_to_loader,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"
DEFAULT_ZIP_PATH = PROJECT_ROOT / "dataset_local" / f"d-{EBRAINS_DATASET_ID}.zip"

# Re-export for backward compatibility
__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_ZIP_PATH",
    "download_minivess",
    "extract_and_reorganise",
    "is_dataset_ready",
    "reorganise_ebrains_to_loader",
]


def main(output_dir: Path | None = None, zip_path: Path | None = None) -> None:
    """Main entry point: download and organise MiniVess dataset."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    zip_path = zip_path or DEFAULT_ZIP_PATH
    download_minivess(output_dir, zip_path=zip_path)


if __name__ == "__main__":
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
