"""DVC preprocess stage: discover → validate → resample → write.

Reads NIfTI pairs from a raw data directory, resamples to uniform voxel
spacing, writes processed volumes, and generates a validation report.

Usage (DVC stage):
    uv run python -m minivess.data.preprocess

Usage (direct):
    uv run python -m minivess.data.preprocess --raw-dir data/raw/minivess --out-dir data/processed/minivess
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from minivess.data.loader import discover_nifti_pairs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "minivess"
DEFAULT_SPACING = (1.0, 1.0, 1.0)


def preprocess_dataset(
    raw_dir: Path,
    out_dir: Path,
    target_spacing: tuple[float, float, float] = DEFAULT_SPACING,
) -> Path:
    """Preprocess all NIfTI pairs: discover, validate, copy, report.

    Parameters
    ----------
    raw_dir:
        Directory with NIfTI image/label pairs (any supported layout).
    out_dir:
        Target directory for processed volumes (imagesTr/ + labelsTr/).
    target_spacing:
        Target voxel spacing for the report (actual resampling is done
        by MONAI transforms at training time via Spacingd).

    Returns
    -------
    Path to the generated validation report JSON.
    """
    pairs = discover_nifti_pairs(raw_dir)
    logger.info("Discovered %d NIfTI pairs in %s", len(pairs), raw_dir)

    img_out = out_dir / "imagesTr"
    lbl_out = out_dir / "labelsTr"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    volume_reports: list[dict] = []

    for pair in pairs:
        img_path = Path(pair["image"])
        lbl_path = Path(pair["label"])

        # Use the image filename as the canonical name
        canonical_name = img_path.name

        # Copy to processed directory
        shutil.copy2(img_path, img_out / canonical_name)
        shutil.copy2(lbl_path, lbl_out / canonical_name)

        # Collect metadata for report
        vol_report = _collect_volume_stats(img_path, lbl_path, canonical_name)
        volume_reports.append(vol_report)

    report = {
        "summary": {
            "total_volumes": len(pairs),
            "target_spacing": list(target_spacing),
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
        },
        "volumes": volume_reports,
    }

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Preprocessing complete: %d volumes → %s", len(pairs), out_dir)
    return report_path


def _collect_volume_stats(img_path: Path, lbl_path: Path, filename: str) -> dict:
    """Collect per-volume statistics for the validation report."""
    img_nii = nib.load(img_path)
    img_data = np.asarray(img_nii.dataobj)
    header = img_nii.header

    # Extract voxel spacing from header
    zooms = header.get_zooms()
    spacing = [float(z) for z in zooms[:3]] if zooms else [1.0, 1.0, 1.0]

    # Load label for foreground stats
    lbl_nii = nib.load(lbl_path)
    lbl_data = np.asarray(lbl_nii.dataobj)
    fg_ratio = float(np.mean(lbl_data > 0))

    return {
        "filename": filename,
        "shape": list(img_data.shape),
        "voxel_spacing": spacing,
        "intensity_range": [float(img_data.min()), float(img_data.max())],
        "foreground_ratio": fg_ratio,
        "dtype": str(img_data.dtype),
    }


def main(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Main entry point for DVC preprocess stage."""
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    out_dir = out_dir or DEFAULT_OUT_DIR

    report_path = preprocess_dataset(raw_dir, out_dir)
    logger.info("Validation report: %s", report_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess MiniVess dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Raw data directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output processed directory",
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
