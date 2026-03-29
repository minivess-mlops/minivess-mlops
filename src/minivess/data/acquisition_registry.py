"""Dataset acquisition registry — metadata and availability checking.

Central registry of all 3 datasets with download methods, authentication
requirements, source formats, and human-readable download instructions.
# TubeNet excluded — olfactory bulb, different organ, only 1 2PM volume. See CLAUDE.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from minivess.config.acquisition_config import DatasetAcquisitionStatus

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetAcquisitionEntry:
    """Acquisition-specific metadata for a dataset.

    Attributes
    ----------
    name:
        Dataset identifier (matches ``KNOWN_DATASETS``).
    source_url:
        Download URL or landing page.
    download_method:
        How to acquire: ``"manual"``, ``"git_clone"``, ``"http_download"``,
        ``"api"``.
    requires_auth:
        Whether authentication/login is required.
    source_format:
        Native format: ``"nifti"``, ``"tiff"``, ``"ome_tiff"``.
    manual_instructions:
        Human-readable download steps for the PhD student.
    expected_checksums:
        Optional filename → SHA256 map for verification.
    """

    name: str
    source_url: str
    download_method: str
    requires_auth: bool
    source_format: str
    manual_instructions: str
    expected_checksums: dict[str, str] | None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ACQUISITION_REGISTRY: dict[str, DatasetAcquisitionEntry] = {
    "minivess": DatasetAcquisitionEntry(
        name="minivess",
        source_url="https://search.kg.ebrains.eu/instances/d05427da-de99-417c-8e78-6bdbfa3494da",
        download_method="manual",
        requires_auth=True,
        source_format="nifti",
        manual_instructions=(
            "1. Go to https://search.kg.ebrains.eu and search for 'MiniVess'\n"
            "2. Sign in with your EBRAINS account (register first if needed)\n"
            "3. Accept the data use agreement\n"
            "4. Download the NIfTI archive and extract to data/raw/minivess/"
        ),
        expected_checksums=None,
    ),
    "deepvess": DatasetAcquisitionEntry(
        name="deepvess",
        source_url=(
            "https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3"
        ),
        download_method="http_download",
        requires_auth=False,
        source_format="tiff",
        manual_instructions=(
            "Automated: Flow 0 downloads from Cornell eCommons bitstream API.\n"
            "Manual alternative:\n"
            "1. Go to https://ecommons.cornell.edu and search for 'DeepVess'\n"
            "2. Download HaftJavaherian_DeepVess2018_Images.zip (1.45 GB)\n"
            "3. Extract TIFF files to data/raw/deepvess/images/ and labels/\n"
            "4. Flow 0 will convert TIFF → NIfTI automatically"
        ),
        expected_checksums=None,
    ),
    # TubeNet excluded — olfactory bulb, different organ, only 1 2PM volume. See CLAUDE.md.
    "vesselnn": DatasetAcquisitionEntry(
        name="vesselnn",
        source_url="https://github.com/petteriTeikari/vesselNN",
        download_method="git_clone",
        requires_auth=False,
        source_format="nifti",
        manual_instructions=(
            "Automated: Flow 0 will run `git clone --depth 1` automatically.\n"
            "Manual alternative: git clone https://github.com/petteriTeikari/vesselNN"
        ),
        expected_checksums=None,
    ),
}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def check_dataset_availability(
    dataset_name: str,
    output_dir: Path,
) -> DatasetAcquisitionStatus:
    """Check if a dataset's files already exist on disk.

    Parameters
    ----------
    dataset_name:
        Name of the dataset (must be in ``ACQUISITION_REGISTRY``).
    output_dir:
        Directory where the dataset should exist.

    Returns
    -------
    ``READY`` if files exist, ``MANUAL_REQUIRED`` if missing,
    ``FAILED`` if dataset is unknown.
    """
    if dataset_name not in ACQUISITION_REGISTRY:
        logger.error("Unknown dataset: %s", dataset_name)
        return DatasetAcquisitionStatus.FAILED

    if not output_dir.is_dir():
        logger.info("Dataset '%s' not found at %s", dataset_name, output_dir)
        return DatasetAcquisitionStatus.MANUAL_REQUIRED

    # Check for images/ and labels/ subdirectories with at least one file
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        logger.info(
            "Dataset '%s' missing images/ or labels/ at %s", dataset_name, output_dir
        )
        return DatasetAcquisitionStatus.MANUAL_REQUIRED

    image_files = list(images_dir.iterdir())
    label_files = list(labels_dir.iterdir())

    if not image_files or not label_files:
        logger.info(
            "Dataset '%s' has empty images/ or labels/ at %s", dataset_name, output_dir
        )
        return DatasetAcquisitionStatus.MANUAL_REQUIRED

    logger.info(
        "Dataset '%s' is READY: %d images, %d labels at %s",
        dataset_name,
        len(image_files),
        len(label_files),
        output_dir,
    )
    return DatasetAcquisitionStatus.READY
