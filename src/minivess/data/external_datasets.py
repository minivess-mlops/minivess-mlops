"""External test dataset configuration and discovery.

Manages external datasets for generalization assessment. Only multiphoton
and two-photon microscopy datasets of mouse brain vasculature are permitted.

No light-sheet, electron microscopy, CT, MRA, OCTA, or human vascular datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid modalities (strict filter)
# ---------------------------------------------------------------------------

VALID_MODALITIES = frozenset(
    {
        "multi-photon microscopy",
        "two-photon microscopy",
    }
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalDatasetConfig:
    """Configuration for an external test dataset.

    Attributes
    ----------
    name:
        Short identifier (e.g. ``"deepvess"``, ``"tubenet_2pm"``).
    source_url:
        Download URL for the dataset.
    modality:
        Imaging modality (must be in :data:`VALID_MODALITIES`).
    organ:
        Target organ (``"brain"``).
    species:
        Species (``"mouse"``).
    resolution_um:
        Voxel spacing in micrometers ``(x, y, z)``.
    n_volumes:
        Number of raw volumes available at source.
    license:
        Dataset license (``"TBD"`` if unverified).
    cite_ref:
        Citation reference key for bibliography.
    """

    name: str
    source_url: str
    modality: str
    organ: str
    species: str
    resolution_um: tuple[float, float, float]
    n_volumes: int
    license: str
    cite_ref: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


EXTERNAL_DATASETS: dict[str, ExternalDatasetConfig] = {
    "deepvess": ExternalDatasetConfig(
        name="deepvess",
        source_url=(
            "https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3"
        ),
        modality="multi-photon microscopy",
        organ="brain",
        species="mouse",
        resolution_um=(1.0, 1.0, 1.7),
        n_volumes=1,
        license="TBD",
        cite_ref="haft_javaherian_2019_deepvess",
    ),
    "tubenet_2pm": ExternalDatasetConfig(
        name="tubenet_2pm",
        source_url=(
            "https://rdr.ucl.ac.uk/articles/dataset/"
            "3D_Microvascular_Image_Data_and_Labels_"
            "for_Machine_Learning/25715604"
        ),
        modality="two-photon microscopy",
        organ="brain",
        species="mouse",
        resolution_um=(0.20, 0.46, 5.20),
        n_volumes=1,
        license="TBD",
        cite_ref="holroyd_2025_tubenet",
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_external_config(config: ExternalDatasetConfig) -> list[str]:
    """Validate an external dataset config against domain constraints.

    Parameters
    ----------
    config:
        The config to validate.

    Returns
    -------
    List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if config.modality not in VALID_MODALITIES:
        errors.append(
            f"Invalid modality '{config.modality}'. "
            f"Must be one of: {sorted(VALID_MODALITIES)}"
        )

    if config.species != "mouse":
        errors.append(f"Invalid species '{config.species}'. Must be 'mouse'.")

    if config.organ != "brain":
        errors.append(f"Invalid organ '{config.organ}'. Must be 'brain'.")

    if len(config.resolution_um) != 3:
        errors.append(f"Resolution must be 3-tuple, got {len(config.resolution_um)}")
    elif any(v <= 0 for v in config.resolution_um):
        errors.append("Resolution values must be positive")

    if not config.source_url:
        errors.append("Source URL must not be empty")

    return errors


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Supported image extensions for external datasets
_IMAGE_EXTENSIONS = frozenset(
    {
        ".nii.gz",
        ".nii",
        ".tif",
        ".tiff",
    }
)


def _get_extension(path: Path) -> str:
    """Get file extension, handling .nii.gz as a single extension."""
    name = path.name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def discover_external_test_pairs(
    data_dir: Path,
    dataset_name: str,
) -> list[dict[str, str]]:
    """Discover image/label pairs for an external dataset.

    Expects directory structure::

        data_dir/
            images/
                vol_001.nii.gz
                vol_002.nii.gz
            labels/
                vol_001.nii.gz
                vol_002.nii.gz

    Parameters
    ----------
    data_dir:
        Root directory for the dataset.
    dataset_name:
        Name of the dataset (for logging).

    Returns
    -------
    List of dicts with ``"image"`` and ``"label"`` paths as strings.
    Returns empty list if directory is missing or has no pairs.
    """
    if not data_dir.is_dir():
        logger.warning("Data directory does not exist: %s", data_dir)
        return []

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.is_dir():
        logger.warning("No images/ subdirectory in %s", data_dir)
        return []

    if not labels_dir.is_dir():
        logger.warning("No labels/ subdirectory in %s", data_dir)
        return []

    # Collect all image files
    image_files = sorted(
        f
        for f in images_dir.iterdir()
        if f.is_file() and _get_extension(f) in _IMAGE_EXTENSIONS
    )

    pairs: list[dict[str, str]] = []
    for img_file in image_files:
        # Find matching label with same name
        label_file = labels_dir / img_file.name
        if label_file.is_file():
            pairs.append(
                {
                    "image": str(img_file),
                    "label": str(label_file),
                }
            )
        else:
            logger.debug(
                "No matching label for %s in %s",
                img_file.name,
                dataset_name,
            )

    logger.info("Discovered %d image/label pairs for %s", len(pairs), dataset_name)
    return pairs
