"""External test dataset configuration and discovery.

Manages external datasets for generalization assessment. Only multiphoton
and two-photon microscopy datasets of mouse brain vasculature are permitted.

No light-sheet, electron microscopy, CT, MRA, OCTA, or human vascular datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies

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
    license_verified: bool = True


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
        license="eCommons-educational",
        cite_ref="haft_javaherian_2019_deepvess",
        license_verified=False,
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
        license="CC-BY-4.0",
        cite_ref="holroyd_2025_tubenet",
        license_verified=True,
    ),
    "vesselnn": ExternalDatasetConfig(
        name="vesselnn",
        source_url="https://github.com/petteriTeikari/vesselNN",
        modality="two-photon microscopy",
        organ="brain",
        species="mouse",
        resolution_um=(0.50, 0.50, 1.0),
        n_volumes=12,
        license="MIT",
        cite_ref="teikari_2016_vesselnn",
        license_verified=True,
    ),
}


# ---------------------------------------------------------------------------
# DVC tracking configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetDVCConfig:
    """DVC tracking configuration for an external dataset.

    Attributes
    ----------
    name:
        Dataset identifier (matches :data:`EXTERNAL_DATASETS` keys).
    dvc_path:
        Relative path to the ``.dvc`` tracking file.
    git_tag_format:
        Git tag format string with ``{version}`` placeholder.
    version:
        Current tracked version (``"0.0.0"`` if not yet tracked).
    """

    name: str
    dvc_path: str
    git_tag_format: str
    version: str = "0.0.0"


DVC_CONFIGS: dict[str, DatasetDVCConfig] = {
    "deepvess": DatasetDVCConfig(
        name="deepvess",
        dvc_path="data/external/deepvess.dvc",
        git_tag_format="data/deepvess/v{version}",
    ),
    "tubenet_2pm": DatasetDVCConfig(
        name="tubenet_2pm",
        dvc_path="data/external/tubenet_2pm.dvc",
        git_tag_format="data/tubenet_2pm/v{version}",
    ),
    "vesselnn": DatasetDVCConfig(
        name="vesselnn",
        dvc_path="data/external/vesselnn.dvc",
        git_tag_format="data/vesselnn/v{version}",
    ),
}


def get_dvc_tracked_datasets() -> list[str]:
    """Return list of dataset names that have DVC tracking configured."""
    return sorted(DVC_CONFIGS.keys())


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


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_external_dataset(
    dataset_name: str,
    target_dir: Path,
) -> Path:
    """Prepare directory structure for an external dataset.

    Creates the ``images/`` and ``labels/`` subdirectories. Actual data
    download must be done manually — this function logs the source URL.

    Parameters
    ----------
    dataset_name:
        Name of the dataset (must be in :data:`EXTERNAL_DATASETS`).
    target_dir:
        Root directory for the downloaded dataset.

    Returns
    -------
    Path to the created target directory.

    Raises
    ------
    KeyError
        If ``dataset_name`` is not in the registry.
    """
    config = EXTERNAL_DATASETS[dataset_name]  # raises KeyError if missing

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "images").mkdir(exist_ok=True)
    (target_dir / "labels").mkdir(exist_ok=True)

    logger.warning(
        "Manual download required for '%s'. Source: %s",
        dataset_name,
        config.source_url,
    )
    return target_dir
