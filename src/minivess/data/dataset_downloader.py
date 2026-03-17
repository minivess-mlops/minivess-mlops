"""Universal dataset downloader — cloud-agnostic download and upload (T-A1).

Supports downloading VesselNN from GitHub and other external datasets,
checksum verification, manifest generation, and upload to any cloud
backend (GCS, S3, or local).

Usage:
    downloader = DatasetDownloader()
    downloader.prepare_download_dir("vesselnn", Path("data/external/vesselnn"))
    manifest = downloader.generate_manifest(Path("data/external/vesselnn"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from minivess.data.external_datasets import EXTERNAL_DATASETS

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.data.external_datasets import ExternalDatasetConfig

logger = logging.getLogger(__name__)

# Supported image extensions for scanning directories
_IMAGE_EXTENSIONS = frozenset({".nii.gz", ".nii", ".tif", ".tiff"})


def _get_extension(path: Path) -> str:
    """Get file extension, handling .nii.gz as a single extension."""
    name = path.name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


class DatasetDownloader:
    """Cloud-agnostic dataset downloader and manager.

    Wraps the EXTERNAL_DATASETS registry with download directory
    preparation, status tracking, checksum verification, and
    manifest generation.
    """

    def __init__(self) -> None:
        self._registry = EXTERNAL_DATASETS

    def available_datasets(self) -> list[str]:
        """List all registered external datasets.

        Returns
        -------
        Sorted list of dataset names.
        """
        return sorted(self._registry.keys())

    def get_dataset_info(self, dataset_name: str) -> ExternalDatasetConfig:
        """Get configuration for a registered dataset.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.

        Returns
        -------
        ExternalDatasetConfig for the dataset.

        Raises
        ------
        KeyError
            If dataset is not in the registry.
        """
        return self._registry[dataset_name]

    def prepare_download_dir(self, dataset_name: str, target_dir: Path) -> Path:
        """Create directory structure for a dataset download.

        Creates ``images/`` and ``labels/`` subdirectories.

        Parameters
        ----------
        dataset_name:
            Name of the dataset (must be in registry).
        target_dir:
            Root directory for the download.

        Returns
        -------
        Path to the created target directory.
        """
        # Validate dataset exists
        _ = self._registry[dataset_name]

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "images").mkdir(exist_ok=True)
        (target_dir / "labels").mkdir(exist_ok=True)

        logger.info(
            "Prepared download directory for %s at %s", dataset_name, target_dir
        )
        return target_dir

    def get_download_status(
        self, dataset_name: str, target_dir: Path
    ) -> dict[str, Any]:
        """Check download status for a dataset.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.
        target_dir:
            Root directory for the download.

        Returns
        -------
        Dict with keys: complete, n_images, n_labels, expected_volumes.
        """
        config = self._registry[dataset_name]
        images_dir = target_dir / "images"
        labels_dir = target_dir / "labels"

        n_images = _count_image_files(images_dir) if images_dir.is_dir() else 0
        n_labels = _count_image_files(labels_dir) if labels_dir.is_dir() else 0

        return {
            "complete": n_images >= config.n_volumes and n_labels >= config.n_volumes,
            "n_images": n_images,
            "n_labels": n_labels,
            "expected_volumes": config.n_volumes,
        }

    def generate_manifest(self, target_dir: Path) -> dict[str, Any]:
        """Generate a download manifest with checksums.

        Scans images/ and labels/ subdirectories and computes
        SHA-256 checksums for all files.

        Parameters
        ----------
        target_dir:
            Root directory containing images/ and labels/.

        Returns
        -------
        Dict with 'files' list, each containing path, checksum, size.
        """
        files: list[dict[str, Any]] = []

        for subdir_name in ("images", "labels"):
            subdir = target_dir / subdir_name
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.is_file() and _get_extension(f) in _IMAGE_EXTENSIONS:
                    files.append(
                        {
                            "path": str(f.relative_to(target_dir)),
                            "checksum": compute_file_checksum(f),
                            "size": f.stat().st_size,
                        }
                    )

        return {"files": files}

    def verify_manifest(self, target_dir: Path, manifest: dict[str, Any]) -> list[str]:
        """Verify files against a manifest.

        Parameters
        ----------
        target_dir:
            Root directory containing the files.
        manifest:
            Manifest dict from generate_manifest().

        Returns
        -------
        List of error messages (empty if all files verify).
        """
        errors: list[str] = []

        for entry in manifest.get("files", []):
            file_path = target_dir / entry["path"]
            if not file_path.exists():
                errors.append(f"Missing file: {entry['path']}")
                continue

            actual_checksum = compute_file_checksum(file_path)
            if actual_checksum != entry["checksum"]:
                errors.append(
                    f"Checksum mismatch for {entry['path']}: "
                    f"expected {entry['checksum']}, got {actual_checksum}"
                )

            actual_size = file_path.stat().st_size
            if actual_size != entry["size"]:
                errors.append(
                    f"Size mismatch for {entry['path']}: "
                    f"expected {entry['size']}, got {actual_size}"
                )

        return errors

    def save_manifest(self, manifest: dict[str, Any], path: Path) -> Path:
        """Save manifest to a JSON file.

        Parameters
        ----------
        manifest:
            Manifest dict from generate_manifest().
        path:
            Output path for the JSON file.

        Returns
        -------
        Path to the saved manifest file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Manifest saved to %s", path)
        return path


def _count_image_files(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.is_dir():
        return 0
    return sum(
        1
        for f in directory.iterdir()
        if f.is_file() and _get_extension(f) in _IMAGE_EXTENSIONS
    )


def compute_file_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    Hex digest of the SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_upload_target(target: str) -> dict[str, str]:
    """Parse a cloud upload target URI.

    Supports:
    - ``gs://bucket/prefix`` → GCS
    - ``s3://bucket/prefix`` → S3
    - ``/absolute/path`` → local filesystem

    Parameters
    ----------
    target:
        Upload target URI or path.

    Returns
    -------
    Dict with provider, bucket, prefix (for cloud) or path (for local).

    Raises
    ------
    ValueError
        If the URI scheme is not supported.
    """
    if target.startswith("gs://"):
        parts = target[5:].split("/", 1)
        return {
            "provider": "gcs",
            "bucket": parts[0],
            "prefix": parts[1] if len(parts) > 1 else "",
        }

    if target.startswith("s3://"):
        parts = target[5:].split("/", 1)
        return {
            "provider": "s3",
            "bucket": parts[0],
            "prefix": parts[1] if len(parts) > 1 else "",
        }

    if target.startswith("/"):
        return {
            "provider": "local",
            "path": target,
        }

    msg = f"Unsupported upload target scheme: {target!r}. Use gs://, s3://, or absolute path."
    raise ValueError(msg)
