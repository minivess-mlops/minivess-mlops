"""DVC version change detection and data versioning utilities.

Provides functions for detecting when .dvc files change,
computing hashes for change detection, and creating version tags.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataVersionInfo:
    """Information about a dataset version.

    Attributes
    ----------
    dataset:
        Dataset name.
    version:
        Version string (e.g., ``"0.1.0"``).
    git_tag:
        Git tag for this version.
    changed_files:
        List of changed .dvc files that triggered this version.
    """

    dataset: str
    version: str
    git_tag: str
    changed_files: list[str] = field(default_factory=list)


def compute_dvc_hash(data_dir: Path) -> str:
    """Compute a deterministic hash of all .dvc files in a directory.

    Parameters
    ----------
    data_dir:
        Directory to scan for .dvc files.

    Returns
    -------
    Hex digest of the combined .dvc file contents.
    """
    hasher = hashlib.sha256()
    dvc_files = sorted(data_dir.glob("**/*.dvc"))
    for dvc_file in dvc_files:
        hasher.update(dvc_file.name.encode())
        hasher.update(dvc_file.read_bytes())
    return hasher.hexdigest()[:16]


def detect_dvc_change(
    data_dir: Path,
    reference_hash: str | None = None,
) -> bool:
    """Detect whether .dvc files have changed.

    Parameters
    ----------
    data_dir:
        Directory to check for .dvc file changes.
    reference_hash:
        Previous hash to compare against. If ``None``, returns True
        if any .dvc files exist (first-run detection).

    Returns
    -------
    True if .dvc files have changed (or exist when no reference).
    """
    dvc_files = list(data_dir.glob("**/*.dvc"))
    if not dvc_files:
        return False

    if reference_hash is None:
        logger.info(
            "No reference hash — treating as changed (%d .dvc files)", len(dvc_files)
        )
        return True

    current_hash = compute_dvc_hash(data_dir)
    changed = current_hash != reference_hash
    if changed:
        logger.info("DVC hash changed: %s → %s", reference_hash, current_hash)
    return changed


def create_data_version_tag(dataset_name: str, version: str) -> str:
    """Create a git tag string for a dataset version.

    Parameters
    ----------
    dataset_name:
        Name of the dataset.
    version:
        Version string.

    Returns
    -------
    Git tag string in format ``data/{dataset}/v{version}``.
    """
    return f"data/{dataset_name}/v{version}"


def get_current_data_version(dataset_name: str) -> str | None:
    """Get the current tracked version for a dataset.

    Parameters
    ----------
    dataset_name:
        Name of the dataset.

    Returns
    -------
    Version string from DVC_CONFIGS, or None if not tracked.
    """
    from minivess.data.external_datasets import DVC_CONFIGS

    config = DVC_CONFIGS.get(dataset_name)
    if config is None:
        return None
    version: str = config.version
    return version
