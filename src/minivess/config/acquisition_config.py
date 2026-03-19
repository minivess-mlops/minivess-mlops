"""Configuration and result types for Flow 0: Data Acquisition.

Defines ``AcquisitionConfig`` for controlling which datasets to acquire,
``DatasetAcquisitionStatus`` for per-dataset state tracking, and
``AcquisitionResult`` for flow output.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Known dataset identifiers — matches acquisition_registry.py
# TubeNet excluded — olfactory bulb, different organ, only 1 2PM volume. See CLAUDE.md.
KNOWN_DATASETS: frozenset[str] = frozenset({"minivess", "deepvess", "vesselnn"})

_DEFAULT_DATASETS: list[str] = sorted(KNOWN_DATASETS)


class DatasetAcquisitionStatus(enum.Enum):
    """Per-dataset acquisition state."""

    READY = "ready"
    DOWNLOADED = "downloaded"
    MANUAL_REQUIRED = "manual_required"
    FAILED = "failed"


@dataclass
class AcquisitionConfig:
    """Configuration for the data acquisition flow.

    Attributes
    ----------
    datasets:
        Which datasets to acquire (default: all 3).
    output_dir:
        Where to write acquired/converted NIfTI files.
    skip_existing:
        Skip download/conversion if output already exists.
    convert_formats:
        Run TIFF → NIfTI conversion after download.
    verify_checksums:
        Verify downloaded files against known checksums.
    """

    datasets: list[str] = field(default_factory=lambda: list(_DEFAULT_DATASETS))
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))
    skip_existing: bool = True
    convert_formats: bool = True
    verify_checksums: bool = True

    def validate(self) -> list[str]:
        """Validate configuration, returning list of error messages."""
        errors: list[str] = []
        if not self.datasets:
            errors.append("datasets list must not be empty")
        for name in self.datasets:
            if name not in KNOWN_DATASETS:
                errors.append(
                    f"Unknown dataset '{name}'. "
                    f"Known datasets: {sorted(KNOWN_DATASETS)}"
                )
        return errors


@dataclass
class AcquisitionResult:
    """Result of the complete data acquisition flow.

    Attributes
    ----------
    datasets_acquired:
        Per-dataset acquisition status.
    total_volumes:
        Total number of volumes across all acquired datasets.
    conversion_log:
        Log of format conversions performed.
    provenance:
        Acquisition metadata for MLflow logging.
    """

    datasets_acquired: dict[str, DatasetAcquisitionStatus]
    total_volumes: int
    conversion_log: list[str]
    provenance: dict[str, Any]
    mlflow_run_id: str | None = None
