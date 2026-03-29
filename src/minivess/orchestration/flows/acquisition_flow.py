"""Prefect Data Acquisition Flow (Flow 0) — Dataset Download & Conversion.

Sits before Flow 1 (Data Engineering). Downloads datasets where possible,
prints manual instructions where not, and converts TIFF → NIfTI.
Uses Prefect @flow and @task decorators for orchestration.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prefect import flow, task

from minivess.config.acquisition_config import (
    AcquisitionConfig,
    AcquisitionResult,
    DatasetAcquisitionStatus,
)
from minivess.data.downloaders import get_downloader
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import FLOW_NAME_ACQUISITION
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# @task functions
# ---------------------------------------------------------------------------


@task(name="check-dataset-status")
def check_dataset_status_task(
    dataset_name: str,
    output_dir: Path,
) -> DatasetAcquisitionStatus:
    """Check if a dataset already exists on disk.

    Parameters
    ----------
    dataset_name:
        Dataset identifier.
    output_dir:
        Directory where the dataset should exist.

    Returns
    -------
    Current acquisition status.
    """
    from minivess.data.acquisition_registry import check_dataset_availability

    status = check_dataset_availability(dataset_name, output_dir)
    logger.info("Dataset '%s' status: %s", dataset_name, status.value)
    return status


@task(name="download-dataset")
def download_dataset_task(
    dataset_name: str,
    output_dir: Path,
) -> DatasetAcquisitionStatus:
    """Attempt automated download of a dataset.

    Parameters
    ----------
    dataset_name:
        Dataset identifier.
    output_dir:
        Target directory for download.

    Returns
    -------
    ``DOWNLOADED`` if successful, ``MANUAL_REQUIRED`` if no downloader,
    ``FAILED`` on error.
    """
    downloader = get_downloader(dataset_name)

    if downloader is None:
        logger.info(
            "No automated downloader for '%s' — manual download required",
            dataset_name,
        )
        return DatasetAcquisitionStatus.MANUAL_REQUIRED

    try:
        downloader(target_dir=output_dir, skip_existing=True)
        logger.info("Successfully downloaded '%s' to %s", dataset_name, output_dir)
        return DatasetAcquisitionStatus.DOWNLOADED
    except Exception:
        logger.exception("Failed to download '%s'", dataset_name)
        return DatasetAcquisitionStatus.FAILED


@task(name="convert-formats")
def convert_formats_task(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Convert TIFF files to NIfTI for a dataset.

    Parameters
    ----------
    dataset_name:
        Dataset identifier.
    input_dir:
        Directory with source TIFF files.
    output_dir:
        Directory for converted NIfTI files.

    Returns
    -------
    List of conversion log messages.
    """
    from minivess.data.acquisition_registry import ACQUISITION_REGISTRY
    from minivess.data.format_conversion import convert_dataset_formats

    entry = ACQUISITION_REGISTRY.get(dataset_name)
    if entry is None or entry.source_format == "nifti":
        logger.info("No format conversion needed for '%s'", dataset_name)
        return []

    # Use default spacing — datasets override via registry
    from minivess.data.external_datasets import EXTERNAL_DATASETS

    ext_config = EXTERNAL_DATASETS.get(dataset_name)
    voxel_spacing = ext_config.resolution_um if ext_config else (1.0, 1.0, 1.0)

    return convert_dataset_formats(
        dataset_name=dataset_name,
        input_dir=input_dir,
        output_dir=output_dir,
        voxel_spacing=voxel_spacing,
    )


@task(name="log-acquisition-provenance")
def log_acquisition_provenance_task(
    datasets_acquired: dict[str, DatasetAcquisitionStatus],
    total_volumes: int,
) -> dict[str, Any]:
    """Compute acquisition provenance metadata for MLflow logging.

    Parameters
    ----------
    datasets_acquired:
        Per-dataset acquisition status.
    total_volumes:
        Total number of volumes acquired.

    Returns
    -------
    Dict of MLflow-compatible params with ``acq_`` prefix.
    """
    provenance: dict[str, Any] = {
        "acq_n_datasets": len(datasets_acquired),
        "acq_total_volumes": total_volumes,
        "acq_timestamp": datetime.now(UTC).isoformat(),
        "acq_statuses": {
            name: status.value for name, status in datasets_acquired.items()
        },
    }
    logger.info("Acquisition provenance: %s", provenance)
    return provenance


@task(name="dvc-commit-datasets")
def dvc_commit_datasets_task(
    data_dir: Path,
    datasets_acquired: list[str],
) -> str | None:
    """Create DVC tracking entries and version tags for acquired datasets.

    Runs ``dvc add`` for each dataset directory and creates a version tag.
    Skips gracefully if DVC is not installed.

    Parameters
    ----------
    data_dir:
        Root data directory containing dataset subdirectories.
    datasets_acquired:
        List of dataset names that were successfully acquired.

    Returns
    -------
    Version tag string if successful, None if DVC is not available.
    """
    import subprocess

    from minivess.data.versioning import create_data_version_tag

    # Check if dvc is installed
    try:
        subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("DVC not installed — skipping dataset versioning")
        return None

    for dataset_name in datasets_acquired:
        dataset_path = data_dir / dataset_name
        if not dataset_path.is_dir():
            logger.info("Skipping DVC add for missing dir: %s", dataset_path)
            continue

        result = subprocess.run(
            ["dvc", "add", str(dataset_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("dvc add failed for '%s': %s", dataset_name, result.stderr)

    # Create a version tag for the acquisition
    tag = create_data_version_tag(
        "minivess",
        datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    logger.info("DVC version tag: %s", tag)
    return tag


@task(name="print-manual-instructions")
def print_manual_instructions_task(
    manual_datasets: list[str],
) -> dict[str, str]:
    """Log download instructions for datasets requiring manual download.

    Parameters
    ----------
    manual_datasets:
        List of dataset names that need manual download.

    Returns
    -------
    Dict of dataset_name → instructions.
    """
    from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

    instructions: dict[str, str] = {}

    for name in manual_datasets:
        entry = ACQUISITION_REGISTRY.get(name)
        if entry is None:
            continue
        instructions[name] = entry.manual_instructions
        logger.warning(
            "MANUAL DOWNLOAD REQUIRED — %s:\n%s",
            name,
            entry.manual_instructions,
        )

    return instructions


# ---------------------------------------------------------------------------
# @flow orchestrator
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_ACQUISITION)
def run_acquisition_flow(
    config: AcquisitionConfig | None = None,
    *,
    trigger_source: str = "manual",
    **kwargs: Any,  # noqa: ARG001
) -> AcquisitionResult:
    """Data Acquisition Flow (Flow 0).

    Downloads datasets where possible, prints manual instructions where not,
    and converts TIFF → NIfTI.

    Parameters
    ----------
    config:
        Acquisition configuration. Uses defaults if None.
    trigger_source:
        What triggered this flow (e.g., ``"manual"``, ``"trigger_chain"``).

    Returns
    -------
    AcquisitionResult with per-dataset status and provenance.
    """
    require_docker_context("acquisition")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("acquisition", logs_dir=logs_dir) as event_logger:
        from minivess.config.acquisition_config import AcquisitionConfig

        if config is None:
            config = AcquisitionConfig()

        logger.info(
            "Starting acquisition flow (trigger: %s, datasets: %s)",
            trigger_source,
            config.datasets,
        )

        datasets_acquired: dict[str, DatasetAcquisitionStatus] = {}
        conversion_log: list[str] = []
        manual_needed: list[str] = []

        for dataset_name in config.datasets:
            dataset_dir = config.output_dir / dataset_name

            # Step 1: Check if already available
            status = check_dataset_status_task(
                dataset_name=dataset_name, output_dir=dataset_dir
            )

            if status == DatasetAcquisitionStatus.READY and config.skip_existing:
                datasets_acquired[dataset_name] = DatasetAcquisitionStatus.READY
            else:
                # Step 2: Try automated download
                status = download_dataset_task(
                    dataset_name=dataset_name, output_dir=dataset_dir
                )
                datasets_acquired[dataset_name] = status

                if status == DatasetAcquisitionStatus.MANUAL_REQUIRED:
                    manual_needed.append(dataset_name)

            # Step 3: Convert formats if needed (runs for READY and DOWNLOADED)
            effective_status = datasets_acquired[dataset_name]
            if config.convert_formats and effective_status in (
                DatasetAcquisitionStatus.DOWNLOADED,
                DatasetAcquisitionStatus.READY,
            ):
                log_entries = convert_formats_task(
                    dataset_name=dataset_name,
                    input_dir=dataset_dir,
                    output_dir=dataset_dir,
                )
                conversion_log.extend(log_entries)

        # Step 4: Print manual instructions
        if manual_needed:
            print_manual_instructions_task(manual_datasets=manual_needed)

        # Step 5: Compute provenance
        total_volumes = sum(
            1
            for s in datasets_acquired.values()
            if s in (DatasetAcquisitionStatus.READY, DatasetAcquisitionStatus.DOWNLOADED)
        )
        provenance = log_acquisition_provenance_task(
            datasets_acquired=datasets_acquired,
            total_volumes=total_volumes,
        )

        logger.info(
            "Acquisition flow complete: %d/%d datasets acquired",
            total_volumes,
            len(config.datasets),
        )

        # --- MLflow logging ---
        _tracking_uri = resolve_tracking_uri()
        mlflow_run_id: str | None = None
        try:
            import mlflow

            from minivess.orchestration.flow_contract import FlowContract

            mlflow.set_tracking_uri(_tracking_uri)
            mlflow.set_experiment("minivess_acquisition")
            with mlflow.start_run(tags={"flow_name": "acquisition"}) as active_run:
                mlflow_run_id = active_run.info.run_id
                mlflow.log_param("acq_n_datasets", len(config.datasets))
                mlflow.log_param("acq_total_volumes", total_volumes)
                for key, value in provenance.items():
                    if isinstance(value, str | int | float | bool):
                        mlflow.log_param(key, value)

            contract = FlowContract(tracking_uri=_tracking_uri)
            contract.log_flow_completion(
                flow_name="acquisition",
                run_id=mlflow_run_id,
            )
        except Exception:
            logger.warning("Failed to log acquisition_flow to MLflow", exc_info=True)

        return AcquisitionResult(
            datasets_acquired=datasets_acquired,
            total_volumes=total_volumes,
            conversion_log=conversion_log,
            provenance=provenance,
            mlflow_run_id=mlflow_run_id,
        )


if __name__ == "__main__":
    run_acquisition_flow()
