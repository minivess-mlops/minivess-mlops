"""Annotation approval pipeline — DVC versioning + MLflow logging.

When an annotator approves a corrected mask in 3D Slicer, this module:
1. Validates the mask file exists
2. Versions it via DVC (dvc add + tag)
3. Logs the annotation event to MLflow
4. Returns an AnnotationApprovalResult with provenance metadata
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used in dataclass fields + function bodies

logger = logging.getLogger(__name__)


@dataclass
class AnnotationApprovalResult:
    """Result from approving an annotation.

    Parameters
    ----------
    volume_id:
        Identifier for the annotated volume.
    label_path:
        Path to the approved label file.
    dvc_tag:
        DVC version tag (e.g. ``annotation/v2026-03-10-abcdef12``).
    mlflow_run_id:
        MLflow run ID for the annotation event.
    """

    volume_id: str
    label_path: Path
    dvc_tag: str
    mlflow_run_id: str


def approve_annotation(
    volume_id: str,
    label_path: Path,
    mlruns_dir: Path,
) -> AnnotationApprovalResult:
    """Approve an annotation: validate, DVC-version, and log to MLflow.

    Parameters
    ----------
    volume_id:
        Volume identifier (e.g. ``mv01``).
    label_path:
        Path to the approved label file (.nii.gz).
    mlruns_dir:
        Path to the MLflow tracking directory.

    Returns
    -------
    AnnotationApprovalResult with provenance metadata.
    """
    if not label_path.exists():
        msg = f"Label file not found: {label_path}"
        raise FileNotFoundError(msg)

    # Generate DVC tag
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    file_hash = _compute_file_hash(label_path)[:8]
    dvc_tag = f"annotation/v{timestamp}-{file_hash}"

    # DVC add
    _dvc_add_label(label_path)

    # Log to MLflow
    mlflow_run_id = _log_annotation_to_mlflow(
        volume_id=volume_id,
        label_path=label_path,
        dvc_tag=dvc_tag,
        mlruns_dir=mlruns_dir,
    )

    logger.info(
        "Annotation approved: volume=%s tag=%s mlflow_run=%s",
        volume_id,
        dvc_tag,
        mlflow_run_id,
    )

    return AnnotationApprovalResult(
        volume_id=volume_id,
        label_path=label_path,
        dvc_tag=dvc_tag,
        mlflow_run_id=mlflow_run_id,
    )


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _dvc_add_label(label_path: Path) -> None:
    """Add label file to DVC tracking.

    Runs ``dvc add <label_path>`` in a subprocess.
    """
    subprocess.run(
        ["dvc", "add", str(label_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _log_annotation_to_mlflow(
    volume_id: str,
    label_path: Path,
    dvc_tag: str,
    mlruns_dir: Path,
) -> str:
    """Log annotation event to MLflow.

    Creates a run in the ``minivess_annotation`` experiment with
    annotation metadata as params.

    Returns the MLflow run ID.
    """
    import mlflow

    mlflow.set_tracking_uri(str(mlruns_dir))
    mlflow.set_experiment("minivess_annotation")

    with mlflow.start_run() as run:
        mlflow.log_param("volume_id", volume_id)
        mlflow.log_param("label_path", str(label_path))
        mlflow.log_param("dvc_tag", dvc_tag)
        mlflow.log_param("annotation_timestamp", datetime.now(UTC).isoformat())
        run_id: str = run.info.run_id
        return run_id
