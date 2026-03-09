"""Tests for annotation approval pipeline (T-ANN.3.1).

When an annotator approves a corrected mask in 3D Slicer, the approved
label is versioned via DVC and logged to MLflow.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from minivess.annotation.approval import (
    approve_annotation,
)


class TestAnnotationApproval:
    """Test the annotation approval → DVC auto-commit pipeline."""

    def test_approval_result_has_volume_id(self, tmp_path: Path) -> None:
        """Result must contain the original volume_id."""
        label_path = tmp_path / "mv01_label.nii.gz"
        _write_mock_nifti(label_path, shape=(16, 16, 8))

        with _mock_dvc_and_mlflow():
            result = approve_annotation(
                volume_id="mv01",
                label_path=label_path,
                mlruns_dir=tmp_path / "mlruns",
            )

        assert result.volume_id == "mv01"

    def test_approved_label_triggers_dvc_add(self, tmp_path: Path) -> None:
        """dvc add should be called for the approved label."""
        label_path = tmp_path / "mv01_label.nii.gz"
        _write_mock_nifti(label_path, shape=(16, 16, 8))

        with _mock_dvc_and_mlflow() as (mock_dvc_add, _):
            approve_annotation(
                volume_id="mv01",
                label_path=label_path,
                mlruns_dir=tmp_path / "mlruns",
            )
            mock_dvc_add.assert_called_once()

    def test_dvc_version_tag_format(self, tmp_path: Path) -> None:
        """DVC tag must follow annotation/v{YYYY-MM-DD}-{hash} format."""
        label_path = tmp_path / "mv01_label.nii.gz"
        _write_mock_nifti(label_path, shape=(16, 16, 8))

        with _mock_dvc_and_mlflow():
            result = approve_annotation(
                volume_id="mv01",
                label_path=label_path,
                mlruns_dir=tmp_path / "mlruns",
            )

        assert result.dvc_tag.startswith("annotation/v")
        # Tag format: annotation/v2026-03-10-abcdef12
        parts = result.dvc_tag.split("-")
        assert len(parts) >= 4  # annotation/vYYYY, MM, DD, hash

    def test_mlflow_logs_annotation_event(self, tmp_path: Path) -> None:
        """MLflow run must be created for the annotation event."""
        label_path = tmp_path / "mv01_label.nii.gz"
        _write_mock_nifti(label_path, shape=(16, 16, 8))

        with _mock_dvc_and_mlflow() as (_, mock_mlflow):
            result = approve_annotation(
                volume_id="mv01",
                label_path=label_path,
                mlruns_dir=tmp_path / "mlruns",
            )

        assert result.mlflow_run_id is not None
        assert len(result.mlflow_run_id) > 0

    def test_result_has_label_path(self, tmp_path: Path) -> None:
        """Result must include the label file path."""
        label_path = tmp_path / "mv01_label.nii.gz"
        _write_mock_nifti(label_path, shape=(16, 16, 8))

        with _mock_dvc_and_mlflow():
            result = approve_annotation(
                volume_id="mv01",
                label_path=label_path,
                mlruns_dir=tmp_path / "mlruns",
            )

        assert result.label_path == label_path


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_mock_nifti(path: Path, shape: tuple[int, ...] = (16, 16, 8)) -> None:
    """Write a minimal binary file to simulate a NIfTI label."""
    # We don't need a real NIfTI for unit tests — just a file that exists
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.int64)
    data[4:12, 4:12, 2:6] = 1
    # Write raw bytes — approval only checks file existence and hash
    path.write_bytes(data.tobytes())


@contextmanager
def _mock_dvc_and_mlflow() -> Generator[tuple[MagicMock, MagicMock]]:
    """Mock DVC and MLflow calls for unit testing."""
    with (
        patch("minivess.annotation.approval._dvc_add_label") as mock_dvc,
        patch("minivess.annotation.approval._log_annotation_to_mlflow") as mock_mlflow,
    ):
        mock_mlflow.return_value = "mock-run-id-12345"
        yield mock_dvc, mock_mlflow
