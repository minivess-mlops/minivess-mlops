"""Unit tests for DVC data pull in training flow (#635, T2.1).

Tests prepare_training_data() which runs DVC pull when training data
is not pre-mounted (cloud execution via SkyPilot).

Uses monkeypatch on subprocess.run inside the function — no real DVC or S3.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest


class TestPrepareTrainingData:
    """Test prepare_training_data() DVC pull logic."""

    def test_skips_pull_when_data_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When data directory has files, DVC pull is skipped."""
        data_dir = tmp_path / "data"
        images_dir = data_dir / "raw" / "minivess" / "imagesTr"
        images_dir.mkdir(parents=True)
        (images_dir / "mv01.nii.gz").touch()

        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        from minivess.orchestration.flows.train_flow import prepare_training_data

        result = prepare_training_data(data_dir=data_dir)
        assert result["pulled"] is False

    def test_runs_dvc_pull_when_data_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When data directory is empty, DVC pull is executed."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        _original_run = subprocess.run

        def _mock_run(
            cmd: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if isinstance(cmd, list) and cmd and cmd[0] == "dvc":
                # Simulate successful DVC pull by creating data
                if "pull" in cmd:
                    img_dir = data_dir / "raw" / "minivess" / "imagesTr"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    (img_dir / "mv01.nii.gz").touch()
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _original_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from minivess.orchestration.flows.train_flow import prepare_training_data

        result = prepare_training_data(data_dir=data_dir)
        assert result["pulled"] is True

    def test_dvc_remote_defaults_to_minio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DVC_REMOTE defaults to 'minio' when not set."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.delenv("DVC_REMOTE", raising=False)
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        calls: list[list[str]] = []
        _original_run = subprocess.run

        def _mock_run(
            cmd: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if isinstance(cmd, list) and cmd and cmd[0] == "dvc":
                calls.append(list(cmd))
                if "pull" in cmd:
                    img_dir = data_dir / "raw" / "minivess" / "imagesTr"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    (img_dir / "mv01.nii.gz").touch()
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _original_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from minivess.orchestration.flows.train_flow import prepare_training_data

        prepare_training_data(data_dir=data_dir)
        pull_calls = [c for c in calls if "pull" in c]
        assert pull_calls
        # Default remote should be minio
        for call in pull_calls:
            r_idx = call.index("-r")
            assert call[r_idx + 1] == "minio"

    def test_uses_remote_storage_when_env_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DVC_REMOTE=remote_storage, uses remote_storage remote (AWS S3 fallback).

        UpCloud archived 2026-03-16 — remote_storage (s3://minivessdataset) is the
        cloud fallback when Network Volume has no data.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("DVC_REMOTE", "remote_storage")
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        calls: list[list[str]] = []
        _original_run = subprocess.run

        def _mock_run(
            cmd: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if isinstance(cmd, list) and cmd and cmd[0] == "dvc":
                calls.append(list(cmd))
                if "pull" in cmd:
                    img_dir = data_dir / "raw" / "minivess" / "imagesTr"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    (img_dir / "mv01.nii.gz").touch()
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return _original_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from minivess.orchestration.flows.train_flow import prepare_training_data

        prepare_training_data(data_dir=data_dir)
        pull_calls = [c for c in calls if "pull" in c]
        assert pull_calls
        for call in pull_calls:
            r_idx = call.index("-r")
            assert call[r_idx + 1] == "remote_storage"
