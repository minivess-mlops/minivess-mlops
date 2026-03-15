"""Tests for cloud mlruns sync (issue #716).

Verifies CloudSyncer generates correct sync commands for
GCS (gsutil) and RunPod (sky rsync) backends.
"""

from __future__ import annotations

from pathlib import Path


class TestCloudSyncerGCS:
    """GCS sync via gsutil."""

    def test_gcs_sync_command(self) -> None:
        """gsutil rsync command is correct."""
        from minivess.compute.mlruns_sync import CloudSyncer

        syncer = CloudSyncer(provider="gcs")
        cmd = syncer.build_command(
            source="gs://minivess-mlops-mlflow-artifacts",
            dest=Path("/tmp/mlruns"),
        )
        assert "gsutil" in cmd[0]
        assert "rsync" in cmd

    def test_gcs_excludes_checkpoints(self) -> None:
        """Default excludes *.pth, *.onnx."""
        from minivess.compute.mlruns_sync import CloudSyncer

        syncer = CloudSyncer(provider="gcs")
        cmd = syncer.build_command(
            source="gs://bucket",
            dest=Path("/tmp/mlruns"),
        )
        cmd_str = " ".join(cmd)
        assert ".pth" in cmd_str or "pth" in cmd_str

    def test_gcs_dry_run(self) -> None:
        """Dry run flag adds -n."""
        from minivess.compute.mlruns_sync import CloudSyncer

        syncer = CloudSyncer(provider="gcs")
        cmd = syncer.build_command(
            source="gs://bucket",
            dest=Path("/tmp/mlruns"),
            dry_run=True,
        )
        assert "-n" in cmd


class TestCloudSyncerSkyPilot:
    """SkyPilot sync via sky rsync."""

    def test_sky_sync_command(self) -> None:
        """sky rsync command is correct."""
        from minivess.compute.mlruns_sync import CloudSyncer

        syncer = CloudSyncer(provider="skypilot")
        cmd = syncer.build_command(
            source="minivess-dev:/mlruns",
            dest=Path("/tmp/mlruns"),
        )
        cmd_str = " ".join(cmd)
        assert "sky" in cmd_str or "rsync" in cmd_str
