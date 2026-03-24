"""Tests for AsyncCheckpointUploader — MLflow-only checkpoint persistence.

H2 design: Local SSD write (sync, fast) + ThreadPoolExecutor background upload
via mlflow.log_artifact(). Replaces the GCS file_mounts which violated the
KG invariant mlflow_only_artifact_contract.

Research: docs/planning/mlflow-async-checkpoint-architecture-research.md
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Test: upload submits to ThreadPoolExecutor (non-blocking)
# ---------------------------------------------------------------------------


class TestAsyncCheckpointUploaderUpload:
    """upload() must be non-blocking — submits to background thread."""

    def test_upload_returns_immediately(self, tmp_path: Path) -> None:
        """upload() should return in <100ms even if mlflow is slow."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        ckpt = tmp_path / "test.pth"
        ckpt.write_bytes(b"fake-checkpoint-data")

        with patch("minivess.pipeline.checkpoint_manager.mlflow") as mock_mlflow:
            # Make log_artifact slow to prove upload() doesn't block
            mock_mlflow.log_artifact.side_effect = lambda *a, **k: time.sleep(0.5)

            uploader = AsyncCheckpointUploader(max_workers=1)
            start = time.monotonic()
            uploader.upload(ckpt, artifact_subdir="checkpoints")
            elapsed = time.monotonic() - start

            # upload() must return immediately (not wait for the 0.5s log_artifact)
            assert elapsed < 0.2, (
                f"upload() took {elapsed:.2f}s — should be non-blocking (<0.2s)"
            )
            uploader.shutdown()

    def test_upload_calls_mlflow_log_artifact(self, tmp_path: Path) -> None:
        """Background thread must call mlflow.log_artifact with correct args."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        ckpt = tmp_path / "best_val_loss.pth"
        ckpt.write_bytes(b"fake-checkpoint-data")

        with patch("minivess.pipeline.checkpoint_manager.mlflow") as mock_mlflow:
            uploader = AsyncCheckpointUploader(max_workers=1)
            uploader.upload(ckpt, artifact_subdir="checkpoints")
            uploader.flush()  # Wait for background upload to complete

            mock_mlflow.log_artifact.assert_called_once_with(str(ckpt), "checkpoints")
            uploader.shutdown()


# ---------------------------------------------------------------------------
# Test: flush() blocks until all uploads complete
# ---------------------------------------------------------------------------


class TestAsyncCheckpointUploaderFlush:
    """flush() must block until all pending uploads are done."""

    def test_flush_waits_for_all_uploads(self, tmp_path: Path) -> None:
        """After flush(), all uploads must be complete."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        call_count = 0

        def slow_log(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            time.sleep(0.1)
            call_count += 1

        with patch("minivess.pipeline.checkpoint_manager.mlflow") as mock_mlflow:
            mock_mlflow.log_artifact.side_effect = slow_log

            uploader = AsyncCheckpointUploader(max_workers=1)
            for i in range(3):
                ckpt = tmp_path / f"epoch_{i}.pth"
                ckpt.write_bytes(b"data")
                uploader.upload(ckpt)

            uploader.flush()
            assert call_count == 3, f"Expected 3 uploads after flush, got {call_count}"
            uploader.shutdown()


# ---------------------------------------------------------------------------
# Test: shutdown() flushes and closes executor
# ---------------------------------------------------------------------------


class TestAsyncCheckpointUploaderShutdown:
    """shutdown() must flush pending uploads then close the executor."""

    def test_shutdown_flushes_and_closes(self, tmp_path: Path) -> None:
        """After shutdown(), no more uploads should be accepted."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        ckpt = tmp_path / "final.pth"
        ckpt.write_bytes(b"data")

        with patch("minivess.pipeline.checkpoint_manager.mlflow") as mock_mlflow:
            uploader = AsyncCheckpointUploader(max_workers=1)
            uploader.upload(ckpt)
            uploader.shutdown()

            # After shutdown, log_artifact should have been called
            mock_mlflow.log_artifact.assert_called_once()

    def test_upload_after_shutdown_raises(self, tmp_path: Path) -> None:
        """upload() after shutdown() must raise RuntimeError."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        ckpt = tmp_path / "late.pth"
        ckpt.write_bytes(b"data")

        with patch("minivess.pipeline.checkpoint_manager.mlflow"):
            uploader = AsyncCheckpointUploader(max_workers=1)
            uploader.shutdown()

            with pytest.raises(RuntimeError, match="shut down"):
                uploader.upload(ckpt)


# ---------------------------------------------------------------------------
# Test: error handling — upload failure logged, not raised
# ---------------------------------------------------------------------------


class TestAsyncCheckpointUploaderErrorHandling:
    """Upload failures must be logged, not crash training."""

    def test_upload_failure_logged_not_raised(self, tmp_path: Path) -> None:
        """If mlflow.log_artifact fails, flush() should not raise."""
        from minivess.pipeline.checkpoint_manager import AsyncCheckpointUploader

        ckpt = tmp_path / "bad.pth"
        ckpt.write_bytes(b"data")

        with patch("minivess.pipeline.checkpoint_manager.mlflow") as mock_mlflow:
            mock_mlflow.log_artifact.side_effect = Exception("Network error")

            uploader = AsyncCheckpointUploader(max_workers=1)
            uploader.upload(ckpt)
            # flush() must NOT propagate the exception — log it instead
            uploader.flush()  # Should not raise
            uploader.shutdown()
