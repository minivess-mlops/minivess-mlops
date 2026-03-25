"""Async checkpoint upload manager — MLflow-only persistence.

H2 design from mlflow-async-checkpoint-architecture-research.md:
  1. torch.save() writes to local SSD (sync, ~50ms for DynUNet)
  2. ThreadPoolExecutor uploads to MLflow artifact store in background
  3. flush() blocks until all uploads complete (call before training ends)
  4. shutdown() flushes + closes executor (call in finally block)

This replaces the SkyPilot file_mounts (MOUNT_CACHED to GCS) which violated
the KG invariant mlflow_only_artifact_contract.

Key facts (from research):
  - mlflow.log_artifact() has NO async variant (GitHub FR #14153 closed)
  - mlflow.log_artifact() IS thread-safe in MLflow 3.x (contextvars, #9235)
  - ThreadPoolExecutor wrapping is the correct pattern
  - GCP L4 local NVMe: 62μs latency, DynUNet 50MB = 50ms write
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AsyncCheckpointUploader:
    """Non-blocking checkpoint uploader backed by ThreadPoolExecutor + mlflow.log_artifact.

    Usage::

        uploader = AsyncCheckpointUploader(max_workers=1)
        # After each epoch:
        torch.save(state_dict, local_path)
        uploader.upload(local_path, artifact_subdir="checkpoints")
        # At training end:
        uploader.shutdown()  # flushes + closes
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[None]] = []
        self._shut_down = False

    def upload(self, local_path: Path, artifact_subdir: str = "checkpoints") -> None:
        """Submit a checkpoint for background upload to MLflow.

        Non-blocking: returns immediately. The actual mlflow.log_artifact()
        call happens in the background thread.

        Parameters
        ----------
        local_path:
            Path to the checkpoint file on local SSD.
        artifact_subdir:
            MLflow artifact subdirectory (default: "checkpoints").

        Raises
        ------
        RuntimeError
            If called after shutdown().
        """
        if self._shut_down:
            msg = "Cannot upload after shut down"
            raise RuntimeError(msg)

        future = self._executor.submit(self._upload_one, local_path, artifact_subdir)
        self._futures.append(future)

    def flush(self) -> None:
        """Block until all pending uploads complete.

        Upload errors are logged but not re-raised — training must not crash
        because of a transient MLflow network failure. The checkpoint is still
        on local SSD and will be picked up by the next flush or by spot
        recovery.
        """
        for future in self._futures:
            try:
                future.result()  # blocks until done
            except Exception:
                logger.warning("Checkpoint upload failed (non-blocking)", exc_info=True)
        self._futures.clear()

    def shutdown(self) -> None:
        """Flush all pending uploads, then close the executor.

        Call this in a ``finally`` block at the end of training.
        """
        self._shut_down = True
        self.flush()
        self._executor.shutdown(wait=True)

    @staticmethod
    def _upload_one(local_path: Path, artifact_subdir: str) -> None:
        """Upload a single checkpoint file to MLflow artifact store."""
        logger.info("Uploading checkpoint: %s → %s/", local_path.name, artifact_subdir)
        mlflow.log_artifact(str(local_path), artifact_subdir)
        logger.info("Upload complete: %s", local_path.name)
