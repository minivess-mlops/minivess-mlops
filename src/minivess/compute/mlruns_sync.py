"""Cloud mlruns sync — pull MLflow artifacts from cloud to local.

Supports GCS (gsutil rsync) and SkyPilot (sky rsync) backends.
Excludes large binary artifacts (checkpoints, ONNX) by default.

Issue: #716
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default exclude patterns for sync (large artifacts belong in registries)
DEFAULT_EXCLUDES = ["*.pth", "*.onnx", "*.ckpt", "*.pt"]


class CloudSyncer:
    """Build sync commands for pulling mlruns from cloud storage.

    Parameters
    ----------
    provider:
        Backend: "gcs" (gsutil rsync) or "skypilot" (sky rsync).
    excludes:
        File patterns to exclude from sync.
    """

    def __init__(
        self,
        provider: str,
        excludes: list[str] | None = None,
    ) -> None:
        if provider not in ("gcs", "skypilot"):
            msg = f"Unsupported provider: {provider}. Use 'gcs' or 'skypilot'."
            raise ValueError(msg)
        self.provider = provider
        self.excludes = excludes if excludes is not None else list(DEFAULT_EXCLUDES)

    def build_command(
        self,
        source: str,
        dest: Path,
        *,
        dry_run: bool = False,
    ) -> list[str]:
        """Build the sync command as a list of arguments.

        Parameters
        ----------
        source:
            Source path. For GCS: "gs://bucket/path". For SkyPilot: "cluster:/path".
        dest:
            Local destination directory.
        dry_run:
            If True, add dry-run flag (no actual transfer).

        Returns
        -------
        Command as list of strings (suitable for subprocess.run).
        """
        if self.provider == "gcs":
            return self._build_gcs_command(source, dest, dry_run=dry_run)
        return self._build_sky_command(source, dest, dry_run=dry_run)

    def _build_gcs_command(
        self, source: str, dest: Path, *, dry_run: bool = False
    ) -> list[str]:
        cmd = ["gsutil", "-m", "rsync", "-r"]
        if dry_run:
            cmd.append("-n")
        for pattern in self.excludes:
            cmd.extend(["-x", pattern])
        cmd.extend([source, str(dest)])
        return cmd

    def _build_sky_command(
        self, source: str, dest: Path, *, dry_run: bool = False
    ) -> list[str]:
        # source format: "cluster_name:/remote/path"
        parts = source.split(":", 1)
        cluster = parts[0]
        remote_path = parts[1] if len(parts) > 1 else "/mlruns"

        cmd = ["sky", "rsync", cluster + ":" + remote_path, str(dest), "--down"]
        if dry_run:
            cmd.append("--dry-run")
        return cmd
