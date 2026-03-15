"""SHA256 checkpoint integrity verification.

Writes a .sha256 sidecar file alongside each checkpoint. On resume,
verifies the checkpoint has not been corrupted (e.g., by spot preemption
interrupting a write, or by storage-layer bit rot).

Uses atomic writes for the sidecar itself — a corrupted sidecar is as
dangerous as a corrupted checkpoint.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from minivess.pipeline.checkpoint_utils import atomic_text_write

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_HASH_BLOCK_SIZE = 1 << 16  # 64 KiB


def compute_checkpoint_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a checkpoint file.

    Parameters
    ----------
    path:
        Path to the checkpoint file.

    Returns
    -------
    Lowercase hex digest string.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_HASH_BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_sha256_sidecar(checkpoint_path: Path) -> Path:
    """Write a .sha256 sidecar file alongside a checkpoint.

    The sidecar file contains the hex digest of the checkpoint file.
    Uses atomic write to prevent half-written sidecars.

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint file (e.g., best_val_loss.pth).

    Returns
    -------
    Path to the sidecar file (e.g., best_val_loss.pth.sha256).
    """
    digest = compute_checkpoint_sha256(checkpoint_path)
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    atomic_text_write(digest + "\n", sidecar_path)
    logger.debug("SHA256 sidecar written: %s", sidecar_path)
    return sidecar_path


def verify_checkpoint_sha256(checkpoint_path: Path) -> bool:
    """Verify a checkpoint against its SHA256 sidecar.

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint file.

    Returns
    -------
    True if the sidecar exists and the hash matches. False if the sidecar
    is missing or the hash does not match.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file itself does not exist.
    """
    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if not sidecar_path.exists():
        logger.warning("No SHA256 sidecar for %s", checkpoint_path)
        return False

    expected = sidecar_path.read_text(encoding="utf-8").strip()
    actual = compute_checkpoint_sha256(checkpoint_path)
    if actual != expected:
        logger.error(
            "SHA256 mismatch for %s: expected %s, got %s",
            checkpoint_path,
            expected,
            actual,
        )
        return False

    return True
