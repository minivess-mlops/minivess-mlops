"""Atomic checkpoint utilities for spot-preemption safety.

Uses tmp file + os.replace() to ensure checkpoint files are never half-written.
On spot instances (RunPod, AWS, GCP), preemption can interrupt a torch.save()
mid-write, corrupting the checkpoint. The atomic pattern guarantees the file
is either fully written or not present at all.

Also provides load_checkpoint_with_fallback() for corruption recovery:
try the primary (latest) checkpoint, fall back to the previous one if corrupt.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Save a PyTorch object atomically using tmp + fsync + os.replace().

    Parameters
    ----------
    obj:
        Object to save (state_dict, checkpoint dict, etc.).
    path:
        Destination file path.

    Raises
    ------
    OSError
        If the underlying torch.save fails (disk full, permissions, etc.).
        The original file (if any) is preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp_path)
        # fsync to ensure data hits disk before rename
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up partial tmp file on any failure
        tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint_with_fallback(
    primary: Path,
    fallback: Path | None = None,
) -> dict[str, Any]:
    """Load checkpoint with corruption fallback.

    Tries loading the primary checkpoint. If it is corrupt (truncated,
    not valid pickle, etc.), falls back to the fallback checkpoint.

    Parameters
    ----------
    primary:
        Path to the primary (latest) checkpoint file.
    fallback:
        Path to the fallback (previous) checkpoint file. If None or
        the file does not exist, FileNotFoundError is raised when
        the primary is corrupt.

    Returns
    -------
    dict
        The loaded checkpoint dictionary.

    Raises
    ------
    FileNotFoundError
        If the primary is corrupt and no valid fallback is available.
    ValueError
        If both primary and fallback are corrupt (Rule #25: loud failure).
    """
    try:
        # SECURITY: weights_only=False -- self-produced checkpoint (state_dict, epoch,
        # optimizer_state_dict, scheduler). See trivy-litellm-secops-double-checking.md
        result: dict[str, Any] = torch.load(
            primary, weights_only=False, map_location="cpu"
        )
        return result
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        logger.warning(
            "Primary checkpoint corrupt (%s): %s. Trying fallback: %s",
            primary,
            exc,
            fallback,
        )
        if fallback is None or not fallback.exists():
            raise FileNotFoundError(
                f"Primary corrupt and no fallback: {primary}"
            ) from exc
        try:
            # SECURITY: weights_only=False -- self-produced checkpoint (fallback path,
            # same format as primary). See trivy-litellm-secops-double-checking.md
            fallback_result: dict[str, Any] = torch.load(
                fallback, weights_only=False, map_location="cpu"
            )
            return fallback_result
        except (RuntimeError, EOFError, pickle.UnpicklingError) as exc2:
            raise ValueError(
                f"Both primary ({primary}) and fallback ({fallback}) corrupt"
            ) from exc2


def atomic_text_write(content: str, path: Path, encoding: str = "utf-8") -> None:
    """Write text atomically using tmp + os.replace().

    Parameters
    ----------
    content:
        Text content to write.
    path:
        Destination file path.
    encoding:
        Text encoding (default utf-8).

    Raises
    ------
    OSError
        If the write fails. The original file (if any) is preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
