"""Atomic checkpoint utilities for spot-preemption safety.

Uses tmp file + os.replace() to ensure checkpoint files are never half-written.
On spot instances (RunPod, AWS, GCP), preemption can interrupt a torch.save()
mid-write, corrupting the checkpoint. The atomic pattern guarantees the file
is either fully written or not present at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Save a PyTorch object atomically using tmp + os.replace().

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
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up partial tmp file on any failure
        tmp_path.unlink(missing_ok=True)
        raise
