from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def save_volume_prediction(
    *,
    output_dir: Path,
    volume_name: str,
    hard_pred: np.ndarray,
    soft_pred: np.ndarray,
) -> Path:
    """Save compressed prediction arrays for one volume.

    Parameters
    ----------
    output_dir:
        Directory to save the .npz file.
    volume_name:
        Name identifier for the volume (used as filename).
    hard_pred:
        Binary prediction mask (D, H, W). Saved as uint8.
    soft_pred:
        Foreground probability map (D, H, W). Saved as float16.

    Returns
    -------
    Path
        Path to the saved .npz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{volume_name}.npz"
    np.savez_compressed(
        path,
        hard_pred=hard_pred.astype(np.uint8),
        soft_pred=soft_pred.astype(np.float16),
    )
    return path


def load_volume_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction arrays from a compressed .npz file.

    Parameters
    ----------
    path:
        Path to the .npz file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(hard_pred, soft_pred)`` arrays.
    """
    data = np.load(path)
    return data["hard_pred"], data["soft_pred"]
