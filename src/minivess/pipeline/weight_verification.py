"""SHA-256 weight verification for HuggingFace model downloads.

Provides verify_weight_sha256() for validating downloaded model weights
against expected SHA-256 hashes. Complements checkpoint_integrity.py
(which handles training checkpoint sidecar verification) by covering
pre-trained weight downloads from HuggingFace Hub.

Two modes:
1. **Verification mode** (expected_sha256 is a hex string): computes the
   hash of the downloaded file and raises ValueError on mismatch.
2. **First-download mode** (expected_sha256 is None): computes and returns
   the hash so the caller can pin it for future downloads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from minivess.pipeline.checkpoint_integrity import compute_checkpoint_sha256

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def verify_weight_sha256(
    path: Path,
    *,
    expected_sha256: str | None,
) -> str | bool:
    """Verify SHA-256 hash of a downloaded model weight file.

    Parameters
    ----------
    path:
        Path to the downloaded weight file.
    expected_sha256:
        Expected lowercase hex SHA-256 digest (64 chars).
        If None, the function computes and returns the hash
        (first-download / hash-discovery mode).

    Returns
    -------
    - If expected_sha256 is None: returns the computed hex digest string.
    - If expected_sha256 matches: returns True.

    Raises
    ------
    FileNotFoundError
        If the weight file does not exist.
    ValueError
        If the computed hash does not match expected_sha256.
    """
    if not path.exists():
        msg = f"Weight file not found: {path}"
        raise FileNotFoundError(msg)

    computed = compute_checkpoint_sha256(path)

    if expected_sha256 is None:
        logger.info(
            "No expected hash provided for %s. Computed SHA-256: %s",
            path.name,
            computed,
        )
        return computed

    if computed != expected_sha256:
        msg = (
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {computed}. "
            f"The file may be corrupted or tampered with."
        )
        raise ValueError(msg)

    logger.debug("SHA-256 verified for %s: %s", path.name, computed)
    return True
