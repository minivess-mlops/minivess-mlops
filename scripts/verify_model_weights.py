"""Verify SHA-256 hashes of downloaded model weights.

Computes and/or verifies SHA-256 hashes for pre-trained model weight files
used in the MinIVess pipeline. Run this after downloading weights on a
trusted network to obtain the hash for pinning in source code.

Supported models:
  - vesselfm: VesselFM (bwittmann/vesselFM on HuggingFace Hub)
  - sam3: SAM3 (facebook/sam3 on HuggingFace Hub, multi-shard)

Usage:
    uv run python scripts/verify_model_weights.py --model vesselfm
    uv run python scripts/verify_model_weights.py --model vesselfm --verify
    uv run python scripts/verify_model_weights.py --model sam3
    uv run python scripts/verify_model_weights.py --path /path/to/weights.pt

CLAUDE.md Rule #16: import re is BANNED.
CLAUDE.md Rule #6:  Use pathlib.Path throughout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _find_vesselfm_cache() -> Path | None:
    """Find the cached VesselFM weights in the HuggingFace cache.

    Returns
    -------
    Path to the cached vesselFM_base.pt file, or None if not found.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        from minivess.adapters.vesselfm import VESSELFM_HF_FILENAME, VESSELFM_HF_REPO

        cached = try_to_load_from_cache(
            repo_id=VESSELFM_HF_REPO,
            filename=VESSELFM_HF_FILENAME,
        )
        if cached is not None and isinstance(cached, str):
            return Path(cached)
    except ImportError:
        logger.warning("huggingface_hub not installed")
    return None


def _find_sam3_cache() -> Path | None:
    """Find the cached SAM3 weights in the HuggingFace cache.

    Returns
    -------
    Path to the cached SAM3 model directory, or None if not found.

    Note: SAM3 is a multi-shard model; the path points to the snapshot
    directory, not a single file. Use HuggingFace revision pinning instead
    of file-level SHA-256 for multi-shard models.
    """
    try:
        from huggingface_hub import scan_cache_dir

        from minivess.adapters.sam3_backbone import SAM3_HF_MODEL_ID

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == SAM3_HF_MODEL_ID:
                # Return the most recent revision's snapshot path
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified)
                if revisions:
                    return revisions[-1].snapshot_path
    except ImportError:
        logger.warning("huggingface_hub not installed")
    return None


def _compute_and_report(path: Path, *, expected: str | None = None) -> int:
    """Compute SHA-256 and optionally verify against expected hash.

    Parameters
    ----------
    path:
        Path to the weight file.
    expected:
        Expected SHA-256 hex digest. If None, just computes and prints.

    Returns
    -------
    Exit code: 0 = success/computed, 1 = mismatch or error.
    """
    from minivess.pipeline.weight_verification import verify_weight_sha256

    try:
        result = verify_weight_sha256(path, expected_sha256=expected)
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if isinstance(result, str):
        # First-download mode: print the hash for pinning
        print(f"SHA-256: {result}")  # noqa: T201
        print(f"File:    {path}")  # noqa: T201
        print()  # noqa: T201
        print("Pin this hash in the source code constant:")  # noqa: T201
        print(f'  VESSELFM_WEIGHT_SHA256 = "{result}"')  # noqa: T201
        return 0

    # Verification succeeded
    logger.info("Verification PASSED for %s", path.name)
    return 0


def main() -> int:
    """Entry point for the weight verification script."""
    parser = argparse.ArgumentParser(
        description="Verify SHA-256 hashes of model weight files."
    )
    parser.add_argument(
        "--model",
        choices=["vesselfm", "sam3"],
        help="Model to verify (looks up cached HuggingFace download).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Direct path to a weight file (overrides --model).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify against the pinned hash in source code (instead of computing).",
    )
    args = parser.parse_args()

    if args.path:
        expected = None
        if args.verify:
            logger.warning("--verify with --path: no expected hash, computing only.")
        return _compute_and_report(args.path, expected=expected)

    if args.model == "vesselfm":
        path = _find_vesselfm_cache()
        if path is None:
            logger.error(
                "VesselFM weights not found in HuggingFace cache. "
                "Download first with: "
                "uv run python -c "
                '"from huggingface_hub import hf_hub_download; '
                "hf_hub_download('bwittmann/vesselFM', 'vesselFM_base.pt')\"",
            )
            return 1

        expected = None
        if args.verify:
            from minivess.adapters.vesselfm import VESSELFM_WEIGHT_SHA256

            if VESSELFM_WEIGHT_SHA256 == "POPULATE_AFTER_FIRST_VERIFIED_DOWNLOAD":
                logger.warning(
                    "VESSELFM_WEIGHT_SHA256 is still the placeholder. "
                    "Run without --verify first to compute the hash."
                )
                expected = None
            else:
                expected = VESSELFM_WEIGHT_SHA256

        return _compute_and_report(path, expected=expected)

    if args.model == "sam3":
        path = _find_sam3_cache()
        if path is None:
            logger.error(
                "SAM3 weights not found in HuggingFace cache. "
                "SAM3 is a gated model — request access at "
                "https://huggingface.co/facebook/sam3 first."
            )
            return 1

        # SAM3 is multi-shard — print the snapshot directory info
        logger.info("SAM3 cache directory: %s", path)
        logger.info(
            "SAM3 uses HuggingFace revision pinning (not file-level SHA-256) "
            "because it is a multi-shard model. Check weights.hf_revision in "
            "configs/model_profiles/sam3_*.yaml."
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
