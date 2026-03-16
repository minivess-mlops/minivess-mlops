"""Configure DVC 'remote_storage' remote for AWS S3 public bucket.

UpCloud archived 2026-03-16 — replaced by AWS S3 public bucket (s3://minivessdataset).
The remote_storage remote requires NO credentials (public bucket).

Data strategy:
  1. Network Volume cache-first: /opt/vol/data/raw/minivess/ (RunPod persistent NVMe)
  2. AWS S3 fallback: s3://minivessdataset (public, no credentials, DVC pull)

Usage:
    uv run python scripts/configure_dvc_remote.py

Prerequisites:
    - .dvc/config already has remote_storage pointing to s3://minivessdataset
    - No .env variables needed for public bucket
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Load .env if dotenv is available (optional)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# remote_storage is a public S3 bucket — no credentials required
REMOTE_NAME = "remote_storage"
REMOTE_URL = "s3://minivessdataset"


def _run_dvc(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a DVC command."""
    cmd = ["dvc", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def configure_remote(env: dict[str, str]) -> None:
    """Ensure remote_storage remote exists in .dvc/config.

    No credentials needed — s3://minivessdataset is a public bucket.
    """
    result = _run_dvc("remote", "list")
    if REMOTE_NAME not in (result.stdout or ""):
        print(f"Adding DVC remote '{REMOTE_NAME}' → {REMOTE_URL}")
        _run_dvc("remote", "add", REMOTE_NAME, REMOTE_URL)
    else:
        print(f"DVC remote '{REMOTE_NAME}' already configured → {REMOTE_URL}")
    # Public bucket: do NOT set access_key_id or secret_access_key
    print(
        f"DVC remote '{REMOTE_NAME}' ready (public S3 bucket, no credentials needed)."
    )


def verify_connectivity(env: dict[str, str]) -> bool:
    """Verify S3 connectivity by checking dvc status against remote_storage."""
    print("Verifying S3 connectivity...")
    result = _run_dvc("status", "-r", REMOTE_NAME)
    if result.returncode != 0:
        print(f"WARNING: DVC remote check failed: {result.stderr.strip()}")
        return False
    print("S3 connectivity verified.")
    return True


def main() -> None:
    """Configure DVC remote_storage (AWS S3 public) and verify connectivity."""
    env: dict[str, str] = {}
    configure_remote(env)
    if not verify_connectivity(env):
        print(
            "WARNING: Could not verify S3 connectivity. "
            "Check network access to s3://minivessdataset (eu-north-1)."
        )
        sys.exit(1)
    print(f"\nDone. To pull data: dvc pull -r {REMOTE_NAME}")
    print("(No push needed — this is a public read-only dataset bucket.)")


if __name__ == "__main__":
    main()
