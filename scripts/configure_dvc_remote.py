"""Configure DVC 'upcloud' remote from .env variables (#631, T0.2).

Reads DVC_S3_ENDPOINT_URL, DVC_S3_ACCESS_KEY, DVC_S3_SECRET_KEY from .env
and writes them to .dvc/config.local (gitignored). Also verifies S3
connectivity by listing the bucket.

Usage:
    uv run python scripts/configure_dvc_remote.py

Prerequisites:
    - .env file with DVC_S3_* variables set (see .env.example)
    - UpCloud Object Storage provisioned (via Pulumi or manual)
    - Get endpoint URL: pulumi stack output s3_endpoint
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Load .env if dotenv is available (optional — works without it if vars are exported)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

REQUIRED_VARS = [
    "DVC_S3_ENDPOINT_URL",
    "DVC_S3_ACCESS_KEY",
    "DVC_S3_SECRET_KEY",
    "DVC_S3_BUCKET",
]


def _check_env() -> dict[str, str]:
    """Validate required environment variables are set."""
    values: dict[str, str] = {}
    missing: list[str] = []
    for var in REQUIRED_VARS:
        val = os.environ.get(var, "")
        if not val:
            missing.append(var)
        else:
            values[var] = val
    if missing:
        msg = (
            f"Missing required .env variables: {', '.join(missing)}. "
            "Set them in .env (see .env.example) or export them."
        )
        raise RuntimeError(msg)
    return values


def _run_dvc(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a DVC command."""
    cmd = ["dvc", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def configure_remote(env: dict[str, str]) -> None:
    """Write upcloud remote credentials to .dvc/config.local."""
    remote_name = "upcloud"
    bucket = env["DVC_S3_BUCKET"]

    # Ensure remote exists in .dvc/config (committed template)
    result = _run_dvc("remote", "list")
    if remote_name not in (result.stdout or ""):
        print(f"Adding DVC remote '{remote_name}' with bucket s3://{bucket}")
        _run_dvc("remote", "add", remote_name, f"s3://{bucket}")

    # Set credentials in .dvc/config.local (gitignored)
    print("Configuring credentials in .dvc/config.local...")
    _run_dvc(
        "remote",
        "modify",
        "--local",
        remote_name,
        "endpointurl",
        env["DVC_S3_ENDPOINT_URL"],
    )
    _run_dvc(
        "remote",
        "modify",
        "--local",
        remote_name,
        "access_key_id",
        env["DVC_S3_ACCESS_KEY"],
    )
    _run_dvc(
        "remote",
        "modify",
        "--local",
        remote_name,
        "secret_access_key",
        env["DVC_S3_SECRET_KEY"],
    )
    print(f"DVC remote '{remote_name}' configured successfully.")


def verify_connectivity(env: dict[str, str]) -> bool:
    """Verify S3 connectivity by checking dvc status against the remote."""
    print("Verifying S3 connectivity...")
    result = _run_dvc("status", "-r", "upcloud")
    if result.returncode != 0:
        print(f"WARNING: DVC remote check failed: {result.stderr.strip()}")
        return False
    print("S3 connectivity verified.")
    return True


def main() -> None:
    """Configure DVC upcloud remote and verify connectivity."""
    env = _check_env()
    configure_remote(env)
    if not verify_connectivity(env):
        print(
            "WARNING: Could not verify S3 connectivity. "
            "Check DVC_S3_ENDPOINT_URL and credentials in .env."
        )
        sys.exit(1)
    print("\nDone. To push data: dvc push -r upcloud")


if __name__ == "__main__":
    main()
