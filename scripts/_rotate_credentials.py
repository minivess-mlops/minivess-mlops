"""Credential rotation utility for MinIVess MLOps.

Generates new random credentials and writes them to .env.
Rule #16 compliant: uses str.partition("=") to parse .env — no regex, no sed.

Usage:
    uv run python scripts/_rotate_credentials.py
    uv run python scripts/_rotate_credentials.py --dry-run

After rotating:
    1. Update OPTUNA_STORAGE_URL in .env to match new POSTGRES_PASSWORD
    2. Restart services: docker compose --env-file .env restart
    3. Encrypt new .env: sops --encrypt .env > .env.enc (team only)
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ENV_FILE = REPO_ROOT / ".env"

# Variables to rotate — generate a new cryptographically secure token for each.
ROTATE_TARGETS: set[str] = {
    "POSTGRES_PASSWORD",
    "MINIO_ROOT_PASSWORD",
    "LANGFUSE_SECRET",
    "LANGFUSE_SALT",
    "GRAFANA_PASSWORD",
}


def rotate(env_file: Path = ENV_FILE, *, dry_run: bool = False) -> int:
    """Rotate credentials in env_file. Returns number of variables rotated."""
    if not env_file.exists():
        print(f"ERROR: {env_file} not found.", file=sys.stderr)
        print("  Run: cp .env.example .env  then fill in your values.", file=sys.stderr)
        sys.exit(1)

    raw = env_file.read_text(encoding="utf-8")
    lines = raw.splitlines()
    new_lines: list[str] = []
    rotated: list[str] = []

    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _, _value = line.partition("=")
        key = key.strip()
        if key in ROTATE_TARGETS:
            new_value = secrets.token_hex(32)
            new_lines.append(f"{key}={new_value}")
            rotated.append(key)
        else:
            new_lines.append(line)

    if dry_run:
        print(
            f"[dry-run] Would rotate {len(rotated)} credentials: {', '.join(rotated)}"
        )
        return len(rotated)

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Rotated {len(rotated)} credentials: {', '.join(rotated)}")
    print()
    print("IMPORTANT next steps:")
    print("  1. Update OPTUNA_STORAGE_URL in .env to match new POSTGRES_PASSWORD")
    print("  2. Restart services: docker compose --env-file .env restart")
    print("  3. Team: re-encrypt: sops --encrypt .env > .env.enc")
    return len(rotated)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    rotate(dry_run=dry)
