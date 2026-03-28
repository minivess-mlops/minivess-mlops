"""Update tests/health_baseline.json with current codebase state.

Run after:
- Fixing lint errors (updates ruff count)
- Adding/removing tests (updates test count)
- Docker rebuild + push (updates deployment state with --docker-pushed)
- Pulumi deploy (updates deployment state with --pulumi-deployed)

Usage:
  uv run python scripts/update_health_baseline.py              # Update test + ruff counts
  uv run python scripts/update_health_baseline.py --docker-pushed   # Mark Docker as fresh
  uv run python scripts/update_health_baseline.py --pulumi-deployed # Mark Pulumi as fresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "tests" / "health_baseline.json"


def _get_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip() or "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_ruff_errors() -> int:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "src/", "tests/", "scripts/", "--output-format", "json"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return 0
    try:
        return len(json.loads(result.stdout))
    except (json.JSONDecodeError, TypeError):
        return -1


def _count_collected_tests() -> int:
    result = subprocess.run(
        [
            "uv", "run", "pytest", "--collect-only", "-q",
            "tests/v2/unit/", "tests/unit/",
            "--timeout=30",
        ],
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "MINIVESS_ALLOW_HOST": "1"},
    )
    for line in result.stdout.splitlines():
        if "tests collected" in line or "test collected" in line:
            first_token = line.strip().split()[0]
            num_str = first_token.split("/")[0]
            try:
                return int(num_str)
            except ValueError:
                pass
    return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Update health baseline")
    parser.add_argument("--docker-pushed", action="store_true", help="Mark Docker as freshly pushed")
    parser.add_argument("--pulumi-deployed", action="store_true", help="Mark Pulumi as freshly deployed")
    args = parser.parse_args()

    # Load existing baseline or create new
    if BASELINE_PATH.exists():
        with BASELINE_PATH.open(encoding="utf-8") as f:
            baseline = json.load(f)
    else:
        baseline = {"schema_version": 1}

    commit = _get_git_sha()
    baseline["updated"] = datetime.now(UTC).isoformat()
    baseline["updated_by_commit"] = commit

    # Update ruff count
    ruff_errors = _count_ruff_errors()
    if ruff_errors >= 0:
        baseline.setdefault("ruff", {})["error_count"] = ruff_errors
        print(f"Ruff errors: {ruff_errors}")

    # Update test count
    collected = _count_collected_tests()
    if collected >= 0:
        baseline.setdefault("test_staging", {})["collected"] = collected
        print(f"Tests collected: {collected}")

    # Update file hashes
    repo_root = Path(__file__).resolve().parent.parent
    deploy_state = baseline.setdefault("deployment_state", {})
    deploy_state["pyproject_toml_sha256"] = _sha256_file(repo_root / "pyproject.toml")
    deploy_state["uv_lock_sha256"] = _sha256_file(repo_root / "uv.lock")

    if args.docker_pushed:
        deploy_state["docker_base_gpu_commit"] = commit
        deploy_state["docker_base_gpu_pushed_to_gar"] = True
        print(f"Docker base marked as pushed at commit {commit}")

    if args.pulumi_deployed:
        deploy_state["pulumi_last_up_commit"] = commit
        deploy_state["pulumi_pending_changes"] = False
        print(f"Pulumi marked as deployed at commit {commit}")

    # Write
    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")

    print(f"Updated {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
