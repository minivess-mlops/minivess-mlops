"""Pre-flight validation for RunPod dev environment (plan v3.1 T0.1-T0.5).

Checks all prerequisites before spending RunPod credits:
1. T0.1: Required env vars (9 total for dev environment)
2. T0.2: SkyPilot RunPod backend + version >= 0.6.0
3. T0.3: UpCloud MLflow health endpoint (with basic auth)
4. T0.4: DVC data synced to UpCloud S3 (350 files)
5. T0.5: Workdir sync size < 100 MB + critical files included

Every check is programmatic — this script NEVER asks the user a question.
Metalearning ML-4: check infrastructure state, don't ask about it.

GitHub: #694

Usage:
    uv run python scripts/validate_runpod_dev_env.py
    # or: make smoke-test-preflight (checks smoke env too)
"""

from __future__ import annotations

import base64
import importlib.metadata
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# T0.1: Required env var check
# ---------------------------------------------------------------------------

# All 9 env vars required for dev_runpod.yaml to work correctly
_REQUIRED_VARS = [
    "RUNPOD_API_KEY",
    "DVC_S3_ENDPOINT_URL",
    "DVC_S3_ACCESS_KEY",
    "DVC_S3_SECRET_KEY",
    "DVC_S3_BUCKET",
    "MLFLOW_CLOUD_URI",
    "MLFLOW_CLOUD_USERNAME",
    "MLFLOW_CLOUD_PASSWORD",
    "HF_TOKEN",
]


def check_env_vars() -> list[str]:
    """Return list of required env var names that are missing or empty.

    Returns:
        List of variable names that are unset or empty. Empty list = all present.
    """
    return [v for v in _REQUIRED_VARS if not os.environ.get(v)]


# ---------------------------------------------------------------------------
# T0.2: SkyPilot version + RunPod backend check
# ---------------------------------------------------------------------------

_MIN_SKYPILOT_VERSION = (0, 6, 0)


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string like '0.7.1' into tuple (0, 7, 1)."""
    parts = version_str.split(".")
    result = []
    for part in parts[:3]:
        # Strip any pre-release suffix (e.g. "0rc1")
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        result.append(int(digits) if digits else 0)
    return tuple(result)


def check_skypilot() -> tuple[bool, str]:
    """Check SkyPilot is installed (>= 0.6.0) and RunPod backend is configured.

    Returns:
        (ok, message): ok=True if all checks pass, message describes result.
    """
    # Version check via importlib.metadata (preferred over subprocess)
    try:
        version_str = importlib.metadata.version("skypilot")
        version = _parse_version(version_str)
        if version < _MIN_SKYPILOT_VERSION:
            min_str = ".".join(str(v) for v in _MIN_SKYPILOT_VERSION)
            return (
                False,
                f"SkyPilot {version_str} < {min_str} required. Run: uv sync --extra infra",
            )
    except importlib.metadata.PackageNotFoundError:
        # Try skypilot-nightly or similar
        for pkg in ["skypilot-nightly", "skypilot[runpod]"]:
            try:
                importlib.metadata.version(pkg)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
        else:
            return False, "SkyPilot not installed. Run: uv sync --extra infra"

    # RunPod backend check via sky check
    for cmd in [
        ["sky", "check", "runpod"],
        ["uv", "run", "sky", "check", "runpod"],
    ]:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return True, f"SkyPilot RunPod backend: OK (version {version_str})"

    return False, (
        "SkyPilot RunPod check failed. "
        "Verify RUNPOD_API_KEY is set and run: sky check runpod"
    )


# ---------------------------------------------------------------------------
# T0.3: MLflow health endpoint with basic auth
# ---------------------------------------------------------------------------


def check_mlflow_health() -> tuple[bool, str]:
    """Test MLflow /health endpoint with basic auth credentials from env.

    Returns:
        (ok, message): ok=True if endpoint returns HTTP 200.
    """
    uri = os.environ.get("MLFLOW_CLOUD_URI", "").rstrip("/")
    if not uri:
        return False, "MLFLOW_CLOUD_URI not set — cannot check MLflow health"

    username = os.environ.get("MLFLOW_CLOUD_USERNAME", "")
    password = os.environ.get("MLFLOW_CLOUD_PASSWORD", "")

    try:
        req = urllib.request.Request(f"{uri}/health")
        if username and password:
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            req.add_header("Authorization", f"Basic {credentials}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, f"MLflow at {uri} is healthy (HTTP 200)"
            return False, f"MLflow returned HTTP {resp.status} (expected 200)"
    except urllib.error.HTTPError as e:
        return False, f"MLflow HTTP error: {e.code} {e.reason} — check auth credentials"
    except urllib.error.URLError as e:
        return False, f"MLflow unreachable: {e.reason} — is the UpCloud server running?"
    except Exception as e:  # noqa: BLE001
        return False, f"MLflow health check failed: {e}"


# ---------------------------------------------------------------------------
# T0.4: DVC status check
# ---------------------------------------------------------------------------


def check_dvc_status() -> tuple[bool, str]:
    """Check DVC remote (upcloud) is in sync with local.

    Returns:
        (ok, message): ok=True if remote shows no new/modified entries.
    """
    result = subprocess.run(
        ["dvc", "status", "-r", "upcloud"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        return False, (
            f"DVC status failed: {result.stderr.strip() or result.stdout.strip()}. "
            "Run: dvc push -r upcloud"
        )

    output = result.stdout.strip()
    # Empty output or "Data and calculation will not change" means in sync
    if not output or "Data and calculation will not change" in output:
        return True, "DVC remote (upcloud) is in sync"

    # Check for "new" or "modified" entries in the output
    if (
        "new" in output.lower()
        or "modified" in output.lower()
        or "deleted" in output.lower()
    ):
        return False, (
            f"DVC remote out of sync:\n{output[:500]}\nRun: dvc push -r upcloud"
        )

    return True, f"DVC remote status: {output[:100] or 'ok'}"


# ---------------------------------------------------------------------------
# T0.5: .skyignore sync size + critical file inclusion
# ---------------------------------------------------------------------------

# Files that MUST be included in the rsync (not excluded by .skyignore)
_CRITICAL_SYNC_FILES = [
    "configs/splits/smoke_test_1fold_4vol.json",
    "configs/experiment/smoke_dynunet.yaml",
    "configs/experiment/smoke_sam3_hybrid.yaml",
]

_SKYIGNORE_EXCLUDES = [
    "/data/",
    "/data_cache/",
    ".dvc/",
    ".git/",
    "mlruns/",
    "mlruns-docker/",
    "checkpoints/",
    "dataset_local/",
    "deployment/pulumi/.venv/",
    "outputs/",
    "__pycache__/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".venv/",
    "venv/",
    ".idea/",
    ".vscode/",
    "deployment/docker/",
]


def _is_excluded(path: Path, root: Path) -> bool:
    """Check if a path should be excluded by .skyignore rules."""
    rel = path.relative_to(root)
    rel_str = str(rel)

    for pattern in _SKYIGNORE_EXCLUDES:
        clean = pattern.lstrip("/")
        if clean.endswith("/"):
            # Directory pattern
            dir_name = clean.rstrip("/")
            parts = rel.parts
            if dir_name in parts:
                return True
            # Check prefix match (e.g. "deployment/pulumi/.venv/")
            if rel_str.startswith(clean) or rel_str.startswith(clean.rstrip("/")):
                return True
        else:
            # File extension or glob pattern
            if path.name.endswith(clean.lstrip("*")) and clean.startswith("*"):
                return True

    return False


def check_sync_size(
    project_root: Path | None = None,
    max_size_mb: float = 100.0,
) -> tuple[int, list[str]]:
    """Walk project directory (respecting .skyignore) and check sync size.

    Args:
        project_root: Root directory to walk. Defaults to repo root.
        max_size_mb: Maximum allowed sync size in MB.

    Returns:
        (size_bytes, missing_critical): size in bytes and list of missing critical files.
    """
    root = project_root or ROOT
    total_bytes = 0
    included_files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path, root):
            continue
        total_bytes += path.stat().st_size
        included_files.append(path)

    # Check critical files are included
    missing_critical = []
    for critical in _CRITICAL_SYNC_FILES:
        critical_path = root / critical
        if critical_path not in included_files and not (root / critical).exists():
            missing_critical.append(critical)
        elif (root / critical).exists() and _is_excluded(root / critical, root):
            missing_critical.append(f"{critical} (incorrectly excluded by .skyignore)")

    return total_bytes, missing_critical


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all pre-flight checks for RunPod dev environment.

    Returns 0 if all pass, 1 if any fail.
    """
    all_ok = True
    print("=== RunPod Dev Environment Pre-Flight Checks ===\n")
    print("Metalearning ML-4: checking infrastructure programmatically\n")

    # T0.1: Env vars
    missing = check_env_vars()
    if missing:
        print(f"FAIL T0.1: Missing env vars: {', '.join(missing)}")
        print("  Add them to .env (see .env.example for all required vars)\n")
        all_ok = False
    else:
        print("PASS T0.1: All 9 required env vars are present.\n")

    # T0.2: SkyPilot
    ok, msg = check_skypilot()
    if ok:
        print(f"PASS T0.2: {msg}\n")
    else:
        print(f"FAIL T0.2: {msg}\n")
        all_ok = False

    # T0.3: MLflow health
    ok, msg = check_mlflow_health()
    if ok:
        print(f"PASS T0.3: {msg}\n")
    else:
        print(f"FAIL T0.3: {msg}\n")
        all_ok = False

    # T0.4: DVC status
    ok, msg = check_dvc_status()
    if ok:
        print(f"PASS T0.4: {msg}\n")
    else:
        print(f"FAIL T0.4: {msg}\n")
        all_ok = False

    # T0.5: Sync size
    size_bytes, missing_critical = check_sync_size(project_root=ROOT)
    size_mb = size_bytes / 1024 / 1024
    if size_mb > 100.0:
        print(f"FAIL T0.5: Sync size {size_mb:.1f} MB exceeds 100 MB limit.")
        print("  Check .skyignore for missing exclusions.\n")
        all_ok = False
    elif missing_critical:
        print(f"FAIL T0.5: Critical files missing from sync: {missing_critical}\n")
        all_ok = False
    else:
        print(
            f"PASS T0.5: Sync size {size_mb:.1f} MB (under 100 MB limit). Critical files included.\n"
        )

    if all_ok:
        print("=== All pre-flight checks passed. Ready to launch RunPod dev job. ===")
        print("  Next step: make dev-gpu MODEL=dynunet")
        return 0

    print("=== Some checks failed. Fix issues above before launching. ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
