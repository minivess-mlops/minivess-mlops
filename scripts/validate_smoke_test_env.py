"""Pre-flight validation for RunPod GPU smoke tests (#637, T3.1, RC13).

Checks all prerequisites before spending money on cloud GPU instances:
1. Required env vars (DVC_S3_*, RUNPOD_API_KEY, MLFLOW_TRACKING_URI)
2. DVC connectivity to UpCloud S3
3. MLflow health endpoint
4. SkyPilot RunPod backend availability

Usage:
    uv run python scripts/validate_smoke_test_env.py
    # or: make smoke-test-preflight
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request


def _check_env_vars() -> list[str]:
    """Check required env vars are set. Returns list of missing vars."""
    required = [
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "DVC_S3_BUCKET",
        "RUNPOD_API_KEY",
        "MLFLOW_TRACKING_URI",
        "HF_TOKEN",
    ]
    return [v for v in required if not os.environ.get(v)]


def _check_mlflow_health() -> bool:
    """Test MLflow health endpoint."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri or not uri.startswith("http"):
        return False
    try:
        req = urllib.request.Request(f"{uri}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  MLflow health check failed: {e}")
        return False


def _check_skypilot() -> bool:
    """Check SkyPilot is installed and RunPod is configured."""
    # Try 'sky' directly first (works in Docker/.venv on PATH),
    # fall back to 'uv run sky' for local dev.
    for cmd in [
        ["sky", "check", "runpod"],
        ["uv", "run", "sky", "check", "runpod"],
    ]:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def main() -> int:
    """Run all pre-flight checks. Returns 0 if all pass, 1 if any fail."""
    all_ok = True

    print("=== Smoke Test Pre-Flight Checks ===\n")

    # 1. Env vars
    missing = _check_env_vars()
    if missing:
        print(f"FAIL: Missing env vars: {', '.join(missing)}")
        print("  Set them in .env and source it, or export directly.\n")
        all_ok = False
    else:
        print("PASS: All required env vars are set.\n")

    # 2. MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow_uri and mlflow_uri.startswith("http"):
        if _check_mlflow_health():
            print(f"PASS: MLflow at {mlflow_uri} is healthy.\n")
        else:
            print("FAIL: MLflow health check failed.\n")
            all_ok = False
    else:
        print("SKIP: MLFLOW_TRACKING_URI not set to HTTP URL — cannot check MLflow.\n")

    # 3. SkyPilot
    if _check_skypilot():
        print("PASS: SkyPilot RunPod backend available.\n")
    else:
        print("FAIL: SkyPilot RunPod check failed.")
        print("  Install: uv sync --extra infra && sky check\n")
        all_ok = False

    if all_ok:
        print("=== All checks passed. Ready for GPU smoke test. ===")
        return 0
    print("=== Some checks failed. Fix issues above before running smoke test. ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
