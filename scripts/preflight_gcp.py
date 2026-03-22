"""GCP preflight checker — run BEFORE sky jobs launch.

Validates ALL prerequisites programmatically. Never asks the user.
Prevents burning cloud credits on FAILED_SETUP jobs.

Usage:
    uv run python scripts/preflight_gcp.py
    # or via run_factorial.sh (called automatically)

Checks:
    1. Docker image exists on GAR
    2. GCS DVC bucket accessible via ADC
    3. DVC data hashes present on GCS
    4. Required env vars set (.env file)
    5. SkyPilot GCP backend configured
    6. SkyPilot YAML parses without errors
    7. dvc.yaml/dvc.lock consistency (tracked vs pushed)
    8. Setup script references only pushed DVC paths

See: docs/planning/dvc-test-suite-improvement.xml
See: .claude/metalearning/2026-03-22-dvc-pull-untested-setup-script-failure.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GAR_IMAGE = "europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest"
GCS_BUCKET = "gs://minivess-mlops-dvc-data"
CHECKPOINT_BUCKET = "gs://minivess-mlops-checkpoints"
SKYPILOT_YAML = REPO_ROOT / "deployment" / "skypilot" / "train_factorial.yaml"
ENV_FILE = REPO_ROOT / ".env"
DVC_YAML = REPO_ROOT / "dvc.yaml"
DVC_LOCK = REPO_ROOT / "dvc.lock"

REQUIRED_ENV_VARS = ["HF_TOKEN", "MLFLOW_TRACKING_URI"]

# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_docker_image() -> tuple[bool, str]:
    """Check if Docker image exists on GAR."""
    result = _run(["docker", "manifest", "inspect", GAR_IMAGE])
    if result.returncode == 0:
        return True, f"Docker image exists: {GAR_IMAGE}"
    return False, f"Docker image NOT found: {GAR_IMAGE}. Rebuild and push."


def check_gcs_bucket() -> tuple[bool, str]:
    """Check if GCS DVC bucket is accessible."""
    result = _run(["gsutil", "ls", GCS_BUCKET + "/"])
    if result.returncode == 0:
        return True, f"GCS bucket accessible: {GCS_BUCKET}"
    return False, (
        f"GCS bucket NOT accessible: {GCS_BUCKET}. "
        "Run: gcloud auth application-default login"
    )


def check_checkpoint_bucket() -> tuple[bool, str]:
    """Check if checkpoint GCS bucket is accessible."""
    result = _run(["gsutil", "ls", CHECKPOINT_BUCKET + "/"])
    if result.returncode == 0:
        return True, f"Checkpoint bucket accessible: {CHECKPOINT_BUCKET}"
    return False, f"Checkpoint bucket NOT accessible: {CHECKPOINT_BUCKET}"


def check_dvc_data_on_gcs() -> tuple[bool, str]:
    """Check if DVC raw data hash exists on GCS."""
    import yaml

    if not DVC_LOCK.exists():
        return False, "dvc.lock not found — run dvc repro first"

    lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    download_stage = lock.get("stages", {}).get("download", {})
    outs = download_stage.get("outs", [])
    if not outs:
        return False, "No outputs in dvc.lock download stage"

    raw_hash = outs[0].get("md5", "")
    if not raw_hash:
        return False, "No md5 hash in dvc.lock for data/raw/minivess"

    # DVC stores hashes on GCS as: files/md5/{first2chars}/{remaining}
    # .dir suffix is preserved for directory manifests
    prefix, suffix = raw_hash[:2], raw_hash[2:]
    gcs_path = f"{GCS_BUCKET}/files/md5/{prefix}/{suffix}"
    result = _run(["gsutil", "ls", gcs_path])
    # DC-5: Check both exit code AND stdout — gsutil ls can return 0 with empty output
    if result.returncode == 0 and result.stdout.strip():
        return True, f"DVC data hash found on GCS: {raw_hash}"
    return False, (
        f"DVC data hash NOT found on GCS: {raw_hash} "
        f"(checked: {gcs_path}). "
        "Run: dvc push data/raw/minivess -r gcs"
    )


def check_dvc_consistency() -> tuple[bool, str]:
    """Check dvc.yaml/dvc.lock consistency — tracked vs locked."""
    import yaml

    if not DVC_YAML.exists():
        return False, "dvc.yaml not found"
    if not DVC_LOCK.exists():
        return False, "dvc.lock not found"

    dvc_yaml = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
    dvc_lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    locked_stages = set(dvc_lock.get("stages", {}).keys())

    issues: list[str] = []
    for stage_name, stage in dvc_yaml.get("stages", {}).items():
        is_frozen = stage.get("frozen", False)
        has_lock = stage_name in locked_stages
        has_outs = bool(stage.get("outs"))

        if has_outs and not has_lock and not is_frozen:
            issues.append(
                f"Stage '{stage_name}' has outputs but no dvc.lock entry "
                "and is NOT frozen — data may not exist on remote"
            )

    if issues:
        return False, "; ".join(issues)
    return True, "dvc.yaml/dvc.lock consistent"


def check_setup_dvc_paths() -> tuple[bool, str]:
    """Check that SkyPilot setup dvc pull paths match locked DVC outputs."""
    import yaml

    if not SKYPILOT_YAML.exists():
        return False, f"SkyPilot YAML not found: {SKYPILOT_YAML}"

    sky_config = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))
    setup = sky_config.get("setup", "")

    # Find dvc pull commands in setup
    dvc_pull_paths: list[str] = []
    for line in setup.splitlines():
        line = line.strip()
        if "dvc pull" in line and "-r gcs" in line:
            # Extract path argument: dvc pull <path> -r gcs
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "pull" and i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if not next_part.startswith("-"):
                        dvc_pull_paths.append(next_part)

    if not dvc_pull_paths and "dvc pull" in setup and "dvc pull data/" not in setup:
        return False, (
            "Setup uses bare 'dvc pull -r gcs' without path filter. "
            "This will fail if any DVC-tracked output is not on GCS. "
            "Use: dvc pull data/raw/minivess -r gcs"
        )

    # Verify each pull path has a dvc.lock entry
    dvc_lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    locked_outs: set[str] = set()
    for stage in dvc_lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            locked_outs.add(out.get("path", ""))

    for pull_path in dvc_pull_paths:
        # Check if pull_path matches or is a parent of any locked output
        matched = any(
            lo.startswith(pull_path) or pull_path.startswith(lo) for lo in locked_outs
        )
        if not matched:
            return False, (
                f"Setup pulls '{pull_path}' but no matching output in dvc.lock. "
                "This path was never pushed to GCS."
            )

    return True, f"Setup DVC paths verified: {dvc_pull_paths or ['path-specific pull']}"


def check_env_vars() -> tuple[bool, str]:
    """Check required env vars in .env file."""
    if not ENV_FILE.exists():
        return False, f".env file not found: {ENV_FILE}. Copy from .env.example"

    env_content = ENV_FILE.read_text(encoding="utf-8")
    missing: list[str] = []
    for var in REQUIRED_ENV_VARS:
        # Check for VAR= with non-empty value
        found = False
        for line in env_content.splitlines():
            if line.startswith(f"{var}=") and len(line) > len(f"{var}="):
                found = True
                break
        if not found:
            missing.append(var)

    if missing:
        return False, f"Missing env vars in .env: {missing}"
    return True, f"All required env vars set: {REQUIRED_ENV_VARS}"


def check_skypilot_gcp() -> tuple[bool, str]:
    """Check if SkyPilot GCP backend is configured."""
    sky_bin = REPO_ROOT / ".venv" / "bin" / "sky"
    if not sky_bin.exists():
        return False, f"sky binary not found: {sky_bin}. Run: uv sync --extra infra"

    result = _run([str(sky_bin), "check", "gcp"])
    if "GCP: enabled" in result.stdout:
        return True, "SkyPilot GCP backend: enabled"
    return False, f"SkyPilot GCP not enabled. Run: {sky_bin} check gcp"


def check_skypilot_yaml() -> tuple[bool, str]:
    """Check if SkyPilot YAML parses without errors."""
    try:
        # Use sky.Task.from_yaml() for full validation
        sky_bin = REPO_ROOT / ".venv" / "bin" / "python"
        result = _run(
            [
                str(sky_bin),
                "-c",
                f"import sky; sky.Task.from_yaml('{SKYPILOT_YAML}')",
            ]
        )
        if result.returncode == 0:
            return True, f"SkyPilot YAML valid: {SKYPILOT_YAML.name}"
        return False, f"SkyPilot YAML invalid: {result.stderr[:200]}"
    except Exception as e:
        return False, f"SkyPilot YAML check failed: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all preflight checks. Returns 0 on success, 1 on failure."""
    checks = [
        ("Docker image on GAR", check_docker_image),
        ("GCS DVC bucket access", check_gcs_bucket),
        ("Checkpoint bucket access", check_checkpoint_bucket),
        ("DVC data on GCS", check_dvc_data_on_gcs),
        ("DVC yaml/lock consistency", check_dvc_consistency),
        ("Setup script DVC paths", check_setup_dvc_paths),
        ("Required env vars", check_env_vars),
        ("SkyPilot GCP backend", check_skypilot_gcp),
        ("SkyPilot YAML validity", check_skypilot_yaml),
    ]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  GCP Preflight Checks                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for name, check_fn in checks:
        try:
            ok, msg = check_fn()
        except Exception as e:
            ok, msg = False, f"Check crashed: {e}"

        status = "✓" if ok else "✗"
        print(f"  [{status}] {name}: {msg}")

        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((name, msg))

    print()
    if failed == 0:
        print(f"  ALL {passed} CHECKS PASSED — safe to launch")
        return 0

    print(f"  {failed} CHECK(S) FAILED — DO NOT LAUNCH")
    print()
    for name, msg in failures:
        print(f"  FIX: {name}")
        print(f"       {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
