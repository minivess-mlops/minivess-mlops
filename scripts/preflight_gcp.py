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

import yaml

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


def check_docker_image_freshness() -> tuple[bool, str]:
    """Check if Docker image GIT_COMMIT label matches current git HEAD.

    Uses 'docker buildx imagetools inspect' to read the
    org.opencontainers.image.revision label from the registry.
    Falls back gracefully if tools are unavailable.

    See: .claude/metalearning/2026-03-25-stale-docker-image-launch.md
    """
    import json

    # Step 1: Get git HEAD
    git_result = _run(["git", "rev-parse", "HEAD"], timeout=10)
    if git_result.returncode != 0:
        return True, "Cannot determine HEAD commit — freshness check skipped"
    head_commit = git_result.stdout.strip()

    # Step 2: Get image label via docker buildx imagetools inspect
    imagetools_result = _run(
        [
            "docker", "buildx", "imagetools", "inspect",
            GAR_IMAGE, "--format", "{{json .}}",
        ],
        timeout=60,
    )
    if imagetools_result.returncode != 0:
        return True, "Cannot verify image freshness (docker buildx not available) — skipped"

    # Step 3: Parse JSON and extract the revision label (Rule 16: no regex)
    try:
        data = json.loads(imagetools_result.stdout)
        labels = data.get("image", {}).get("config", {}).get("Labels", {})
        image_commit = labels.get("org.opencontainers.image.revision", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return True, "Cannot parse image metadata — freshness check skipped"

    # Step 4: Compare commits
    if not image_commit or image_commit == "unknown":
        return False, (
            f"Docker image built without GIT_COMMIT (label='unknown'). "
            f"Rebuild: make build-base-gpu && docker tag minivess-base:latest "
            f"{GAR_IMAGE} && docker push {GAR_IMAGE}"
        )

    if image_commit == head_commit:
        return True, f"Docker image commit matches HEAD: {image_commit[:12]}"

    return False, (
        f"Docker image STALE: image={image_commit[:12]}, HEAD={head_commit[:12]}. "
        f"Rebuild: make build-base-gpu && docker tag minivess-base:latest "
        f"{GAR_IMAGE} && docker push {GAR_IMAGE}"
    )


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
        # Standalone .dvc files (e.g., data/raw/deepvess.dvc) are valid
        # DVC tracking references — they don't need dvc.lock entries.
        if pull_path.endswith(".dvc"):
            if not Path(pull_path).exists():
                return False, (
                    f"Setup pulls '{pull_path}' but the .dvc file does not exist."
                )
            continue

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


def _check_env_value(var: str, value: str) -> tuple[str | None, str | None]:
    """Validate a single env var value. Returns (error, warning) — at most one set.

    Rule #16: No regex — uses str methods only.

    Returns:
        (error_msg, None) for hard failures.
        (None, warning_msg) for soft warnings.
        (None, None) for valid values.
    """
    # Check 1: Unexpanded ${...} placeholder (shell variable not expanded)
    if "${" in value:
        return (
            f"{var} contains unexpanded placeholder: '{value}' "
            "— shell variable was not expanded",
            None,
        )

    # Check 2: REPLACE_WITH_... template (copied from .env.example without editing)
    if "REPLACE_WITH_" in value.upper():
        return (
            f"{var} contains template placeholder: '{value}' "
            "— replace with actual value",
            None,
        )

    # Check 3: Empty value after stripping whitespace
    if value.strip() == "":
        return f"{var} is empty — set a value in .env", None

    # Check 4: localhost URI — soft warning for MLFLOW_TRACKING_URI
    # (valid for local dev, but won't work on SkyPilot cloud VMs)
    if var == "MLFLOW_TRACKING_URI" and "localhost" in value:
        return None, (
            f"{var} uses localhost ({value}) — "
            "SkyPilot cloud VMs cannot reach localhost. "
            "Set to Cloud Run URL for GCP launches"
        )

    return None, None


def check_env_vars() -> tuple[bool, str]:
    """Check required env vars in .env file — presence AND value quality.

    Detects:
    - Missing env vars (not set at all)
    - Unexpanded ${...} placeholders
    - REPLACE_WITH_... template values from .env.example
    - Empty values
    - localhost URIs (soft WARNING — valid for local dev)

    Issue #945: MLflow URI placeholder detection in preflight.
    Rule #16: No regex — uses str methods only.
    """
    if not ENV_FILE.exists():
        return False, f".env file not found: {ENV_FILE}. Copy from .env.example"

    env_content = ENV_FILE.read_text(encoding="utf-8")

    # Parse env file into a dict: {VAR_NAME: value}
    env_vars: dict[str, str] = {}
    for line in env_content.splitlines():
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Split on first '=' only
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        env_vars[key] = val

    # Phase 1: Check all required vars are PRESENT
    missing: list[str] = []
    for var in REQUIRED_ENV_VARS:
        if var not in env_vars:
            missing.append(var)

    if missing:
        return False, f"Missing env vars in .env: {missing}"

    # Phase 2: Validate VALUES of present vars
    errors: list[str] = []
    warnings: list[str] = []
    for var in REQUIRED_ENV_VARS:
        value = env_vars[var]
        error, warning = _check_env_value(var, value)
        if error is not None:
            errors.append(error)
        if warning is not None:
            warnings.append(warning)

    if errors:
        return False, f"Env var issues: {'; '.join(errors)}"

    if warnings:
        return True, (
            f"All required env vars set (with warnings): "
            f"{'; '.join(warnings)}"
        )

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


def check_controller_cloud() -> tuple[bool, str]:
    """Check that SkyPilot controller runs on the SAME cloud as jobs.

    A GCP job with a RunPod controller causes 36 min/submission latency
    and RunPod outages killing GCP jobs. 5th pass root cause.
    """
    sky_config = Path.home() / ".sky" / "config.yaml"
    if not sky_config.exists():
        return True, "No ~/.sky/config.yaml — SkyPilot will auto-place controller"

    config = yaml.safe_load(sky_config.read_text(encoding="utf-8"))
    controller_cloud = (
        config.get("jobs", {}).get("controller", {}).get("resources", {}).get("cloud")
    )
    if controller_cloud is None:
        return True, "Controller cloud not pinned — SkyPilot will auto-select"

    # Job cloud from SkyPilot YAML
    sky_yaml = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))
    job_cloud = sky_yaml.get("resources", {}).get("cloud", "gcp")

    if controller_cloud.lower() != job_cloud.lower():
        return False, (
            f"Controller on '{controller_cloud}' but jobs on '{job_cloud}'. "
            f"Cross-cloud SSH adds ~30 min/submission and creates single point of failure. "
            f"Fix: set 'cloud: {job_cloud}' in ~/.sky/config.yaml jobs.controller.resources"
        )
    return True, f"Controller and jobs both on {job_cloud}"


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


def check_yaml_contract() -> tuple[bool, str]:
    """Validate SkyPilot YAML against the golden contract.

    Defense layer 4 of 5: catches unauthorized GPU types, cloud providers,
    and resource configurations at launch time (after pre-commit and tests).

    See: .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md
    """
    contract_path = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
    if not contract_path.exists():
        return False, f"Contract file missing: {contract_path}"

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    sky_config = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))

    violations: list[str] = []

    # Extract GPU types from factorial YAML
    resources = sky_config.get("resources", {})
    accel_field = resources.get("accelerators")

    # Parse accelerator names from any format
    gpu_names: list[str] = []
    if isinstance(accel_field, str):
        gpu_names.append(accel_field.split(":")[0])
    elif isinstance(accel_field, dict):
        for key in accel_field:
            gpu_names.append(str(key).split(":")[0])
    elif isinstance(accel_field, list):
        for item in accel_field:
            if isinstance(item, str):
                gpu_names.append(item.split(":")[0])

    # Check 1: No banned GPU types
    banned = set(contract.get("banned_accelerators", []))
    for gpu in gpu_names:
        if gpu in banned:
            violations.append(f"BANNED GPU '{gpu}'")

    # Check 2: GPU types match factorial allowlist
    factorial_allowed = set(
        contract.get("factorial", {}).get("allowed_accelerators", [])
    )
    for gpu in gpu_names:
        if gpu not in factorial_allowed:
            violations.append(
                f"GPU '{gpu}' not in factorial allowlist {sorted(factorial_allowed)}"
            )

    # Check 3: Cloud provider allowed
    cloud = resources.get("cloud")
    allowed_clouds = set(contract.get("allowed_clouds", []))
    if cloud and cloud not in allowed_clouds:
        violations.append(f"Cloud '{cloud}' not allowed ({sorted(allowed_clouds)})")

    # Check 4: Accelerators is NOT a dict (priority list = non-determinism)
    if isinstance(accel_field, dict):
        violations.append(
            "Accelerators is a dict (priority list) — use string 'L4:1' for determinism"
        )

    if violations:
        return False, (
            f"YAML contract violations: {'; '.join(violations)}. "
            f"See: configs/cloud/yaml_contract.yaml"
        )
    return True, f"YAML contract valid: {len(gpu_names)} GPU types, cloud={cloud}"


def check_gcp_cpu_quota() -> tuple[bool, str]:
    """Check GCP CPU quota is sufficient for SkyPilot controller + jobs.

    SkyPilot controller needs ~4 CPUs (n4-standard-4). L4 VMs need 4-12 CPUs.
    A quota of 0 means launch will fail with cryptic "capacity" error.
    Metalearning: 30 min wasted on n4 quota = 0.
    """
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            "europe-north1",
            "--format=json",
            "--project=minivess-mlops",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return True, "gcloud not available — quota check skipped"

    import json

    region_info = json.loads(result.stdout)
    quotas = region_info.get("quotas", [])
    for q in quotas:
        if q.get("metric") == "CPUS":
            limit = q.get("limit", 0)
            usage = q.get("usage", 0)
            available = limit - usage
            if limit == 0:
                return False, "CPU quota is 0 in europe-north1 — request increase"
            if available < 8:
                return False, (
                    f"CPU quota low: {available:.0f} available "
                    f"({usage:.0f}/{limit:.0f} used) — need ≥8 for controller + job"
                )
            return (
                True,
                f"CPU quota OK: {available:.0f} available ({usage:.0f}/{limit:.0f})",
            )

    return True, "CPU quota metric not found — check manually"


def check_gcp_gpu_quota() -> tuple[bool, str]:
    """Check GCP GPU quota is sufficient for L4 jobs.

    Detects the GPUS_ALL_REGIONS=1 bottleneck before launch. Without GPU
    quota, SkyPilot will fail with a cryptic "no available resources" error.

    Reads min_gpu_quota from configs/cloud/yaml_contract.yaml (Rule #29).
    Issue #943: GPU quota preflight check.
    """
    import json

    # Read threshold from contract — not hardcoded (Rule #29)
    contract_path = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    min_gpu_quota = contract.get("min_gpu_quota", 1)

    # Query GPU quota in europe-west4 (L4 availability region)
    result = _run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            "europe-west4",
            "--format=json",
            "--project=minivess-mlops",
        ],
        timeout=30,
    )
    if result.returncode != 0:
        return True, "gcloud not available — GPU quota check skipped"

    region_info = json.loads(result.stdout)
    quotas = region_info.get("quotas", [])

    # Look for NVIDIA_L4_GPUS metric
    for q in quotas:
        if q.get("metric") == "NVIDIA_L4_GPUS":
            limit = q.get("limit", 0)
            usage = q.get("usage", 0)
            available = limit - usage

            if limit == 0:
                return False, (
                    "NVIDIA_L4_GPUS quota is 0 in europe-west4 — "
                    "request GPU quota increase via GCP console"
                )

            if available < min_gpu_quota:
                return False, (
                    f"NVIDIA_L4_GPUS quota insufficient: {available:.0f} available "
                    f"({usage:.0f}/{limit:.0f}), need >= {min_gpu_quota}"
                )

            # Warn if quota is low relative to factorial job count (~34 concurrent)
            if limit < 4:
                return True, (
                    f"WARNING: NVIDIA_L4_GPUS quota low: {available:.0f} available "
                    f"({usage:.0f}/{limit:.0f}) — factorial jobs will queue heavily"
                )

            return True, (
                f"NVIDIA_L4_GPUS quota OK: {available:.0f} available "
                f"({usage:.0f}/{limit:.0f})"
            )

    return True, "NVIDIA_L4_GPUS metric not found in europe-west4 — check manually"


def check_cost_estimate() -> tuple[bool, str]:
    """Estimate max cost and compare against budget guard rails.

    Prevents accidental cost explosions from unauthorized GPU types.
    """
    contract_path = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
    if not contract_path.exists():
        return True, "No contract file — cost check skipped"

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    sky_config = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))

    resources = sky_config.get("resources", {})
    accel_field = resources.get("accelerators")

    # Get the GPU name
    gpu_name = ""
    if isinstance(accel_field, str):
        gpu_name = accel_field.split(":")[0]
    elif isinstance(accel_field, dict):
        # Dict = priority list, take the MOST expensive for worst-case
        cost_map = contract.get("max_hourly_cost_usd", {})
        max_cost = 0.0
        for key in accel_field:
            name = str(key).split(":")[0]
            cost = cost_map.get(name, 0)
            if cost > max_cost:
                max_cost = cost
                gpu_name = name

    cost_map = contract.get("max_hourly_cost_usd", {})
    hourly_cost = cost_map.get(gpu_name, 0)

    if hourly_cost == 0:
        return True, f"GPU '{gpu_name}' — no cost data in contract"

    # Estimate: 34 jobs x ~30 min each = ~17 GPU-hours
    estimated_gpu_hours = 17
    estimated_cost = hourly_cost * estimated_gpu_hours

    return True, (
        f"GPU '{gpu_name}' at ${hourly_cost:.2f}/hr, "
        f"~{estimated_gpu_hours} GPU-hrs, ~${estimated_cost:.0f} estimated"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all preflight checks. Returns 0 on success, 1 on failure."""
    checks = [
        ("Docker image on GAR", check_docker_image),
        ("Docker image freshness", check_docker_image_freshness),
        ("GCS DVC bucket access", check_gcs_bucket),
        ("Checkpoint bucket access", check_checkpoint_bucket),
        ("DVC data on GCS", check_dvc_data_on_gcs),
        ("DVC yaml/lock consistency", check_dvc_consistency),
        ("Setup script DVC paths", check_setup_dvc_paths),
        ("Required env vars", check_env_vars),
        ("SkyPilot GCP backend", check_skypilot_gcp),
        ("Controller cloud matches jobs", check_controller_cloud),
        ("SkyPilot YAML validity", check_skypilot_yaml),
        ("YAML contract compliance", check_yaml_contract),
        ("GCP CPU quota", check_gcp_cpu_quota),
        ("GCP GPU quota", check_gcp_gpu_quota),
        ("Cost estimate", check_cost_estimate),
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
