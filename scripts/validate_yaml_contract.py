"""Validate SkyPilot YAMLs against the golden contract.

Pre-commit hook + standalone validator that catches unauthorized YAML
modifications BEFORE they are committed. Defense layer 1 of 5.

Checks:
    1. GPU types in accelerators match configs/cloud/yaml_contract.yaml
    2. No banned GPU types (T4, V100, etc.)
    3. Cloud providers match the two-provider architecture
    4. No unauthorized top-level keys
    5. No unauthorized resource keys
    6. Factorial YAML has EXACTLY L4 on GCP with spot enabled

Usage:
    # Pre-commit hook (runs on staged files):
    uv run python scripts/validate_yaml_contract.py

    # Standalone (validate all SkyPilot YAMLs):
    uv run python scripts/validate_yaml_contract.py --all

    # Check a specific file:
    uv run python scripts/validate_yaml_contract.py deployment/skypilot/train_factorial.yaml

See: .claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md
See: .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
SKYPILOT_DIR = REPO_ROOT / "deployment" / "skypilot"
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"


def load_contract() -> dict:
    """Load the golden contract YAML."""
    if not CONTRACT_PATH.exists():
        print(f"FATAL: Contract file missing: {CONTRACT_PATH}")
        sys.exit(2)
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def extract_accelerator_names(accel_field: object) -> list[str]:
    """Extract GPU type names from SkyPilot accelerator field.

    SkyPilot supports multiple formats:
        - String: "L4:1"
        - Dict: {L4: 1, A100: 1}
        - List: ["L4:1", "A100:1"]
        - None (local, no GPU)
    """
    if accel_field is None:
        return []

    names: list[str] = []
    if isinstance(accel_field, str):
        # "L4:1" → "L4"
        names.append(accel_field.split(":")[0])
    elif isinstance(accel_field, dict):
        # {L4: 1, A100-80GB: 1} → ["L4", "A100-80GB"]
        for key in accel_field:
            names.append(str(key).split(":")[0])
    elif isinstance(accel_field, list):
        for item in accel_field:
            if isinstance(item, str):
                names.append(item.split(":")[0])
            elif isinstance(item, dict):
                for key in item:
                    names.append(str(key).split(":")[0])
    return names


def _is_volume_yaml(config: dict) -> bool:
    """Detect SkyPilot volume definition files (not task files).

    Volume YAMLs have 'type' like 'runpod-network-volume' and 'size' like '50Gi'.
    They use a different schema than task YAMLs and are excluded from task validation.
    """
    return "type" in config and ("size" in config or "infra" in config)


def validate_yaml(yaml_path: Path, contract: dict) -> list[str]:
    """Validate a single SkyPilot YAML against the contract.

    Returns list of violation messages (empty = valid).
    """
    violations: list[str] = []
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        violations.append(f"{yaml_path.name}: Did not parse to a dict")
        return violations

    # Skip volume definition YAMLs — they have a different schema
    if _is_volume_yaml(config):
        return violations

    fname = yaml_path.name

    # ── Check 1: No banned GPU types ────────────────────────────────────
    resources = config.get("resources", {})
    accel_field = resources.get("accelerators")
    gpu_names = extract_accelerator_names(accel_field)

    banned = set(contract.get("banned_accelerators", []))
    for gpu in gpu_names:
        if gpu in banned:
            violations.append(
                f"{fname}: BANNED GPU type '{gpu}' in accelerators. "
                f"Banned list: {sorted(banned)}"
            )

    # ── Check 2: GPU types match allowed list for cloud ─────────────────
    cloud = resources.get("cloud")
    if cloud and cloud in contract.get("allowed_accelerators", {}):
        allowed_for_cloud = set(contract["allowed_accelerators"][cloud])
        for gpu in gpu_names:
            if gpu not in allowed_for_cloud:
                violations.append(
                    f"{fname}: GPU '{gpu}' is NOT in allowed list for {cloud}. "
                    f"Allowed: {sorted(allowed_for_cloud)}. "
                    f"Adding a GPU type requires EXPLICIT user authorization."
                )

    # Also check any_of and ordered blocks (multi-region failover)
    for block_name in ("any_of", "ordered"):
        block = resources.get(block_name, [])
        for i, option in enumerate(block):
            if isinstance(option, dict):
                opt_cloud = option.get("cloud", cloud)
                opt_accel = option.get("accelerators")
                opt_gpus = extract_accelerator_names(opt_accel)
                for gpu in opt_gpus:
                    if gpu in banned:
                        violations.append(
                            f"{fname}: BANNED GPU '{gpu}' in {block_name}[{i}]"
                        )
                    if opt_cloud and opt_cloud in contract.get(
                        "allowed_accelerators", {}
                    ):
                        allowed = set(contract["allowed_accelerators"][opt_cloud])
                        if gpu not in allowed:
                            violations.append(
                                f"{fname}: GPU '{gpu}' NOT allowed for "
                                f"{opt_cloud} in {block_name}[{i}]. "
                                f"Allowed: {sorted(allowed)}"
                            )

    # ── Check 3: Cloud provider allowed ─────────────────────────────────
    allowed_clouds = set(contract.get("allowed_clouds", []))
    if cloud and cloud not in allowed_clouds:
        violations.append(
            f"{fname}: Cloud '{cloud}' not in allowed list: {sorted(allowed_clouds)}"
        )
    for block_name in ("any_of", "ordered"):
        block = resources.get(block_name, [])
        for i, option in enumerate(block):
            if isinstance(option, dict):
                opt_cloud = option.get("cloud")
                if opt_cloud and opt_cloud not in allowed_clouds:
                    violations.append(
                        f"{fname}: Cloud '{opt_cloud}' in {block_name}[{i}] "
                        f"not allowed. Allowed: {sorted(allowed_clouds)}"
                    )

    # ── Check 4: No unauthorized top-level keys ────────────────────────
    allowed_top = set(contract.get("allowed_top_level_keys", []))
    for key in config:
        if key not in allowed_top:
            violations.append(
                f"{fname}: Unauthorized top-level key '{key}'. "
                f"Allowed: {sorted(allowed_top)}"
            )

    # ── Check 5: No unauthorized resource keys ─────────────────────────
    allowed_res = set(contract.get("allowed_resource_keys", []))
    for key in resources:
        if key not in allowed_res:
            violations.append(
                f"{fname}: Unauthorized resource key '{key}'. "
                f"Allowed: {sorted(allowed_res)}"
            )

    # ── Check 6: Factorial-specific strict checks ──────────────────────
    if yaml_path.name == "train_factorial.yaml":
        factorial = contract.get("factorial", {})

        # Only allowed accelerators
        factorial_allowed = set(factorial.get("allowed_accelerators", []))
        for gpu in gpu_names:
            if gpu not in factorial_allowed:
                violations.append(
                    f"{fname}: FACTORIAL VIOLATION — GPU '{gpu}' not in "
                    f"factorial allowlist {sorted(factorial_allowed)}. "
                    f"This YAML controls 34+ jobs. Unauthorized GPU types "
                    f"can multiply cost by 5.5x."
                )

        # Must use GCP
        req_cloud = factorial.get("required_cloud")
        if req_cloud and cloud != req_cloud:
            violations.append(
                f"{fname}: FACTORIAL — must use cloud={req_cloud}, got {cloud}"
            )

        # Must use spot
        if factorial.get("require_spot") and not resources.get("use_spot"):
            violations.append(f"{fname}: FACTORIAL — use_spot must be true")

        # Must use Docker
        image_id = resources.get("image_id", "")
        if factorial.get("require_docker_image") and not image_id:
            violations.append(
                f"{fname}: FACTORIAL — image_id required (bare VM banned)"
            )

    return violations


def main() -> int:
    """Validate SkyPilot YAMLs. Returns 0 on success, 1 on violations."""
    contract = load_contract()

    # Determine which files to validate
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        yaml_files = sorted(SKYPILOT_DIR.glob("*.yaml"))
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        yaml_files = [Path(sys.argv[1])]
    else:
        # Default: validate all SkyPilot YAMLs (suitable for pre-commit)
        yaml_files = sorted(SKYPILOT_DIR.glob("*.yaml"))

    all_violations: list[str] = []

    for yaml_path in yaml_files:
        if not yaml_path.exists():
            all_violations.append(f"File not found: {yaml_path}")
            continue
        violations = validate_yaml(yaml_path, contract)
        all_violations.extend(violations)

    if all_violations:
        print("YAML Contract Violations Found:")
        print("=" * 60)
        for v in all_violations:
            print(f"  VIOLATION: {v}")
        print()
        print("Fix: Remove unauthorized GPU types, keys, or cloud providers.")
        print("Contract: configs/cloud/yaml_contract.yaml")
        print(
            "See: .claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md"
        )
        return 1

    print(f"YAML contract validation passed ({len(yaml_files)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
