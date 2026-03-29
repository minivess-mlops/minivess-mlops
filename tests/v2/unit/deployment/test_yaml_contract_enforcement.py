"""YAML Contract Enforcement Tests — Defense layer 3 of 5.

Catches unauthorized YAML modifications at test time (make test-staging).
Every test reads from the golden contract (configs/cloud/yaml_contract.yaml)
to avoid hardcoded allowlists in tests.

Defends against:
    - Unauthorized GPU types (A100 fallback incident, 2026-03-24)
    - Banned GPU types (T4, V100, etc.)
    - Unauthorized cloud providers (AWS incident, 2026-03-16)
    - SkyPilot YAML key drift (job_recovery removal, SkyPilot v1.0)
    - Factorial YAML cost escalation (5.5x A100 vs L4)
    - Accelerator mismatch between cloud config and SkyPilot YAML

See: .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md
See: .claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants — read from contract, never hardcode
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
SKYPILOT_DIR = REPO_ROOT / "deployment" / "skypilot"
CLOUD_CONFIG_DIR = REPO_ROOT / "configs" / "cloud"
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"
GCP_SPOT_CONFIG = CLOUD_CONFIG_DIR / "gcp_spot.yaml"
RUNPOD_DEV_CONFIG = CLOUD_CONFIG_DIR / "runpod_dev.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def contract() -> dict:
    """Load the golden YAML contract."""
    return _load_yaml(CONTRACT_PATH)


@pytest.fixture
def factorial_config() -> dict:
    """Load the factorial SkyPilot YAML."""
    return _load_yaml(FACTORIAL_YAML)


def _is_volume_yaml(path: Path) -> bool:
    """Detect SkyPilot volume definition files (not task files)."""
    config = _load_yaml(path)
    return "type" in config and ("size" in config or "infra" in config)


def _discover_skypilot_yamls() -> list[Path]:
    """Find all SkyPilot task YAML files (excludes volume definitions)."""
    if not SKYPILOT_DIR.exists():
        return []
    return sorted(p for p in SKYPILOT_DIR.glob("*.yaml") if not _is_volume_yaml(p))


def _extract_accelerator_names(accel_field: object) -> list[str]:
    """Extract GPU type names from any SkyPilot accelerator format."""
    if accel_field is None:
        return []
    names: list[str] = []
    if isinstance(accel_field, str):
        names.append(accel_field.split(":")[0])
    elif isinstance(accel_field, dict):
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


# ===========================================================================
# 1. Accelerator allowlist enforcement (per cloud)
# ===========================================================================


class TestAcceleratorsMatchCloudConfig:
    """GPU types in SkyPilot YAML must match configs/cloud/yaml_contract.yaml."""

    def test_factorial_accelerators_match_contract(
        self, contract: dict, factorial_config: dict
    ) -> None:
        """train_factorial.yaml accelerators MUST be EXACTLY the factorial allowlist."""
        factorial_allowed = set(
            contract.get("factorial", {}).get("allowed_accelerators", [])
        )
        accel_field = factorial_config.get("resources", {}).get("accelerators")
        actual_gpus = set(_extract_accelerator_names(accel_field))

        unauthorized = actual_gpus - factorial_allowed
        assert not unauthorized, (
            f"UNAUTHORIZED GPU types in train_factorial.yaml: {unauthorized}. "
            f"Only {sorted(factorial_allowed)} are allowed. "
            f"A100 was removed — 5.5x cost, never authorized by user. "
            f"See: .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md"
        )

    def test_factorial_accelerators_match_gcp_spot_config(self) -> None:
        """train_factorial.yaml accelerators must match configs/cloud/gcp_spot.yaml."""
        if not GCP_SPOT_CONFIG.exists():
            pytest.skip("configs/cloud/gcp_spot.yaml not found")

        gcp_config = _load_yaml(GCP_SPOT_CONFIG)
        factorial = _load_yaml(FACTORIAL_YAML)

        # Extract accelerator names from cloud config
        cloud_accels = gcp_config.get("accelerators", [])
        if isinstance(cloud_accels, list):
            allowed = {a.split(":")[0] for a in cloud_accels}
        elif isinstance(cloud_accels, str):
            allowed = {cloud_accels.split(":")[0]}
        else:
            allowed = set()

        # Extract from factorial
        accel_field = factorial.get("resources", {}).get("accelerators")
        actual = set(_extract_accelerator_names(accel_field))

        unauthorized = actual - allowed
        assert not unauthorized, (
            f"GPU types in train_factorial.yaml ({sorted(actual)}) "
            f"don't match configs/cloud/gcp_spot.yaml ({sorted(allowed)}). "
            f"SkyPilot YAML must use ONLY GPUs declared in the cloud config."
        )

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_unauthorized_gpu_types(self, yaml_path: Path, contract: dict) -> None:
        """Every SkyPilot YAML must only use GPUs allowed for its cloud provider."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        cloud = resources.get("cloud")
        accel_field = resources.get("accelerators")
        gpu_names = _extract_accelerator_names(accel_field)

        if not cloud or not gpu_names:
            return  # No cloud or no GPU (local)

        allowed_per_cloud = contract.get("allowed_accelerators", {})
        if cloud in allowed_per_cloud:
            allowed = set(allowed_per_cloud[cloud])
            unauthorized = set(gpu_names) - allowed
            assert not unauthorized, (
                f"{yaml_path.name}: Unauthorized GPU types {unauthorized} "
                f"for cloud={cloud}. Allowed: {sorted(allowed)}. "
                f"Adding a GPU type requires EXPLICIT user authorization."
            )


# ===========================================================================
# 2. Banned GPU types (global)
# ===========================================================================


class TestNoBannedGpuTypes:
    """No banned GPU type may appear in any SkyPilot YAML."""

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_banned_accelerators(self, yaml_path: Path, contract: dict) -> None:
        """Banned GPU types (T4, V100, etc.) must NEVER appear."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        accel_field = resources.get("accelerators")
        gpu_names = set(_extract_accelerator_names(accel_field))

        banned = set(contract.get("banned_accelerators", []))
        found_banned = gpu_names & banned
        assert not found_banned, (
            f"{yaml_path.name}: BANNED GPU types found: {found_banned}. "
            f"T4 (Turing, no BF16) and V100 (Volta, no BF16) are banned."
        )

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_banned_in_any_of(self, yaml_path: Path, contract: dict) -> None:
        """Banned GPU types must not appear in any_of blocks either."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        any_of = resources.get("any_of", [])
        banned = set(contract.get("banned_accelerators", []))

        for i, option in enumerate(any_of):
            if isinstance(option, dict):
                opt_accel = option.get("accelerators")
                opt_gpus = set(_extract_accelerator_names(opt_accel))
                found = opt_gpus & banned
                assert not found, (
                    f"{yaml_path.name}: BANNED GPU in any_of[{i}]: {found}"
                )


# ===========================================================================
# 3. Cloud provider enforcement
# ===========================================================================


class TestCloudProviderAllowlist:
    """Only GCP and RunPod are allowed. No AWS, Lambda, UpCloud."""

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_cloud_in_allowlist(self, yaml_path: Path, contract: dict) -> None:
        """Cloud provider must be in the two-provider allowlist."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        cloud = resources.get("cloud")
        allowed = set(contract.get("allowed_clouds", []))

        if cloud:
            assert cloud in allowed, (
                f"{yaml_path.name}: Cloud '{cloud}' not in allowed list "
                f"{sorted(allowed)}. Only GCP + RunPod are authorized."
            )

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_unauthorized_cloud_in_any_of(
        self, yaml_path: Path, contract: dict
    ) -> None:
        """any_of blocks must not reference unauthorized clouds."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        any_of = resources.get("any_of", [])
        allowed = set(contract.get("allowed_clouds", []))

        for i, option in enumerate(any_of):
            if isinstance(option, dict):
                opt_cloud = option.get("cloud")
                if opt_cloud:
                    assert opt_cloud in allowed, (
                        f"{yaml_path.name}: any_of[{i}] uses cloud "
                        f"'{opt_cloud}' — not in allowed list "
                        f"{sorted(allowed)}"
                    )


# ===========================================================================
# 4. YAML schema drift detection
# ===========================================================================


class TestYamlSchemaDrift:
    """Detect unauthorized keys in SkyPilot YAML files."""

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_unauthorized_top_level_keys(
        self, yaml_path: Path, contract: dict
    ) -> None:
        """Top-level keys must be in the contract allowlist."""
        config = _load_yaml(yaml_path)
        allowed = set(contract.get("allowed_top_level_keys", []))
        actual = set(config.keys())
        unauthorized = actual - allowed
        assert not unauthorized, (
            f"{yaml_path.name}: Unauthorized top-level keys: {unauthorized}. "
            f"Allowed: {sorted(allowed)}. "
            f"Adding new keys requires updating configs/cloud/yaml_contract.yaml."
        )

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_unauthorized_resource_keys(
        self, yaml_path: Path, contract: dict
    ) -> None:
        """Resource keys must be in the contract allowlist."""
        config = _load_yaml(yaml_path)
        resources = config.get("resources", {})
        if not resources:
            return
        allowed = set(contract.get("allowed_resource_keys", []))
        actual = set(resources.keys())
        unauthorized = actual - allowed
        assert not unauthorized, (
            f"{yaml_path.name}: Unauthorized resource keys: {unauthorized}. "
            f"Allowed: {sorted(allowed)}."
        )


# ===========================================================================
# 5. Factorial YAML strict contract
# ===========================================================================


class TestFactorialYamlStrictContract:
    """The factorial YAML controls 34+ jobs. Strictest validation."""

    def test_factorial_uses_exactly_l4(self, factorial_config: dict) -> None:
        """Factorial must use L4 and ONLY L4 (may be in ordered: blocks)."""
        resources = factorial_config.get("resources", {})
        gpu_names: list[str] = []
        # Direct accelerators field
        gpu_names.extend(_extract_accelerator_names(resources.get("accelerators")))
        # ordered: and any_of: blocks (multi-region failover)
        for block_name in ("ordered", "any_of"):
            for option in resources.get(block_name, []):
                if isinstance(option, dict):
                    gpu_names.extend(
                        _extract_accelerator_names(option.get("accelerators"))
                    )
        unique_gpus = sorted(set(gpu_names))
        assert unique_gpus == ["L4"], (
            f"Factorial YAML must use ONLY L4. Got: {unique_gpus}. "
            f"A100 was added without authorization (5.5x cost)."
        )

    def test_factorial_uses_gcp(self, factorial_config: dict) -> None:
        """Factorial must use GCP (same region as GCS, Cloud SQL, MLflow)."""
        cloud = factorial_config.get("resources", {}).get("cloud")
        assert cloud == "gcp", f"Factorial must use cloud=gcp, got: {cloud}"

    def test_factorial_uses_spot(self, factorial_config: dict) -> None:
        """Factorial must use spot instances (60-91% cheaper)."""
        use_spot = factorial_config.get("resources", {}).get("use_spot")
        assert use_spot is True, "Factorial must use spot instances"

    def test_factorial_uses_docker(self, factorial_config: dict) -> None:
        """Factorial must use Docker image (bare VM banned)."""
        image_id = factorial_config.get("resources", {}).get("image_id", "")
        assert image_id, "Factorial must have image_id (bare VM banned)"
        assert image_id.startswith("docker:"), (
            f"image_id must start with 'docker:': {image_id}"
        )

    def test_factorial_no_any_of_block(self, factorial_config: dict) -> None:
        """Factorial MUST NOT have any_of (would introduce non-determinism)."""
        resources = factorial_config.get("resources", {})
        assert "any_of" not in resources, (
            "Factorial YAML must NOT have any_of block. "
            "any_of introduces non-determinism — different jobs get different "
            "GPU types depending on spot availability. "
            "YAML is the contract: same YAML = same resources."
        )

    def test_factorial_accelerators_not_dict(self, factorial_config: dict) -> None:
        """Factorial accelerators must be a string like 'L4:1', not a dict.

        Dict format {L4: 1, A100: 1} is a SkyPilot priority list — it means
        'try L4 first, fall back to A100'. This is the EXACT pattern that
        caused the unauthorized A100 incident. String format 'L4:1' means
        'L4 ONLY, fail if unavailable' — which is the correct behavior.
        """
        accel_field = factorial_config.get("resources", {}).get("accelerators")
        assert not isinstance(accel_field, dict), (
            f"Factorial accelerators must be a string (e.g., 'L4:1'), "
            f"not a dict. Dict format {{L4: 1, A100: 1}} creates a "
            f"PRIORITY LIST that silently falls back to expensive GPUs. "
            f"Got: {accel_field}"
        )


# ===========================================================================
# 6. Cross-file consistency (cloud config ↔ SkyPilot YAML)
# ===========================================================================


class TestCrossFileConsistency:
    """SkyPilot YAMLs must be consistent with configs/cloud/*.yaml."""

    def test_gcp_skypilot_yamls_match_gcp_spot_config(self) -> None:
        """All GCP SkyPilot YAMLs must use GPUs from gcp_spot.yaml."""
        if not GCP_SPOT_CONFIG.exists():
            pytest.skip("gcp_spot.yaml not found")

        gcp_config = _load_yaml(GCP_SPOT_CONFIG)
        cloud_accels = gcp_config.get("accelerators", [])
        if isinstance(cloud_accels, list):
            allowed = {a.split(":")[0] for a in cloud_accels}
        elif isinstance(cloud_accels, str):
            allowed = {cloud_accels.split(":")[0]}
        else:
            allowed = set()

        for yaml_path in _discover_skypilot_yamls():
            config = _load_yaml(yaml_path)
            resources = config.get("resources", {})
            if resources.get("cloud") != "gcp":
                continue

            accel_field = resources.get("accelerators")
            actual = set(_extract_accelerator_names(accel_field))
            unauthorized = actual - allowed
            assert not unauthorized, (
                f"{yaml_path.name} uses GPU types {unauthorized} not in "
                f"configs/cloud/gcp_spot.yaml ({sorted(allowed)}). "
                f"Cloud config is the source of truth for allowed GPUs."
            )

    def test_runpod_skypilot_yamls_match_runpod_config(self) -> None:
        """All RunPod SkyPilot YAMLs must use GPUs from runpod_dev.yaml."""
        if not RUNPOD_DEV_CONFIG.exists():
            pytest.skip("runpod_dev.yaml not found")

        runpod_config = _load_yaml(RUNPOD_DEV_CONFIG)
        cloud_accels = runpod_config.get("accelerators", [])
        if isinstance(cloud_accels, list):
            allowed = {a.split(":")[0] for a in cloud_accels}
        elif isinstance(cloud_accels, str):
            allowed = {cloud_accels.split(":")[0]}
        else:
            allowed = set()

        for yaml_path in _discover_skypilot_yamls():
            config = _load_yaml(yaml_path)
            resources = config.get("resources", {})
            if resources.get("cloud") != "runpod":
                continue

            accel_field = resources.get("accelerators")
            actual = set(_extract_accelerator_names(accel_field))
            unauthorized = actual - allowed
            assert not unauthorized, (
                f"{yaml_path.name} uses GPU types {unauthorized} not in "
                f"configs/cloud/runpod_dev.yaml ({sorted(allowed)})"
            )


# ===========================================================================
# 7. Contract file integrity
# ===========================================================================


class TestContractFileIntegrity:
    """The contract file itself must be well-formed and complete."""

    def test_contract_file_exists(self) -> None:
        """configs/cloud/yaml_contract.yaml must exist."""
        assert CONTRACT_PATH.exists(), (
            f"Golden contract missing: {CONTRACT_PATH}. "
            f"This file is the source of truth for YAML validation."
        )

    def test_contract_has_required_sections(self, contract: dict) -> None:
        """Contract must have all required sections."""
        required = {
            "allowed_accelerators",
            "banned_accelerators",
            "allowed_clouds",
            "allowed_top_level_keys",
            "allowed_resource_keys",
            "factorial",
        }
        actual = set(contract.keys())
        missing = required - actual
        assert not missing, f"Contract missing required sections: {missing}"

    def test_contract_banned_list_includes_t4(self, contract: dict) -> None:
        """T4 must always be in the banned list."""
        banned = contract.get("banned_accelerators", [])
        assert "T4" in banned, "T4 MUST be in banned_accelerators — Turing, no BF16."

    def test_contract_gcp_includes_a100_fallback_tiers(self, contract: dict) -> None:
        """A100 + A100-80GB authorized as spot fallback tiers (2026-03-28)."""
        gcp_allowed = contract.get("allowed_accelerators", {}).get("gcp", [])
        assert "A100" in gcp_allowed, (
            "A100 (40GB) must be in GCP allowed accelerators as L4 fallback tier 1."
        )
        assert "A100-80GB" in gcp_allowed, (
            "A100-80GB must be in GCP allowed accelerators as L4 fallback tier 2."
        )
        assert "H100" not in gcp_allowed, (
            "H100 must NOT be in GCP allowed accelerators — not authorized."
        )
