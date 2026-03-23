"""Cloud infrastructure config validation — single-source YAML architecture.

Validates the config chain: configs/cloud/*.yaml → controller placement,
infrastructure parameters, and factorial config references.

Issue #913: SkyPilot controller was on RunPod for GCP jobs (36 min/submission).
Root cause: ~/.sky/config.yaml had cloud:runpod, never updated for GCP.
Fix: Repo-level config with controller.cloud == provider.

Plan: docs/planning/v0-2_archive/original_docs/infrastructure-performance-audit.xml
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CLOUD_CONFIG_DIR = Path("configs/cloud")
FACTORIAL_CONFIG_DIR = Path("configs/factorial")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cloud_configs() -> list[Path]:
    """Return all cloud config YAMLs (excluding non-cloud files like quotas)."""
    if not CLOUD_CONFIG_DIR.exists():
        return []
    return [
        p
        for p in sorted(CLOUD_CONFIG_DIR.glob("*.yaml"))
        if p.stem not in ("gcp_quotas", "yaml_contract")
    ]


def _factorial_configs() -> list[Path]:
    """Return all factorial experiment config YAMLs."""
    if not FACTORIAL_CONFIG_DIR.exists():
        return []
    return sorted(FACTORIAL_CONFIG_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Controller cloud must match provider (5th pass root cause)
# ---------------------------------------------------------------------------


class TestControllerCloudMatch:
    """Controller MUST run on same cloud as jobs — no cross-cloud SSH."""

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_controller_cloud_matches_provider(self, config_path: Path) -> None:
        """controller.cloud must equal top-level provider field."""
        cfg = _load(config_path)
        provider = cfg.get("provider")
        controller = cfg.get("controller")

        if provider == "local":
            assert controller is None, (
                f"{config_path.stem}: local provider should have controller: null"
            )
            return

        assert controller is not None, (
            f"{config_path.stem}: cloud provider '{provider}' must have a controller block"
        )
        assert controller.get("cloud") == provider, (
            f"{config_path.stem}: controller.cloud='{controller.get('cloud')}' "
            f"!= provider='{provider}'. Cross-cloud SSH adds ~30 min/submission "
            f"and creates single point of failure. "
            f"See: .claude/metalearning/2026-03-23-skypilot-controller-on-wrong-cloud.md"
        )


# ---------------------------------------------------------------------------
# Infrastructure block required for cloud providers
# ---------------------------------------------------------------------------


class TestInfrastructureBlock:
    """Cloud configs must have infrastructure block with required fields."""

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_infrastructure_block_exists(self, config_path: Path) -> None:
        """Cloud configs must have an infrastructure section."""
        cfg = _load(config_path)
        if cfg.get("provider") == "local":
            return  # Local doesn't need infrastructure block
        assert "infrastructure" in cfg, (
            f"{config_path.stem}: missing 'infrastructure' block "
            f"(needs parallel_submissions, rate_limit_seconds)"
        )

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_parallel_submissions_is_positive_int(self, config_path: Path) -> None:
        """parallel_submissions must be a positive integer."""
        cfg = _load(config_path)
        if cfg.get("provider") == "local":
            return
        infra = cfg.get("infrastructure", {})
        ps = infra.get("parallel_submissions")
        assert ps is not None, (
            f"{config_path.stem}: missing infrastructure.parallel_submissions"
        )
        assert isinstance(ps, int) and ps >= 1, (
            f"{config_path.stem}: parallel_submissions must be int >= 1, got {ps}"
        )

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_rate_limit_seconds_exists(self, config_path: Path) -> None:
        """rate_limit_seconds must exist for cloud providers."""
        cfg = _load(config_path)
        if cfg.get("provider") == "local":
            return
        infra = cfg.get("infrastructure", {})
        assert "rate_limit_seconds" in infra, (
            f"{config_path.stem}: missing infrastructure.rate_limit_seconds"
        )


# ---------------------------------------------------------------------------
# Cloud config schema — required keys
# ---------------------------------------------------------------------------


class TestCloudConfigSchema:
    """All cloud configs must have required top-level keys."""

    REQUIRED_KEYS = {"provider", "docker_registry", "docker_image"}

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_has_required_keys(self, config_path: Path) -> None:
        cfg = _load(config_path)
        missing = self.REQUIRED_KEYS - set(cfg.keys())
        assert not missing, f"{config_path.stem} missing keys: {missing}"

    @pytest.mark.parametrize(
        "config_path",
        _cloud_configs(),
        ids=[p.stem for p in _cloud_configs()],
    )
    def test_no_t4_in_accelerators(self, config_path: Path) -> None:
        """T4 BANNED in all cloud configs (Turing, no BF16)."""
        cfg = _load(config_path)
        accels = cfg.get("accelerators")
        if accels is None:
            return
        accel_str = str(accels)
        assert "T4" not in accel_str, f"{config_path.stem}: T4 is BANNED (no BF16)"


# ---------------------------------------------------------------------------
# Factorial config references valid cloud configs
# ---------------------------------------------------------------------------


class TestFactorialInfraReferences:
    """Factorial configs must reference valid cloud configs."""

    @pytest.mark.parametrize(
        "config_path",
        _factorial_configs(),
        ids=[p.stem for p in _factorial_configs()],
    )
    def test_cloud_config_reference_valid(self, config_path: Path) -> None:
        """infrastructure.cloud_config must point to an existing cloud config."""
        cfg = _load(config_path)
        infra = cfg.get("infrastructure", {})
        cloud_config_name = infra.get("cloud_config")
        if cloud_config_name is None:
            pytest.skip(f"No infrastructure.cloud_config in {config_path.stem}")
        cloud_path = CLOUD_CONFIG_DIR / f"{cloud_config_name}.yaml"
        assert cloud_path.exists(), (
            f"{config_path.stem} references cloud_config='{cloud_config_name}' "
            f"but {cloud_path} does not exist"
        )

    @pytest.mark.parametrize(
        "config_path",
        _factorial_configs(),
        ids=[p.stem for p in _factorial_configs()],
    )
    def test_skypilot_yaml_reference_valid(self, config_path: Path) -> None:
        """infrastructure.skypilot_yaml must point to an existing file (or be null for local)."""
        cfg = _load(config_path)
        infra = cfg.get("infrastructure", {})
        sky_yaml = infra.get("skypilot_yaml")
        cloud_config_name = infra.get("cloud_config", "")
        if sky_yaml is None:
            # Local configs don't need SkyPilot YAML — verify it's actually local
            assert cloud_config_name == "local", (
                f"{config_path.stem}: skypilot_yaml is null but cloud_config "
                f"is '{cloud_config_name}' (not 'local'). Cloud configs MUST have skypilot_yaml."
            )
            return
        assert Path(sky_yaml).exists(), (
            f"{config_path.stem} references {sky_yaml} but it does not exist"
        )


# ---------------------------------------------------------------------------
# No hardcoded parallelism in scripts
# ---------------------------------------------------------------------------


class TestNoHardcodedParallelism:
    """run_factorial.sh must read parallel_submissions from config, not hardcode."""

    def test_no_hardcoded_parallel_in_script(self) -> None:
        """Script must not contain PARALLEL_SUBMISSIONS=N as a constant."""
        script_path = Path("scripts/run_factorial.sh")
        if not script_path.exists():
            pytest.skip("run_factorial.sh not found")
        content = script_path.read_text(encoding="utf-8")
        # Allow: reading from config. Ban: hardcoded assignment.
        for n in (2, 4, 8):
            pattern = f"PARALLEL_SUBMISSIONS={n}"
            # Allow in comments
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert pattern not in stripped, (
                    f"run_factorial.sh hardcodes {pattern}. "
                    "Read from configs/cloud/*.yaml instead."
                )
