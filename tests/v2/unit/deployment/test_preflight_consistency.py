"""Preflight checker consistency tests — last gate before spending cloud credits.

Tests verify that preflight_gcp.py's hardcoded constants match canonical config
sources, that the YAML contract validator works, and that region configs parse.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 5 (P1): T5.1 – T5.4
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
PREFLIGHT = REPO_ROOT / "scripts" / "preflight_gcp.py"
GCP_SPOT = REPO_ROOT / "configs" / "cloud" / "gcp_spot.yaml"
CONTRACT = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
REGIONS_DIR = REPO_ROOT / "configs" / "cloud" / "regions"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _extract_preflight_constant(name: str) -> str:
    """Extract a string constant from preflight_gcp.py using ast."""
    source = PREFLIGHT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    msg = f"Constant '{name}' not found in {PREFLIGHT}"
    raise ValueError(msg)


def _extract_preflight_list_constant(name: str) -> list[str]:
    """Extract a list of string constants from preflight_gcp.py using ast."""
    source = PREFLIGHT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.List)
                ):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    ]
    msg = f"List constant '{name}' not found in {PREFLIGHT}"
    raise ValueError(msg)


# ===========================================================================
# T5.1: Preflight detects stale Docker image (tested indirectly via constants)
# ===========================================================================


class TestPreflightDockerImage:
    """Preflight Docker image freshness check has correct structure."""

    def test_preflight_has_check_docker_image_freshness(self) -> None:
        """preflight_gcp.py defines check_docker_image_freshness function."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "check_docker_image_freshness" in func_names

    def test_preflight_freshness_uses_oci_revision_label(self) -> None:
        """check_docker_image_freshness reads OCI revision label."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        assert "org.opencontainers.image.revision" in source


# ===========================================================================
# T5.2: Preflight config references match across files
# ===========================================================================


class TestPreflightConfig:
    """Preflight constants must match canonical config sources."""

    def test_preflight_gar_image_matches_config(self) -> None:
        gar_image = _extract_preflight_constant("GAR_IMAGE")
        gcp_spot = _load_yaml(GCP_SPOT)
        assert gar_image == gcp_spot["docker_image"]

    def test_preflight_skypilot_yaml_path_exists(self) -> None:
        """SKYPILOT_YAML path from preflight must reference an existing file."""
        # The constant uses REPO_ROOT / "deployment" / "skypilot" / "train_factorial.yaml"
        # which is a Path, not a string. We verify the file exists.
        expected_path = REPO_ROOT / "deployment" / "skypilot" / "train_factorial.yaml"
        assert expected_path.exists(), f"SKYPILOT_YAML target does not exist: {expected_path}"

    def test_preflight_required_env_vars_complete(self) -> None:
        """REQUIRED_ENV_VARS includes HF_TOKEN and MLFLOW_TRACKING_URI."""
        env_vars = _extract_preflight_list_constant("REQUIRED_ENV_VARS")
        assert "HF_TOKEN" in env_vars
        assert "MLFLOW_TRACKING_URI" in env_vars


# ===========================================================================
# T5.3: YAML contract validator catches unauthorized GPU types
# ===========================================================================


class TestContractValidation:
    """Tests for the contract validation logic in preflight_gcp.py."""

    def test_contract_rejects_h100_in_skypilot_yaml(self, tmp_path: Path) -> None:
        """A SkyPilot YAML with H100 must fail contract validation."""
        # Import the check function
        import importlib.util

        spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Create a temp SkyPilot YAML with H100
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            yaml.dump({
                "name": "test",
                "resources": {"accelerators": "H100:1", "cloud": "gcp"},
            }),
            encoding="utf-8",
        )

        # Monkey-patch SKYPILOT_YAML to point to the bad file
        original = mod.SKYPILOT_YAML
        mod.SKYPILOT_YAML = bad_yaml
        try:
            ok, msg = mod.check_yaml_contract()
            assert not ok, f"H100 should be rejected but passed: {msg}"
        finally:
            mod.SKYPILOT_YAML = original

    def test_contract_accepts_l4_accelerator(self, tmp_path: Path) -> None:
        """A SkyPilot YAML with L4 passes contract validation."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        good_yaml = tmp_path / "good.yaml"
        good_yaml.write_text(
            yaml.dump({
                "name": "test",
                "resources": {"accelerators": "L4:1", "cloud": "gcp"},
            }),
            encoding="utf-8",
        )

        original = mod.SKYPILOT_YAML
        mod.SKYPILOT_YAML = good_yaml
        try:
            ok, msg = mod.check_yaml_contract()
            assert ok, f"L4 should be accepted but failed: {msg}"
        finally:
            mod.SKYPILOT_YAML = original

    def test_contract_rejects_t4_in_any_position(self, tmp_path: Path) -> None:
        """T4 must be rejected (globally banned)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        t4_yaml = tmp_path / "t4.yaml"
        t4_yaml.write_text(
            yaml.dump({
                "name": "test",
                "resources": {"accelerators": "T4:1", "cloud": "gcp"},
            }),
            encoding="utf-8",
        )

        original = mod.SKYPILOT_YAML
        mod.SKYPILOT_YAML = t4_yaml
        try:
            ok, msg = mod.check_yaml_contract()
            assert not ok, f"T4 should be rejected but passed: {msg}"
            assert "BANNED" in msg or "not in factorial" in msg
        finally:
            mod.SKYPILOT_YAML = original


# ===========================================================================
# T5.4: Region config files parse correctly with required fields
# ===========================================================================


class TestRegionConfigSchema:
    """Every region config must parse and have the required structure."""

    def test_all_region_configs_parse_without_error(self) -> None:
        for path in sorted(REGIONS_DIR.glob("*.yaml")):
            config = _load_yaml(path)
            assert config is not None, f"Failed to parse {path.name}"

    def test_all_region_configs_have_description(self) -> None:
        for path in sorted(REGIONS_DIR.glob("*.yaml")):
            config = _load_yaml(path)
            assert "description" in config, f"{path.name} missing 'description'"

    def test_all_region_configs_have_regions_gcp(self) -> None:
        for path in sorted(REGIONS_DIR.glob("*.yaml")):
            config = _load_yaml(path)
            assert "regions" in config, f"{path.name} missing 'regions'"
            assert "gcp" in config["regions"], f"{path.name} missing 'regions.gcp'"

    def test_region_entries_have_region_and_zones(self) -> None:
        for path in sorted(REGIONS_DIR.glob("*.yaml")):
            config = _load_yaml(path)
            for gpu_type, entries in config["regions"]["gcp"].items():
                for i, entry in enumerate(entries):
                    assert "region" in entry, (
                        f"{path.name}: {gpu_type}[{i}] missing 'region'"
                    )
                    assert "zones" in entry, (
                        f"{path.name}: {gpu_type}[{i}] missing 'zones'"
                    )
                    assert isinstance(entry["zones"], int) and entry["zones"] > 0, (
                        f"{path.name}: {gpu_type}[{i}] zones must be int > 0"
                    )
