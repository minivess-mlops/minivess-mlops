"""Region injection: generate SkyPilot YAML with ordered: block from region configs.

Tests for src/minivess/cloud/region_injection.py — the function that reads a region
config and injects an ordered: priority list into a base SkyPilot YAML template.

Plan: docs/planning/cold-start-prompt-composable-regions-phase2.md (Task 1)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
REGION_DIR = REPO_ROOT / "configs" / "cloud" / "regions"
SKYPILOT_YAML = REPO_ROOT / "deployment" / "skypilot" / "train_factorial.yaml"
YAML_CONTRACT = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_yaml_path() -> Path:
    """Path to the real base SkyPilot YAML."""
    return SKYPILOT_YAML


@pytest.fixture()
def europe_us_region_path() -> Path:
    """Path to the default europe_us region config."""
    return REGION_DIR / "europe_us.yaml"


@pytest.fixture()
def base_yaml_content() -> dict:
    """Parsed base SkyPilot YAML."""
    return yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))


@pytest.fixture()
def europe_us_regions() -> dict:
    """Parsed europe_us region config."""
    path = REGION_DIR / "europe_us.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test: generate_skypilot_yaml produces valid output
# ---------------------------------------------------------------------------


class TestRegionInjectionGeneratesValidYaml:
    """Generated YAML must be valid and contain correct ordered: block."""

    def test_returns_path_to_yaml_file(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        assert result.exists()
        assert result.suffix == ".yaml"

    def test_output_is_valid_yaml(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert isinstance(content, dict)

    def test_has_ordered_block(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert "ordered" in content["resources"], "Missing ordered: block in resources"

    def test_ordered_has_correct_region_count(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        ordered = content["resources"]["ordered"]
        # europe_us.yaml has 6 entries (3 EU + 3 US)
        assert len(ordered) == 6

    def test_ordered_entries_have_accelerators_and_region(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        for entry in content["resources"]["ordered"]:
            assert "accelerators" in entry, f"Missing accelerators in ordered entry: {entry}"
            assert "region" in entry, f"Missing region in ordered entry: {entry}"

    def test_ordered_preserves_region_order(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        regions = [e["region"] for e in content["resources"]["ordered"]]
        # First 3 must be EU, last 3 must be US
        assert regions[0].startswith("europe-"), f"First region not EU: {regions[0]}"
        assert regions[-1].startswith("us-"), f"Last region not US: {regions[-1]}"

    def test_preserves_non_resource_keys(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        """All non-resource sections (envs, setup, run) must survive."""
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        # file_mounts removed (competing persistence mechanism — MLflow GCS artifact store
        # is the only checkpoint persistence mechanism now)
        for key in ("name", "envs", "setup", "run"):
            assert key in content, f"Generated YAML missing '{key}'"


# ---------------------------------------------------------------------------
# Test: default accelerators removed when ordered: is present
# ---------------------------------------------------------------------------


class TestRemovesDefaultAccelerators:
    """When ordered: is injected, the standalone accelerators: key must be removed."""

    def test_no_accelerators_key_in_resources(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert "accelerators" not in content["resources"], (
            "accelerators: must be removed when ordered: is present"
        )

    def test_base_yaml_has_accelerators(self, base_yaml_content: dict) -> None:
        """Sanity check: base YAML DOES have accelerators: L4:1."""
        assert "accelerators" in base_yaml_content["resources"]


# ---------------------------------------------------------------------------
# Test: no region config → use static YAML unchanged
# ---------------------------------------------------------------------------


class TestNoRegionConfigBypass:
    """When region_config is None/empty, return the original YAML path unchanged."""

    def test_none_region_config_returns_base_path(
        self, base_yaml_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=None,
            output_dir=tmp_path,
        )
        # Should return the original base path, not a generated file
        assert result == base_yaml_path


# ---------------------------------------------------------------------------
# Test: generated YAML passes sky.Task.from_yaml()
# ---------------------------------------------------------------------------


class TestSkyPilotParsing:
    """Generated YAML must be parseable by SkyPilot."""

    def test_generated_yaml_passes_sky_task_from_yaml(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        sky = pytest.importorskip("sky", reason="SkyPilot not installed")
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        # This must not raise
        task = sky.Task.from_yaml(str(result))
        assert task is not None


# ---------------------------------------------------------------------------
# Test: accelerator type extracted from base YAML
# ---------------------------------------------------------------------------


class TestAcceleratorExtraction:
    """The ordered: entries use the GPU type from the region config key (L4)."""

    def test_accelerator_matches_region_config_gpu_type(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        for entry in content["resources"]["ordered"]:
            assert entry["accelerators"] == "L4:1", (
                f"Expected L4:1, got {entry['accelerators']}"
            )


# ---------------------------------------------------------------------------
# Test: works with all region configs
# ---------------------------------------------------------------------------


class TestAllRegionConfigs:
    """Generate YAML from every region config in the regions/ directory."""

    @pytest.mark.parametrize(
        "region_path",
        sorted(REGION_DIR.glob("*.yaml")),
        ids=lambda p: p.stem,
    )
    def test_generates_valid_yaml_for_each_config(
        self, base_yaml_path: Path, region_path: Path, tmp_path: Path
    ) -> None:
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=region_path,
            output_dir=tmp_path,
        )
        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert "ordered" in content["resources"]
        assert "accelerators" not in content["resources"]
        assert len(content["resources"]["ordered"]) > 0


# ---------------------------------------------------------------------------
# Test: contract enforcement — generated YAML obeys yaml_contract.yaml (#944)
# ---------------------------------------------------------------------------


class TestRegionInjectionContractEnforcement:
    """Generated YAML must satisfy the golden contract (configs/cloud/yaml_contract.yaml).

    Issue #944, Task 1.3: Region injection YAML validity tests.
    """

    @pytest.fixture()
    def generated_yaml_content(
        self, base_yaml_path: Path, europe_us_region_path: Path, tmp_path: Path
    ) -> dict:
        """Generate a SkyPilot YAML and return its parsed content."""
        from minivess.cloud.region_injection import generate_skypilot_yaml

        result_path = generate_skypilot_yaml(
            base_yaml_path=base_yaml_path,
            region_config_path=europe_us_region_path,
            output_dir=tmp_path,
        )
        return yaml.safe_load(result_path.read_text(encoding="utf-8"))

    @pytest.fixture()
    def contract(self) -> dict:
        """Parse the golden YAML contract."""
        return yaml.safe_load(YAML_CONTRACT.read_text(encoding="utf-8"))

    def test_generated_yaml_preserves_job_recovery(
        self, generated_yaml_content: dict
    ) -> None:
        """job_recovery must be inside resources (not top-level) after injection."""
        resources = generated_yaml_content["resources"]
        assert "job_recovery" in resources, (
            "job_recovery missing from resources after region injection — "
            "generate_skypilot_yaml must preserve all existing resource keys"
        )
        # Verify it is NOT at the top level (SkyPilot expects it under resources)
        assert "job_recovery" not in generated_yaml_content or "job_recovery" in resources, (
            "job_recovery found at top level — must be nested under resources"
        )

    def test_no_banned_gpu_in_ordered(
        self, generated_yaml_content: dict, contract: dict
    ) -> None:
        """No accelerator in the ordered: block may contain a banned GPU name."""
        banned = contract["banned_accelerators"]
        ordered = generated_yaml_content["resources"]["ordered"]
        for entry in ordered:
            accel = entry["accelerators"]
            # Extract GPU name (e.g., "L4" from "L4:1")
            gpu_name = accel.split(":")[0]
            for banned_gpu in banned:
                assert banned_gpu not in gpu_name, (
                    f"Banned GPU '{banned_gpu}' found in ordered entry: {accel}"
                )

    def test_ordered_gpu_in_contract_allowlist(
        self, generated_yaml_content: dict, contract: dict
    ) -> None:
        """Every GPU in ordered: must appear in the contract allowlist for its cloud."""
        cloud = generated_yaml_content["resources"].get("cloud", "gcp")
        allowed = contract["allowed_accelerators"].get(cloud, [])
        ordered = generated_yaml_content["resources"]["ordered"]
        for entry in ordered:
            accel = entry["accelerators"]
            gpu_name = accel.split(":")[0]
            assert gpu_name in allowed, (
                f"GPU '{gpu_name}' not in contract allowlist for {cloud}: {allowed}"
            )

    def test_required_keys_in_generated_yaml(
        self, generated_yaml_content: dict
    ) -> None:
        """Generated YAML must contain all required top-level keys."""
        required_keys = {"name", "resources", "setup", "run"}
        missing = required_keys - set(generated_yaml_content.keys())
        assert not missing, (
            f"Generated YAML missing required top-level keys: {missing}"
        )
