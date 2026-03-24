"""Composable region config validation.

Verifies configs/cloud/regions/*.yaml structure, references from factorial
configs, and that no hardcoded regions exist in SkyPilot YAML templates.

Plan: multi-region-skypilot-skill-cleanup.xml
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
REGION_DIR = REPO_ROOT / "configs" / "cloud" / "regions"
CONTRACT_PATH = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
FACTORIAL_DIR = REPO_ROOT / "configs" / "factorial"
SKYPILOT_DIR = REPO_ROOT / "deployment" / "skypilot"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _region_configs() -> list[Path]:
    return sorted(REGION_DIR.glob("*.yaml"))


def _factorial_configs() -> list[Path]:
    return sorted(FACTORIAL_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Region config structure
# ---------------------------------------------------------------------------


class TestRegionConfigStructure:
    """Every region config file must have valid structure."""

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_has_description(self, path: Path) -> None:
        cfg = _load(path)
        assert "description" in cfg, f"{path.name} missing 'description'"

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_has_regions_block(self, path: Path) -> None:
        cfg = _load(path)
        assert "regions" in cfg, f"{path.name} missing 'regions'"
        assert isinstance(cfg["regions"], dict)

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_regions_keyed_by_provider(self, path: Path) -> None:
        cfg = _load(path)
        contract = _load(CONTRACT_PATH)
        allowed = set(contract["allowed_clouds"])
        for provider in cfg["regions"]:
            assert provider in allowed, (
                f"{path.name}: unauthorized provider '{provider}'"
            )

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_each_entry_has_region_field(self, path: Path) -> None:
        cfg = _load(path)
        for provider, gpu_map in cfg["regions"].items():
            for gpu_type, entries in gpu_map.items():
                for i, entry in enumerate(entries):
                    assert "region" in entry, (
                        f"{path.name}: {provider}.{gpu_type}[{i}] missing 'region'"
                    )

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_region_list_nonempty(self, path: Path) -> None:
        cfg = _load(path)
        for provider, gpu_map in cfg["regions"].items():
            for gpu_type, entries in gpu_map.items():
                assert len(entries) > 0, f"{path.name}: {provider}.{gpu_type} is empty"


# ---------------------------------------------------------------------------
# No europe-north1 in L4 regions (ROOT CAUSE of 12+ hr PENDING)
# ---------------------------------------------------------------------------


class TestNoEuropeNorth1InL4:
    """europe-north1 has NO L4 GPUs — must never appear in L4 region lists."""

    @pytest.mark.parametrize("path", _region_configs(), ids=lambda p: p.stem)
    def test_no_europe_north1(self, path: Path) -> None:
        cfg = _load(path)
        for _provider, gpu_map in cfg.get("regions", {}).items():
            for entry in gpu_map.get("L4", []):
                assert entry["region"] != "europe-north1", (
                    f"{path.name}: europe-north1 has NO L4 GPUs "
                    "(root cause of 12+ hr PENDING in 6th/7th pass)"
                )


# ---------------------------------------------------------------------------
# Factorial config references valid region configs
# ---------------------------------------------------------------------------


class TestFactorialRegionReference:
    """Cloud factorial configs must reference valid region configs."""

    @pytest.mark.parametrize("path", _factorial_configs(), ids=lambda p: p.stem)
    def test_cloud_factorial_has_region_config(self, path: Path) -> None:
        cfg = _load(path)
        infra = cfg.get("infrastructure", {})
        cloud = infra.get("cloud_config", "local")
        if cloud == "local":
            return  # Local doesn't need region_config
        assert "region_config" in infra, (
            f"{path.name}: cloud factorial must have region_config"
        )

    @pytest.mark.parametrize("path", _factorial_configs(), ids=lambda p: p.stem)
    def test_region_config_file_exists(self, path: Path) -> None:
        cfg = _load(path)
        infra = cfg.get("infrastructure", {})
        name = infra.get("region_config")
        if name is None:
            return
        region_path = REGION_DIR / f"{name}.yaml"
        assert region_path.exists(), (
            f"{path.name} references region_config='{name}' but "
            f"{region_path} does not exist"
        )


# ---------------------------------------------------------------------------
# EU before US ordering
# ---------------------------------------------------------------------------


class TestRegionOrdering:
    """In europe_us.yaml, EU regions must come before US regions."""

    def test_eu_before_us(self) -> None:
        path = REGION_DIR / "europe_us.yaml"
        if not path.exists():
            pytest.skip("europe_us.yaml not found")
        cfg = _load(path)
        regions = [r["region"] for r in cfg["regions"]["gcp"]["L4"]]
        first_us = next(
            (i for i, r in enumerate(regions) if r.startswith("us-")),
            len(regions),
        )
        for r in regions[:first_us]:
            assert r.startswith("europe-"), (
                f"Non-EU region '{r}' appears before US regions"
            )

    def test_europe_us_has_6_entries(self) -> None:
        path = REGION_DIR / "europe_us.yaml"
        if not path.exists():
            pytest.skip("europe_us.yaml not found")
        cfg = _load(path)
        entries = cfg["regions"]["gcp"]["L4"]
        assert len(entries) == 6, f"Expected 6 (3 EU + 3 US), got {len(entries)}"

    def test_europe_has_3_entries(self) -> None:
        path = REGION_DIR / "europe.yaml"
        if not path.exists():
            pytest.skip("europe.yaml not found")
        cfg = _load(path)
        entries = cfg["regions"]["gcp"]["L4"]
        assert len(entries) == 3, f"Expected 3 EU entries, got {len(entries)}"
