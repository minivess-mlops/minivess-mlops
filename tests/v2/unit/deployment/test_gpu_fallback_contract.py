"""GPU fallback and YAML contract tests — prevents unauthorized cost escalation.

After the unauthorized A100 incident (5.5x cost, 2026-03-24), the YAML contract
system was created. These tests verify: right GPUs allowed, right order, right
cost limits, right per-model spot overrides.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 2 (P0): T2.1 – T2.6
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
GCP_SPOT = REPO_ROOT / "configs" / "cloud" / "gcp_spot.yaml"
DEBUG_YAML = REPO_ROOT / "configs" / "factorial" / "debug.yaml"
PAPER_FULL = REPO_ROOT / "configs" / "factorial" / "paper_full.yaml"
REGIONS_DIR = REPO_ROOT / "configs" / "cloud" / "regions"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ===========================================================================
# T2.1: YAML contract allows exactly L4, A100, A100-80GB for GCP
# ===========================================================================


class TestContractAllowedGpus:
    """Contract GCP allowed list must be exactly {L4, A100, A100-80GB}."""

    def test_contract_gcp_allowed_accelerators_exact(self) -> None:
        contract = _load_yaml(CONTRACT)
        gcp_allowed = contract["allowed_accelerators"]["gcp"]
        assert gcp_allowed == ["L4", "A100", "A100-80GB"]

    def test_contract_gcp_does_not_include_h100(self) -> None:
        contract = _load_yaml(CONTRACT)
        gcp_allowed = contract["allowed_accelerators"]["gcp"]
        assert "H100" not in gcp_allowed

    def test_contract_gcp_does_not_include_banned_gpus(self) -> None:
        contract = _load_yaml(CONTRACT)
        gcp_allowed = set(contract["allowed_accelerators"]["gcp"])
        banned = contract["banned_accelerators"]
        for gpu in banned:
            assert gpu not in gcp_allowed, (
                f"Banned GPU '{gpu}' found in GCP allowed list"
            )


# ===========================================================================
# T2.2: Factorial contract allows exactly L4, A100, A100-80GB
# ===========================================================================


class TestFactorialContract:
    """Factorial section of the contract — strictest cost control."""

    def test_factorial_contract_allowed_accelerators(self) -> None:
        contract = _load_yaml(CONTRACT)
        factorial_allowed = contract["factorial"]["allowed_accelerators"]
        assert factorial_allowed == ["L4", "A100", "A100-80GB"]

    def test_factorial_contract_requires_gcp(self) -> None:
        contract = _load_yaml(CONTRACT)
        assert contract["factorial"]["required_cloud"] == "gcp"

    def test_factorial_contract_requires_docker(self) -> None:
        contract = _load_yaml(CONTRACT)
        assert contract["factorial"]["require_docker_image"] is True


# ===========================================================================
# T2.3: Region configs: L4 before A100 in GPU ordering
# ===========================================================================


class TestGpuOrdering:
    """In all region configs, L4 must precede A100 in iteration order."""

    def test_region_config_gpu_ordering_l4_before_a100(self) -> None:
        """For every region config, L4 index < A100 index < A100-80GB index."""
        for region_path in sorted(REGIONS_DIR.glob("*.yaml")):
            config = _load_yaml(region_path)
            gpu_types = list(config.get("regions", {}).get("gcp", {}).keys())

            if "L4" in gpu_types and "A100" in gpu_types:
                assert gpu_types.index("L4") < gpu_types.index("A100"), (
                    f"{region_path.name}: L4 must precede A100, got {gpu_types}"
                )
            if "L4" in gpu_types and "A100-80GB" in gpu_types:
                assert gpu_types.index("L4") < gpu_types.index("A100-80GB"), (
                    f"{region_path.name}: L4 must precede A100-80GB, got {gpu_types}"
                )
            if "A100" in gpu_types and "A100-80GB" in gpu_types:
                assert gpu_types.index("A100") < gpu_types.index("A100-80GB"), (
                    f"{region_path.name}: A100 must precede A100-80GB, got {gpu_types}"
                )

    def test_us_central1_region_has_l4_first(self) -> None:
        config = _load_yaml(REGIONS_DIR / "us_central1.yaml")
        first_gpu = next(iter(config["regions"]["gcp"]))
        assert first_gpu == "L4", f"Expected L4 first, got {first_gpu}"

    def test_europe_a100_fallback_has_l4_first(self) -> None:
        config = _load_yaml(REGIONS_DIR / "europe_a100_fallback.yaml")
        first_gpu = next(iter(config["regions"]["gcp"]))
        assert first_gpu == "L4", f"Expected L4 first, got {first_gpu}"


# ===========================================================================
# T2.4: gcp_spot.yaml accelerators match contract GCP list
# ===========================================================================


class TestGcpSpotMatchesContract:
    """gcp_spot.yaml accelerator list must be a subset of the contract."""

    def test_gcp_spot_accelerators_subset_of_contract(self) -> None:
        gcp_spot = _load_yaml(GCP_SPOT)
        contract = _load_yaml(CONTRACT)
        allowed = set(contract["allowed_accelerators"]["gcp"])

        for accel in gcp_spot["accelerators"]:
            # Strip ":1" suffix
            gpu_name = accel.split(":")[0]
            assert gpu_name in allowed, (
                f"gcp_spot.yaml accelerator '{gpu_name}' not in contract "
                f"allowed list {sorted(allowed)}"
            )

    def test_gcp_spot_accelerators_order_matches_contract(self) -> None:
        """gcp_spot.yaml ordering follows contract priority (L4 first)."""
        gcp_spot = _load_yaml(GCP_SPOT)
        contract = _load_yaml(CONTRACT)
        contract_order = contract["allowed_accelerators"]["gcp"]

        spot_names = [a.split(":")[0] for a in gcp_spot["accelerators"]]

        for i in range(len(spot_names) - 1):
            assert contract_order.index(spot_names[i]) < contract_order.index(
                spot_names[i + 1]
            ), (
                f"gcp_spot.yaml ordering wrong: '{spot_names[i]}' should come "
                f"before '{spot_names[i + 1]}' per contract"
            )


# ===========================================================================
# T2.5: use_spot override works per-model in factorial config
# ===========================================================================


class TestUseSpotOverride:
    """SAM3 models must have use_spot: false (80% preemption on 25+ min jobs)."""

    def test_sam3_hybrid_use_spot_false(self) -> None:
        debug = _load_yaml(DEBUG_YAML)
        assert debug["model_overrides"]["sam3_hybrid"]["use_spot"] is False

    def test_sam3_topolora_use_spot_false(self) -> None:
        debug = _load_yaml(DEBUG_YAML)
        assert debug["model_overrides"]["sam3_topolora"]["use_spot"] is False

    def test_dynunet_no_use_spot_override(self) -> None:
        """DynUNet should not have use_spot: false (benefits from spot)."""
        debug = _load_yaml(DEBUG_YAML)
        overrides = debug.get("model_overrides", {})
        if "dynunet" in overrides:
            assert overrides["dynunet"].get("use_spot", True) is True

    def test_paper_full_sam3_use_spot_matches_debug(self) -> None:
        """Rule 27: debug = production — SAM3 use_spot must match."""
        debug = _load_yaml(DEBUG_YAML)
        paper = _load_yaml(PAPER_FULL)

        for model in ("sam3_hybrid", "sam3_topolora"):
            debug_spot = debug["model_overrides"][model]["use_spot"]
            paper_spot = paper["model_overrides"][model]["use_spot"]
            assert debug_spot == paper_spot, (
                f"{model}: debug use_spot={debug_spot} != paper use_spot={paper_spot}"
            )


# ===========================================================================
# T2.6: Max hourly cost per GPU type is set in contract
# ===========================================================================


class TestMaxHourlyCost:
    """Every allowed GPU must have a max_hourly_cost_usd entry."""

    def test_every_allowed_gpu_has_max_hourly_cost(self) -> None:
        contract = _load_yaml(CONTRACT)
        cost_map = contract["max_hourly_cost_usd"]

        for provider in ("gcp", "runpod"):
            for gpu in contract["allowed_accelerators"][provider]:
                if gpu is None:
                    continue  # local: null
                assert gpu in cost_map, (
                    f"GPU '{gpu}' allowed for {provider} but has no "
                    f"max_hourly_cost_usd entry"
                )

    def test_l4_cheaper_than_a100_in_contract(self) -> None:
        contract = _load_yaml(CONTRACT)
        costs = contract["max_hourly_cost_usd"]
        assert costs["L4"] < costs["A100"]

    def test_a100_cheaper_than_a100_80gb_in_contract(self) -> None:
        contract = _load_yaml(CONTRACT)
        costs = contract["max_hourly_cost_usd"]
        assert costs["A100"] < costs["A100-80GB"]
