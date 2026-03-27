"""Shift-left FinOps: deterministic cost governance tests.

Policy-as-code checks that prevent cost regressions by validating
Pulumi config, SkyPilot YAMLs, Dockerfiles, and experiment configs
at PR review time — not after a surprise €89 bill.

Reference: https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEPLOY = _REPO_ROOT / "deployment"
_CONFIGS = _REPO_ROOT / "configs"


def _load_gar_config() -> dict:
    """Load canonical GAR config (single source of truth for registry)."""
    gar_path = _CONFIGS / "registry" / "gar.yaml"
    with gar_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestGARRegionGovernance:
    """GAR must be co-located with GPU training regions to avoid egress costs."""

    def test_gar_region_has_l4_gpus(self) -> None:
        """GAR region must be in a region that has L4 GPUs."""
        gar = _load_gar_config()
        gar_server = gar["server"]  # e.g., "europe-west4-docker.pkg.dev"
        gar_region = gar_server.split("-docker.pkg.dev")[0]

        # Regions with L4 GPUs (verified via `sky show-gpus L4 --cloud gcp`)
        l4_regions = {
            "europe-west4", "europe-west1", "europe-west3", "europe-west2",
            "us-central1", "us-east1", "us-west1", "us-east4",
            "asia-northeast3", "asia-east1", "asia-southeast1",
        }
        assert gar_region in l4_regions, (
            f"GAR region {gar_region} has no L4 GPUs — all Docker pulls will be "
            f"cross-region egress. Move GAR to a region with L4 GPUs."
        )

    def test_gar_region_matches_primary_gpu_region(self) -> None:
        """GAR region must match the first region in the active region config."""
        gar = _load_gar_config()
        gar_server = gar["server"]
        gar_region = gar_server.split("-docker.pkg.dev")[0]

        # Read the debug factorial config to get the region config name
        debug_yaml = _CONFIGS / "factorial" / "debug.yaml"
        with debug_yaml.open(encoding="utf-8") as f:
            debug = yaml.safe_load(f)

        region_config_name = debug.get("region_config", "europe")
        region_config_path = _CONFIGS / "cloud" / "regions" / f"{region_config_name}.yaml"
        with region_config_path.open(encoding="utf-8") as f:
            region_config = yaml.safe_load(f)

        # Get the first (highest priority) region
        first_region = region_config["regions"]["gcp"]["L4"][0]["region"]
        assert gar_region == first_region, (
            f"GAR region ({gar_region}) != primary GPU region ({first_region}). "
            f"This causes cross-region egress on every Docker pull."
        )

    def test_no_intercontinental_fallback_in_debug(self) -> None:
        """Debug experiments must NOT use region configs with US/Asia fallback."""
        debug_yaml = _CONFIGS / "factorial" / "debug.yaml"
        with debug_yaml.open(encoding="utf-8") as f:
            debug = yaml.safe_load(f)

        region_config_name = debug.get("region_config", "europe")
        region_config_path = _CONFIGS / "cloud" / "regions" / f"{region_config_name}.yaml"
        with region_config_path.open(encoding="utf-8") as f:
            region_config = yaml.safe_load(f)

        regions = [r["region"] for r in region_config["regions"]["gcp"]["L4"]]
        non_eu = [r for r in regions if not r.startswith("europe-")]
        assert not non_eu, (
            f"Debug region config '{region_config_name}' contains non-EU regions: {non_eu}. "
            f"Intercontinental Docker pulls cost €0.068/GB. Use europe_strict or europe."
        )


class TestSkyPilotCostGovernance:
    """SkyPilot YAMLs must use cost-efficient defaults."""

    def test_spot_instance_used_by_default(self) -> None:
        """train_factorial.yaml must use spot instances by default."""
        yaml_path = _DEPLOY / "skypilot" / "train_factorial.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["resources"]["use_spot"] is True, (
            "train_factorial.yaml must use spot by default (use_spot: true)"
        )

    def test_max_restarts_bounded(self) -> None:
        """max_restarts_on_errors must be <= 5 to prevent infinite preemption loops."""
        yaml_path = _DEPLOY / "skypilot" / "train_factorial.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        max_restarts = config.get("job_recovery", {}).get("max_restarts_on_errors", 0)
        assert max_restarts <= 5, (
            f"max_restarts_on_errors={max_restarts} is too high — "
            f"each restart re-pulls a 6.4 GB Docker image"
        )

    def test_disk_size_not_excessive(self) -> None:
        """Disk size must be <= 200 GB to prevent storage waste."""
        yaml_path = _DEPLOY / "skypilot" / "train_factorial.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        disk_size = config["resources"].get("disk_size", 0)
        assert disk_size <= 200, f"disk_size={disk_size} GB is excessive"


class TestControllerGovernance:
    """SkyPilot controller must be cost-efficient."""

    def test_controller_cpus_bounded(self) -> None:
        """Controller must not request more than 2+ CPUs."""
        sky_yaml = _REPO_ROOT / ".sky.yaml"
        with sky_yaml.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cpus = config["jobs"]["controller"]["resources"].get("cpus", "8+")
        # Parse "2+" to get the number
        cpu_min = int(str(cpus).rstrip("+"))
        assert cpu_min <= 4, (
            f"Controller requests {cpus} CPUs — a job scheduler doesn't need "
            f"more than 2-4. Use e2-medium ($0.034/hr) not n4-standard-4 ($0.169/hr)."
        )

    def test_controller_region_matches_gar(self) -> None:
        """Controller should be in the same region as GAR."""
        sky_yaml = _REPO_ROOT / ".sky.yaml"
        with sky_yaml.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        infra = config["jobs"]["controller"]["resources"]["infra"]
        controller_region = infra.split("/")[1]  # "gcp/europe-west4" → "europe-west4"

        gar = _load_gar_config()
        gar_region = gar["server"].split("-docker.pkg.dev")[0]

        assert controller_region == gar_region, (
            f"Controller ({controller_region}) != GAR ({gar_region}). "
            f"Co-locate for lowest latency."
        )


class TestDockerImageGovernance:
    """Docker images must be built efficiently."""

    def test_base_image_uses_multistage_build(self) -> None:
        """Dockerfile.base must use multi-stage build (builder + runner)."""
        dockerfile = _DEPLOY / "docker" / "Dockerfile.base"
        content = dockerfile.read_text(encoding="utf-8")
        assert "AS builder" in content, "Dockerfile.base must have a builder stage"
        assert "AS runner" in content, "Dockerfile.base must have a runner stage"

    def test_no_dev_deps_in_production_image(self) -> None:
        """Production Docker image must use --no-dev to exclude dev dependencies."""
        dockerfile = _DEPLOY / "docker" / "Dockerfile.base"
        content = dockerfile.read_text(encoding="utf-8")
        assert "--no-dev" in content, (
            "Dockerfile.base must use uv sync --no-dev for production images"
        )
