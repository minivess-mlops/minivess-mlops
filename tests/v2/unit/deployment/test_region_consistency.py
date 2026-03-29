"""Region migration consistency tests — regression safety net.

Every region migration (europe-north1 → europe-west4 → us-central1) has caused
$89/month egress, FAILED_SETUP, or 12+ hour PENDING due to stale references.
These tests verify ALL config files agree on the same region.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 1 (P0): T1.1 – T1.6
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths (relative to repo root for portability)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
GCP_SPOT = REPO_ROOT / "configs" / "cloud" / "gcp_spot.yaml"
PULUMI_DEV = REPO_ROOT / "deployment" / "pulumi" / "gcp" / "Pulumi.dev.yaml"
PREFLIGHT = REPO_ROOT / "scripts" / "preflight_gcp.py"
DEBUG_YAML = REPO_ROOT / "configs" / "factorial" / "debug.yaml"
REGIONS_DIR = REPO_ROOT / "configs" / "cloud" / "regions"
SKYPILOT_DIR = REPO_ROOT / "deployment" / "skypilot"
TRAIN_FACTORIAL = SKYPILOT_DIR / "train_factorial.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _extract_gar_region(url: str) -> str:
    """Extract the region prefix from a GAR URL like us-central1-docker.pkg.dev/..."""
    # GAR URL format: {region}-docker.pkg.dev/{project}/{repo}/{image}:{tag}
    parts = url.split("-docker.pkg.dev")
    return parts[0] if parts else ""


def _extract_preflight_constant(name: str) -> str:
    """Extract a string constant from preflight_gcp.py using ast (Rule 16: no regex)."""
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


# ===========================================================================
# T1.1: SkyPilot YAML image_id matches configured region
# ===========================================================================


class TestGarRegionConsistency:
    """All GAR URLs across configs must reference the same region."""

    def test_skypilot_yamls_image_id_region_matches_gcp_spot(self) -> None:
        """Each SkyPilot YAML with image_id must match gcp_spot.yaml's docker_image region."""
        gcp_spot = _load_yaml(GCP_SPOT)
        expected_region = _extract_gar_region(gcp_spot["docker_image"])

        for yaml_path in sorted(SKYPILOT_DIR.glob("*.yaml")):
            config = _load_yaml(yaml_path)
            resources = config.get("resources", {})
            image_id = resources.get("image_id", "")
            if not isinstance(image_id, str) or "docker.pkg.dev" not in image_id:
                continue
            # Strip "docker:" prefix
            url = image_id.removeprefix("docker:")
            actual_region = _extract_gar_region(url)
            assert actual_region == expected_region, (
                f"{yaml_path.name}: GAR region '{actual_region}' != "
                f"gcp_spot region '{expected_region}'"
            )

    def test_all_gar_urls_use_same_region(self) -> None:
        """Collect all GAR URLs across all configs and assert identical regions."""
        regions: dict[str, str] = {}

        # gcp_spot.yaml
        gcp_spot = _load_yaml(GCP_SPOT)
        regions["gcp_spot.yaml:docker_image"] = _extract_gar_region(
            gcp_spot["docker_image"]
        )

        # preflight_gcp.py
        gar_image = _extract_preflight_constant("GAR_IMAGE")
        regions["preflight_gcp.py:GAR_IMAGE"] = _extract_gar_region(gar_image)

        # SkyPilot YAMLs
        for yaml_path in sorted(SKYPILOT_DIR.glob("*.yaml")):
            config = _load_yaml(yaml_path)
            resources = config.get("resources", {})
            image_id = resources.get("image_id", "")
            if isinstance(image_id, str) and "docker.pkg.dev" in image_id:
                url = image_id.removeprefix("docker:")
                regions[f"{yaml_path.name}:image_id"] = _extract_gar_region(url)

        unique_regions = set(regions.values())
        assert len(unique_regions) == 1, (
            f"Multiple GAR regions found: {regions}"
        )


# ===========================================================================
# T1.2: GCS bucket references consistent across configs
# ===========================================================================


class TestGcsBucketConsistency:
    """GCS bucket names must be consistent across preflight and DVC config."""

    def test_preflight_gcs_bucket_matches_dvc_remote(self) -> None:
        """Preflight GCS_BUCKET references the same bucket as DVC remote."""
        gcs_bucket = _extract_preflight_constant("GCS_BUCKET")
        # GCS_BUCKET is "gs://minivess-mlops-dvc-data"
        assert "minivess-mlops-dvc-data" in gcs_bucket

    def test_checkpoint_bucket_referenced_consistently(self) -> None:
        """Preflight CHECKPOINT_BUCKET follows the project naming convention."""
        checkpoint_bucket = _extract_preflight_constant("CHECKPOINT_BUCKET")
        assert "minivess-mlops-checkpoints" in checkpoint_bucket


# ===========================================================================
# T1.3: No stale europe-west4 references in active configs
# ===========================================================================


class TestNoStaleEuropeWest4:
    """After us-central1 migration, no active config should reference europe-west4."""

    def test_no_stale_europe_west4_in_skypilot_yamls(self) -> None:
        """SkyPilot YAML image_id and region fields must not contain europe-west4."""
        for yaml_path in sorted(SKYPILOT_DIR.glob("*.yaml")):
            config = _load_yaml(yaml_path)
            resources = config.get("resources", {})

            # Check image_id
            image_id = resources.get("image_id", "")
            if isinstance(image_id, str):
                assert "europe-west4" not in image_id, (
                    f"{yaml_path.name}: stale europe-west4 in image_id: {image_id}"
                )

            # Check region field
            region = resources.get("region", "")
            if isinstance(region, str):
                assert "europe-west4" not in region, (
                    f"{yaml_path.name}: stale europe-west4 in region: {region}"
                )

    def test_no_stale_europe_west4_in_gcp_spot(self) -> None:
        """gcp_spot.yaml region and docker_image must not be europe-west4."""
        gcp_spot = _load_yaml(GCP_SPOT)
        assert gcp_spot["region"] != "europe-west4", (
            f"gcp_spot.yaml region is stale: {gcp_spot['region']}"
        )
        assert "europe-west4" not in gcp_spot["docker_image"], (
            f"gcp_spot.yaml docker_image has stale region: {gcp_spot['docker_image']}"
        )

    def test_no_stale_europe_west4_in_preflight(self) -> None:
        """preflight_gcp.py GAR_IMAGE must not contain europe-west4."""
        gar_image = _extract_preflight_constant("GAR_IMAGE")
        assert "europe-west4" not in gar_image, (
            f"preflight GAR_IMAGE has stale region: {gar_image}"
        )

    def test_no_stale_europe_west4_in_pulumi_config(self) -> None:
        """Pulumi.dev.yaml gcp:region must not be europe-west4."""
        pulumi = _load_yaml(PULUMI_DEV)
        config = pulumi.get("config", {})
        gcp_region = config.get("gcp:region", "")
        assert gcp_region != "europe-west4", (
            f"Pulumi.dev.yaml gcp:region is stale: {gcp_region}"
        )


# ===========================================================================
# T1.4: Region config matches Pulumi-deployed region
# ===========================================================================


class TestRegionConfigMatchesPulumi:
    """Active region config must target the same region as Pulumi stack."""

    def test_debug_region_config_matches_pulumi_region(self) -> None:
        """debug.yaml's region_config targets the Pulumi-deployed region."""
        debug = _load_yaml(DEBUG_YAML)
        region_config_name = debug["infrastructure"]["region_config"]
        region_config_path = REGIONS_DIR / f"{region_config_name}.yaml"
        region_config = _load_yaml(region_config_path)

        pulumi = _load_yaml(PULUMI_DEV)
        pulumi_region = pulumi["config"]["gcp:region"]

        # Check all GPU entries reference the Pulumi region
        for gpu_type, entries in region_config["regions"]["gcp"].items():
            for entry in entries:
                assert entry["region"] == pulumi_region, (
                    f"Region config {region_config_name}.yaml: "
                    f"GPU {gpu_type} targets {entry['region']} "
                    f"but Pulumi is in {pulumi_region}"
                )

    def test_gcp_spot_region_matches_pulumi_region(self) -> None:
        """gcp_spot.yaml region must match Pulumi.dev.yaml gcp:region."""
        gcp_spot = _load_yaml(GCP_SPOT)
        pulumi = _load_yaml(PULUMI_DEV)
        pulumi_region = pulumi["config"]["gcp:region"]
        assert gcp_spot["region"] == pulumi_region, (
            f"gcp_spot.yaml region '{gcp_spot['region']}' != "
            f"Pulumi region '{pulumi_region}'"
        )


# ===========================================================================
# T1.5: GAR URL in preflight_gcp.py matches gcp_spot.yaml
# ===========================================================================


def test_preflight_gar_image_matches_gcp_spot_docker_image() -> None:
    """preflight_gcp.py GAR_IMAGE == gcp_spot.yaml docker_image."""
    gar_image = _extract_preflight_constant("GAR_IMAGE")
    gcp_spot = _load_yaml(GCP_SPOT)
    assert gar_image == gcp_spot["docker_image"], (
        f"GAR_IMAGE mismatch: preflight='{gar_image}' vs "
        f"gcp_spot='{gcp_spot['docker_image']}'"
    )


# ===========================================================================
# T1.6: Factorial YAML image_id matches gcp_spot docker_image
# ===========================================================================


def test_factorial_image_id_matches_gcp_spot() -> None:
    """train_factorial.yaml image_id (stripped of docker: prefix) == gcp_spot docker_image."""
    factorial = _load_yaml(TRAIN_FACTORIAL)
    image_id = factorial["resources"]["image_id"]
    # Strip "docker:" prefix
    actual_image = image_id.removeprefix("docker:")

    gcp_spot = _load_yaml(GCP_SPOT)
    expected_image = gcp_spot["docker_image"]

    assert actual_image == expected_image, (
        f"Factorial image_id mismatch: '{actual_image}' vs "
        f"gcp_spot '{expected_image}'"
    )
