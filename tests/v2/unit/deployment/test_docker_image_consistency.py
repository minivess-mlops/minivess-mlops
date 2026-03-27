"""Cross-file Docker GAR image path consistency tests.

Validates that the GAR image path in preflight_gcp.py matches all
SkyPilot YAMLs and cloud config files. Drift in any file causes
SkyPilot to pull the wrong Docker image, silently breaking all jobs.

Source of truth: scripts/preflight_gcp.py::GAR_IMAGE
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

PREFLIGHT_PATH = Path("scripts/preflight_gcp.py")
SKYPILOT_DIR = Path("deployment/skypilot")
CLOUD_CONFIG = Path("configs/cloud/gcp_spot.yaml")

# SkyPilot YAMLs that use the GAR image (not Docker Hub images)
GAR_SKYPILOT_YAMLS = [
    SKYPILOT_DIR / "train_factorial.yaml",
    SKYPILOT_DIR / "train_production.yaml",
    SKYPILOT_DIR / "train_hpo.yaml",
    SKYPILOT_DIR / "hpo_grid_worker.yaml",
    SKYPILOT_DIR / "smoke_test_gcp.yaml",
]


def _extract_gar_image() -> str:
    """Extract GAR_IMAGE constant from preflight_gcp.py via AST."""
    source = PREFLIGHT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "GAR_IMAGE"
                    and isinstance(node.value, ast.Constant)
                ):
                    return str(node.value.value)
    msg = "GAR_IMAGE constant not found in preflight_gcp.py"
    raise ValueError(msg)


class TestDockerImageConsistency:
    """GAR image path must be consistent across all config files."""

    def test_gar_image_source_constant_exists(self) -> None:
        """preflight_gcp.py must define GAR_IMAGE constant."""
        gar_image = _extract_gar_image()
        assert "europe-north1-docker.pkg.dev" in gar_image
        assert "minivess-mlops" in gar_image

    @pytest.mark.parametrize("yaml_path", GAR_SKYPILOT_YAMLS, ids=lambda p: p.name)
    def test_skypilot_yamls_use_correct_image(self, yaml_path: Path) -> None:
        """SkyPilot YAML image_id must match GAR_IMAGE."""
        gar_image = _extract_gar_image()

        with yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # image_id can be under resources or at top level
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")

        # Strip "docker:" prefix for comparison
        actual_image = image_id.removeprefix("docker:")
        assert actual_image == gar_image, (
            f"{yaml_path.name}: image_id '{actual_image}' != GAR_IMAGE '{gar_image}'"
        )

    def test_cloud_config_matches_gar_image(self) -> None:
        """configs/cloud/gcp_spot.yaml docker_image must match GAR_IMAGE."""
        gar_image = _extract_gar_image()

        with CLOUD_CONFIG.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        docker_image = config.get("docker_image", "")
        assert docker_image == gar_image, (
            f"gcp_spot.yaml docker_image '{docker_image}' != GAR_IMAGE '{gar_image}'"
        )
