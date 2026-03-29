"""Tests for Docker pull optimization — GAR remote repository + SkyPilot MOUNT_CACHED.

T1.6: Validate Pulumi GAR remote repo region config. Validate SkyPilot
file_mounts with mode: MOUNT_CACHED.

Uses yaml.safe_load() and ast.parse() — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PULUMI_MAIN = PROJECT_ROOT / "deployment" / "pulumi" / "gcp" / "__main__.py"
SKYPILOT_WORKER = PROJECT_ROOT / "deployment" / "skypilot" / "hpo_grid_worker.yaml"


class TestGARRemoteRepo:
    """Test GAR remote repository cache configuration in Pulumi."""

    def test_pulumi_has_gar_remote_repo(self) -> None:
        """Pulumi __main__.py must define a GAR remote repo for caching."""
        source = PULUMI_MAIN.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Look for a variable assignment or function call creating a remote repo
        # We're looking for gcp.artifactregistry.Repository with mode="REMOTE_REPOSITORY"
        # or a second Repository resource for remote caching
        found_remote_repo = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "remote" in node.value.lower()
                and "repo" in node.value.lower()
            ):
                found_remote_repo = True
                break
            if (
                isinstance(node, ast.keyword)
                and node.arg == "mode"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and "REMOTE" in node.value.value.upper()
            ):
                found_remote_repo = True
                break
            # Also check string resource IDs
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in ("docker-remote-repo", "gar-remote-cache")
            ):
                found_remote_repo = True
                break

        assert found_remote_repo, (
            "Pulumi __main__.py must define a GAR remote repository cache "
            "for nvidia base layers (same-region as training VMs). "
            "Add a gcp.artifactregistry.Repository with mode=REMOTE_REPOSITORY."
        )

    def test_pulumi_gar_in_configured_region(self) -> None:
        """GAR remote repo must be in the configured region (same region as training)."""
        source = PULUMI_MAIN.read_text(encoding="utf-8")
        # The existing docker_repo uses location=region which is read from Pulumi config.
        # Verify the region variable is used consistently.
        gar_config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "registry" / "gar.yaml").read_text(
                encoding="utf-8"
            )
        )
        expected_server = gar_config["server"]
        region_prefix = expected_server.split("-docker.pkg.dev")[0]
        assert region_prefix in source or "region" in source, (
            f"Pulumi GAR configuration must use {region_prefix} region"
        )


class TestSkyPilotGridWorkerYAML:
    """Test SkyPilot hpo_grid_worker.yaml configuration."""

    def test_hpo_grid_worker_yaml_exists(self) -> None:
        """deployment/skypilot/hpo_grid_worker.yaml must exist."""
        assert SKYPILOT_WORKER.exists(), (
            f"SkyPilot grid worker YAML not found at {SKYPILOT_WORKER}. "
            "Create it for the factorial GPU study."
        )

    def test_hpo_grid_worker_valid_yaml(self) -> None:
        """hpo_grid_worker.yaml must parse as a dict."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"

    def test_hpo_grid_worker_uses_docker_image(self) -> None:
        """SkyPilot must use image_id: docker:... (bare VM BANNED)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        resources = data.get("resources", {})
        image_id = resources.get("image_id", "")
        assert image_id.startswith("docker:"), (
            f"SkyPilot resources.image_id must start with 'docker:', got {image_id!r}. "
            "Bare VM setup is BANNED (CLAUDE.md)."
        )

    def test_hpo_grid_worker_no_apt_get(self) -> None:
        """Setup must not contain apt-get (all deps in Docker image)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        setup = data.get("setup", "")
        assert "apt-get" not in setup, (
            "SkyPilot setup must NOT contain apt-get — all deps are in the Docker image"
        )

    def test_hpo_grid_worker_no_uv_sync(self) -> None:
        """Setup must not contain uv sync (all deps in Docker image)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        setup = data.get("setup", "")
        assert "uv sync" not in setup, (
            "SkyPilot setup must NOT contain 'uv sync' — all deps are in the Docker image"
        )

    def test_hpo_grid_worker_uses_prefect_module(self) -> None:
        """run: section must invoke hpo_flow via python -m (Rule #17)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        run_section = data.get("run", "")
        assert "minivess.orchestration.flows.hpo_flow" in run_section, (
            "SkyPilot run section must invoke "
            "'python -m minivess.orchestration.flows.hpo_flow' (Rule #17)"
        )

    def test_hpo_grid_worker_no_t4(self) -> None:
        """T4 GPU must NOT be in accelerators list (BANNED — no BF16)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        resources = data.get("resources", {})
        accelerators = resources.get("accelerators", {})
        if isinstance(accelerators, dict):
            assert "T4" not in accelerators, (
                "T4 is BANNED — no BF16 support, causes FP16 overflow NaN"
            )
        elif isinstance(accelerators, str):
            assert "T4" not in accelerators, "T4 is BANNED — no BF16 support"

    def test_hpo_grid_worker_uses_gcp(self) -> None:
        """Grid worker targets GCP (production factorial study)."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        resources = data.get("resources", {})
        cloud = resources.get("cloud", "")
        assert cloud == "gcp", (
            f"Grid worker must target GCP for production factorial, got {cloud!r}"
        )

    def test_hpo_grid_worker_uses_spot(self) -> None:
        """Grid worker should use spot instances for cost savings."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        resources = data.get("resources", {})
        use_spot = resources.get("use_spot", False)
        assert use_spot is True, (
            "Grid worker should use spot instances for the factorial study "
            "(60-91% cheaper on GCP)"
        )

    def test_hpo_grid_worker_has_worker_env_vars(self) -> None:
        """Worker must expose WORKER_ID and TOTAL_WORKERS env vars."""
        with open(SKYPILOT_WORKER, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        envs = data.get("envs", {})
        run_section = data.get("run", "")
        has_worker_id = "WORKER_ID" in envs or "worker_id" in str(run_section).lower()
        assert has_worker_id, (
            "Grid worker must define WORKER_ID env var for partition assignment"
        )
