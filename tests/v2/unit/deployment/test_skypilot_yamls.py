"""SkyPilot YAML validation tests — catch config errors before burning cloud credits.

Uses sky.Task.from_yaml() for structural validation + custom assertions for
project-specific invariants (T4 ban, env vars, Docker image, banned commands).

Issue #908. See: docs/planning/skypilot-fake-mock-ssh-test-suite-plan.md
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKYPILOT_DIR = Path("deployment/skypilot")
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Structural validation via sky.Task.from_yaml()
# ---------------------------------------------------------------------------


class TestSkyPilotYamlParsing:
    """Every SkyPilot YAML must parse without errors."""

    def test_factorial_yaml_parses(self) -> None:
        """train_factorial.yaml must be valid SkyPilot YAML."""
        try:
            import sky

            task = sky.Task.from_yaml(str(FACTORIAL_YAML))
            assert task is not None
        except ImportError:
            pytest.skip("skypilot not installed")

    def test_factorial_yaml_has_resources(self) -> None:
        """Must declare resources (accelerators, cloud)."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "resources" in config, "Missing resources section"

    def test_factorial_yaml_has_setup(self) -> None:
        """Must have a setup section."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "setup" in config, "Missing setup section"
        assert len(config["setup"].strip()) > 0, "Empty setup section"

    def test_factorial_yaml_has_run(self) -> None:
        """Must have a run section."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "run" in config, "Missing run section"
        assert len(config["run"].strip()) > 0, "Empty run section"


# ---------------------------------------------------------------------------
# T4 ban (Turing, no BF16 → FP16 overflow → NaN in SAM3)
# ---------------------------------------------------------------------------


class TestNoT4:
    """T4 is banned — Turing architecture has no BF16 support."""

    def test_no_t4_in_accelerators(self) -> None:
        """accelerators must NOT include T4."""
        config = _load_yaml(FACTORIAL_YAML)
        accels = config.get("resources", {}).get("accelerators", {})
        if isinstance(accels, dict):
            assert "T4" not in accels, (
                f"T4 banned (no BF16). Accelerators: {accels}"
            )
        elif isinstance(accels, str):
            assert "T4" not in accels


# ---------------------------------------------------------------------------
# Spot instances
# ---------------------------------------------------------------------------


class TestSpotEnabled:
    """Production tasks must use spot instances for cost savings."""

    def test_spot_enabled(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        use_spot = config.get("resources", {}).get("use_spot", False)
        assert use_spot is True, "use_spot must be true for cost savings"


# ---------------------------------------------------------------------------
# Docker image (no bare VM)
# ---------------------------------------------------------------------------


class TestDockerImage:
    """SkyPilot YAML must use Docker image, not bare VM."""

    def test_image_id_set(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        image_id = config.get("resources", {}).get("image_id", "")
        assert image_id, "image_id not set — bare VM is banned"
        assert image_id.startswith("docker:"), (
            f"image_id must start with 'docker:': {image_id}"
        )

    def test_image_id_points_to_gar(self) -> None:
        """Docker image must be from GAR (same region as training VMs)."""
        config = _load_yaml(FACTORIAL_YAML)
        image_id = config.get("resources", {}).get("image_id", "")
        assert "europe-north1-docker.pkg.dev" in image_id, (
            f"Image must be from GAR europe-north1: {image_id}"
        )


# ---------------------------------------------------------------------------
# Required env vars
# ---------------------------------------------------------------------------


class TestRequiredEnvVars:
    """All env vars needed by train_flow.py must be declared in the YAML."""

    def test_required_envs_declared(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        required = {
            "MODEL_FAMILY",
            "LOSS_NAME",
            "FOLD_ID",
            "WITH_AUX_CALIB",
            "MAX_EPOCHS",
            "EXPERIMENT_NAME",
            "POST_TRAINING_METHODS",
        }
        declared = set(envs.keys())
        missing = required - declared
        assert not missing, f"Missing env vars in SkyPilot YAML: {missing}"

    def test_hf_token_declared(self) -> None:
        """HF_TOKEN must be in envs (needed for SAM3/VesselFM weight download)."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "HF_TOKEN" in envs, "HF_TOKEN not in envs — SAM3 will fail"


# ---------------------------------------------------------------------------
# Banned commands in setup
# ---------------------------------------------------------------------------


class TestSetupBannedCommands:
    """Setup must NOT install packages — everything is in the Docker image."""

    def test_no_apt_get(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "apt-get" not in setup, "apt-get banned in setup (use Docker image)"

    def test_no_uv_sync(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "uv sync" not in setup, "uv sync banned in setup (use Docker image)"

    def test_no_pip_install(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "pip install" not in setup, "pip install banned in setup (use Docker image)"

    def test_no_git_clone(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "git clone" not in setup, "git clone banned in setup (use Docker image)"


# ---------------------------------------------------------------------------
# DVC pull path validation
# ---------------------------------------------------------------------------


class TestSetupDvcPull:
    """Setup DVC pull must use path-specific pull, not bare dvc pull."""

    def test_no_bare_dvc_pull(self) -> None:
        """Must NOT use bare 'dvc pull -r gcs' — will fail on unpushed tracked files."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        for line in setup.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped and "-r gcs" in stripped:
                # Must have a path argument between "pull" and "-r"
                parts = stripped.split()
                pull_idx = parts.index("pull") if "pull" in parts else -1
                if pull_idx >= 0:
                    next_parts = parts[pull_idx + 1 :]
                    # Filter out flags
                    path_args = [p for p in next_parts if not p.startswith("-")]
                    assert len(path_args) > 0, (
                        f"Bare 'dvc pull -r gcs' found (no path filter). "
                        f"Use: dvc pull data/raw/minivess -r gcs. Line: {stripped}"
                    )

    def test_setup_has_error_handling_for_dvc(self) -> None:
        """DVC pull must have error handling (|| { exit 1; })."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "exit 1" in setup, (
            "Setup must exit on DVC pull failure (no silent continuation)"
        )


# ---------------------------------------------------------------------------
# Run section guards
# ---------------------------------------------------------------------------


class TestRunSectionGuards:
    """Run section must verify prerequisites before training."""

    def test_run_checks_splits_file(self) -> None:
        """Run must check splits.json exists before proceeding."""
        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")
        assert "splits.json" in run, (
            "Run section must check for splits.json existence"
        )
