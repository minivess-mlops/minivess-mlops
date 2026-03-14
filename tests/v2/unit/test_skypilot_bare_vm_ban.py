"""Enforce Docker mandate: NO bare-VM commands in SkyPilot YAMLs.

All SkyPilot YAMLs must use Docker image_id for dependency management.
Setup sections are DATA ONLY — DVC pull, config copy, directory creation.

BANNED in setup: apt-get, uv sync, git clone, pip install
See: .claude/metalearning/2026-03-14-skypilot-bare-vm-docker-violation.md
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SKYPILOT_DIR = Path("deployment/skypilot")

# Commands that indicate bare-VM setup (BANNED by Docker mandate)
_BANNED_SETUP_COMMANDS = [
    "apt-get",
    "uv sync",
    "git clone",
    "pip install",
    "conda install",
    "pip3 install",
]


def _all_skypilot_yamls() -> list[Path]:
    """Return all YAML files in deployment/skypilot/."""
    return sorted(_SKYPILOT_DIR.glob("*.yaml"))


class TestNoBareVmSkyPilotYamls:
    """Ensure no SkyPilot YAML uses bare-VM dependency installation."""

    def test_no_bare_vm_commands_in_setup(self) -> None:
        """No SkyPilot YAML setup: section may contain banned commands."""
        violations: list[str] = []
        for yaml_path in _all_skypilot_yamls():
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            setup = config.get("setup", "")
            for cmd in _BANNED_SETUP_COMMANDS:
                if cmd in setup:
                    violations.append(f"{yaml_path.name}: setup contains '{cmd}'")

        assert not violations, (
            "Bare-VM commands found in SkyPilot YAML setup sections "
            "(Docker mandate violation):\n  " + "\n  ".join(violations)
        )

    def test_all_yamls_use_docker_image_id(self) -> None:
        """Every SkyPilot YAML must specify image_id: docker:... in resources."""
        missing: list[str] = []
        for yaml_path in _all_skypilot_yamls():
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            resources = config.get("resources", {})
            image_id = resources.get("image_id", "")
            if not str(image_id).startswith("docker:"):
                missing.append(f"{yaml_path.name}: no docker: image_id")

        assert not missing, (
            "SkyPilot YAMLs without Docker image_id "
            "(bare-VM approach is BANNED):\n  " + "\n  ".join(missing)
        )

    def test_no_uv_run_in_setup_or_run(self) -> None:
        """uv binary is NOT in the runner image — use python/dvc directly."""
        violations: list[str] = []
        for yaml_path in _all_skypilot_yamls():
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            for section in ("setup", "run"):
                content = config.get(section, "")
                if "uv run " in content:
                    violations.append(
                        f"{yaml_path.name}: {section} contains 'uv run' "
                        "(uv not in runner image — use python/dvc directly)"
                    )

        assert not violations, (
            "SkyPilot YAMLs using 'uv run' (not available in Docker runner):\n  "
            + "\n  ".join(violations)
        )

    def test_no_deleted_bare_vm_yamls_exist(self) -> None:
        """train_generic.yaml and train_hpo_sweep.yaml must not exist."""
        for name in ("train_generic.yaml", "train_hpo_sweep.yaml"):
            path = _SKYPILOT_DIR / name
            assert not path.exists(), (
                f"{name} still exists — this bare-VM YAML was scheduled for deletion. "
                "All training must use Docker image_id pattern (see smoke_test_gpu.yaml)."
            )
