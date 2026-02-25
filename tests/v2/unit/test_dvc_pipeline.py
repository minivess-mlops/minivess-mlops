"""Tests for DVC pipeline configuration.

Verifies that dvc.yaml defines valid stages with consistent dependencies,
and that stage commands reference existing modules.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DVC_YAML = PROJECT_ROOT / "dvc.yaml"


class TestDvcYamlValid:
    """dvc.yaml should be valid YAML with expected stage structure."""

    def test_dvc_yaml_exists(self) -> None:
        assert DVC_YAML.exists(), f"dvc.yaml not found at {DVC_YAML}"

    def test_dvc_yaml_parseable(self) -> None:
        with DVC_YAML.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "stages" in config

    def test_stages_have_cmd(self) -> None:
        with DVC_YAML.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for name, stage in config["stages"].items():
            assert "cmd" in stage, f"Stage '{name}' missing 'cmd'"

    def test_download_stage_exists(self) -> None:
        with DVC_YAML.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "download" in config["stages"], "download stage should exist"
        download = config["stages"]["download"]
        assert download.get("frozen") is True, "download stage should be frozen"

    def test_preprocess_stage_deps(self) -> None:
        with DVC_YAML.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        preprocess = config["stages"]["preprocess"]
        deps = preprocess.get("deps", [])
        assert any("preprocess.py" in d for d in deps), "preprocess should depend on preprocess.py"

    def test_preprocess_module_exists(self) -> None:
        """The preprocess module referenced in dvc.yaml should exist."""
        module_path = PROJECT_ROOT / "src" / "minivess" / "data" / "preprocess.py"
        assert module_path.exists(), f"preprocess.py not found at {module_path}"


class TestStageConsistency:
    """Stage dependencies should be internally consistent."""

    def test_preprocess_outputs_exist_as_validate_deps(self) -> None:
        with DVC_YAML.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        preprocess_outs = config["stages"]["preprocess"].get("outs", [])
        validate_deps = config["stages"].get("validate_data", {}).get("deps", [])

        # At least one preprocess output should be a validate_data dependency
        if validate_deps:
            out_set = set(preprocess_outs)
            dep_set = set(validate_deps)
            assert out_set & dep_set, (
                "preprocess outputs should feed into validate_data deps"
            )
