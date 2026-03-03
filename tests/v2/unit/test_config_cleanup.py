"""Tests for config cleanup — legacy rename, Justfile, experiment CLI (#294)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class TestLegacyDefaultsRenamed:
    """Verify the old defaults.yaml is renamed."""

    def test_old_defaults_yaml_does_not_exist(self) -> None:
        assert not (CONFIGS_DIR / "defaults.yaml").exists(), (
            "configs/defaults.yaml should be renamed to _legacy_v01_defaults.yaml"
        )

    def test_legacy_defaults_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "_legacy_v01_defaults.yaml").exists()

    def test_new_base_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "base.yaml").exists()

    def test_old_experiments_dir_does_not_exist(self) -> None:
        """configs/experiments/ should be fully deleted (migrated to configs/experiment/)."""
        assert not (CONFIGS_DIR / "experiments").exists(), (
            "configs/experiments/ should be deleted — use configs/experiment/ instead"
        )


class TestNoCodeReferencesOldDefaults:
    """Verify no source code references the old defaults.yaml path."""

    def test_no_src_references(self) -> None:
        """No Python files in src/ should reference configs/defaults.yaml."""
        src_dir = PROJECT_ROOT / "src"
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "configs/defaults.yaml" not in content, (
                f"{py_file} still references configs/defaults.yaml"
            )


class TestJustfileExists:
    """Verify Justfile is present."""

    def test_justfile_exists(self) -> None:
        # just accepts both 'justfile' and 'Justfile'
        assert (PROJECT_ROOT / "justfile").exists() or (
            PROJECT_ROOT / "Justfile"
        ).exists()

    def test_justfile_has_experiment_recipe(self) -> None:
        path = PROJECT_ROOT / "justfile"
        if not path.exists():
            path = PROJECT_ROOT / "Justfile"
        content = path.read_text(encoding="utf-8")
        assert "experiment NAME" in content


class TestRunExperimentCLI:
    """Verify --experiment flag is the sole config loading path."""

    def test_experiment_flag_accepted(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_experiment import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["--experiment", "dynunet_losses"])
        assert args.experiment == "dynunet_losses"

    def test_experiment_flag_is_required(self) -> None:
        """--experiment is required — no --config fallback."""
        import pytest

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_experiment import _build_arg_parser

        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
