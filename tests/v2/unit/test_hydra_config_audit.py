"""Tests for Hydra config audit script (#279).

Covers:
- Config file discovery
- Required field validation
- Unused config detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigDiscovery:
    """Test discovery of YAML config files."""

    def test_discover_yaml_files(self, tmp_path: Path) -> None:
        from minivess.config.audit import discover_config_files

        # Create some YAML files
        (tmp_path / "exp1.yaml").write_text("loss: dice_ce\n", encoding="utf-8")
        (tmp_path / "exp2.yaml").write_text("loss: focal_ce\n", encoding="utf-8")
        (tmp_path / "not_yaml.txt").write_text("not yaml\n", encoding="utf-8")

        files = discover_config_files(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".yaml" for f in files)

    def test_discover_recursive(self, tmp_path: Path) -> None:
        from minivess.config.audit import discover_config_files

        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.yaml").write_text("key: value\n", encoding="utf-8")

        files = discover_config_files(tmp_path, recursive=True)
        assert len(files) == 1


class TestConfigValidation:
    """Test required field validation."""

    def test_validate_experiment_config(self, tmp_path: Path) -> None:
        from minivess.config.audit import validate_experiment_config

        config = {
            "experiment_name": "test",
            "losses": ["dice_ce"],
            "model_family": "dynunet",
        }
        issues = validate_experiment_config(config, config_path=tmp_path / "test.yaml")
        # Should pass with all required fields
        assert len(issues) == 0

    def test_validate_missing_required_fields(self, tmp_path: Path) -> None:
        from minivess.config.audit import validate_experiment_config

        config = {"losses": ["dice_ce"]}  # missing experiment_name and model_family
        issues = validate_experiment_config(config, config_path=tmp_path / "test.yaml")
        assert len(issues) > 0
        assert any("experiment_name" in i["field"] for i in issues)


class TestAuditReport:
    """Test audit report generation."""

    def test_generate_audit_report(self) -> None:
        from minivess.config.audit import generate_audit_report

        issues = [
            {
                "file": "exp1.yaml",
                "field": "model_family",
                "severity": "error",
                "message": "Missing required field",
            },
            {
                "file": "exp2.yaml",
                "field": "losses",
                "severity": "warning",
                "message": "Empty losses list",
            },
        ]
        report = generate_audit_report(issues)
        assert "exp1.yaml" in report
        assert "model_family" in report
        assert isinstance(report, str)
