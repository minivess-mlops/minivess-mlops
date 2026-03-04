"""Tests for pre-commit capability schema check (Phase 5, #337).

Verifies that the capability schema check:
1. Returns no errors for valid schema
2. Detects missing models
3. Detects invalid loss references
4. CLI returns correct exit codes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from minivess.testing.capability_discovery import check_consistency

if TYPE_CHECKING:
    from pathlib import Path


class TestCheckConsistency:
    """check_consistency returns errors for invalid schemas."""

    def test_valid_schema_no_errors(self) -> None:
        errors = check_consistency()
        assert errors == []

    def test_detects_missing_model(self, tmp_path: Path) -> None:
        """Schema with ModelFamily member missing from both lists."""
        # Write a YAML missing a model
        from minivess.config.models import ModelFamily

        all_models = [m.value for m in ModelFamily]
        # Include all but one in either list
        yaml_data = {
            "version": "1.0",
            "implemented_models": all_models[:5],
            "not_implemented": all_models[5:-1],  # skip last
            "loss_exclusions": {},
            "post_training_exclusions": {},
            "ensemble_exclusions": {},
            "deployment_exclusions": {},
            "model_default_loss": {},
            "model_extra_losses": {},
        }
        path = tmp_path / "test_caps.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh)

        errors = check_consistency(path)
        assert len(errors) >= 1
        assert any("not in" in e for e in errors)

    def test_detects_invalid_default_loss(self, tmp_path: Path) -> None:
        """Schema with model_default_loss referencing nonexistent loss."""
        from minivess.config.models import ModelFamily

        all_models = [m.value for m in ModelFamily]
        yaml_data = {
            "version": "1.0",
            "implemented_models": all_models[:5],
            "not_implemented": all_models[5:],
            "loss_exclusions": {},
            "post_training_exclusions": {},
            "ensemble_exclusions": {},
            "deployment_exclusions": {},
            "model_default_loss": {all_models[0]: "totally_fake_loss"},
            "model_extra_losses": {},
        }
        path = tmp_path / "test_caps.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh)

        errors = check_consistency(path)
        assert any("not in known losses" in e for e in errors)

    def test_detects_invalid_extra_loss(self, tmp_path: Path) -> None:
        """Schema with model_extra_losses referencing nonexistent loss."""
        from minivess.config.models import ModelFamily

        all_models = [m.value for m in ModelFamily]
        yaml_data = {
            "version": "1.0",
            "implemented_models": all_models[:5],
            "not_implemented": all_models[5:],
            "loss_exclusions": {},
            "post_training_exclusions": {},
            "ensemble_exclusions": {},
            "deployment_exclusions": {},
            "model_default_loss": {},
            "model_extra_losses": {all_models[0]: ["nonexistent_loss"]},
        }
        path = tmp_path / "test_caps.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh)

        errors = check_consistency(path)
        assert any("not in known losses" in e for e in errors)


class TestCapabilityCheckCli:
    """CLI entry point returns correct exit codes."""

    def test_check_succeeds(self) -> None:
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "minivess.testing.capability_discovery",
                "--check",
            ],
            capture_output=True,
            text=True,
            cwd="/home/petteri/Dropbox/github-personal/minivess-mlops",
        )
        assert result.returncode == 0
        assert "consistent" in result.stdout.lower()

    def test_check_with_invalid_yaml_fails(self, tmp_path: Path) -> None:
        import subprocess

        # Write invalid YAML
        yaml_data = {
            "version": "1.0",
            "implemented_models": ["dynunet"],
            "not_implemented": [],
            "loss_exclusions": {},
            "model_default_loss": {},
            "model_extra_losses": {},
        }
        path = tmp_path / "bad.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh)

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "minivess.testing.capability_discovery",
                "--check",
                "--yaml",
                str(path),
            ],
            capture_output=True,
            text=True,
            cwd="/home/petteri/Dropbox/github-personal/minivess-mlops",
        )
        assert result.returncode == 1
