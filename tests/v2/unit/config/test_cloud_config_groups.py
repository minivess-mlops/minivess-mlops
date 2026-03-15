"""Tests for Hydra cloud + registry + lab/user config groups (#708).

Verifies:
1. All required config group directories and files exist
2. YAML files are valid and parseable
3. base.yaml includes cloud, registry, lab, user in defaults
4. Lab/user directories have .gitignore
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIGS_ROOT = Path("configs")


class TestCloudConfigGroupsExist:
    """All cloud provider config files must exist."""

    @pytest.mark.parametrize(
        "filename",
        ["gcp_spot.yaml", "lambda.yaml", "runpod_dev.yaml", "local.yaml"],
    )
    def test_cloud_config_exists(self, filename: str) -> None:
        path = CONFIGS_ROOT / "cloud" / filename
        assert path.exists(), f"Missing cloud config: {path}"

    @pytest.mark.parametrize(
        "filename", ["gcp_spot.yaml", "lambda.yaml", "runpod_dev.yaml", "local.yaml"]
    )
    def test_cloud_config_valid_yaml(self, filename: str) -> None:
        path = CONFIGS_ROOT / "cloud" / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{filename} must be a valid YAML dict"
        assert "provider" in data, f"{filename} must have a 'provider' key"

    @pytest.mark.parametrize(
        "filename", ["gcp_spot.yaml", "lambda.yaml", "runpod_dev.yaml"]
    )
    def test_cloud_config_has_accelerators(self, filename: str) -> None:
        """Non-local cloud configs must specify accelerators."""
        path = CONFIGS_ROOT / "cloud" / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "accelerators" in data, f"{filename} must have 'accelerators' key"
        assert data["accelerators"] is not None, (
            f"{filename} accelerators must not be null"
        )


class TestRegistryConfigGroupsExist:
    """Docker registry config files must exist."""

    @pytest.mark.parametrize("filename", ["ghcr.yaml", "gar.yaml"])
    def test_registry_config_exists(self, filename: str) -> None:
        path = CONFIGS_ROOT / "registry" / filename
        assert path.exists(), f"Missing registry config: {path}"

    @pytest.mark.parametrize("filename", ["ghcr.yaml", "gar.yaml"])
    def test_registry_config_valid_yaml(self, filename: str) -> None:
        path = CONFIGS_ROOT / "registry" / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "registry" in data
        assert "server" in data


class TestLabUserConfigGroupsExist:
    """Lab and user config directories must exist with defaults."""

    @pytest.mark.parametrize("group", ["lab", "user"])
    def test_default_yaml_exists(self, group: str) -> None:
        path = CONFIGS_ROOT / group / "default.yaml"
        assert path.exists(), f"Missing {group}/default.yaml"

    @pytest.mark.parametrize("group", ["lab", "user"])
    def test_gitignore_exists(self, group: str) -> None:
        path = CONFIGS_ROOT / group / ".gitignore"
        assert path.exists(), f"Missing {group}/.gitignore"

    @pytest.mark.parametrize("group", ["lab", "user"])
    def test_default_is_empty_or_dict(self, group: str) -> None:
        path = CONFIGS_ROOT / group / "default.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # default.yaml should be empty dict or None (no overrides)
        assert data is None or isinstance(data, dict)


class TestBaseYamlIncludesCloudGroups:
    """base.yaml defaults list must include cloud, registry, lab, user."""

    def test_base_yaml_defaults(self) -> None:
        base = yaml.safe_load((CONFIGS_ROOT / "base.yaml").read_text(encoding="utf-8"))
        defaults = base.get("defaults", [])
        # Extract group names from defaults list
        group_names: set[str] = set()
        for item in defaults:
            if isinstance(item, dict):
                group_names.update(item.keys())
            elif isinstance(item, str):
                group_names.add(item)

        for required in ("cloud", "registry", "lab", "user"):
            assert required in group_names, (
                f"base.yaml defaults must include '{required}' config group"
            )
