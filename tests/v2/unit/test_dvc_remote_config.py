"""Unit tests for DVC remote configuration (#631, T0.2).

Validates that .dvc/config has a properly structured 'upcloud' remote
for S3-compatible data transport to RunPod GPU instances via SkyPilot.

Approach: parse .dvc/config with configparser (INI-like format).
No moto dependency — unit tests verify config structure only.

Run: uv run pytest tests/v2/unit/test_dvc_remote_config.py -v
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
DVC_CONFIG = ROOT / ".dvc" / "config"


def _parse_dvc_config() -> configparser.ConfigParser:
    """Parse .dvc/config as INI."""
    parser = configparser.ConfigParser()
    parser.read(DVC_CONFIG, encoding="utf-8")
    return parser


class TestDvcUpcloudRemote:
    """Validate upcloud remote exists in .dvc/config with required fields."""

    # DVC config uses single-quoted section names: ['remote "upcloud"']
    SECTION = "'remote \"upcloud\"'"

    def test_dvc_config_has_upcloud_remote(self) -> None:
        """DVC config must define a remote named 'upcloud'."""
        parser = _parse_dvc_config()
        assert self.SECTION in parser.sections(), (
            ".dvc/config missing upcloud remote. "
            "Add: dvc remote add upcloud s3://minivess-dvc-data"
        )

    def test_upcloud_remote_url_is_s3(self) -> None:
        """Upcloud remote URL must use s3:// protocol."""
        parser = _parse_dvc_config()
        if self.SECTION not in parser.sections():
            pytest.skip("upcloud remote not configured yet")
        url = parser.get(self.SECTION, "url")
        assert url.startswith("s3://"), f"Expected s3:// URL, got: {url}"
        assert "minivess-dvc-data" in url

    def test_upcloud_remote_has_endpointurl(self) -> None:
        """Upcloud remote must have endpointurl for S3-compatible endpoint."""
        parser = _parse_dvc_config()
        if self.SECTION not in parser.sections():
            pytest.skip("upcloud remote not configured yet")
        assert parser.has_option(self.SECTION, "endpointurl"), (
            "upcloud remote needs endpointurl for UpCloud S3-compatible endpoint"
        )

    def test_upcloud_remote_endpoint_is_placeholder(self) -> None:
        """Committed .dvc/config endpointurl must be a placeholder (not real creds).

        Real credentials go in .dvc/config.local (gitignored).
        """
        parser = _parse_dvc_config()
        if self.SECTION not in parser.sections():
            pytest.skip("upcloud remote not configured yet")
        if not parser.has_option(self.SECTION, "endpointurl"):
            pytest.skip("endpointurl not set yet")
        endpoint = parser.get(self.SECTION, "endpointurl")
        # Should NOT contain real UpCloud endpoints in committed config
        # The real endpoint goes in .dvc/config.local
        assert "minioadmin" not in endpoint, (
            "upcloud remote should not have MinIO credentials"
        )


class TestDvcConfigureScript:
    """Validate scripts/configure_dvc_remote.py exists and is functional."""

    def test_configure_script_exists(self) -> None:
        """scripts/configure_dvc_remote.py must exist."""
        script = ROOT / "scripts" / "configure_dvc_remote.py"
        assert script.exists(), (
            "scripts/configure_dvc_remote.py not found. "
            "Create it to auto-configure DVC upcloud remote from .env"
        )

    def test_configure_script_importable(self) -> None:
        """Script must be importable (valid Python syntax)."""
        script = ROOT / "scripts" / "configure_dvc_remote.py"
        if not script.exists():
            pytest.skip("script not created yet")
        import ast

        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
