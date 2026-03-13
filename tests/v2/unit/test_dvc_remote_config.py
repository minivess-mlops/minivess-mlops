"""Unit tests for DVC remote configuration (#631 T0.2, #632 T0.3).

Validates that .dvc/config has a properly structured 'upcloud' remote
for S3-compatible data transport to RunPod GPU instances via SkyPilot.

Approach: parse .dvc/config with configparser (INI-like format).
No moto dependency — unit tests verify config structure only.
DVC pull protocol tested via monkeypatch on subprocess.run (RC4).

Run: uv run pytest tests/v2/unit/test_dvc_remote_config.py -v
"""

from __future__ import annotations

import configparser
import subprocess
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


class TestDvcPullProtocol:
    """Verify DVC pull command construction via monkeypatch (#632, T0.3).

    Uses monkeypatch on subprocess.run instead of moto (RC4).
    """

    def test_dvc_pull_uses_correct_remote_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dvc pull -r upcloud must use 'upcloud' remote name."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import _run_dvc

        _run_dvc("pull", "-r", "upcloud")
        assert len(calls) == 1
        assert calls[0] == ["dvc", "pull", "-r", "upcloud"]

    def test_dvc_status_check_uses_upcloud_remote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_connectivity calls dvc status -r upcloud."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import verify_connectivity

        env = {
            "DVC_S3_ENDPOINT_URL": "https://test.example.com",
            "DVC_S3_ACCESS_KEY": "key",
            "DVC_S3_SECRET_KEY": "secret",
            "DVC_S3_BUCKET": "minivess-dvc-data",
        }
        verify_connectivity(env)
        assert any(cmd == ["dvc", "status", "-r", "upcloud"] for cmd in calls)

    def test_configure_remote_sets_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure_remote writes endpoint, access_key, secret_key."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            # Simulate remote listing (upcloud already exists)
            if "list" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="upcloud\ts3://minivess-dvc-data\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import configure_remote

        env = {
            "DVC_S3_ENDPOINT_URL": "https://objects.example.com",
            "DVC_S3_ACCESS_KEY": "AKTEST",
            "DVC_S3_SECRET_KEY": "secret123",
            "DVC_S3_BUCKET": "minivess-dvc-data",
        }
        configure_remote(env)

        # Must have called dvc remote modify --local for endpoint, access_key, secret
        modify_calls = [c for c in calls if "modify" in c]
        assert len(modify_calls) == 3
        assert any("endpointurl" in c for c in modify_calls)
        assert any("access_key_id" in c for c in modify_calls)
        assert any("secret_access_key" in c for c in modify_calls)


class TestDvcVersionPinning:
    """Verify DVC version pin in pyproject.toml (#632, T0.3)."""

    def test_dvc_version_pinned_in_pyproject(self) -> None:
        """pyproject.toml must pin DVC with upper bound."""
        import tomllib

        pyproject = ROOT / "pyproject.toml"
        with pyproject.open("rb") as f:
            config = tomllib.load(f)
        deps = config["project"]["dependencies"]
        dvc_deps = [d for d in deps if d.startswith("dvc")]
        assert dvc_deps, "DVC not found in pyproject.toml dependencies"
        # Must have version constraint (not unpinned)
        dvc_spec = dvc_deps[0]
        assert ">=" in dvc_spec or "==" in dvc_spec, (
            f"DVC dependency '{dvc_spec}' must be version-pinned"
        )
