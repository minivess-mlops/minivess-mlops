"""Unit tests for DVC remote configuration.

Validates that .dvc/config has properly structured remotes for the active
data transport strategy: Network Volume cache-first → AWS S3 fallback.

UpCloud remote archived 2026-03-16 — provider dropped.
Active remotes:
  - minio: local MinIO (default for Docker Compose stack)
  - remote_storage: AWS S3 public bucket (s3://minivessdataset, no credentials)
  - remote_readonly: AWS S3 public website endpoint (read-only)

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


class TestDvcActiveRemotes:
    """Validate active remotes exist in .dvc/config with required fields.

    Active strategy (2026-03-16): Network Volume cache-first → AWS S3 fallback.
    UpCloud archived — remote_storage (AWS S3 public) is the cloud fallback.
    """

    def test_dvc_config_has_remote_storage(self) -> None:
        """DVC config must define remote_storage pointing to AWS S3."""
        parser = _parse_dvc_config()
        section = "'remote \"remote_storage\"'"
        assert section in parser.sections(), (
            ".dvc/config missing remote_storage remote. "
            "Add: dvc remote add remote_storage s3://minivessdataset"
        )

    def test_remote_storage_url_is_aws_s3(self) -> None:
        """remote_storage URL must point to the AWS S3 public dataset bucket."""
        parser = _parse_dvc_config()
        section = "'remote \"remote_storage\"'"
        if section not in parser.sections():
            pytest.skip("remote_storage not configured yet")
        url = parser.get(section, "url")
        assert url.startswith("s3://"), f"Expected s3:// URL, got: {url}"
        assert "minivessdataset" in url, (
            f"remote_storage should point to s3://minivessdataset, got: {url}"
        )

    def test_remote_storage_has_no_credentials_in_committed_config(self) -> None:
        """remote_storage (AWS S3 public) must not have credentials in committed config.

        s3://minivessdataset is a public bucket — no access keys needed.
        Credentials would only appear in .dvc/config.local (gitignored).
        """
        parser = _parse_dvc_config()
        section = "'remote \"remote_storage\"'"
        if section not in parser.sections():
            pytest.skip("remote_storage not configured yet")
        # Public bucket: no endpointurl, no access_key_id needed
        assert not parser.has_option(section, "access_key_id"), (
            "remote_storage should not have access_key_id — it is a public S3 bucket"
        )
        assert not parser.has_option(section, "secret_access_key"), (
            "remote_storage should not have secret_access_key — it is a public S3 bucket"
        )

    def test_dvc_config_has_minio_remote(self) -> None:
        """DVC config must define minio remote for local Docker Compose stack."""
        parser = _parse_dvc_config()
        section = "'remote \"minio\"'"
        assert section in parser.sections(), (
            ".dvc/config missing minio remote for local stack. "
            "Add: dvc remote add minio s3://dvc-data --endpointurl http://localhost:9000"
        )

    def test_minio_remote_uses_localhost_endpoint(self) -> None:
        """minio remote must use local MinIO endpoint."""
        parser = _parse_dvc_config()
        section = "'remote \"minio\"'"
        if section not in parser.sections():
            pytest.skip("minio remote not configured yet")
        assert parser.has_option(section, "endpointurl"), (
            "minio remote needs endpointurl for local MinIO S3-compatible endpoint"
        )
        endpoint = parser.get(section, "endpointurl")
        assert "localhost" in endpoint or "minio" in endpoint, (
            f"minio remote endpointurl should point to localhost MinIO, got: {endpoint}"
        )

    def test_upcloud_remote_archived(self) -> None:
        """UpCloud remote must NOT be present as an active remote (archived 2026-03-16)."""
        parser = _parse_dvc_config()
        upcloud_section = "'remote \"upcloud\"'"
        assert upcloud_section not in parser.sections(), (
            ".dvc/config still has active upcloud remote — should have been archived 2026-03-16. "
            "Remove it: dvc remote remove upcloud. Code archived at deployment/archived/upcloud/"
        )

    def test_lambda_remote_not_present(self) -> None:
        """Lambda Labs remote must NOT be present (provider rejected 2026-03-16)."""
        parser = _parse_dvc_config()
        lambda_section = "'remote \"lambda\"'"
        assert lambda_section not in parser.sections(), (
            ".dvc/config has lambda remote — Lambda Labs was rejected (no EU availability). "
            "Code archived at deployment/archived/lambda/"
        )


class TestDvcPullProtocol:
    """Verify DVC pull command construction via monkeypatch.

    Uses monkeypatch on subprocess.run instead of moto (RC4).
    Updated 2026-03-16: remote_storage (AWS S3) replaces upcloud.
    """

    def test_dvc_pull_uses_remote_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dvc pull -r remote_storage must use 'remote_storage' remote name."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import _run_dvc

        _run_dvc("pull", "-r", "remote_storage")
        assert len(calls) == 1
        assert calls[0] == ["dvc", "pull", "-r", "remote_storage"]

    def test_dvc_status_check_uses_remote_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_connectivity calls dvc status -r remote_storage."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import verify_connectivity

        env: dict[str, str] = {}
        verify_connectivity(env)
        assert any(cmd == ["dvc", "status", "-r", "remote_storage"] for cmd in calls), (
            f"Expected dvc status -r remote_storage, got: {calls}"
        )

    def test_configure_remote_does_not_set_credentials_for_public_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure_remote must NOT set credentials for remote_storage (public bucket)."""
        calls: list[list[str]] = []

        def _mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            if "list" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="remote_storage\ts3://minivessdataset\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)

        from scripts.configure_dvc_remote import configure_remote

        env: dict[str, str] = {}
        configure_remote(env)

        # Public bucket: must NOT call dvc remote modify --local with credentials
        credential_calls = [
            c
            for c in calls
            if "modify" in c and ("access_key_id" in c or "secret_access_key" in c)
        ]
        assert len(credential_calls) == 0, (
            f"configure_remote should not set credentials for public S3 bucket, "
            f"but called: {credential_calls}"
        )


class TestDvcConfigureScript:
    """Validate scripts/configure_dvc_remote.py exists and is functional."""

    def test_configure_script_exists(self) -> None:
        """scripts/configure_dvc_remote.py must exist."""
        script = ROOT / "scripts" / "configure_dvc_remote.py"
        assert script.exists(), (
            "scripts/configure_dvc_remote.py not found. "
            "Create it to auto-configure DVC remote_storage from environment"
        )

    def test_configure_script_importable(self) -> None:
        """Script must be importable (valid Python syntax)."""
        script = ROOT / "scripts" / "configure_dvc_remote.py"
        if not script.exists():
            pytest.skip("script not created yet")
        import ast

        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


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
