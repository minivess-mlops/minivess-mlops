"""Tests for MLflow backend standardization (#274).

Covers:
- resolve_tracking_uri() preference order
- Backend type detection (local vs server)
- Warning on local filesystem backend in production
- Migration helper detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestResolveTrackingUri:
    """Test resolve_tracking_uri() priority order."""

    def test_explicit_uri_takes_priority(self) -> None:
        from minivess.observability.tracking import resolve_tracking_uri

        result = resolve_tracking_uri(
            tracking_uri="http://mlflow:5000", use_dynaconf=False
        )
        assert result == "http://mlflow:5000"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow:5000")
        result = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert result == "http://env-mlflow:5000"

    def test_default_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        result = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert result == "mlruns"

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow:5000")
        result = resolve_tracking_uri(
            tracking_uri="http://explicit:5000", use_dynaconf=False
        )
        assert result == "http://explicit:5000"


class TestTrackingUriCredentialEncoding:
    """resolve_tracking_uri() must percent-encode credentials (#628).

    Passwords with @, :, /, %, + produce malformed URIs if not encoded.
    """

    def test_password_with_at_sign(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Password containing '@' must be percent-encoded."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "p@ssword")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        # @ in password must be encoded as %40
        assert "%40" in uri
        # The URI must still be parseable and point to the right host
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        assert parsed.hostname == "mlflow"
        assert parsed.port == 5000

    def test_password_with_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Password containing ':' must be percent-encoded."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass:word")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "%3A" in uri

    def test_password_with_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Password containing '/' must be percent-encoded."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass/word")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "%2F" in uri

    def test_password_with_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Password containing '%' must be percent-encoded."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "100%safe")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "%25" in uri

    def test_username_with_special_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Username with special chars must also be encoded."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user@org")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "plain")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "%40" in uri

    def test_encoded_uri_roundtrips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """URI with encoded credentials must parse back correctly."""
        from urllib.parse import unquote, urlparse

        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "p@ss:w/rd%2B")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        parsed = urlparse(uri)
        assert parsed.hostname == "mlflow"
        assert parsed.port == 5000
        assert unquote(parsed.username or "") == "admin"
        assert unquote(parsed.password or "") == "p@ss:w/rd%2B"

    def test_no_port_uri_with_special_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """URI without explicit port must also encode credentials."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "p@ss")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "%40" in uri
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        assert parsed.hostname == "mlflow.example.com"


class TestBackendTypeDetection:
    """Test detection of backend type (local vs server)."""

    def test_detect_local_filesystem_backend(self) -> None:
        from minivess.observability.mlflow_backend import detect_backend_type

        assert detect_backend_type("mlruns") == "local"
        assert detect_backend_type("./mlruns") == "local"
        assert detect_backend_type("/tmp/mlruns") == "local"

    def test_detect_server_backend(self) -> None:
        from minivess.observability.mlflow_backend import detect_backend_type

        assert detect_backend_type("http://mlflow:5000") == "server"
        assert detect_backend_type("https://mlflow.example.com") == "server"

    def test_detect_postgres_backend(self) -> None:
        from minivess.observability.mlflow_backend import detect_backend_type

        assert (
            detect_backend_type("postgresql://user:pass@localhost:5432/mlflow")
            == "database"
        )

    def test_detect_sqlite_backend(self) -> None:
        from minivess.observability.mlflow_backend import detect_backend_type

        assert detect_backend_type("sqlite:///mlflow.db") == "database"

    def test_detect_postgres_scheme_without_ql(self) -> None:
        """postgres:// (without 'ql' suffix) must be detected as database (#629).

        UpCloud managed PostgreSQL returns URIs starting with postgres://
        which is valid per PostgreSQL docs but was not detected.
        """
        from minivess.observability.mlflow_backend import detect_backend_type

        assert (
            detect_backend_type("postgres://user:pass@host:5432/mlflow") == "database"
        )

    def test_detect_postgres_with_options(self) -> None:
        """postgres:// with query params must also be detected."""
        from minivess.observability.mlflow_backend import detect_backend_type

        assert (
            detect_backend_type("postgres://user:pass@host:5432/mlflow?sslmode=require")
            == "database"
        )


class TestBackendWarning:
    """Test that local backends emit warnings in non-dev environments."""

    def test_warn_local_backend_in_staging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from minivess.observability.mlflow_backend import warn_if_local_backend

        with caplog.at_level(logging.WARNING):
            warn_if_local_backend("mlruns", environment="staging")
        assert "local filesystem" in caplog.text.lower()

    def test_no_warn_local_backend_in_dev(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from minivess.observability.mlflow_backend import warn_if_local_backend

        with caplog.at_level(logging.WARNING):
            warn_if_local_backend("mlruns", environment="dev")
        assert "local filesystem" not in caplog.text.lower()

    def test_no_warn_server_backend(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from minivess.observability.mlflow_backend import warn_if_local_backend

        with caplog.at_level(logging.WARNING):
            warn_if_local_backend("http://mlflow:5000", environment="staging")
        assert "local filesystem" not in caplog.text.lower()


class TestMigrationDetection:
    """Test detection of local mlruns/ for migration guidance."""

    def test_detect_local_mlruns_exists(self, tmp_path: Path) -> None:
        from minivess.observability.mlflow_backend import check_local_mlruns

        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        (mlruns / "0").mkdir()  # default experiment

        result = check_local_mlruns(tmp_path)
        assert result["has_local_mlruns"] is True
        assert result["n_experiments"] >= 1

    def test_detect_no_local_mlruns(self, tmp_path: Path) -> None:
        from minivess.observability.mlflow_backend import check_local_mlruns

        result = check_local_mlruns(tmp_path)
        assert result["has_local_mlruns"] is False
