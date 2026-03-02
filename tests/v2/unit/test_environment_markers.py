"""Unit tests for environment-aware pytest markers (Task 6).

Tests the requires_mlflow_server marker, environment utilities,
and auto-skip behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestRequiresMlflowServerMarker:
    """Test that requires_mlflow_server marker is registered."""

    def test_marker_is_registered(self, pytestconfig: pytest.Config) -> None:
        """requires_mlflow_server marker should not trigger strict-markers warning."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        assert "requires_mlflow_server" in marker_names


class TestEnvironmentUtilities:
    """Test environment detection utilities."""

    def test_is_mlflow_server_backend_http(self) -> None:
        """http:// URIs should return True."""
        from minivess.config.environment import is_mlflow_server_backend

        assert is_mlflow_server_backend("http://localhost:5000") is True

    def test_is_mlflow_server_backend_https(self) -> None:
        """https:// URIs should return True."""
        from minivess.config.environment import is_mlflow_server_backend

        assert is_mlflow_server_backend("https://mlflow.example.com") is True

    def test_is_mlflow_server_backend_filesystem(self) -> None:
        """Filesystem URIs should return False."""
        from minivess.config.environment import is_mlflow_server_backend

        assert is_mlflow_server_backend("mlruns") is False

    def test_get_active_environment_returns_string(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_active_environment() should return a string."""
        from minivess.config.settings import clear_settings_cache

        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        clear_settings_cache()

        from minivess.config.environment import get_active_environment

        env = get_active_environment()
        assert isinstance(env, str)
        assert len(env) > 0
