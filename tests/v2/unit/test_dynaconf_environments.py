"""Unit tests for Dynaconf environment tiers (Task 2).

Tests that all environments (default, development, staging, production)
have correct settings, especially mlflow_tracking_uri and debug flags.
"""

from __future__ import annotations

import pytest


class TestDynaconfEnvironmentTiers:
    """Test environment-specific overrides in Dynaconf TOML files."""

    def test_default_env_mlflow_uri_is_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default env MLFLOW_TRACKING_URI should start with http://."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.MLFLOW_TRACKING_URI.startswith("http://")

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_all_envs_have_nonempty_mlflow_uri(
        self, monkeypatch: pytest.MonkeyPatch, env: str
    ) -> None:
        """All environments should have a non-empty MLFLOW_TRACKING_URI."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", env)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert isinstance(settings.MLFLOW_TRACKING_URI, str)
        assert len(settings.MLFLOW_TRACKING_URI) > 0

    def test_staging_debug_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Staging environment DEBUG should be False."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", "staging")
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.DEBUG is False

    def test_development_debug_is_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Development environment DEBUG should be True."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", "development")
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.DEBUG is True

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_all_envs_share_same_server_uri(
        self, monkeypatch: pytest.MonkeyPatch, env: str
    ) -> None:
        """All environments should use the same Docker server URI."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", env)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.MLFLOW_TRACKING_URI == "http://localhost:5000"

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_all_envs_have_test_markers(
        self, monkeypatch: pytest.MonkeyPatch, env: str
    ) -> None:
        """All environments should have non-empty test_markers setting."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", env)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert isinstance(settings.TEST_MARKERS, str)
        assert len(settings.TEST_MARKERS) > 0
