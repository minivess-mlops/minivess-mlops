"""Unit tests for Dynaconf environment tiers (Task 2).

Tests that all environments (default, development, staging, production)
have correct settings. Per Rule #22, MLFLOW_TRACKING_URI is NOT a Dynaconf
setting — it is consumed directly from the env var via resolve_tracking_uri().
Tests that previously checked settings.MLFLOW_TRACKING_URI now verify
resolve_tracking_uri() behaviour instead (Issue #541).
"""

from __future__ import annotations

import pytest


class TestDynaconfEnvironmentTiers:
    """Test environment-specific overrides in Dynaconf TOML files."""

    def test_resolve_tracking_uri_returns_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_tracking_uri() must return MLFLOW_TRACKING_URI env var value.

        Per Rule #22, MLflow URI is NOT in Dynaconf — it comes from the env var.
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-test:5000")
        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri()
        assert uri.startswith("http://")
        assert uri == "http://mlflow-test:5000"

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_resolve_tracking_uri_independent_of_dynaconf_env(
        self, monkeypatch: pytest.MonkeyPatch, env: str
    ) -> None:
        """resolve_tracking_uri() must return env var regardless of Dynaconf env tier.

        MLflow URI is global (Rule #22) — different Dynaconf environment tiers
        must NOT override it. The URI comes from MLFLOW_TRACKING_URI only.
        """
        monkeypatch.setenv("ENV_FOR_DYNACONF", env)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-shared:5000")
        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri()
        assert isinstance(uri, str)
        assert len(uri) > 0
        assert uri == "http://mlflow-shared:5000"

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
