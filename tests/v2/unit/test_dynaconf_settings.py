"""Unit tests for Dynaconf settings singleton (Task 1).

Tests the settings.py module which provides a cached Dynaconf instance
that reads from configs/deployment/*.toml with environment-based overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestDynaconfSettingsSingleton:
    """Test get_settings() returns a properly configured Dynaconf instance."""

    def test_get_settings_returns_dynaconf_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_settings() should return a Dynaconf instance."""
        from dynaconf import Dynaconf

        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert isinstance(settings, Dynaconf)

    def test_project_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should contain PROJECT_NAME from settings.toml [default]."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.PROJECT_NAME == "minivess-mlops-v2"

    def test_resolve_tracking_uri_reads_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_tracking_uri() must return the MLFLOW_TRACKING_URI env var.

        Per Rule #22, MLFLOW_TRACKING_URI is NOT a Dynaconf setting — it is read
        directly from the environment via resolve_tracking_uri() (Issue #541).
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-unit-test:5000")
        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri()
        assert isinstance(uri, str)
        assert len(uri) > 0
        assert uri == "http://mlflow-unit-test:5000"

    def test_debug_exists_and_is_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should contain DEBUG as a boolean."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert isinstance(settings.DEBUG, bool)

    def test_singleton_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two calls to get_settings() should return the exact same object."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_staging_debug_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In staging environment, DEBUG should be False."""
        monkeypatch.setenv("ENV_FOR_DYNACONF", "staging")
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()
        assert settings.DEBUG is False

    def test_missing_secrets_toml_does_not_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing .secrets.toml should not raise an error."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        # If this raises, the test fails
        settings = get_settings()
        assert settings.PROJECT_NAME is not None

    def test_clear_settings_cache_resets_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """clear_settings_cache() should reset so next call creates new object."""
        monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)
        from minivess.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2
