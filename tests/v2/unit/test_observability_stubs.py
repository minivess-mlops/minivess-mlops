"""Unit tests for Sentry/PostHog observability stubs (PR-3, T3.1).

Tests that:
- init_sentry() initializes Sentry SDK when SENTRY_DSN is set
- init_sentry() is a no-op when SENTRY_DSN is empty/unset
- init_posthog() initializes PostHog when POSTHOG_KEY is set
- init_posthog() is a no-op when POSTHOG_KEY is empty/unset
- Neither function raises ImportError when sentry-sdk/posthog are not installed
- init_monitoring() is a convenience wrapper that calls both

Closes: #841
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# T3.1a: Sentry stub tests
# ---------------------------------------------------------------------------


class TestInitSentry:
    """Sentry SDK initialization stub."""

    def test_noop_when_dsn_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_sentry() returns False when SENTRY_DSN is empty."""
        monkeypatch.setenv("SENTRY_DSN", "")
        from minivess.observability.monitoring import init_sentry

        result = init_sentry()
        assert result is False

    def test_noop_when_dsn_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_sentry() returns False when SENTRY_DSN is not in env."""
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        from minivess.observability.monitoring import init_sentry

        result = init_sentry()
        assert result is False

    def test_initializes_when_dsn_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_sentry() returns True when SENTRY_DSN has a value."""
        monkeypatch.setenv(
            "SENTRY_DSN", "https://examplePublicKey@o0.ingest.sentry.io/0"
        )
        mock_sentry = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            from minivess.observability.monitoring import init_sentry

            result = init_sentry()
            assert result is True
            mock_sentry.init.assert_called_once()

    def test_no_import_error_when_package_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_sentry() gracefully handles missing sentry-sdk package."""
        monkeypatch.setenv(
            "SENTRY_DSN", "https://examplePublicKey@o0.ingest.sentry.io/0"
        )
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            from minivess.observability.monitoring import init_sentry

            result = init_sentry()
            assert result is False


# ---------------------------------------------------------------------------
# T3.1b: PostHog stub tests
# ---------------------------------------------------------------------------


class TestInitPostHog:
    """PostHog analytics initialization stub."""

    def test_noop_when_key_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_posthog() returns False when POSTHOG_KEY is empty."""
        monkeypatch.setenv("POSTHOG_KEY", "")
        from minivess.observability.monitoring import init_posthog

        result = init_posthog()
        assert result is False

    def test_noop_when_key_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_posthog() returns False when POSTHOG_KEY is not in env."""
        monkeypatch.delenv("POSTHOG_KEY", raising=False)
        from minivess.observability.monitoring import init_posthog

        result = init_posthog()
        assert result is False

    def test_initializes_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_posthog() returns True when POSTHOG_KEY has a value."""
        monkeypatch.setenv("POSTHOG_KEY", "phc_testkey123")
        mock_posthog = MagicMock()
        with patch.dict("sys.modules", {"posthog": mock_posthog}):
            from minivess.observability.monitoring import init_posthog

            result = init_posthog()
            assert result is True

    def test_no_import_error_when_package_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_posthog() gracefully handles missing posthog package."""
        monkeypatch.setenv("POSTHOG_KEY", "phc_testkey123")
        with patch.dict("sys.modules", {"posthog": None}):
            from minivess.observability.monitoring import init_posthog

            result = init_posthog()
            assert result is False


# ---------------------------------------------------------------------------
# T3.1c: Convenience wrapper test
# ---------------------------------------------------------------------------


class TestInitMonitoring:
    """init_monitoring() convenience function."""

    def test_calls_both_stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """init_monitoring() returns a dict with both results."""
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        monkeypatch.delenv("POSTHOG_KEY", raising=False)
        from minivess.observability.monitoring import init_monitoring

        result = init_monitoring()
        assert isinstance(result, dict)
        assert "sentry" in result
        assert "posthog" in result
        assert result["sentry"] is False
        assert result["posthog"] is False
