"""Unit tests for MLflow backend health check (Task 5).

Tests the health.py module which provides HealthCheckResult dataclass
and check_mlflow_health() function for both filesystem and server backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """HealthCheckResult should have healthy, backend_type, message, uri."""
        from minivess.observability.health import HealthCheckResult

        result = HealthCheckResult(
            healthy=True,
            backend_type="server",
            message="OK",
            uri="http://localhost:5000",
        )
        assert result.healthy is True
        assert result.backend_type == "server"
        assert result.message == "OK"
        assert result.uri == "http://localhost:5000"

    def test_dataclass_is_frozen(self) -> None:
        """HealthCheckResult should be immutable."""
        from minivess.observability.health import HealthCheckResult

        result = HealthCheckResult(
            healthy=True,
            backend_type="filesystem",
            message="OK",
            uri="mlruns",
        )
        with pytest.raises(AttributeError):
            result.healthy = False  # type: ignore[misc]


class TestCheckMlflowHealth:
    """Test check_mlflow_health() for server and filesystem backends."""

    def test_filesystem_uri_creates_dir_if_missing(self, tmp_path: Path) -> None:
        """Filesystem URI should create directory and return healthy."""
        from minivess.observability.health import check_mlflow_health

        mlruns_dir = tmp_path / "mlruns"
        assert not mlruns_dir.exists()
        result = check_mlflow_health(str(mlruns_dir))
        assert result.healthy is True
        assert result.backend_type == "filesystem"
        assert mlruns_dir.exists()

    def test_filesystem_uri_existing_dir_is_healthy(self, tmp_path: Path) -> None:
        """Filesystem URI with existing directory should return healthy."""
        from minivess.observability.health import check_mlflow_health

        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()
        result = check_mlflow_health(str(mlruns_dir))
        assert result.healthy is True
        assert result.backend_type == "filesystem"

    def test_server_uri_mocked_200_is_healthy(self) -> None:
        """Server URI returning 200 should be healthy."""
        from minivess.observability.health import check_mlflow_health

        with patch("minivess.observability.health.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.status = 200
            result = check_mlflow_health("http://localhost:5000")

        assert result.healthy is True
        assert result.backend_type == "server"

    def test_server_uri_connection_error_is_unhealthy(self) -> None:
        """Server URI with ConnectionError should be unhealthy."""
        from urllib.error import URLError

        from minivess.observability.health import check_mlflow_health

        with patch(
            "minivess.observability.health.urlopen",
            side_effect=URLError("Connection refused"),
        ):
            result = check_mlflow_health("http://localhost:5000")

        assert result.healthy is False
        assert result.backend_type == "server"
        assert "Connection refused" in result.message

    def test_no_arg_uses_resolve_tracking_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No-argument call should use resolve_tracking_uri() to get URI."""
        from minivess.observability.health import check_mlflow_health

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        # With Dynaconf active, this resolves to http://localhost:5000
        # Mock the server check to avoid actual connection
        with patch("minivess.observability.health.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.status = 200
            result = check_mlflow_health()

        assert result.uri  # Should have resolved to something

    def test_preflight_includes_mlflow_health(self, tmp_path: Path) -> None:
        """run_preflight should include an mlflow_health check."""
        from minivess.pipeline.preflight import run_preflight

        data_dir = tmp_path / "data" / "imagesTr"
        data_dir.mkdir(parents=True)
        (data_dir / "test.nii.gz").write_bytes(b"fake")
        labels_dir = tmp_path / "data" / "labelsTr"
        labels_dir.mkdir()
        (labels_dir / "test.nii.gz").write_bytes(b"fake")

        result = run_preflight(tmp_path / "data")
        check_names = [c.name for c in result.checks]
        assert "mlflow_health" in check_names
