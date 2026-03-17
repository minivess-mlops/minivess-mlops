"""Tests for serving endpoint health check in the deploy flow.

PR-D T3 (Issue #827): Health check task that verifies BentoML serving
endpoint is responsive after deployment.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestHealthCheckEndpoint:
    """Verify health check detects server state."""

    def test_health_check_endpoint_success(self) -> None:
        """Health check returns True when server responds 200."""
        from minivess.serving.deploy_health_check import check_health_endpoint

        # Mock a successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch(
            "minivess.serving.deploy_health_check._http_get",
            return_value=mock_response,
        ):
            result = check_health_endpoint("http://localhost:3000")
            assert result.healthy is True
            assert result.status_code == 200

    def test_health_check_endpoint_down(self) -> None:
        """Health check returns False when server is unreachable."""
        from minivess.serving.deploy_health_check import check_health_endpoint

        with patch(
            "minivess.serving.deploy_health_check._http_get",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = check_health_endpoint("http://localhost:3000")
            assert result.healthy is False
            assert result.error is not None


class TestHealthCheckInference:
    """Verify health check can run inference."""

    def test_health_check_inference(self) -> None:
        """Inference health check sends volume and gets response."""
        from minivess.serving.deploy_health_check import (
            check_inference_endpoint,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_shape": [1, 2, 8, 8, 4],
            "inference_time_ms": 42.0,
        }

        with patch(
            "minivess.serving.deploy_health_check._http_post",
            return_value=mock_response,
        ):
            result = check_inference_endpoint(
                "http://localhost:3000",
                input_shape=(1, 1, 8, 8, 4),
            )
            assert result.inference_ok is True
            assert result.output_shape == [1, 2, 8, 8, 4]


class TestHealthCheckResponseShape:
    """Verify response shape validation."""

    def test_health_check_response_shape(self) -> None:
        """Response shape must be 5D (B, C, D, H, W)."""
        from minivess.serving.deploy_health_check import validate_response_shape

        assert validate_response_shape([1, 2, 8, 8, 4]) is True
        assert validate_response_shape([1, 2, 32, 32, 16]) is True
        # Wrong number of dims
        assert validate_response_shape([1, 2, 8]) is False
        assert validate_response_shape([]) is False

    def test_health_check_response_batch_dim(self) -> None:
        """First dimension (batch) must be >= 1."""
        from minivess.serving.deploy_health_check import validate_response_shape

        assert validate_response_shape([0, 2, 8, 8, 4]) is False
        assert validate_response_shape([1, 2, 8, 8, 4]) is True


class TestHealthCheckMLflowLogging:
    """Verify health check results are logged to MLflow."""

    def test_health_check_mlflow_logging(self) -> None:
        """Health check result is structured for MLflow logging."""
        from minivess.serving.deploy_health_check import (
            HealthCheckResult,
            build_health_check_params,
        )

        result = HealthCheckResult(
            healthy=True,
            status_code=200,
            inference_ok=True,
            output_shape=[1, 2, 8, 8, 4],
        )
        params = build_health_check_params(result)

        assert "deploy/health_check_passed" in params
        assert params["deploy/health_check_passed"] == "True"
        assert "deploy/inference_check_passed" in params
        assert params["deploy/inference_check_passed"] == "True"
        assert "deploy/output_shape" in params
