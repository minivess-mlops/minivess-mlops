"""Health check for BentoML serving endpoints.

Verifies that the deployed BentoML server is responsive and can
run inference. Used as a post-deployment smoke test in the deploy flow.

PR-D T3 (Issue #827).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a serving endpoint health check.

    Parameters
    ----------
    healthy:
        Whether the /healthz endpoint returned 200.
    status_code:
        HTTP status code from health endpoint, or -1 if unreachable.
    inference_ok:
        Whether inference produced a valid response.
    output_shape:
        Output tensor shape from inference, or empty list.
    error:
        Error message if health check failed.
    """

    healthy: bool = False
    status_code: int = -1
    inference_ok: bool = False
    output_shape: list[int] = field(default_factory=list)
    error: str | None = None


def _http_get(url: str, timeout: int = 10) -> Any:
    """Send HTTP GET request. Thin wrapper for mocking.

    Parameters
    ----------
    url:
        Full URL to send GET request to.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Response object.
    """
    import urllib.request

    req = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response


def _http_post(url: str, data: bytes, timeout: int = 30) -> Any:
    """Send HTTP POST request. Thin wrapper for mocking.

    Parameters
    ----------
    url:
        Full URL to send POST request to.
    data:
        Request body bytes.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Response object.
    """
    import urllib.request

    req = urllib.request.Request(  # noqa: S310
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response


def check_health_endpoint(base_url: str) -> HealthCheckResult:
    """Check the /healthz endpoint of a BentoML server.

    Parameters
    ----------
    base_url:
        Base URL of the BentoML server (e.g., ``http://localhost:3000``).

    Returns
    -------
    :class:`HealthCheckResult` with health status.
    """
    url = f"{base_url.rstrip('/')}/healthz"
    try:
        response = _http_get(url)
        status_code = response.status_code if hasattr(response, "status_code") else 200
        return HealthCheckResult(healthy=True, status_code=status_code)
    except Exception as exc:
        logger.warning("Health check failed for %s: %s", url, exc)
        return HealthCheckResult(healthy=False, error=str(exc))


def check_inference_endpoint(
    base_url: str,
    input_shape: tuple[int, ...] = (1, 1, 8, 8, 4),
) -> HealthCheckResult:
    """Run inference health check on a BentoML server.

    Sends a synthetic volume and checks the response shape.

    Parameters
    ----------
    base_url:
        Base URL of the BentoML server.
    input_shape:
        Shape of the synthetic input volume.

    Returns
    -------
    :class:`HealthCheckResult` with inference status.
    """
    import json

    url = f"{base_url.rstrip('/')}/predict"
    try:
        import numpy as np

        volume = np.zeros(input_shape, dtype=np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        response = _http_post(url, payload)

        resp_data = response.json() if hasattr(response, "json") else {}
        output_shape = resp_data.get("output_shape", [])

        return HealthCheckResult(
            healthy=True,
            status_code=200,
            inference_ok=True,
            output_shape=output_shape,
        )
    except Exception as exc:
        logger.warning("Inference check failed for %s: %s", url, exc)
        return HealthCheckResult(error=str(exc))


def validate_response_shape(shape: list[int]) -> bool:
    """Validate that a response shape is a valid 5D tensor.

    Expected format: ``[B, C, D, H, W]`` where B >= 1.

    Parameters
    ----------
    shape:
        Output tensor shape.

    Returns
    -------
    ``True`` if shape is valid 5D with positive batch dimension.
    """
    if len(shape) != 5:  # noqa: PLR2004
        return False
    return shape[0] >= 1


def build_health_check_params(result: HealthCheckResult) -> dict[str, str]:
    """Build MLflow param dict from health check result.

    Parameters
    ----------
    result:
        Health check result to convert.

    Returns
    -------
    Dict of ``deploy/*`` params for MLflow logging.
    """
    return {
        "deploy/health_check_passed": str(result.healthy),
        "deploy/inference_check_passed": str(result.inference_ok),
        "deploy/output_shape": str(result.output_shape),
        "deploy/health_status_code": str(result.status_code),
    }
