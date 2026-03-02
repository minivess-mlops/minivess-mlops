"""Inference client protocol and implementations.

Defines a Protocol for inference clients and provides:
- ``LocalInferenceClient`` — in-process via ModelRegistryServer (for CI/tests)
- ``RemoteInferenceClient`` — HTTP to BentoML server (for production)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from minivess.serving.api_models import SegmentationRequest, SegmentationResponse
    from minivess.serving.model_registry_server import ModelRegistryServer

logger = logging.getLogger(__name__)


@runtime_checkable
class InferenceClient(Protocol):
    """Protocol for inference clients."""

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Run segmentation inference."""
        ...

    def health(self) -> dict[str, Any]:
        """Return server health status."""
        ...


class LocalInferenceClient:
    """In-process inference client using ModelRegistryServer directly.

    Useful for CI, testing, and single-machine deployments.
    """

    def __init__(self, server: ModelRegistryServer) -> None:
        self._server = server

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Delegate to the local ModelRegistryServer."""
        return self._server.predict(request)

    def health(self) -> dict[str, Any]:
        """Delegate health check to server."""
        return self._server.health()


class RemoteInferenceClient:
    """HTTP-based inference client for production BentoML deployments.

    Parameters
    ----------
    server_url:
        Base URL of the BentoML server (e.g. ``http://localhost:3000``).
    timeout_s:
        Request timeout in seconds.
    """

    def __init__(self, server_url: str, *, timeout_s: float = 120.0) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout_s = timeout_s

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Send inference request to remote BentoML server.

        Note: This is a stub implementation. Full HTTP transport
        will be wired when BentoML multi-model service is deployed.
        """
        msg = (
            "RemoteInferenceClient.predict() is a stub. "
            "Wire HTTP transport when BentoML multi-model service is deployed."
        )
        raise NotImplementedError(msg)

    def health(self) -> dict[str, Any]:
        """Check remote server health via HTTP."""
        import urllib.request

        try:
            url = f"{self._server_url}/health"
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                import json

                result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return result
        except Exception:
            logger.warning("Remote health check failed", exc_info=True)
            return {"status": "unreachable", "url": self._server_url}
