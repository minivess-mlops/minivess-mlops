"""ServiceAdapter ABC — base class for all dashboard data source adapters.

Each adapter connects to one upstream service (MLflow, Prefect, BentoML, etc.)
and provides a cached query interface. Adapters are independently testable
with mocked HTTP responses.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterStatus:
    """Health status of an adapter."""

    healthy: bool
    service_name: str
    message: str = ""
    last_checked: float = field(default_factory=time.time)


class ServiceAdapter(ABC):
    """Base class for all dashboard service adapters.

    Provides connection management, caching with TTL, and health status.

    Parameters
    ----------
    service_name:
        Human-readable service name (e.g. "MLflow", "Prefect").
    base_url:
        Base URL for the service.
    cache_ttl_s:
        Cache time-to-live in seconds.
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        *,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._healthy: bool | None = None

    @abstractmethod
    def _fetch(self, endpoint: str) -> dict[str, Any]:
        """Fetch data from the upstream service.

        Subclasses implement the actual HTTP/DB call.
        """

    def query(self, endpoint: str) -> dict[str, Any]:
        """Query the adapter with caching.

        Returns cached result if within TTL, otherwise fetches fresh data.
        On connection error, returns empty dict (never raises).
        """
        now = time.time()
        cached = self._cache.get(endpoint)
        if cached is not None:
            cached_time, cached_data = cached
            if now - cached_time < self.cache_ttl_s:
                return cached_data

        try:
            data = self._fetch(endpoint)
            self._cache[endpoint] = (now, data)
            self._healthy = True
            return data
        except Exception:
            logger.warning(
                "%s adapter: failed to fetch %s",
                self.service_name,
                endpoint,
                exc_info=True,
            )
            self._healthy = False
            return {}

    def status(self) -> AdapterStatus:
        """Return current health status."""
        return AdapterStatus(
            healthy=self._healthy is True,
            service_name=self.service_name,
            message="ok" if self._healthy else "unavailable",
        )
