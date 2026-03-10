"""Aggregated health adapter — QA monitoring merged into dashboard (#342).

Queries all service adapters and applies threshold rules to generate
alerts. Replaces the separate QA Prefect flow.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minivess.dashboard.adapters.base import AdapterStatus, ServiceAdapter

logger = logging.getLogger(__name__)


@dataclass
class HealthAlert:
    """A single health alert."""

    severity: str  # "critical", "warning", "info"
    source: str  # adapter name
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealthSummary:
    """Aggregated health summary across all adapters."""

    overall: str  # "healthy", "degraded", "unhealthy"
    alerts: list[HealthAlert] = field(default_factory=list)
    adapter_statuses: list[AdapterStatus] = field(default_factory=list)
    last_checked: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "overall": self.overall,
            "alerts": [
                {
                    "severity": a.severity,
                    "source": a.source,
                    "message": a.message,
                }
                for a in self.alerts
            ],
            "adapter_statuses": [
                {
                    "service_name": s.service_name,
                    "healthy": s.healthy,
                    "message": s.message,
                }
                for s in self.adapter_statuses
            ],
            "last_checked": self.last_checked,
        }


class HealthAdapter:
    """Aggregated health check across all service adapters.

    Queries each adapter's status and applies threshold rules.
    Results are cached and refreshed on a configurable interval.
    """

    def __init__(self, adapters: list[ServiceAdapter]) -> None:
        self._adapters = adapters
        self._interval_s = float(
            os.environ.get("DASHBOARD_HEALTH_CHECK_INTERVAL_S", "300")
        )
        self._last_summary: HealthSummary | None = None

    @property
    def check_interval_s(self) -> float:
        """Health check interval in seconds."""
        return self._interval_s

    def check_all(self) -> HealthSummary:
        """Run health checks across all adapters.

        Returns a HealthSummary with overall status and any alerts.
        """
        statuses: list[AdapterStatus] = []
        alerts: list[HealthAlert] = []

        for adapter in self._adapters:
            status = adapter.status()
            statuses.append(status)
            if not status.healthy:
                alerts.append(
                    HealthAlert(
                        severity="warning",
                        source=status.service_name,
                        message=f"{status.service_name} is unavailable",
                    )
                )

        # Determine overall status
        n_unhealthy = sum(1 for s in statuses if not s.healthy)
        if n_unhealthy == 0:
            overall = "healthy"
        elif n_unhealthy < len(statuses):
            overall = "degraded"
        else:
            overall = "unhealthy"

        summary = HealthSummary(
            overall=overall,
            alerts=alerts,
            adapter_statuses=statuses,
        )
        self._last_summary = summary
        return summary

    @property
    def last_summary(self) -> HealthSummary | None:
        """Last cached health summary."""
        return self._last_summary
