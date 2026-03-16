"""Prometheus Alertmanager integration for drift detection alerts.

Three-layer alerting (Level 4 MLOps mandate):
1. Prometheus Alertmanager → webhook + JSONL log (production alerts)
2. Grafana built-in alerts → visual dashboard alerts
3. MLflow run tags → audit trail for compliance

Alert rules evaluate drift scores and performance metrics against
configurable thresholds. When thresholds are breached, alerts fire
through all configured channels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DriftAlert:
    """A single drift or performance alert.

    Immutable record of a triggered alert, persisted to JSONL
    and optionally forwarded to Alertmanager webhook.
    """

    alert_name: str
    severity: str  # "info", "warning", "critical"
    drift_score: float
    n_drifted_features: int
    batch_id: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/JSONL output."""
        return {
            "alert_name": self.alert_name,
            "severity": self.severity,
            "drift_score": self.drift_score,
            "n_drifted_features": self.n_drifted_features,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AlertManager:
    """Dispatch alerts to JSONL log and optional webhook.

    Handles the first alert layer (Prometheus Alertmanager integration).
    Grafana alerts (layer 2) are configured in dashboard JSON.
    MLflow audit trail (layer 3) is handled by the flow that calls this.
    """

    def __init__(
        self,
        jsonl_path: Path | None = None,
        webhook_url: str | None = None,
    ) -> None:
        self._jsonl_path = jsonl_path
        self._webhook_url = webhook_url

    @property
    def webhook_url(self) -> str | None:
        """The configured webhook URL for Alertmanager."""
        return self._webhook_url

    def fire(self, alert: DriftAlert) -> None:
        """Dispatch an alert to all configured channels.

        1. Append to JSONL log file (always, if path configured)
        2. POST to Alertmanager webhook (if URL configured)
        """
        self._log_jsonl(alert)
        if self._webhook_url:
            self._post_webhook(alert)

    def _log_jsonl(self, alert: DriftAlert) -> None:
        """Append alert to JSONL log file."""
        if self._jsonl_path is None:
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict(), default=str) + "\n")

    def _post_webhook(self, alert: DriftAlert) -> None:
        """POST alert to Alertmanager webhook."""
        if self._webhook_url is None:
            return

        import contextlib
        import urllib.request

        payload = json.dumps(
            [
                {
                    "labels": {
                        "alertname": alert.alert_name,
                        "severity": alert.severity,
                        "batch_id": alert.batch_id,
                    },
                    "annotations": {
                        "drift_score": str(alert.drift_score),
                        "n_drifted_features": str(alert.n_drifted_features),
                    },
                }
            ]
        ).encode("utf-8")

        req = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with contextlib.suppress(Exception):
            urllib.request.urlopen(req, timeout=5)  # noqa: S310


def evaluate_drift_alert(
    drift_score: float,
    n_drifted_features: int,
    drift_threshold: float,
    batch_id: str,
) -> DriftAlert | None:
    """Evaluate whether a drift score warrants an alert.

    Returns:
        DriftAlert if threshold is exceeded, None otherwise.
    """
    if drift_score < drift_threshold:
        return None

    severity = "critical" if drift_score > 0.7 else "warning"
    return DriftAlert(
        alert_name="drift_detected",
        severity=severity,
        drift_score=drift_score,
        n_drifted_features=n_drifted_features,
        batch_id=batch_id,
        timestamp=datetime.now(UTC),
    )


def evaluate_performance_alert(
    current_dice: float,
    baseline_dice: float,
    drop_threshold: float,
    batch_id: str,
) -> DriftAlert | None:
    """Evaluate whether a performance drop warrants an alert.

    Returns:
        DriftAlert if performance dropped more than drop_threshold.
    """
    drop = baseline_dice - current_dice
    if drop < drop_threshold:
        return None

    severity = "critical" if drop > 0.2 else "warning"
    return DriftAlert(
        alert_name="performance_drop",
        severity=severity,
        drift_score=drop,
        n_drifted_features=0,
        batch_id=batch_id,
        timestamp=datetime.now(UTC),
        metadata={"current_dice": current_dice, "baseline_dice": baseline_dice},
    )
