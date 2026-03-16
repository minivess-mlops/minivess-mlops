"""Tests for Prometheus Alertmanager integration.

TDD RED phase for Task T-B3 (Issue #764).
Three-layer alerting: Alertmanager + Grafana + MLflow audit trail.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class TestDriftAlert:
    """Test the DriftAlert dataclass."""

    def test_drift_alert_fields(self) -> None:
        from minivess.observability.alerting import DriftAlert

        alert = DriftAlert(
            alert_name="drift_detected",
            severity="warning",
            drift_score=0.33,
            n_drifted_features=3,
            batch_id="drift-batch-2",
            timestamp=datetime(2026, 3, 16, 12, 0, 0),
        )
        assert alert.alert_name == "drift_detected"
        assert alert.severity == "warning"
        assert alert.drift_score == 0.33

    def test_drift_alert_is_dataclass(self) -> None:
        import dataclasses

        from minivess.observability.alerting import DriftAlert

        assert dataclasses.is_dataclass(DriftAlert)

    def test_drift_alert_to_dict(self) -> None:
        from minivess.observability.alerting import DriftAlert

        alert = DriftAlert(
            alert_name="performance_drop",
            severity="critical",
            drift_score=0.8,
            n_drifted_features=7,
            batch_id="drift-batch-5",
            timestamp=datetime(2026, 3, 16, 14, 0, 0),
        )
        d = alert.to_dict()
        assert isinstance(d, dict)
        assert d["alert_name"] == "performance_drop"
        assert d["severity"] == "critical"


class TestAlertManager:
    """Test the alert dispatch manager."""

    def test_alert_manager_instantiation(self) -> None:
        from minivess.observability.alerting import AlertManager

        mgr = AlertManager()
        assert mgr is not None

    def test_alert_manager_log_jsonl(self, tmp_path: Path) -> None:
        from minivess.observability.alerting import AlertManager, DriftAlert

        log_path = tmp_path / "alerts.jsonl"
        mgr = AlertManager(jsonl_path=log_path)
        alert = DriftAlert(
            alert_name="test_alert",
            severity="info",
            drift_score=0.1,
            n_drifted_features=1,
            batch_id="batch-1",
            timestamp=datetime(2026, 3, 16, 12, 0, 0),
        )
        mgr.fire(alert)
        assert log_path.exists()
        line = log_path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["alert_name"] == "test_alert"

    def test_alert_manager_multiple_alerts(self, tmp_path: Path) -> None:
        from minivess.observability.alerting import AlertManager, DriftAlert

        log_path = tmp_path / "alerts.jsonl"
        mgr = AlertManager(jsonl_path=log_path)
        for i in range(3):
            mgr.fire(
                DriftAlert(
                    alert_name=f"alert_{i}",
                    severity="warning",
                    drift_score=0.1 * (i + 1),
                    n_drifted_features=i + 1,
                    batch_id=f"batch-{i}",
                    timestamp=datetime(2026, 3, 16, 12, i, 0),
                )
            )
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_alert_manager_webhook_url_configurable(self) -> None:
        from minivess.observability.alerting import AlertManager

        mgr = AlertManager(webhook_url="http://localhost:9093/api/v1/alerts")
        assert mgr.webhook_url == "http://localhost:9093/api/v1/alerts"


class TestAlertRules:
    """Test alert rule evaluation."""

    def test_evaluate_drift_alert_fires_on_threshold(self) -> None:
        from minivess.observability.alerting import evaluate_drift_alert

        result = evaluate_drift_alert(
            drift_score=0.5,
            n_drifted_features=4,
            drift_threshold=0.3,
            batch_id="batch-3",
        )
        assert result is not None
        assert result.severity in ("warning", "critical")

    def test_evaluate_drift_alert_no_fire_below_threshold(self) -> None:
        from minivess.observability.alerting import evaluate_drift_alert

        result = evaluate_drift_alert(
            drift_score=0.1,
            n_drifted_features=1,
            drift_threshold=0.3,
            batch_id="batch-1",
        )
        assert result is None

    def test_evaluate_performance_drop_alert(self) -> None:
        from minivess.observability.alerting import evaluate_performance_alert

        result = evaluate_performance_alert(
            current_dice=0.65,
            baseline_dice=0.85,
            drop_threshold=0.10,
            batch_id="batch-4",
        )
        assert result is not None
        assert result.alert_name == "performance_drop"

    def test_no_performance_alert_when_stable(self) -> None:
        from minivess.observability.alerting import evaluate_performance_alert

        result = evaluate_performance_alert(
            current_dice=0.84,
            baseline_dice=0.85,
            drop_threshold=0.10,
            batch_id="batch-1",
        )
        assert result is None
