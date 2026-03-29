"""Integration tests across monitoring modules: manifest + queue + anomaly detection.

End-to-end tests verifying the full monitoring pipeline:
1. Create manifest → 2. Parse queue → 3. Update statuses → 4. Detect anomalies.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 3 (P0): T3.8, T3.9
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path


def _make_job_data(
    job_id: int,
    condition: str,
    model: str,
    status: str = "PENDING",
    launched_at: str = "2026-03-28T09:00:00Z",
    ended_at: str | None = None,
    spot: bool = True,
    expected: float = 10.0,
    hourly_rate: float = 0.22,
) -> dict:
    return {
        "job_id": job_id,
        "condition_name": condition,
        "model": model,
        "loss": "dice_ce",
        "fold": 0,
        "spot": spot,
        "expected_duration_minutes": expected,
        "warn_duration_minutes": 30.0,
        "cancel_duration_minutes": 50.0,
        "hourly_rate": hourly_rate,
        "status": status,
        "launched_at": launched_at,
        "ended_at": ended_at,
    }


class TestMonitorIntegration:
    """End-to-end: manifest → queue update → anomaly detection."""

    def test_integration_manifest_queue_anomaly(self, tmp_path: Path) -> None:
        """Full pipeline: create manifest, detect anomaly on overrun job."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        # 1. Create manifest with 2 RUNNING jobs (one overrunning)
        manifest_data = {
            "experiment_name": "integration_test",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 10.0,
            "budget_cap_usd": 20.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T09:00:00Z",
            "jobs": [
                _make_job_data(
                    165, "dynunet-dice_ce-calibfalse-f0", "dynunet",
                    status="RUNNING", launched_at="2026-03-28T09:00:00Z",
                ),
                _make_job_data(
                    166, "sam3_hybrid-cbdice_cldice-f0", "sam3_hybrid",
                    status="RUNNING", launched_at="2026-03-28T09:00:00Z",
                    spot=False, expected=25.0, hourly_rate=0.74,
                ),
            ],
        }

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # 2. Run anomaly detection 35 min later (warn threshold = 30)
        fixed_now = datetime(2026, 3, 28, 9, 35, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        # Both jobs exceed warn threshold (35 > 30)
        warn_alerts = [a for a in alerts if a.alert_type == "DURATION_WARN"]
        assert len(warn_alerts) >= 2

    def test_integration_all_succeeded_fires_summary(self, tmp_path: Path) -> None:
        """All jobs SUCCEEDED produces ALL_TERMINAL summary."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        manifest_data = {
            "experiment_name": "integration_test",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 100.0,
            "budget_cap_usd": 200.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T09:00:00Z",
            "jobs": [
                _make_job_data(
                    i, f"job-{i}", "dynunet",
                    status="SUCCEEDED",
                    launched_at="2026-03-28T09:00:00Z",
                    ended_at="2026-03-28T09:12:00Z",
                )
                for i in range(3)
            ],
        }

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        summaries = [a for a in alerts if a.alert_type == "ALL_TERMINAL"]
        assert len(summaries) == 1
        assert summaries[0].action == "SUMMARY"
        assert "3 succeeded" in summaries[0].message


class TestManifestRoundtrip11thPass:
    """T3.9: manifest round-trip with realistic 11th pass data."""

    def test_11th_pass_manifest_roundtrip(self, tmp_path: Path) -> None:
        """Serialize/deserialize exact 11th pass Phase 1 jobs."""
        from minivess.compute.job_manifest import JobManifest

        manifest_data = {
            "experiment_name": "debug_factorial",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 8.0,
            "budget_cap_usd": 15.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T17:00:00Z",
            "jobs": [
                {
                    "job_id": 165,
                    "condition_name": "dynunet-dice_ce-calibfalse-f0",
                    "model": "dynunet",
                    "loss": "dice_ce",
                    "fold": 0,
                    "spot": True,
                    "expected_duration_minutes": 10.0,
                    "warn_duration_minutes": 30.0,
                    "cancel_duration_minutes": 50.0,
                    "hourly_rate": 0.22,
                    "status": "PENDING",
                    "launched_at": "2026-03-28T17:00:00Z",
                },
                {
                    "job_id": 166,
                    "condition_name": "sam3_hybrid-cbdice_cldice-calibfalse-f0",
                    "model": "sam3_hybrid",
                    "loss": "cbdice_cldice",
                    "fold": 0,
                    "spot": False,
                    "expected_duration_minutes": 25.0,
                    "warn_duration_minutes": 75.0,
                    "cancel_duration_minutes": 75.0,
                    "hourly_rate": 0.74,
                    "status": "PENDING",
                    "launched_at": "2026-03-28T17:01:00Z",
                },
                {
                    "job_id": 167,
                    "condition_name": "sam3_vanilla-zeroshot-minivess-f0",
                    "model": "sam3_vanilla",
                    "loss": "none",
                    "fold": 0,
                    "spot": True,
                    "expected_duration_minutes": 15.0,
                    "warn_duration_minutes": 45.0,
                    "cancel_duration_minutes": 75.0,
                    "hourly_rate": 0.22,
                    "status": "PENDING",
                    "launched_at": "2026-03-28T17:02:00Z",
                },
            ],
        }

        # Write and read back
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest_data), encoding="utf-8")
        original = JobManifest.from_json(path)

        out_path = tmp_path / "manifest_out.json"
        original.to_json(out_path)
        roundtripped = JobManifest.from_json(out_path)

        # Verify all fields match
        assert roundtripped.experiment_name == "debug_factorial"
        assert roundtripped.budget_warning_usd == 8.0
        assert roundtripped.budget_cap_usd == 15.0
        assert len(roundtripped.jobs) == 3

        # Verify specific job details
        dynunet = next(j for j in roundtripped.jobs if j.model == "dynunet")
        assert dynunet.spot is True
        assert dynunet.hourly_rate == 0.22
        assert dynunet.expected_duration_minutes == 10.0

        sam3h = next(j for j in roundtripped.jobs if j.model == "sam3_hybrid")
        assert sam3h.spot is False
        assert sam3h.hourly_rate == 0.74
