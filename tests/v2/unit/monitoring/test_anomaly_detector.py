"""Tests for the anomaly detector — the CORE of the factorial monitor.

RED phase: these tests validate EVERY failure mode from passes 4, 10, and 11.
The anomaly detector takes a JobManifest and returns Alert objects.

Plan: experiment-harness-improvement-plan.xml Task T1.3
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path


def _make_manifest_data(
    jobs: list[dict],
    budget_warning: float = 10.0,
    budget_cap: float = 15.0,
    pending_timeout: float = 30.0,
    kill_switch_threshold: int = 3,
    kill_switch_window: float = 5.0,
    batch_failure_pct: float = 0.5,
) -> dict:
    """Helper to create a manifest dict with defaults."""
    return {
        "experiment_name": "test",
        "config_file": "configs/factorial/debug.yaml",
        "budget_warning_usd": budget_warning,
        "budget_cap_usd": budget_cap,
        "pending_timeout_minutes": pending_timeout,
        "kill_switch_threshold": kill_switch_threshold,
        "kill_switch_window_minutes": kill_switch_window,
        "batch_failure_pct": batch_failure_pct,
        "launched_at": "2026-03-28T09:00:00Z",
        "jobs": jobs,
    }


def _make_job(
    job_id: int = 1,
    status: str = "RUNNING",
    model: str = "dynunet",
    launched_at: str = "2026-03-28T09:00:00Z",
    ended_at: str | None = None,
    expected: float = 10.0,
    warn: float = 30.0,
    cancel: float = 50.0,
    hourly_rate: float = 0.22,
    spot: bool = True,
    failure_category: str | None = None,
) -> dict:
    """Helper to create a job dict."""
    return {
        "job_id": job_id,
        "condition_name": f"{model}-test-f0",
        "model": model,
        "loss": "dice_ce",
        "fold": 0,
        "spot": spot,
        "expected_duration_minutes": expected,
        "warn_duration_minutes": warn,
        "cancel_duration_minutes": cancel,
        "hourly_rate": hourly_rate,
        "status": status,
        "launched_at": launched_at,
        "ended_at": ended_at,
        "failure_category": failure_category,
    }


class TestPendingTimeout:
    """PENDING timeout detection — prevents 11th pass repeat."""

    def test_pending_timeout_triggers_warn(self, tmp_path: Path) -> None:
        """Job PENDING for 35 min (threshold 30) triggers WARN."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="PENDING", launched_at="2026-03-28T09:00:00Z")],
            pending_timeout=30.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 9, 35, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        warns = [a for a in alerts if a.alert_type == "PENDING_TIMEOUT"]
        assert len(warns) >= 1
        assert warns[0].severity == "WARN"

    def test_pending_under_threshold_no_alert(self, tmp_path: Path) -> None:
        """Job PENDING for 25 min (threshold 30) does NOT trigger."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="PENDING", launched_at="2026-03-28T09:00:00Z")],
            pending_timeout=30.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 9, 25, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        pending_alerts = [a for a in alerts if a.alert_type == "PENDING_TIMEOUT"]
        assert len(pending_alerts) == 0


class TestDurationAnomalies:
    """Duration-based anomaly detection — prevents 10th pass repeat."""

    def test_running_over_warn_triggers_warn(self, tmp_path: Path) -> None:
        """Job RUNNING for 35 min with warn at 30 triggers WARN."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="RUNNING", warn=30.0, cancel=50.0)]
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 9, 35, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        warns = [a for a in alerts if a.alert_type == "DURATION_WARN"]
        assert len(warns) >= 1

    def test_running_over_cancel_triggers_critical_cancel(self, tmp_path: Path) -> None:
        """Job RUNNING for 55 min with cancel at 50 triggers CRITICAL + CANCEL_JOB."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="RUNNING", warn=30.0, cancel=50.0)]
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 9, 55, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        cancels = [a for a in alerts if a.alert_type == "DURATION_CANCEL"]
        assert len(cancels) >= 1
        assert cancels[0].severity == "CRITICAL"
        assert cancels[0].action == "CANCEL_JOB"


class TestFailedSetup:
    """FAILED_SETUP immediate detection — prevents 4th pass repeat."""

    def test_failed_setup_triggers_high(self, tmp_path: Path) -> None:
        """FAILED_SETUP status triggers HIGH alert immediately."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="FAILED_SETUP", ended_at="2026-03-28T09:10:00Z")]
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        setup_alerts = [a for a in alerts if a.alert_type == "FAILED_SETUP"]
        assert len(setup_alerts) >= 1
        assert setup_alerts[0].severity == "HIGH"


class TestKillSwitch:
    """Kill-switch: cascading failure detection."""

    def test_three_identical_failures_triggers_emergency(self, tmp_path: Path) -> None:
        """3 FAILED jobs with same category in 5 min triggers EMERGENCY."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(job_id=i, status="FAILED", ended_at=f"2026-03-28T09:0{i}:00Z",
                      failure_category="DVC_PULL_FAILED")
            for i in range(1, 4)
        ]
        data = _make_manifest_data(jobs=jobs, kill_switch_threshold=3, kill_switch_window=5.0)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        kill_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_IDENTICAL"]
        assert len(kill_alerts) >= 1
        assert kill_alerts[0].severity == "EMERGENCY"

    def test_batch_failure_percentage_triggers_emergency(self, tmp_path: Path) -> None:
        """60% of jobs FAILED (threshold 50%) triggers EMERGENCY."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = []
        for i in range(12):
            jobs.append(_make_job(job_id=i, status="FAILED",
                                  ended_at="2026-03-28T09:10:00Z"))
        for i in range(12, 20):
            jobs.append(_make_job(job_id=i, status="SUCCEEDED",
                                  ended_at="2026-03-28T09:10:00Z"))

        data = _make_manifest_data(jobs=jobs, batch_failure_pct=0.5)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        batch_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_BATCH"]
        assert len(batch_alerts) >= 1
        assert batch_alerts[0].severity == "EMERGENCY"


class TestBudget:
    """Budget monitoring — prevents cost overruns."""

    def test_budget_warning_triggers_warn(self, tmp_path: Path) -> None:
        """Total cost $12 with budget_warning $10 triggers WARN."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        # 1 job running for 60 min at $12/hr = $12
        data = _make_manifest_data(
            jobs=[_make_job(hourly_rate=12.0, launched_at="2026-03-28T09:00:00Z")],
            budget_warning=10.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        budget_warns = [a for a in alerts if a.alert_type == "BUDGET_WARN"]
        assert len(budget_warns) >= 1

    def test_budget_cap_triggers_emergency_cancel_all(self, tmp_path: Path) -> None:
        """Total cost $16 with budget_cap $15 triggers EMERGENCY CANCEL_ALL."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(hourly_rate=16.0, launched_at="2026-03-28T09:00:00Z")],
            budget_cap=15.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        cap_alerts = [a for a in alerts if a.alert_type == "BUDGET_CAP"]
        assert len(cap_alerts) >= 1
        assert cap_alerts[0].severity == "EMERGENCY"
        assert cap_alerts[0].action == "CANCEL_ALL"


class TestHealthyState:
    """No alerts when everything is healthy."""

    def test_no_alerts_when_all_healthy(self, tmp_path: Path) -> None:
        """All jobs RUNNING within expected durations, no alerts."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="RUNNING", warn=30.0, cancel=50.0)],
            budget_warning=100.0,
            budget_cap=200.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # 5 min elapsed — well within thresholds
        fixed_now = datetime(2026, 3, 28, 9, 5, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        assert len(alerts) == 0


class TestAllTerminal:
    """Summary alert when all jobs are terminal."""

    def test_all_terminal_returns_summary(self, tmp_path: Path) -> None:
        """All jobs terminal produces INFO summary alert."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[
                _make_job(job_id=1, status="SUCCEEDED", ended_at="2026-03-28T09:12:00Z"),
                _make_job(job_id=2, status="FAILED", ended_at="2026-03-28T09:15:00Z"),
            ],
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        summaries = [a for a in alerts if a.alert_type == "ALL_TERMINAL"]
        assert len(summaries) == 1
        assert summaries[0].severity == "INFO"
        assert summaries[0].action == "SUMMARY"


class TestHistoricalScenarios:
    """Historical scenario replays with EXACT alert type/severity/action checks."""

    def test_pass4_scenario_8_failed_setup_triggers_kill_switch(
        self, tmp_path: Path
    ) -> None:
        """8 identical FAILED_SETUP with kill_switch_threshold=3 fires KILL_SWITCH."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(
                job_id=i,
                status="FAILED_SETUP",
                ended_at=f"2026-03-22T09:0{i}:00Z",
                failure_category="DOCKER_PULL_FAILED",
            )
            for i in range(1, 9)
        ]
        data = _make_manifest_data(
            jobs=jobs, kill_switch_threshold=3, kill_switch_window=10.0
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        kill_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_IDENTICAL"]
        assert len(kill_alerts) >= 1
        assert kill_alerts[0].severity == "EMERGENCY"
        assert kill_alerts[0].action == "CANCEL_ALL"

    def test_pass10_scenario_12h_stuck_job_triggers_cancel(
        self, tmp_path: Path
    ) -> None:
        """Job running 12h for 5-min task: DURATION_CANCEL fires."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[
                _make_job(
                    job_id=154,
                    status="RUNNING",
                    expected=5.0,
                    warn=15.0,
                    cancel=50.0,
                    launched_at="2026-03-27T09:00:00Z",
                )
            ],
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 27, 21, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        cancels = [a for a in alerts if a.alert_type == "DURATION_CANCEL"]
        assert len(cancels) >= 1
        assert cancels[0].severity == "CRITICAL"
        assert cancels[0].action == "CANCEL_JOB"

    def test_pass10_scenario_also_triggers_duration_warn(
        self, tmp_path: Path
    ) -> None:
        """12h stuck job triggers BOTH DURATION_WARN and DURATION_CANCEL."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[
                _make_job(
                    job_id=154,
                    status="RUNNING",
                    expected=5.0,
                    warn=15.0,
                    cancel=50.0,
                    launched_at="2026-03-27T09:00:00Z",
                )
            ],
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 27, 21, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        warn = [a for a in alerts if a.alert_type == "DURATION_WARN"]
        cancel = [a for a in alerts if a.alert_type == "DURATION_CANCEL"]
        assert len(warn) >= 1
        assert len(cancel) >= 1

    def test_pass11_scenario_3_pending_10h_triggers_timeout(
        self, tmp_path: Path
    ) -> None:
        """3 PENDING 10h jobs trigger exactly 3 PENDING_TIMEOUT alerts."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(job_id=159 + i, status="PENDING", launched_at="2026-03-28T09:00:00Z")
            for i in range(3)
        ]
        data = _make_manifest_data(jobs=jobs, pending_timeout=30.0)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 19, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        pending_alerts = [a for a in alerts if a.alert_type == "PENDING_TIMEOUT"]
        assert len(pending_alerts) == 3


class TestAlertDeduplication:
    """Alert deduplication: _already_fired set prevents duplicate alerts."""

    def test_alert_deduplication_check_all_twice(self, tmp_path: Path) -> None:
        """Calling check_all() twice returns zero alerts the second time."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="PENDING", launched_at="2026-03-28T09:00:00Z")],
            pending_timeout=30.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            first = detector.check_all()
            second = detector.check_all()

        assert len(first) >= 1
        assert len(second) == 0  # All deduplicated

    def test_different_alert_types_not_deduplicated(self, tmp_path: Path) -> None:
        """DURATION_WARN and DURATION_CANCEL are different types — both fire."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(status="RUNNING", warn=30.0, cancel=50.0)],
            budget_warning=999.0,
            budget_cap=9999.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        types = {a.alert_type for a in alerts}
        assert "DURATION_WARN" in types
        assert "DURATION_CANCEL" in types


class TestBudgetAlarms:
    """Budget alarm at warning AND cap thresholds."""

    def test_budget_warning_threshold_fires(self, tmp_path: Path) -> None:
        """Cost above warning but below cap fires BUDGET_WARN."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        # 1 job: 60 min at $9/hr = $9 (above $8 warning, below $15 cap)
        data = _make_manifest_data(
            jobs=[_make_job(hourly_rate=9.0, launched_at="2026-03-28T09:00:00Z")],
            budget_warning=8.0,
            budget_cap=15.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        budget_warns = [a for a in alerts if a.alert_type == "BUDGET_WARN"]
        assert len(budget_warns) >= 1
        assert budget_warns[0].severity == "WARN"

    def test_budget_cap_threshold_fires_cancel_all(self, tmp_path: Path) -> None:
        """Cost above cap fires BUDGET_CAP with CANCEL_ALL."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(hourly_rate=16.0, launched_at="2026-03-28T09:00:00Z")],
            budget_warning=8.0,
            budget_cap=15.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        cap_alerts = [a for a in alerts if a.alert_type == "BUDGET_CAP"]
        assert len(cap_alerts) >= 1
        assert cap_alerts[0].severity == "EMERGENCY"
        assert cap_alerts[0].action == "CANCEL_ALL"

    def test_budget_below_warning_no_alert(self, tmp_path: Path) -> None:
        """Cost below warning triggers no budget alerts."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(hourly_rate=0.22, launched_at="2026-03-28T09:00:00Z")],
            budget_warning=100.0,
            budget_cap=200.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        fixed_now = datetime(2026, 3, 28, 9, 5, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        budget_alerts = [
            a for a in alerts if a.alert_type in ("BUDGET_WARN", "BUDGET_CAP")
        ]
        assert len(budget_alerts) == 0


class TestKillSwitchExtended:
    """Extended kill switch edge cases."""

    def test_kill_switch_3_identical_failures(self, tmp_path: Path) -> None:
        """3 OOM failures trigger KILL_SWITCH_IDENTICAL."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = []
        for i in range(3):
            jobs.append(
                _make_job(
                    job_id=i,
                    status="FAILED",
                    ended_at=f"2026-03-28T09:0{i}:00Z",
                    failure_category="OOM",
                )
            )
        for i in range(3, 5):
            jobs.append(
                _make_job(
                    job_id=i,
                    status="FAILED",
                    ended_at=f"2026-03-28T09:0{i}:00Z",
                    failure_category="SPOT_PREEMPTION",
                )
            )
        data = _make_manifest_data(
            jobs=jobs, kill_switch_threshold=3, kill_switch_window=10.0
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        kill_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_IDENTICAL"]
        assert len(kill_alerts) >= 1  # OOM cluster triggers

    def test_kill_switch_does_not_fire_at_2_identical(self, tmp_path: Path) -> None:
        """2 identical failures (threshold=3) does NOT trigger kill switch."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(
                job_id=i,
                status="FAILED",
                ended_at=f"2026-03-28T09:0{i}:00Z",
                failure_category="OOM",
            )
            for i in range(2)
        ]
        data = _make_manifest_data(
            jobs=jobs, kill_switch_threshold=3, kill_switch_window=10.0
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        kill_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_IDENTICAL"]
        assert len(kill_alerts) == 0

    def test_kill_switch_batch_failure_percentage(self, tmp_path: Path) -> None:
        """60% failure rate (threshold 50%) triggers KILL_SWITCH_BATCH."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = []
        for i in range(6):
            jobs.append(
                _make_job(job_id=i, status="FAILED", ended_at="2026-03-28T09:10:00Z")
            )
        for i in range(6, 10):
            jobs.append(
                _make_job(
                    job_id=i, status="SUCCEEDED", ended_at="2026-03-28T09:10:00Z"
                )
            )
        data = _make_manifest_data(jobs=jobs, batch_failure_pct=0.5)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        batch_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_BATCH"]
        assert len(batch_alerts) >= 1
        assert batch_alerts[0].severity == "EMERGENCY"


class TestScenarioReplays:
    """Replay actual failure scenarios from passes 4, 10, and 11."""

    def test_10th_pass_12h_stuck_job(self, tmp_path: Path) -> None:
        """Job #154 running 720 min (12h) with expected 5 min."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        data = _make_manifest_data(
            jobs=[_make_job(
                job_id=154, status="RUNNING", expected=5.0,
                warn=15.0, cancel=25.0,
                launched_at="2026-03-27T09:00:00Z",
            )],
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # 12 hours later
        fixed_now = datetime(2026, 3, 27, 21, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        # Should have BOTH warn and cancel alerts
        warn = [a for a in alerts if a.alert_type == "DURATION_WARN"]
        cancel = [a for a in alerts if a.alert_type == "DURATION_CANCEL"]
        assert len(warn) >= 1
        assert len(cancel) >= 1
        assert cancel[0].action == "CANCEL_JOB"

    def test_11th_pass_10h_pending(self, tmp_path: Path) -> None:
        """3 jobs PENDING for 600+ min."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(job_id=159, status="PENDING", launched_at="2026-03-28T09:00:00Z"),
            _make_job(job_id=160, status="PENDING", launched_at="2026-03-28T09:00:00Z"),
            _make_job(job_id=161, status="PENDING", launched_at="2026-03-28T09:00:00Z"),
        ]
        data = _make_manifest_data(jobs=jobs, pending_timeout=30.0)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # 10 hours later
        fixed_now = datetime(2026, 3, 28, 19, 0, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            detector = AnomalyDetector(manifest)
            alerts = detector.check_all()

        pending_alerts = [a for a in alerts if a.alert_type == "PENDING_TIMEOUT"]
        assert len(pending_alerts) == 3  # One per job

    def test_4th_pass_cascading_failed_setup(self, tmp_path: Path) -> None:
        """8 FAILED_SETUP with same error — kill switch after 3rd."""
        from minivess.compute.anomaly_detector import AnomalyDetector
        from minivess.compute.job_manifest import JobManifest

        jobs = [
            _make_job(
                job_id=i, status="FAILED_SETUP",
                ended_at=f"2026-03-22T09:0{i}:00Z",
                failure_category="DVC_PULL_FAILED",
            )
            for i in range(1, 9)
        ]
        data = _make_manifest_data(
            jobs=jobs,
            kill_switch_threshold=3,
            kill_switch_window=10.0,
        )
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        detector = AnomalyDetector(manifest)
        alerts = detector.check_all()

        kill_alerts = [a for a in alerts if a.alert_type == "KILL_SWITCH_IDENTICAL"]
        assert len(kill_alerts) >= 1
        setup_alerts = [a for a in alerts if a.alert_type == "FAILED_SETUP"]
        assert len(setup_alerts) >= 1
