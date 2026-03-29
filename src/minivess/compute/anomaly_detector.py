"""Anomaly detector for factorial experiment monitoring.

The CORE of the monitoring system. Takes a JobManifest (updated with current
queue data) and returns Alert objects for each detected anomaly.

Detects ALL failure modes from passes 4, 10, and 11:
- PENDING timeout (11th pass: 10h no intervention)
- Duration overrun (10th pass: 12h stuck job)
- FAILED_SETUP cascade (4th pass: 8 identical failures)
- Kill switch (batch failure percentage)
- Budget overrun

All thresholds come from the manifest — nothing is hardcoded here.

See: docs/planning/v0-2_archive/original_docs/experiment-harness-improvement-plan.xml T1.3
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from minivess.compute.job_manifest import JobManifest, JobRecord  # noqa: TC001

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A monitoring alert produced by the anomaly detector.

    Attributes
    ----------
    severity:
        Alert level: INFO, WARN, HIGH, CRITICAL, EMERGENCY.
    alert_type:
        Classification: PENDING_TIMEOUT, DURATION_WARN, DURATION_CANCEL,
        FAILED_SETUP, KILL_SWITCH_IDENTICAL, KILL_SWITCH_BATCH,
        BUDGET_WARN, BUDGET_CAP, ALL_TERMINAL.
    action:
        What to do: LOG, NOTIFY, CANCEL_JOB, CANCEL_ALL, SUMMARY.
    job_id:
        Affected job ID (None for batch-level alerts).
    message:
        Human-readable description.
    timestamp:
        ISO 8601 UTC when the alert was created.
    """

    severity: str
    alert_type: str
    action: str
    job_id: int | None
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class AnomalyDetector:
    """Detects anomalies in a factorial experiment by inspecting the manifest.

    Parameters
    ----------
    manifest:
        The current job manifest with updated statuses.
    """

    def __init__(self, manifest: JobManifest) -> None:
        self.manifest = manifest
        self._already_fired: set[tuple[int | None, str]] = set()

    def check_all(self) -> list[Alert]:
        """Run all anomaly checks and return combined alerts.

        Returns
        -------
        List of Alert objects, possibly empty if everything is healthy.
        """
        alerts: list[Alert] = []

        # Check for all-terminal first (summary)
        if self.manifest.all_terminal:
            alerts.extend(self._check_all_terminal())
            # Also check for failed setup and kill switch on terminal jobs
            alerts.extend(self._check_failed_setup())
            alerts.extend(self._check_kill_switch())
            return alerts

        alerts.extend(self._check_pending_timeouts())
        alerts.extend(self._check_duration_anomalies())
        alerts.extend(self._check_failed_setup())
        alerts.extend(self._check_kill_switch())
        alerts.extend(self._check_budget())

        return alerts

    def _check_pending_timeouts(self) -> list[Alert]:
        """Check for jobs stuck in PENDING beyond the timeout threshold."""
        alerts: list[Alert] = []
        timeout = self.manifest.pending_timeout_minutes

        for job in self.manifest.jobs:
            if job.status != "PENDING":
                continue
            elapsed = job.elapsed_minutes
            if elapsed > timeout:
                key = (job.job_id, "PENDING_TIMEOUT")
                if key not in self._already_fired:
                    alerts.append(Alert(
                        severity="WARN",
                        alert_type="PENDING_TIMEOUT",
                        action="NOTIFY",
                        job_id=job.job_id,
                        message=(
                            f"Job {job.job_id} ({job.condition_name}) PENDING for "
                            f"{elapsed:.0f} min (threshold: {timeout:.0f} min). "
                            f"Consider region fallback or GPU type change."
                        ),
                    ))
                    self._already_fired.add(key)

        return alerts

    def _check_duration_anomalies(self) -> list[Alert]:
        """Check for RUNNING jobs exceeding warn/cancel durations."""
        alerts: list[Alert] = []

        for job in self.manifest.jobs:
            if job.status != "RUNNING":
                continue

            elapsed = job.elapsed_minutes

            # Cancel threshold (highest priority — check first)
            if elapsed > job.cancel_duration_minutes:
                key = (job.job_id, "DURATION_CANCEL")
                if key not in self._already_fired:
                    alerts.append(Alert(
                        severity="CRITICAL",
                        alert_type="DURATION_CANCEL",
                        action="CANCEL_JOB",
                        job_id=job.job_id,
                        message=(
                            f"Job {job.job_id} ({job.condition_name}) RUNNING for "
                            f"{elapsed:.0f} min — exceeds cancel threshold "
                            f"({job.cancel_duration_minutes:.0f} min). AUTO-CANCELLING."
                        ),
                    ))
                    self._already_fired.add(key)

            # Warn threshold
            if elapsed > job.warn_duration_minutes:
                key = (job.job_id, "DURATION_WARN")
                if key not in self._already_fired:
                    alerts.append(Alert(
                        severity="WARN",
                        alert_type="DURATION_WARN",
                        action="NOTIFY",
                        job_id=job.job_id,
                        message=(
                            f"Job {job.job_id} ({job.condition_name}) RUNNING for "
                            f"{elapsed:.0f} min — exceeds warn threshold "
                            f"({job.warn_duration_minutes:.0f} min). "
                            f"Expected: {job.expected_duration_minutes:.0f} min."
                        ),
                    ))
                    self._already_fired.add(key)

        return alerts

    def _check_failed_setup(self) -> list[Alert]:
        """Check for FAILED_SETUP jobs (immediate alert)."""
        alerts: list[Alert] = []

        for job in self.manifest.jobs:
            if job.status != "FAILED_SETUP":
                continue
            key = (job.job_id, "FAILED_SETUP")
            if key not in self._already_fired:
                alerts.append(Alert(
                    severity="HIGH",
                    alert_type="FAILED_SETUP",
                    action="NOTIFY",
                    job_id=job.job_id,
                    message=(
                        f"Job {job.job_id} ({job.condition_name}) FAILED_SETUP. "
                        f"Category: {job.failure_category or 'unknown'}. "
                        f"Recoveries: {job.recovery_count}."
                    ),
                ))
                self._already_fired.add(key)

        return alerts

    def _check_kill_switch(self) -> list[Alert]:
        """Check for cascading failures that warrant batch cancellation."""
        alerts: list[Alert] = []
        failed_jobs = [
            j for j in self.manifest.jobs
            if j.status in ("FAILED", "FAILED_SETUP")
        ]

        if not failed_jobs:
            return alerts

        # Check 1: Identical failure category within time window
        category_jobs: dict[str, list[JobRecord]] = defaultdict(list)
        for job in failed_jobs:
            cat = job.failure_category or "unknown"
            category_jobs[cat].append(job)

        for cat, cat_jobs in category_jobs.items():
            if len(cat_jobs) >= self.manifest.kill_switch_threshold:
                # Check if failures are within the time window
                end_times: list[datetime] = []
                for j in cat_jobs:
                    if j.ended_at:
                        end_times.append(datetime.fromisoformat(j.ended_at))

                if len(end_times) >= self.manifest.kill_switch_threshold:
                    end_times.sort()
                    # Check sliding window: any N consecutive within window?
                    n = self.manifest.kill_switch_threshold
                    window_min = self.manifest.kill_switch_window_minutes
                    for i in range(len(end_times) - n + 1):
                        span = (end_times[i + n - 1] - end_times[i]).total_seconds() / 60.0
                        if span <= window_min:
                            key = (None, f"KILL_SWITCH_IDENTICAL_{cat}")
                            if key not in self._already_fired:
                                alerts.append(Alert(
                                    severity="EMERGENCY",
                                    alert_type="KILL_SWITCH_IDENTICAL",
                                    action="CANCEL_ALL",
                                    job_id=None,
                                    message=(
                                        f"KILL SWITCH: {len(cat_jobs)} jobs failed with "
                                        f"category '{cat}' — {n}+ within "
                                        f"{window_min:.0f} min window. "
                                        f"Cancelling all active jobs."
                                    ),
                                ))
                                self._already_fired.add(key)
                            break

        # Check 2: Batch failure percentage
        terminal = self.manifest.terminal_jobs
        if terminal:
            failed_count = sum(
                1 for j in terminal if j.status in ("FAILED", "FAILED_SETUP")
            )
            failure_rate = failed_count / len(terminal)
            if failure_rate > self.manifest.batch_failure_pct:
                key = (None, "KILL_SWITCH_BATCH")
                if key not in self._already_fired:
                    alerts.append(Alert(
                        severity="EMERGENCY",
                        alert_type="KILL_SWITCH_BATCH",
                        action="CANCEL_ALL",
                        job_id=None,
                        message=(
                            f"KILL SWITCH: {failed_count}/{len(terminal)} "
                            f"({failure_rate:.0%}) jobs failed — exceeds "
                            f"{self.manifest.batch_failure_pct:.0%} threshold."
                        ),
                    ))
                    self._already_fired.add(key)

        return alerts

    def _check_budget(self) -> list[Alert]:
        """Check total cost against budget warning and cap thresholds."""
        alerts: list[Alert] = []
        total_cost = self.manifest.total_cost

        # Budget cap (highest priority)
        if total_cost > self.manifest.budget_cap_usd:
            key = (None, "BUDGET_CAP")
            if key not in self._already_fired:
                alerts.append(Alert(
                    severity="EMERGENCY",
                    alert_type="BUDGET_CAP",
                    action="CANCEL_ALL",
                    job_id=None,
                    message=(
                        f"BUDGET CAP EXCEEDED: ${total_cost:.2f} > "
                        f"${self.manifest.budget_cap_usd:.2f}. "
                        f"CANCELLING ALL ACTIVE JOBS."
                    ),
                ))
                self._already_fired.add(key)

        # Budget warning
        elif total_cost > self.manifest.budget_warning_usd:
            key = (None, "BUDGET_WARN")
            if key not in self._already_fired:
                alerts.append(Alert(
                    severity="WARN",
                    alert_type="BUDGET_WARN",
                    action="NOTIFY",
                    job_id=None,
                    message=(
                        f"Budget warning: ${total_cost:.2f} > "
                        f"${self.manifest.budget_warning_usd:.2f} warning threshold. "
                        f"Cap: ${self.manifest.budget_cap_usd:.2f}."
                    ),
                ))
                self._already_fired.add(key)

        return alerts

    def _check_all_terminal(self) -> list[Alert]:
        """Produce summary alert when all jobs are terminal."""
        terminal = self.manifest.terminal_jobs
        succeeded = sum(1 for j in terminal if j.status == "SUCCEEDED")
        failed = sum(
            1 for j in terminal if j.status in ("FAILED", "FAILED_SETUP")
        )
        cancelled = sum(1 for j in terminal if j.status == "CANCELLED")
        total_cost = self.manifest.total_cost

        return [Alert(
            severity="INFO",
            alert_type="ALL_TERMINAL",
            action="SUMMARY",
            job_id=None,
            message=(
                f"ALL JOBS TERMINAL: {succeeded} succeeded, {failed} failed, "
                f"{cancelled} cancelled. Total cost: ${total_cost:.2f}. "
                f"Success rate: {self.manifest.success_rate:.0%}."
            ),
        )]
