"""Tests for scripts/analyze_factorial_run.py — preemption rate monitoring.

Issue: #962 (TDD Task 3.5)
Verifies compute_preemption_metrics() correctly counts preempted jobs,
calculates preemption rates, and raises on empty input.
"""

from __future__ import annotations

import pytest


def _make_job(
    job_id: int,
    name: str,
    recoveries: int = 0,
    duration_seconds: float = 300.0,
    recovery_seconds: float = 0.0,
) -> dict:
    """Helper to build a minimal job dict matching sky jobs queue JSON schema."""
    return {
        "job_id": job_id,
        "job_name": name,
        "status": "SUCCEEDED",
        "#RECOVERIES": recoveries,
        "duration_seconds": duration_seconds,
        "recovery_seconds": recovery_seconds,
    }


class TestPreemptionCountCorrect:
    """3 jobs, 1 with recoveries=2 -> preempted_count=1."""

    def test_preemption_count_correct(self) -> None:
        from scripts.analyze_factorial_run import compute_preemption_metrics

        jobs = [
            _make_job(1, "dynunet-dice_ce-f0", recoveries=0),
            _make_job(2, "dynunet-cbdice-f0", recoveries=2, recovery_seconds=120.0),
            _make_job(3, "segresnet-dice_ce-f0", recoveries=0),
        ]
        result = compute_preemption_metrics(jobs)
        assert result["preempted_count"] == 1
        assert result["total_jobs"] == 3


class TestPreemptionRateCorrect:
    """10 jobs, 3 preempted -> rate=0.3."""

    def test_preemption_rate_correct(self) -> None:
        from scripts.analyze_factorial_run import compute_preemption_metrics

        jobs = []
        for i in range(10):
            recoveries = 1 if i < 3 else 0
            recovery_sec = 60.0 if i < 3 else 0.0
            jobs.append(
                _make_job(
                    i,
                    f"job-{i}",
                    recoveries=recoveries,
                    recovery_seconds=recovery_sec,
                )
            )
        result = compute_preemption_metrics(jobs)
        assert result["preemption_rate"] == pytest.approx(0.3)


class TestEmptyRaisesValueError:
    """Empty job list -> ValueError (Rule #25: loud failures)."""

    def test_empty_raises_value_error(self) -> None:
        from scripts.analyze_factorial_run import compute_preemption_metrics

        with pytest.raises(ValueError, match="empty"):
            compute_preemption_metrics([])


class TestZeroPreemptions:
    """5 jobs, 0 recoveries -> rate=0.0."""

    def test_zero_preemptions(self) -> None:
        from scripts.analyze_factorial_run import compute_preemption_metrics

        jobs = [_make_job(i, f"job-{i}", recoveries=0) for i in range(5)]
        result = compute_preemption_metrics(jobs)
        assert result["preemption_rate"] == 0.0
        assert result["preempted_count"] == 0
        assert result["avg_recovery_seconds"] == 0.0


class TestOutputTypes:
    """Verify rate is float in [0,1], count is int."""

    def test_output_types(self) -> None:
        from scripts.analyze_factorial_run import compute_preemption_metrics

        jobs = [
            _make_job(0, "job-0", recoveries=0),
            _make_job(1, "job-1", recoveries=1, recovery_seconds=30.0),
        ]
        result = compute_preemption_metrics(jobs)

        assert isinstance(result["total_jobs"], int)
        assert isinstance(result["preempted_count"], int)
        assert isinstance(result["preemption_rate"], float)
        assert isinstance(result["avg_recovery_seconds"], float)
        assert 0.0 <= result["preemption_rate"] <= 1.0
