"""Tests for the job manifest data model used by the factorial monitor.

RED phase: these tests define the contract for JobRecord and JobManifest
before any implementation exists. All tests should FAIL initially.

Plan: experiment-harness-improvement-plan.xml Task T1.1
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# JobRecord tests
# ---------------------------------------------------------------------------


class TestJobRecord:
    """JobRecord dataclass: per-job metadata + runtime state."""

    def test_job_record_from_dict(self) -> None:
        """JobRecord can be constructed from a dict with all required fields."""
        from minivess.compute.job_manifest import JobRecord

        data = {
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
            "launched_at": "2026-03-28T09:00:00Z",
        }
        record = JobRecord.from_dict(data)
        assert record.job_id == 165
        assert record.condition_name == "dynunet-dice_ce-calibfalse-f0"
        assert record.model == "dynunet"
        assert record.spot is True
        assert record.hourly_rate == 0.22

    def test_job_record_to_dict_roundtrip(self) -> None:
        """JobRecord serializes to dict and deserializes back identically."""
        from minivess.compute.job_manifest import JobRecord

        data = {
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
            "launched_at": "2026-03-28T09:00:00Z",
        }
        original = JobRecord.from_dict(data)
        roundtripped = JobRecord.from_dict(original.to_dict())
        assert original.to_dict() == roundtripped.to_dict()

    def test_job_record_elapsed_minutes(self) -> None:
        """elapsed_minutes computes time since launched_at."""
        from minivess.compute.job_manifest import JobRecord

        record = JobRecord.from_dict({
            "job_id": 1,
            "condition_name": "test",
            "model": "dynunet",
            "loss": "dice_ce",
            "fold": 0,
            "spot": True,
            "expected_duration_minutes": 10.0,
            "warn_duration_minutes": 30.0,
            "cancel_duration_minutes": 50.0,
            "hourly_rate": 0.22,
            "status": "RUNNING",
            "launched_at": "2026-03-28T09:00:00Z",
        })

        # Mock datetime.now to return a fixed time 15 min after launch
        fixed_now = datetime(2026, 3, 28, 9, 15, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            assert record.elapsed_minutes == pytest.approx(15.0, abs=0.1)

    def test_job_record_elapsed_minutes_none_launched(self) -> None:
        """elapsed_minutes returns 0 if launched_at is None."""
        from minivess.compute.job_manifest import JobRecord

        record = JobRecord.from_dict({
            "job_id": None,
            "condition_name": "test",
            "model": "dynunet",
            "loss": "dice_ce",
            "fold": 0,
            "spot": True,
            "expected_duration_minutes": 10.0,
            "warn_duration_minutes": 30.0,
            "cancel_duration_minutes": 50.0,
            "hourly_rate": 0.22,
            "status": "PENDING",
            "launched_at": None,
        })
        assert record.elapsed_minutes == 0.0

    def test_job_record_cost_usd(self) -> None:
        """cost_usd = elapsed_hours * hourly_rate."""
        from minivess.compute.job_manifest import JobRecord

        record = JobRecord.from_dict({
            "job_id": 1,
            "condition_name": "test",
            "model": "sam3_hybrid",
            "loss": "cbdice_cldice",
            "fold": 0,
            "spot": False,
            "expected_duration_minutes": 25.0,
            "warn_duration_minutes": 75.0,
            "cancel_duration_minutes": 75.0,
            "hourly_rate": 0.74,
            "status": "RUNNING",
            "launched_at": "2026-03-28T09:00:00Z",
        })

        # 15 min elapsed = 0.25 hours * $0.74/hr = $0.185
        fixed_now = datetime(2026, 3, 28, 9, 15, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            assert record.cost_usd == pytest.approx(0.185, abs=0.01)


# ---------------------------------------------------------------------------
# JobManifest tests
# ---------------------------------------------------------------------------


class TestJobManifest:
    """JobManifest: experiment-level container for all job records."""

    @pytest.fixture()
    def sample_manifest_data(self) -> dict:
        """Sample manifest JSON structure."""
        return {
            "experiment_name": "debug_factorial",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 8.0,
            "budget_cap_usd": 15.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T09:00:00Z",
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
                    "status": "SUCCEEDED",
                    "launched_at": "2026-03-28T09:00:00Z",
                    "ended_at": "2026-03-28T09:12:00Z",
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
                    "status": "RUNNING",
                    "launched_at": "2026-03-28T09:00:00Z",
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
                    "launched_at": "2026-03-28T09:01:00Z",
                },
            ],
        }

    def test_manifest_from_json_file(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """JobManifest loads from a JSON file."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")

        manifest = JobManifest.from_json(path)
        assert manifest.experiment_name == "debug_factorial"
        assert manifest.budget_cap_usd == 15.0
        assert len(manifest.jobs) == 3

    def test_manifest_to_json_file_roundtrip(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """JobManifest writes to JSON and reads back identically."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")

        original = JobManifest.from_json(path)
        out_path = tmp_path / "manifest_out.json"
        original.to_json(out_path)
        roundtripped = JobManifest.from_json(out_path)

        assert original.experiment_name == roundtripped.experiment_name
        assert original.budget_cap_usd == roundtripped.budget_cap_usd
        assert len(original.jobs) == len(roundtripped.jobs)

    def test_manifest_missing_required_field_raises(self, tmp_path: Path) -> None:
        """JobManifest raises ValueError when required field missing."""
        from minivess.compute.job_manifest import JobManifest

        data = {"experiment_name": "test", "jobs": []}
        # Missing budget_cap_usd and other required fields
        path = tmp_path / "bad_manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises((ValueError, KeyError)):
            JobManifest.from_json(path)

    def test_manifest_job_count(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """job_count returns correct count."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)
        assert manifest.job_count == 3

    def test_manifest_active_jobs(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """active_jobs returns only non-terminal jobs."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        active = manifest.active_jobs
        assert len(active) == 2  # RUNNING + PENDING
        statuses = {j.status for j in active}
        assert "SUCCEEDED" not in statuses

    def test_manifest_terminal_jobs(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """terminal_jobs returns only terminal jobs."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        terminal = manifest.terminal_jobs
        assert len(terminal) == 1
        assert terminal[0].status == "SUCCEEDED"

    def test_manifest_total_cost(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """total_cost computes sum of elapsed_hours * hourly_rate."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # SUCCEEDED job: 12 min at $0.22/hr = $0.044
        # RUNNING job: ongoing, cost depends on elapsed time
        # PENDING job: 0 cost (not yet running)
        cost = manifest.total_cost
        assert cost >= 0.04  # At least the SUCCEEDED job's cost

    def test_manifest_success_rate(
        self, tmp_path: Path, sample_manifest_data: dict
    ) -> None:
        """success_rate = SUCCEEDED / total terminal jobs."""
        from minivess.compute.job_manifest import JobManifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(sample_manifest_data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # 1 SUCCEEDED out of 1 terminal job = 100%
        assert manifest.success_rate == pytest.approx(1.0)
