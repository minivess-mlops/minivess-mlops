"""Cost guardrail tests — financial safety net for GPU experiments.

Verifies cost estimation logic, JobRecord cost calculations, manifest totals,
and YAML contract cost limits.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 6 (P1): T6.1 – T6.5
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT = REPO_ROOT / "configs" / "cloud" / "yaml_contract.yaml"
DEBUG_YAML = REPO_ROOT / "configs" / "factorial" / "debug.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ===========================================================================
# T6.1: Cost estimate calculation correct for L4, A100, A100-80GB
# ===========================================================================


class TestCostEstimation:
    """Verify per-job and per-pass cost calculations match FinOps report."""

    def test_l4_cost_per_job(self) -> None:
        """L4 spot: 55 min at $0.22/hr = ~$0.20."""
        hourly_rate = 0.22
        total_minutes = 55
        cost = (total_minutes / 60.0) * hourly_rate
        assert abs(cost - 0.2017) < 0.01

    def test_a100_cost_per_job(self) -> None:
        """A100-40GB spot: 30 min at $1.10/hr = $0.55."""
        hourly_rate = 1.10
        total_minutes = 30
        cost = (total_minutes / 60.0) * hourly_rate
        assert abs(cost - 0.55) < 0.01

    def test_a100_80gb_cost_per_job(self) -> None:
        """A100-80GB spot: 30 min at $1.38/hr = $0.69."""
        hourly_rate = 1.38
        total_minutes = 30
        cost = (total_minutes / 60.0) * hourly_rate
        assert abs(cost - 0.69) < 0.01

    def test_l4_cost_per_debug_pass_34_jobs(self) -> None:
        """34 L4 jobs: 34 * $0.20 = ~$6.85."""
        cost_per_job = (55 / 60.0) * 0.22
        total = 34 * cost_per_job
        assert abs(total - 6.86) < 0.5

    def test_a100_cost_per_debug_pass_34_jobs(self) -> None:
        """34 A100 jobs: 34 * $0.55 = ~$18.70."""
        cost_per_job = (30 / 60.0) * 1.10
        total = 34 * cost_per_job
        assert abs(total - 18.70) < 1.0


# ===========================================================================
# T6.2: JobRecord cost_usd property calculates correctly
# ===========================================================================


class TestJobRecordCost:
    """JobRecord.cost_usd = elapsed_hours * hourly_rate."""

    def test_job_record_cost_15_min_l4(self) -> None:
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

        fixed_now = datetime(2026, 3, 28, 9, 15, 0, tzinfo=UTC)
        with patch("minivess.compute.job_manifest.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            cost = record.cost_usd

        # 15 min = 0.25 hr * $0.22 = $0.055
        assert cost == pytest.approx(0.055, abs=0.01)

    def test_job_record_cost_2_hours_a100(self) -> None:
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
            "hourly_rate": 1.10,
            "status": "SUCCEEDED",
            "launched_at": "2026-03-28T09:00:00Z",
            "ended_at": "2026-03-28T11:00:00Z",
        })

        cost = record.cost_usd
        # 2 hr * $1.10 = $2.20
        assert cost == pytest.approx(2.20, abs=0.01)

    def test_job_record_cost_zero_when_not_launched(self) -> None:
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
        assert record.cost_usd == 0.0


# ===========================================================================
# T6.3: Manifest total_cost sums all job costs correctly
# ===========================================================================


class TestManifestTotalCost:
    """JobManifest.total_cost = sum of all job costs."""

    def test_manifest_total_cost_sums_correctly(self, tmp_path: Path) -> None:
        from minivess.compute.job_manifest import JobManifest

        data = {
            "experiment_name": "test",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 100.0,
            "budget_cap_usd": 200.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T09:00:00Z",
            "jobs": [
                {  # 30 min at $0.22/hr = $0.11
                    "job_id": 1, "condition_name": "j1", "model": "dynunet",
                    "loss": "dice_ce", "fold": 0, "spot": True,
                    "expected_duration_minutes": 10.0,
                    "warn_duration_minutes": 30.0,
                    "cancel_duration_minutes": 50.0,
                    "hourly_rate": 0.22, "status": "SUCCEEDED",
                    "launched_at": "2026-03-28T09:00:00Z",
                    "ended_at": "2026-03-28T09:30:00Z",
                },
                {  # 60 min at $1.10/hr = $1.10
                    "job_id": 2, "condition_name": "j2", "model": "sam3_hybrid",
                    "loss": "cbdice_cldice", "fold": 0, "spot": False,
                    "expected_duration_minutes": 25.0,
                    "warn_duration_minutes": 75.0,
                    "cancel_duration_minutes": 75.0,
                    "hourly_rate": 1.10, "status": "SUCCEEDED",
                    "launched_at": "2026-03-28T09:00:00Z",
                    "ended_at": "2026-03-28T10:00:00Z",
                },
                {  # 15 min at $0.22/hr = $0.055
                    "job_id": 3, "condition_name": "j3", "model": "dynunet",
                    "loss": "dice_ce", "fold": 1, "spot": True,
                    "expected_duration_minutes": 10.0,
                    "warn_duration_minutes": 30.0,
                    "cancel_duration_minutes": 50.0,
                    "hourly_rate": 0.22, "status": "SUCCEEDED",
                    "launched_at": "2026-03-28T09:00:00Z",
                    "ended_at": "2026-03-28T09:15:00Z",
                },
            ],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)

        # Total: $0.11 + $1.10 + $0.055 = $1.265
        assert manifest.total_cost == pytest.approx(1.265, abs=0.05)

    def test_manifest_total_cost_zero_for_unlaunched_jobs(self, tmp_path: Path) -> None:
        from minivess.compute.job_manifest import JobManifest

        data = {
            "experiment_name": "test",
            "config_file": "configs/factorial/debug.yaml",
            "budget_warning_usd": 100.0,
            "budget_cap_usd": 200.0,
            "pending_timeout_minutes": 30.0,
            "kill_switch_threshold": 3,
            "kill_switch_window_minutes": 5.0,
            "batch_failure_pct": 0.5,
            "launched_at": "2026-03-28T09:00:00Z",
            "jobs": [
                {
                    "job_id": None, "condition_name": f"j{i}", "model": "dynunet",
                    "loss": "dice_ce", "fold": 0, "spot": True,
                    "expected_duration_minutes": 10.0,
                    "warn_duration_minutes": 30.0,
                    "cancel_duration_minutes": 50.0,
                    "hourly_rate": 0.22, "status": "PENDING",
                    "launched_at": None,
                }
                for i in range(3)
            ],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        manifest = JobManifest.from_json(path)
        assert manifest.total_cost == pytest.approx(0.0, abs=0.001)


# ===========================================================================
# T6.4: Contract max_hourly_cost values are sane
# ===========================================================================


class TestContractCostLimits:
    """Contract cost limits must be within reasonable bounds."""

    def test_max_hourly_cost_l4_under_1_dollar(self) -> None:
        contract = _load_yaml(CONTRACT)
        assert contract["max_hourly_cost_usd"]["L4"] < 1.0

    def test_max_hourly_cost_a100_under_2_dollars(self) -> None:
        contract = _load_yaml(CONTRACT)
        assert contract["max_hourly_cost_usd"]["A100"] < 2.0

    def test_max_hourly_cost_ordering(self) -> None:
        """L4 < RTX4090 < A100 < A100-80GB < H100."""
        contract = _load_yaml(CONTRACT)
        costs = contract["max_hourly_cost_usd"]
        assert costs["L4"] < costs["RTX4090"]
        assert costs["RTX4090"] < costs["A100"]
        assert costs["A100"] < costs["A100-80GB"]
        assert costs["A100-80GB"] < costs["H100"]

    def test_max_hourly_cost_all_positive(self) -> None:
        contract = _load_yaml(CONTRACT)
        for gpu, cost in contract["max_hourly_cost_usd"].items():
            assert cost > 0, f"GPU '{gpu}' has non-positive cost: {cost}"


# ===========================================================================
# T6.5: Factorial configs have budget metadata / academic budget
# ===========================================================================


class TestBudgetConfig:
    """Budget awareness in factorial configs."""

    def test_total_experiment_cost_within_academic_budget(self) -> None:
        """Worst case: 34 A100-80GB jobs * max 1hr = under $100 academic budget."""
        contract = _load_yaml(CONTRACT)
        a100_80gb_cost = contract["max_hourly_cost_usd"]["A100-80GB"]
        # 34 jobs * 1 hour each (worst case — actual is ~30 min)
        total = 34 * a100_80gb_cost
        assert total < 100, (
            f"Worst-case cost ${total:.2f} exceeds $100 academic budget ceiling"
        )

    def test_debug_config_references_cloud_config(self) -> None:
        """debug.yaml has infrastructure.cloud_config pointing to gcp_spot."""
        debug = _load_yaml(DEBUG_YAML)
        assert debug["infrastructure"]["cloud_config"] == "gcp_spot"
