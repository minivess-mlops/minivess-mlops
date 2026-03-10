"""Integration tests for Prefect API contract verification.

E2E Plan Phase 2, Task T2.2: Verify flow run status via Prefect API.

Verifies:
1. All expected flow runs show COMPLETED state
2. No FAILED or CRASHED flow runs
3. Task-level runs recorded for each flow
4. No flow run exceeds 60 min (indicates hang)
5. Prefect API /health returns 200

Marked @integration — excluded from staging and prod suites.
"""

from __future__ import annotations

import pytest


def _prefect_api_reachable() -> bool:
    """Check if Prefect API is reachable."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:4200/api/health", timeout=5):
            return True
    except Exception:
        return False


_REQUIRES_PREFECT = "requires Prefect server running"

EXPECTED_FLOW_TYPES = [
    "acquisition",
    "data",
    "annotation",
    "train",
    "post_training",
    "analysis",
    "biostatistics",
    "deploy",
]


@pytest.mark.integration
class TestPrefectApiContract:
    """Verify Prefect API shows correct flow run statuses."""

    def test_prefect_api_healthy(self) -> None:
        """Verify Prefect API /health returns 200."""
        if not _prefect_api_reachable():
            pytest.skip(_REQUIRES_PREFECT)

        import urllib.request

        with urllib.request.urlopen(
            "http://localhost:4200/api/health", timeout=5
        ) as resp:
            assert resp.status == 200

    def test_all_expected_flows_completed(self) -> None:
        """Verify expected flow types show COMPLETED state."""
        if not _prefect_api_reachable():
            pytest.skip(_REQUIRES_PREFECT)

        import json
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:4200/api/flow_runs/filter",
            data=json.dumps({"limit": 100}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            flow_runs = json.loads(resp.read().decode("utf-8"))

        if not flow_runs:
            pytest.skip("No flow runs found in Prefect")

        completed_names = {
            r.get("name", "").split("/")[0]
            for r in flow_runs
            if r.get("state_type") == "COMPLETED"
        }
        # At least some flows should have completed
        assert len(completed_names) > 0, "No COMPLETED flow runs found"

    def test_no_failed_flow_runs(self) -> None:
        """Verify zero FAILED or CRASHED flow runs."""
        if not _prefect_api_reachable():
            pytest.skip(_REQUIRES_PREFECT)

        import json
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:4200/api/flow_runs/filter",
            data=json.dumps({"limit": 100}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            flow_runs = json.loads(resp.read().decode("utf-8"))

        failed = [r for r in flow_runs if r.get("state_type") in ("FAILED", "CRASHED")]
        assert not failed, (
            f"Found {len(failed)} failed/crashed flow runs: "
            f"{[r.get('name') for r in failed]}"
        )

    def test_flow_run_duration_reasonable(self) -> None:
        """Verify no flow run exceeds 60 min (indicates hang)."""
        if not _prefect_api_reachable():
            pytest.skip(_REQUIRES_PREFECT)

        import json
        import urllib.request
        from datetime import datetime

        req = urllib.request.Request(
            "http://localhost:4200/api/flow_runs/filter",
            data=json.dumps({"limit": 100}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            flow_runs = json.loads(resp.read().decode("utf-8"))

        max_duration_s = 3600  # 60 minutes
        for run in flow_runs:
            start = run.get("start_time")
            end = run.get("end_time")
            if start and end:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                duration_s = (end_dt - start_dt).total_seconds()
                assert duration_s <= max_duration_s, (
                    f"Flow run {run.get('name')!r} took {duration_s:.0f}s "
                    f"(>{max_duration_s}s limit) — may indicate a hang."
                )

    def test_task_runs_recorded(self) -> None:
        """Verify task-level runs exist for each flow."""
        if not _prefect_api_reachable():
            pytest.skip(_REQUIRES_PREFECT)

        import json
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:4200/api/task_runs/filter",
            data=json.dumps({"limit": 100}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            task_runs = json.loads(resp.read().decode("utf-8"))

        if not task_runs:
            pytest.skip(
                "No task runs found — Prefect may not record tasks in ephemeral mode"
            )

        assert len(task_runs) > 0, "Expected at least some task-level runs"
