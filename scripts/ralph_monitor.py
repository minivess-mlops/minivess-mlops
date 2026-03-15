"""Ralph Monitor CLI — Cloud GPU job monitoring with automatic diagnosis.

Polls SkyPilot managed job status at configurable intervals and diagnoses
failures using known failure patterns. Outputs structured JSONL diagnoses.

Usage:
    # Monitor a specific job
    uv run python scripts/ralph_monitor.py --job-id 4 --poll-interval 30

    # Monitor the latest job
    uv run python scripts/ralph_monitor.py --latest --poll-interval 30

    # Diagnose the most recent failure without polling
    uv run python scripts/ralph_monitor.py --diagnose-last

See: docs/planning/ralph-loop-for-cloud-monitoring.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from minivess.compute.ralph_monitor import (  # noqa: E402
    DiagnosisRecord,
    analyze_logs,
    append_diagnosis,
    parse_job_status,
)

_DIAGNOSIS_FILE = _ROOT / "outputs" / "ralph_diagnoses.jsonl"
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "CANCELLED"}


def _run_sky_command(args: list[str]) -> str:
    """Run a sky CLI command, trying direct then uv run."""
    for prefix in [[], ["uv", "run"]]:
        result = subprocess.run(
            [*prefix, "sky", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
    return result.stdout + result.stderr


def _get_job_queue() -> str:
    """Get sky jobs queue output."""
    return _run_sky_command(["jobs", "queue"])


def _get_job_logs(job_id: int) -> str:
    """Get logs for a specific job."""
    return _run_sky_command(["jobs", "logs", str(job_id), "--no-follow"])


def _find_latest_job_id(queue_output: str) -> int | None:
    """Find the highest job ID from queue output."""
    max_id = None
    for line in queue_output.split("\n"):
        parts = line.split()
        if parts and parts[0].isdigit():
            job_id = int(parts[0])
            if max_id is None or job_id > max_id:
                max_id = job_id
    return max_id


def monitor_job(job_id: int, poll_interval: int = 30) -> str:
    """Monitor a SkyPilot job until completion.

    Returns the final status.
    """
    print(f"=== Ralph Monitor: Watching job {job_id} (poll every {poll_interval}s) ===")
    last_status = None

    while True:
        queue_output = _get_job_queue()
        status = parse_job_status(queue_output, job_id=job_id)

        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Job {job_id}: {status}")
            last_status = status

        if status in _TERMINAL_STATUSES:
            print(f"\n=== Job {job_id} reached terminal status: {status} ===")
            if status in {"FAILED", "FAILED_SETUP"}:
                diagnose_job(job_id, status)
            return status

        if status is None:
            print(f"WARNING: Job {job_id} not found in queue")
            return "NOT_FOUND"

        time.sleep(poll_interval)


def diagnose_job(job_id: int, status: str) -> DiagnosisRecord:
    """Fetch logs and diagnose a failed job."""
    print(f"\n--- Diagnosing job {job_id} ({status}) ---")
    logs = _get_job_logs(job_id)
    failure_info = analyze_logs(logs, status)

    record = DiagnosisRecord(
        job_id=job_id,
        status=status,
        category=failure_info.category,
        error_line=failure_info.matched_line[:200],
        root_cause=failure_info.root_cause,
        fix_suggestion=failure_info.root_cause,
        auto_fixable=failure_info.auto_fixable,
    )

    append_diagnosis(record, _DIAGNOSIS_FILE)
    print(f"Category:     {record.category}")
    print(f"Root cause:   {record.root_cause}")
    print(f"Auto-fixable: {record.auto_fixable}")
    print(f"Diagnosis saved to: {_DIAGNOSIS_FILE}")
    return record


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ralph Monitor — SkyPilot job diagnosis"
    )
    parser.add_argument("--job-id", type=int, help="Job ID to monitor")
    parser.add_argument("--latest", action="store_true", help="Monitor the latest job")
    parser.add_argument(
        "--diagnose-last", action="store_true", help="Diagnose last failed job"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30, help="Poll interval in seconds"
    )
    args = parser.parse_args()

    if args.diagnose_last:
        queue_output = _get_job_queue()
        job_id = _find_latest_job_id(queue_output)
        if job_id is None:
            print("No jobs found in queue")
            return 1
        status = parse_job_status(queue_output, job_id)
        if status in {"FAILED", "FAILED_SETUP"}:
            diagnose_job(job_id, status)
        else:
            print(f"Latest job {job_id} status is {status} (not a failure)")
        return 0

    job_id = args.job_id
    if args.latest:
        queue_output = _get_job_queue()
        job_id = _find_latest_job_id(queue_output)
        if job_id is None:
            print("No jobs found in queue")
            return 1

    if job_id is None:
        parser.error("Specify --job-id or --latest")
        return 1

    final_status = monitor_job(job_id, args.poll_interval)
    return 0 if final_status == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
