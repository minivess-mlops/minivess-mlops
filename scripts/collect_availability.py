"""Collect GPU availability data from SkyPilot job queue.

Appends one JSONL record per poll to outputs/availability_data/availability.jsonl.
Designed to run every 2-5 minutes via cron or Claude Code /schedule.

Usage:
    python scripts/collect_availability.py

Output schema (one JSON object per line):
    {
        "timestamp": "2026-03-29T01:30:00Z",
        "job_id": 171,
        "job_name": "dynunet-dice_ce-calibfalse-f0",
        "requested": "1x[A100-80GB:1, A100:1, L4:1][Spot]",
        "status": "PENDING",
        "total_duration_minutes": 91.4,
        "region": "us-central1",
        "cloud": "gcp",
        "hour_utc": 1,
        "day_of_week": "Sunday",
        "status_changed": false,
        "previous_status": "STARTING"
    }

Analysis: Load with DuckDB:
    SELECT hour_utc, status, COUNT(*) as samples
    FROM read_json_auto('outputs/availability_data/availability.jsonl')
    GROUP BY hour_utc, status
    ORDER BY hour_utc;
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "outputs" / "availability_data" / "availability.jsonl"
STATE_FILE = REPO_ROOT / "outputs" / "availability_data" / "last_status.json"

KNOWN_STATUSES = frozenset({
    "PENDING", "STARTING", "RUNNING", "SUCCEEDED",
    "FAILED", "FAILED_SETUP", "CANCELLED", "RECOVERING",
})


def parse_queue_line(line: str) -> dict | None:
    """Parse a single sky jobs queue line into a record."""
    tokens = line.split()
    if not tokens:
        return None

    try:
        job_id = int(tokens[0])
    except (ValueError, IndexError):
        return None

    status = None
    for token in tokens:
        if token in KNOWN_STATUSES:
            status = token
            break

    if status is None:
        return None

    name = tokens[2] if len(tokens) > 2 else ""

    # Extract requested resources (between name and "ago")
    ago_indices = [i for i, t in enumerate(tokens) if t == "ago"]
    requested = ""
    if ago_indices:
        first_ago = ago_indices[0]
        submitted_start = max(3, first_ago - 2)
        req_tokens = tokens[3:submitted_start]
        requested = " ".join(req_tokens)

    # Parse total duration
    total_dur_str = ""
    if ago_indices:
        last_ago = ago_indices[-1]
        # Find status index
        status_idx = tokens.index(status) if status in tokens else -1
        if status_idx > last_ago + 1:
            dur_tokens = tokens[last_ago + 1 : status_idx - 1]
            total_dur_str = " ".join(dur_tokens)

    total_minutes = 0.0
    for part in total_dur_str.split():
        part = part.strip()
        if part.endswith("h"):
            total_minutes += float(part[:-1]) * 60
        elif part.endswith("m"):
            total_minutes += float(part[:-1])
        elif part.endswith("s"):
            total_minutes += float(part[:-1]) / 60.0
        elif part.endswith("d"):
            total_minutes += float(part[:-1]) * 24 * 60

    return {
        "job_id": job_id,
        "job_name": name,
        "requested": requested,
        "status": status,
        "total_duration_minutes": round(total_minutes, 1),
    }


def collect() -> None:
    """Run sky jobs queue and append availability records."""
    now = datetime.now(UTC)

    try:
        result = subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "sky"), "jobs", "queue"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        record = {
            "timestamp": now.isoformat(),
            "error": str(exc),
            "hour_utc": now.hour,
            "day_of_week": now.strftime("%A"),
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return

    # Load previous status for change detection
    previous_statuses: dict[int, str] = {}
    if STATE_FILE.exists():
        try:
            previous_statuses = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            previous_statuses = {int(k): v for k, v in previous_statuses.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    current_statuses: dict[int, str] = {}

    for line in output.split("\n"):
        parsed = parse_queue_line(line.strip())
        if parsed is None:
            continue

        # Only track active (non-cancelled, non-old) jobs
        job_id = parsed["job_id"]
        if parsed["status"] == "CANCELLED":
            continue

        prev = previous_statuses.get(job_id, "UNKNOWN")
        status_changed = prev != parsed["status"]

        record = {
            "timestamp": now.isoformat(),
            "job_id": job_id,
            "job_name": parsed["job_name"],
            "requested": parsed["requested"],
            "status": parsed["status"],
            "total_duration_minutes": parsed["total_duration_minutes"],
            "region": "us-central1",
            "cloud": "gcp",
            "hour_utc": now.hour,
            "day_of_week": now.strftime("%A"),
            "status_changed": status_changed,
            "previous_status": prev,
        }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        current_statuses[job_id] = parsed["status"]

    # Save current statuses for next poll
    STATE_FILE.write_text(json.dumps(current_statuses), encoding="utf-8")


if __name__ == "__main__":
    collect()
    print(f"Collected at {datetime.now(UTC).isoformat()} → {OUTPUT_FILE}")
