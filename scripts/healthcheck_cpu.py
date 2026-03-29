#!/usr/bin/env python3
"""Docker health check for CPU flow containers.

Reads events.jsonl staleness and returns exit 0 (healthy) or exit 1 (unhealthy).
For CPU flows that don't have GPU heartbeat monitoring.

Uses ONLY stdlib — no minivess package imports.

Environment variables:
    LOGS_DIR: Directory containing events.jsonl (default: /app/logs)
    STALL_THRESHOLD_MINUTES: Events stale after N minutes (default: 30)
    HEALTH_GRACE_PERIOD_MINUTES: Ignore during startup (default: 10)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def check_cpu_flow_health() -> tuple[bool, str]:
    """Check if CPU flow is producing events.

    Returns (healthy: bool, message: str).
    """
    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    stall_threshold_min = float(os.environ.get("STALL_THRESHOLD_MINUTES", "30"))
    grace_period_min = float(os.environ.get("HEALTH_GRACE_PERIOD_MINUTES", "10"))

    events_path = logs_dir / "events.jsonl"

    # During grace period, always healthy
    container_start = Path("/proc/1/stat").stat().st_mtime if Path("/proc/1/stat").exists() else time.time()
    uptime_minutes = (time.time() - container_start) / 60.0
    if uptime_minutes < grace_period_min:
        return True, f"Grace period ({uptime_minutes:.1f}/{grace_period_min:.0f} min)"

    # After grace, events.jsonl must exist
    if not events_path.exists():
        return False, f"events.jsonl not found at {events_path} after {uptime_minutes:.0f} min"

    # Check last modification time of events.jsonl
    try:
        mtime = events_path.stat().st_mtime
        age_minutes = (time.time() - mtime) / 60.0

        if age_minutes > stall_threshold_min:
            return False, f"events.jsonl stale: {age_minutes:.1f} min since last write (threshold: {stall_threshold_min:.0f})"

        return True, f"events.jsonl fresh: {age_minutes:.1f} min since last write"
    except OSError as e:
        return False, f"Cannot stat events.jsonl: {e}"


def main() -> None:
    """Entry point for Docker HEALTHCHECK CMD."""
    healthy, message = check_cpu_flow_health()
    print(message)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
