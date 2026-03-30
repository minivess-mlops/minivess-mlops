#!/usr/bin/env python3
"""Docker health check for training containers.

Reads heartbeat.json and returns exit 0 (healthy) or exit 1 (unhealthy)
based on staleness. Designed for Docker HEALTHCHECK CMD.

Uses ONLY stdlib — no minivess package imports (must work in any container).

Environment variables:
    LOGS_DIR: Directory containing heartbeat.json (default: /app/logs)
    STALL_THRESHOLD_MINUTES: Heartbeat stale after N minutes (default: 30)
    HEALTH_GRACE_PERIOD_MINUTES: Ignore during startup (default: 10)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def check_heartbeat_health() -> tuple[bool, str]:
    """Check if training heartbeat is fresh.

    Returns (healthy: bool, message: str).
    """
    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    stall_threshold_min = float(os.environ.get("STALL_THRESHOLD_MINUTES", "30"))
    grace_period_min = float(os.environ.get("HEALTH_GRACE_PERIOD_MINUTES", "10"))

    heartbeat_path = logs_dir / "heartbeat.json"

    # During grace period, always healthy (container starting up)
    container_start = Path("/proc/1/stat").stat().st_mtime if Path("/proc/1/stat").exists() else time.time()
    uptime_minutes = (time.time() - container_start) / 60.0
    if uptime_minutes < grace_period_min:
        return True, f"Grace period ({uptime_minutes:.1f}/{grace_period_min:.0f} min)"

    # After grace period, heartbeat.json must exist
    if not heartbeat_path.exists():
        return False, f"heartbeat.json not found at {heartbeat_path} after {uptime_minutes:.0f} min"

    # Check staleness
    try:
        data = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        ts_str = data.get("timestamp", "")
        ts = datetime.fromisoformat(ts_str)
        age_minutes = (datetime.now(UTC) - ts).total_seconds() / 60.0

        if age_minutes > stall_threshold_min:
            return False, f"Heartbeat stale: {age_minutes:.1f} min old (threshold: {stall_threshold_min:.0f} min)"

        return True, f"Heartbeat fresh: {age_minutes:.1f} min old"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return False, f"Heartbeat malformed: {e}"


def main() -> None:
    """Entry point for Docker HEALTHCHECK CMD."""
    healthy, message = check_heartbeat_health()
    print(message)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
