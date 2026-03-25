"""Tests for training heartbeat zombie detection (issue #963).

Verifies that:
- is_heartbeat_stale function exists in train_flow.py (structural, staging-safe)
- Stale heartbeats (>30 min) are correctly detected
- Fresh heartbeats are not flagged as stale

Task 3.6 from debug run plan.
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestHeartbeatStructural:
    """Staging-safe structural tests using ast.parse (no imports needed)."""

    def test_is_heartbeat_stale_function_exists(self) -> None:
        """is_heartbeat_stale must exist in train_flow.py."""
        source = Path(
            "src/minivess/orchestration/flows/train_flow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        ]
        assert "is_heartbeat_stale" in names

    def test_heartbeat_stale_detects_old(self) -> None:
        """A heartbeat older than threshold_minutes should be stale."""
        from datetime import datetime, timedelta, timezone

        from minivess.orchestration.flows.train_flow import is_heartbeat_stale

        old = (
            datetime.now(timezone.utc) - timedelta(minutes=31)
        ).isoformat()
        assert is_heartbeat_stale({"timestamp": old}) is True

    def test_heartbeat_fresh_not_stale(self) -> None:
        """A heartbeat just created should not be stale."""
        from datetime import datetime, timezone

        from minivess.orchestration.flows.train_flow import is_heartbeat_stale

        fresh = datetime.now(timezone.utc).isoformat()
        assert is_heartbeat_stale({"timestamp": fresh}) is False
