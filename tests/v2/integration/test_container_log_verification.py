"""Integration tests for container log verification.

E2E Plan Phase 2, Task T2.3: Container log verification with JSONL events.

Verifies:
1. JSONL events file exists in logs volume
2. Each flow has at least flow_start + flow_end events
3. No ERROR-level log lines
4. All JSONL lines are parseable with json.loads()
5. All flow_end events have status="completed"

Rule #16: No regex. Use json.loads() for parsing JSONL events.

Marked @integration — excluded from staging and prod suites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REQUIRES_LOGS = "requires logs volume with JSONL events"


def _find_logs_dir() -> Path | None:
    """Find the logs directory with JSONL events."""
    candidates = [
        Path("/app/logs"),  # Docker container path
        Path(__file__).resolve().parents[4] / "outputs" / "debug" / "logs",
        Path(__file__).resolve().parents[4] / "logs",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


@pytest.mark.integration
class TestContainerLogVerification:
    """Verify container logs contain valid JSONL events."""

    @pytest.fixture(scope="class")
    def logs_dir(self) -> Path:
        """Find the logs directory or skip."""
        found = _find_logs_dir()
        if found is None:
            pytest.skip(_REQUIRES_LOGS)
        return found

    @pytest.fixture(scope="class")
    def jsonl_events(self, logs_dir: Path) -> list[dict]:
        """Parse all JSONL event files in logs directory."""
        events: list[dict] = []
        for jsonl_file in logs_dir.glob("*.jsonl"):
            content = jsonl_file.read_text(encoding="utf-8")
            for line_num, line in enumerate(content.strip().split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    pytest.fail(
                        f"Invalid JSONL at {jsonl_file.name}:{line_num}: {e}\n"
                        f"Line content: {line[:200]}"
                    )
        return events

    def test_jsonl_events_file_exists(self, logs_dir: Path) -> None:
        """Verify at least one .jsonl file exists in logs volume."""
        jsonl_files = list(logs_dir.glob("*.jsonl"))
        assert jsonl_files, (
            f"No .jsonl files found in {logs_dir}. "
            f"StructuredLogger must write events to JSONL files."
        )

    def test_jsonl_events_parseable(self, jsonl_events: list[dict]) -> None:
        """Every line in JSONL files passes json.loads()."""
        # The fixture already validates parsing; just ensure we got events
        assert len(jsonl_events) > 0, "No JSONL events parsed from log files"

    def test_each_flow_has_start_and_end_events(self, jsonl_events: list[dict]) -> None:
        """For each flow name, verify both flow_start and flow_end events."""
        if not jsonl_events:
            pytest.skip("No JSONL events available")

        # Group events by flow_name
        flows_seen: dict[str, set[str]] = {}
        for event in jsonl_events:
            flow_name = event.get("flow_name", "")
            event_type = event.get("event_type", "")
            if flow_name:
                flows_seen.setdefault(flow_name, set()).add(event_type)

        assert flows_seen, "No flow events found in JSONL"

        for flow_name, event_types in flows_seen.items():
            assert "flow_start" in event_types, (
                f"Flow {flow_name!r} has no flow_start event"
            )
            assert "flow_end" in event_types, (
                f"Flow {flow_name!r} has no flow_end event"
            )

    def test_no_error_level_in_container_logs(self, logs_dir: Path) -> None:
        """Parse container logs, verify no ERROR lines."""
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            pytest.skip("No .log files found")

        errors: list[str] = []
        for log_file in log_files:
            content = log_file.read_text(encoding="utf-8")
            for line_num, line in enumerate(content.split("\n"), 1):
                # Check for ERROR level using string operations (no regex per Rule #16)
                if " ERROR " in line or line.startswith("ERROR"):
                    errors.append(f"{log_file.name}:{line_num}: {line.strip()[:200]}")

        assert not errors, (
            f"Found {len(errors)} ERROR lines in container logs:\n"
            + "\n".join(errors[:10])
        )

    def test_flow_end_events_have_status_completed(
        self, jsonl_events: list[dict]
    ) -> None:
        """All flow_end events have status='completed'."""
        if not jsonl_events:
            pytest.skip("No JSONL events available")

        flow_end_events = [e for e in jsonl_events if e.get("event_type") == "flow_end"]
        if not flow_end_events:
            pytest.skip("No flow_end events found")

        for event in flow_end_events:
            status = event.get("status", "")
            assert status == "completed", (
                f"flow_end event for {event.get('flow_name')!r} has "
                f"status={status!r}, expected 'completed'"
            )
