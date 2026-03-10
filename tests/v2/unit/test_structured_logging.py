"""Tests for structured JSONL logging for flow/task/agent transitions.

Rule #16: No regex. Use json.loads() for parsing JSONL events.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestFlowEvent:
    """FlowEvent must produce valid JSONL with required fields."""

    def test_flow_start_event_has_required_fields(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.flow_start(flow_name="training-flow")
        data = json.loads(event.to_jsonl())
        assert data["event_type"] == "flow_start"
        assert data["flow_name"] == "training-flow"
        assert "timestamp" in data
        assert "event_id" in data

    def test_flow_end_event_has_status_and_duration(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.flow_end(
            flow_name="training-flow", status="completed", duration_ms=12345
        )
        data = json.loads(event.to_jsonl())
        assert data["event_type"] == "flow_end"
        assert data["status"] == "completed"
        assert data["duration_ms"] == 12345

    def test_task_start_event(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.task_start(
            flow_name="training-flow", task_name="train_one_fold"
        )
        data = json.loads(event.to_jsonl())
        assert data["event_type"] == "task_start"
        assert data["task_name"] == "train_one_fold"

    def test_task_end_event(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.task_end(
            flow_name="training-flow",
            task_name="train_one_fold",
            status="completed",
            duration_ms=5000,
        )
        data = json.loads(event.to_jsonl())
        assert data["event_type"] == "task_end"
        assert data["status"] == "completed"
        assert data["duration_ms"] == 5000

    def test_timestamp_is_utc_iso(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.flow_start(flow_name="test")
        data = json.loads(event.to_jsonl())
        # Must end with +00:00 or Z for UTC
        ts = data["timestamp"]
        assert "+00:00" in ts or ts.endswith("Z"), f"Timestamp not UTC: {ts}"

    def test_to_jsonl_is_single_line(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        event = FlowEvent.flow_start(flow_name="test-flow")
        jsonl = event.to_jsonl()
        assert "\n" not in jsonl, "JSONL must be a single line"

    def test_event_id_is_unique(self) -> None:
        from minivess.observability.structured_logging import FlowEvent

        e1 = FlowEvent.flow_start(flow_name="test")
        e2 = FlowEvent.flow_start(flow_name="test")
        d1 = json.loads(e1.to_jsonl())
        d2 = json.loads(e2.to_jsonl())
        assert d1["event_id"] != d2["event_id"]


class TestStructuredLogger:
    """StructuredLogger must emit JSONL events via Python logging."""

    def test_emit_logs_jsonl_line(self, caplog: pytest.LogCaptureFixture) -> None:
        from minivess.observability.structured_logging import StructuredLogger

        sl = StructuredLogger(flow_name="test-flow")
        with caplog.at_level(logging.INFO, logger="minivess.structured"):
            sl.log_flow_start()

        jsonl_records = [r for r in caplog.records if r.name == "minivess.structured"]
        assert len(jsonl_records) >= 1
        data = json.loads(jsonl_records[0].message)
        assert data["event_type"] == "flow_start"

    def test_log_task_start_and_end(self, caplog: pytest.LogCaptureFixture) -> None:
        from minivess.observability.structured_logging import StructuredLogger

        sl = StructuredLogger(flow_name="test-flow")
        with caplog.at_level(logging.INFO, logger="minivess.structured"):
            sl.log_task_start("my_task")
            sl.log_task_end("my_task", status="completed", duration_ms=100)

        jsonl_records = [r for r in caplog.records if r.name == "minivess.structured"]
        assert len(jsonl_records) >= 2
        start_data = json.loads(jsonl_records[0].message)
        end_data = json.loads(jsonl_records[1].message)
        assert start_data["event_type"] == "task_start"
        assert end_data["event_type"] == "task_end"

    def test_log_flow_end(self, caplog: pytest.LogCaptureFixture) -> None:
        from minivess.observability.structured_logging import StructuredLogger

        sl = StructuredLogger(flow_name="test-flow")
        with caplog.at_level(logging.INFO, logger="minivess.structured"):
            sl.log_flow_end(status="completed", duration_ms=9999)

        jsonl_records = [r for r in caplog.records if r.name == "minivess.structured"]
        assert len(jsonl_records) >= 1
        data = json.loads(jsonl_records[0].message)
        assert data["status"] == "completed"
        assert data["duration_ms"] == 9999
