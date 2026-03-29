"""Tests for StructuredEventLogger — JSONL + heartbeat for training monitoring.

Verifies that training events are written as valid JSONL lines and that
heartbeat.json is updated atomically.
"""

from __future__ import annotations

import json
from pathlib import Path


class TestStructuredEventLogger:
    """StructuredEventLogger must write valid JSONL and heartbeat."""

    def test_event_logger_writes_jsonl(self, tmp_path: Path) -> None:
        """log_event must append a valid JSONL line to events.jsonl."""
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_event("epoch_end", {"epoch": 5, "val_dice": 0.82, "vram_gb": 9.1})

        events_path = tmp_path / "events.jsonl"
        assert events_path.exists(), "events.jsonl not created"

        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert "timestamp" in event
        assert event["event_type"] == "epoch_end"
        assert event["epoch"] == 5
        assert event["val_dice"] == 0.82

    def test_heartbeat_writes_json(self, tmp_path: Path) -> None:
        """update_heartbeat must write valid heartbeat.json."""
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.update_heartbeat(epoch=5, loss=0.45, vram_gb=9.2)

        heartbeat_path = tmp_path / "heartbeat.json"
        assert heartbeat_path.exists(), "heartbeat.json not created"

        data = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        assert data["epoch"] == 5
        assert data["loss"] == 0.45
        assert data["status"] == "training"
        assert "timestamp" in data

    def test_event_logger_is_noop_without_dir(self) -> None:
        """StructuredEventLogger(output_dir=None) must not crash."""
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=None)
        # These should all be no-ops, no crash
        logger.log_event("epoch_end", {"epoch": 1})
        logger.update_heartbeat(epoch=1, loss=0.5)

    def test_multiple_events_append(self, tmp_path: Path) -> None:
        """Multiple log_event calls must append (not overwrite)."""
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_event("epoch_start", {"epoch": 1})
        logger.log_event("epoch_end", {"epoch": 1, "loss": 0.9})
        logger.log_event("epoch_start", {"epoch": 2})

        events_path = tmp_path / "events.jsonl"
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        types = [json.loads(line)["event_type"] for line in lines]
        assert types == ["epoch_start", "epoch_end", "epoch_start"]
