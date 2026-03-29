"""Tests for structured epoch progress logging (Phase 3, Task 3.1).

Verifies log_epoch_complete(), log_training_start(), log_training_end()
write JSONL events with force-flush and update heartbeat.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


class TestLogEpochComplete:
    """StructuredEventLogger.log_epoch_complete() emits epoch JSONL event."""

    def test_method_exists(self) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=None)
        assert hasattr(logger, "log_epoch_complete")

    def test_writes_jsonl_event(self, tmp_path: Path) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_epoch_complete(
            epoch=5,
            max_epochs=20,
            train_loss=0.34,
            val_loss=0.28,
            val_dice=0.81,
            lr=0.0001,
            epoch_wall_s=180.0,
            gpu_util_pct=94,
            vram_used_mb=2416,
        )

        events_path = tmp_path / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        event = json.loads(lines[-1])
        assert event["event_type"] == "epoch_complete"
        assert event["epoch"] == 5
        assert event["max_epochs"] == 20
        assert event["train_loss"] == 0.34
        assert event["val_dice"] == 0.81

    def test_updates_heartbeat(self, tmp_path: Path) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_epoch_complete(
            epoch=10, max_epochs=20, train_loss=0.25,
            val_loss=0.22, val_dice=0.85, lr=0.00005,
            epoch_wall_s=175.0, gpu_util_pct=96, vram_used_mb=2400,
        )

        hb = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
        assert hb["epoch"] == 10
        assert hb["max_epochs"] == 20
        assert "eta_s" in hb

    def test_noop_when_no_output_dir(self) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=None)
        # Should not raise
        logger.log_epoch_complete(
            epoch=1, max_epochs=10, train_loss=0.5,
            val_loss=0.4, val_dice=0.6, lr=0.001,
            epoch_wall_s=60.0, gpu_util_pct=90, vram_used_mb=2000,
        )


class TestLogTrainingStartEnd:
    """Start/end events for run-level tracking."""

    def test_log_training_start(self, tmp_path: Path) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_training_start(
            model_family="dynunet",
            loss_name="dice_ce",
            num_folds=3,
            max_epochs=20,
        )

        events = tmp_path / "events.jsonl"
        line = json.loads(events.read_text(encoding="utf-8").strip().split("\n")[-1])
        assert line["event_type"] == "training_start"
        assert line["model_family"] == "dynunet"

    def test_log_training_end(self, tmp_path: Path) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_training_end(
            status="completed",
            total_wall_s=3600.0,
            folds_completed=3,
        )

        events = tmp_path / "events.jsonl"
        line = json.loads(events.read_text(encoding="utf-8").strip().split("\n")[-1])
        assert line["event_type"] == "training_end"
        assert line["status"] == "completed"


class TestForceFlush:
    """Epoch logging must force-flush stdout for Docker log visibility."""

    @patch.object(sys.stdout, "flush")
    @patch.object(sys.stderr, "flush")
    def test_epoch_flushes_stdout(self, mock_stderr_flush, mock_stdout_flush, tmp_path: Path) -> None:
        from minivess.observability.structured_logging import StructuredEventLogger

        logger = StructuredEventLogger(output_dir=tmp_path)
        logger.log_epoch_complete(
            epoch=1, max_epochs=10, train_loss=0.5,
            val_loss=0.4, val_dice=0.6, lr=0.001,
            epoch_wall_s=60.0, gpu_util_pct=90, vram_used_mb=2000,
        )
        mock_stdout_flush.assert_called()
        mock_stderr_flush.assert_called()
