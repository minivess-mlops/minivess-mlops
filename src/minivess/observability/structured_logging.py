"""Structured JSONL logging for Prefect flow/task/agent transitions.

Emits machine-parseable JSONL events via Python's logging module.
Events are logged to the ``minivess.structured`` logger at INFO level,
which Prefect captures and displays in its UI.

Usage::

    sl = StructuredLogger(flow_name="training-flow")
    sl.log_flow_start()
    sl.log_task_start("train_one_fold")
    sl.log_task_end("train_one_fold", status="completed", duration_ms=5000)
    sl.log_flow_end(status="completed", duration_ms=60000)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("minivess.structured")


@dataclass(frozen=True)
class FlowEvent:
    """A single structured event for flow/task/agent transitions."""

    event_type: str
    flow_name: str
    timestamp: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to a single-line JSON string."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def flow_start(cls, flow_name: str, **kwargs: Any) -> FlowEvent:
        return cls(
            event_type="flow_start",
            flow_name=flow_name,
            timestamp=datetime.now(UTC).isoformat(),
            **kwargs,
        )

    @classmethod
    def flow_end(
        cls,
        flow_name: str,
        status: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> FlowEvent:
        return cls(
            event_type="flow_end",
            flow_name=flow_name,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            duration_ms=duration_ms,
            **kwargs,
        )

    @classmethod
    def task_start(cls, flow_name: str, task_name: str, **kwargs: Any) -> FlowEvent:
        return cls(
            event_type="task_start",
            flow_name=flow_name,
            task_name=task_name,
            timestamp=datetime.now(UTC).isoformat(),
            **kwargs,
        )

    @classmethod
    def task_end(
        cls,
        flow_name: str,
        task_name: str,
        status: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> FlowEvent:
        return cls(
            event_type="task_end",
            flow_name=flow_name,
            task_name=task_name,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            duration_ms=duration_ms,
            **kwargs,
        )


class StructuredEventLogger:
    """Disk-based JSONL event logger for training monitoring.

    Writes training events (epoch_start, epoch_end, checkpoint_saved) to
    events.jsonl and a heartbeat.json file for quick status checks.

    This SUPPLEMENTS MLflow tracking — it does NOT replace it. The JSONL
    output is for LLM-parseable monitoring of cloud GPU jobs where
    ``sky jobs logs`` is slow/blocking/unstructured.

    Parameters
    ----------
    output_dir:
        Directory to write events.jsonl and heartbeat.json.
        If None, all operations are no-ops (graceful degradation).
    """

    def __init__(self, output_dir: Path | None) -> None:
        self._output_dir = output_dir
        self._events_path = Path(output_dir) / "events.jsonl" if output_dir else None
        self._heartbeat_path = Path(output_dir) / "heartbeat.json" if output_dir else None

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a structured event to events.jsonl."""
        if not self._events_path:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def update_heartbeat(self, **kwargs: Any) -> None:
        """Write a heartbeat.json file for quick status checks."""
        if not self._heartbeat_path:
            return
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "training",
            **kwargs,
        }
        self._heartbeat_path.write_text(
            json.dumps(data, separators=(",", ":")), encoding="utf-8"
        )

    def log_epoch_complete(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        lr: float,
        epoch_wall_s: float,
        gpu_util_pct: int = 0,
        vram_used_mb: int = 0,
    ) -> None:
        """Log epoch completion event + update heartbeat + force-flush stdout.

        This is the primary training progress signal visible in Docker logs,
        Prefect UI, and events.jsonl.
        """
        # Compute ETA from average epoch time
        remaining = max_epochs - epoch
        eta_s = remaining * epoch_wall_s if epoch > 0 else 0.0

        self.log_event("epoch_complete", {
            "epoch": epoch,
            "max_epochs": max_epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": lr,
            "epoch_wall_s": epoch_wall_s,
            "gpu_util_pct": gpu_util_pct,
            "vram_used_mb": vram_used_mb,
            "eta_s": eta_s,
        })

        self.update_heartbeat(
            epoch=epoch,
            max_epochs=max_epochs,
            train_loss=train_loss,
            val_dice=val_dice,
            eta_s=eta_s,
        )

        # Force-flush for Docker log visibility (defense-in-depth over PYTHONUNBUFFERED=1)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

    def log_training_start(
        self,
        *,
        model_family: str,
        loss_name: str,
        num_folds: int,
        max_epochs: int,
    ) -> None:
        """Log training start event."""
        self.log_event("training_start", {
            "model_family": model_family,
            "loss_name": loss_name,
            "num_folds": num_folds,
            "max_epochs": max_epochs,
        })

    def log_training_end(
        self,
        *,
        status: str,
        total_wall_s: float,
        folds_completed: int,
    ) -> None:
        """Log training end event."""
        self.log_event("training_end", {
            "status": status,
            "total_wall_s": total_wall_s,
            "folds_completed": folds_completed,
        })


class StructuredLogger:
    """Convenience wrapper that emits FlowEvents via Python logging."""

    def __init__(self, flow_name: str) -> None:
        self.flow_name = flow_name

    def _emit(self, event: FlowEvent) -> None:
        logger.info(event.to_jsonl())

    def log_flow_start(self) -> None:
        self._emit(FlowEvent.flow_start(self.flow_name))

    def log_flow_end(self, status: str, duration_ms: int) -> None:
        self._emit(FlowEvent.flow_end(self.flow_name, status, duration_ms))

    def log_task_start(self, task_name: str) -> None:
        self._emit(FlowEvent.task_start(self.flow_name, task_name))

    def log_task_end(self, task_name: str, status: str, duration_ms: int) -> None:
        self._emit(FlowEvent.task_end(self.flow_name, task_name, status, duration_ms))
