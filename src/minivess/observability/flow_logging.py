"""JSONL flow logging — Issue #503.

configure_flow_logging() adds a JSONL FileHandler to the named logger so that
training logs appear in three places simultaneously:
  1. Terminal (already works via PYTHONUNBUFFERED=1)
  2. Prefect UI (via PREFECT_LOGGING_EXTRA_LOGGERS env var)
  3. Durable /app/logs/train.jsonl on the named Docker volume

Call once at the top of training_flow() and hpo_flow() entrypoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


class _JsonlFormatter(logging.Formatter):
    """Format each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload)


def configure_flow_logging(
    logs_dir: Path,
    logger_name: str = "minivess",
) -> None:
    """Add a JSONL FileHandler to *logger_name*. Idempotent — safe to call multiple times.

    Args:
        logs_dir: Directory for the JSONL log file. Must be a :class:`pathlib.Path`.
                  The directory is created if it does not exist.
        logger_name: Name of the Python logger to attach the handler to.

    Raises:
        TypeError: If *logs_dir* is not a :class:`pathlib.Path`.
    """
    if not isinstance(logs_dir, Path):
        raise TypeError(
            f"logs_dir must be a pathlib.Path, got {type(logs_dir).__name__!r}. "
            "Never pass a str — use Path(logs_dir) at the call site."
        )

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train.jsonl"

    logger = logging.getLogger(logger_name)

    # Idempotency check — avoid duplicate handlers on repeated calls
    if any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    ):
        return

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JsonlFormatter())
    logger.addHandler(handler)
