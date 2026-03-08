"""Tests for configure_flow_logging() — Issue #503.

Tests verify:
- JSONL FileHandler is added to the named logger
- Output is valid JSON (no regex — CLAUDE.md Rule #16)
- Function is idempotent (no duplicate handlers)
- Rejects str path (requires pathlib.Path)

Plan: docs/planning/overnight-child-prefect-docker.xml Phase 2 (T-PD.2.1)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_configure_flow_logging_adds_jsonl_handler(tmp_path: Path) -> None:
    """configure_flow_logging() must add a FileHandler writing to logs_dir/train.jsonl."""
    from minivess.observability.flow_logging import configure_flow_logging

    logger_name = "minivess.test_adds_handler"
    configure_flow_logging(logs_dir=tmp_path, logger_name=logger_name)

    logger = logging.getLogger(logger_name)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    expected = str(tmp_path / "train.jsonl")
    assert file_handlers[0].baseFilename == expected


def test_jsonl_output_is_parseable(tmp_path: Path) -> None:
    """Each line written by the handler must be valid JSON (no regex)."""
    from minivess.observability.flow_logging import configure_flow_logging

    logger_name = "minivess.test_jsonl_parseable"
    configure_flow_logging(logs_dir=tmp_path, logger_name=logger_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.info("test message from unit test")

    # Flush handlers
    for h in logger.handlers:
        h.flush()

    log_file = tmp_path / "train.jsonl"
    assert log_file.exists(), "train.jsonl must be created"
    lines = [
        ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) >= 1
    for line in lines:
        parsed = json.loads(line)  # No regex — CLAUDE.md Rule #16
        assert "level" in parsed
        assert "msg" in parsed
        assert "ts" in parsed


def test_flow_logging_does_not_add_duplicate_handlers(tmp_path: Path) -> None:
    """Calling configure_flow_logging twice must not add a second handler (idempotent)."""
    from minivess.observability.flow_logging import configure_flow_logging

    logger_name = "minivess.test_idempotent"
    configure_flow_logging(logs_dir=tmp_path, logger_name=logger_name)
    configure_flow_logging(logs_dir=tmp_path, logger_name=logger_name)

    logger = logging.getLogger(logger_name)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, (
        f"Expected 1 FileHandler after 2 calls, got {len(file_handlers)}"
    )


def test_configure_flow_logging_requires_path_not_string(tmp_path: Path) -> None:
    """configure_flow_logging must reject str (requires pathlib.Path)."""
    import pytest

    from minivess.observability.flow_logging import configure_flow_logging

    with pytest.raises(TypeError):
        configure_flow_logging(logs_dir=str(tmp_path))  # type: ignore[arg-type]
