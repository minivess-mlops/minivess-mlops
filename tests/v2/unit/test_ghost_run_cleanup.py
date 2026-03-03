"""Tests for ghost run cleanup (#275).

Covers:
- Signal handler registration
- Ghost run detection
- Run status marking on interruption
- Cleanup script logic
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock

import pytest


class TestSignalHandlerRegistration:
    """Test SIGTERM/SIGINT handler setup."""

    def test_register_signal_handlers(self) -> None:
        from minivess.observability.ghost_cleanup import register_graceful_shutdown

        mock_tracker = MagicMock()
        register_graceful_shutdown(mock_tracker, run_id="test-run-123")

        # Verify handlers are registered (not default)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not signal.SIG_DFL

    def test_handler_marks_run_failed(self) -> None:
        from minivess.observability.ghost_cleanup import _create_shutdown_handler

        mock_client = MagicMock()
        handler = _create_shutdown_handler(mock_client, run_id="test-run-123")

        # Call handler (simulating SIGTERM)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)

        mock_client.set_terminated.assert_called_once_with(
            "test-run-123", status="FAILED"
        )


class TestGhostRunDetection:
    """Test detection of orphaned RUNNING runs."""

    def test_find_ghost_runs(self) -> None:
        from minivess.observability.ghost_cleanup import find_ghost_runs

        mock_client = MagicMock()

        # Create mock runs: 1 RUNNING (ghost), 1 FINISHED
        mock_running = MagicMock()
        mock_running.info.run_id = "ghost-123"
        mock_running.info.status = "RUNNING"
        mock_running.info.start_time = 1000000  # epoch ms

        mock_client.search_runs.return_value = [mock_running]

        ghosts = find_ghost_runs(mock_client, experiment_ids=["0"])
        assert len(ghosts) == 1
        assert ghosts[0].info.run_id == "ghost-123"

    def test_no_ghost_runs(self) -> None:
        from minivess.observability.ghost_cleanup import find_ghost_runs

        mock_client = MagicMock()
        mock_client.search_runs.return_value = []

        ghosts = find_ghost_runs(mock_client, experiment_ids=["0"])
        assert len(ghosts) == 0


class TestGhostRunCleanup:
    """Test cleanup of ghost runs."""

    def test_cleanup_marks_as_failed(self) -> None:
        from minivess.observability.ghost_cleanup import cleanup_ghost_runs

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "ghost-456"
        mock_run.info.status = "RUNNING"

        result = cleanup_ghost_runs(mock_client, ghost_runs=[mock_run], dry_run=False)
        assert result["cleaned"] == 1
        mock_client.set_terminated.assert_called_once()

    def test_cleanup_dry_run(self) -> None:
        from minivess.observability.ghost_cleanup import cleanup_ghost_runs

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "ghost-789"
        mock_run.info.status = "RUNNING"

        result = cleanup_ghost_runs(mock_client, ghost_runs=[mock_run], dry_run=True)
        assert result["cleaned"] == 0
        assert result["would_clean"] == 1
        mock_client.set_terminated.assert_not_called()
