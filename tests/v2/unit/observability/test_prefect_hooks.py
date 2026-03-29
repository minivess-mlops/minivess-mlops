"""Tests for Prefect task timing hooks (Phase 5, Task 5.1)."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestCreateTaskTimingHooks:
    def test_returns_two_callables(self) -> None:
        from minivess.observability.prefect_hooks import create_task_timing_hooks

        on_complete, on_fail = create_task_timing_hooks()
        assert callable(on_complete)
        assert callable(on_fail)

    def test_on_completion_logs_info(self) -> None:
        from minivess.observability.prefect_hooks import create_task_timing_hooks

        on_complete, _ = create_task_timing_hooks()

        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task_run = MagicMock()
        mock_task_run.start_time = None
        mock_state = MagicMock()

        # Should not raise
        on_complete(mock_task, mock_task_run, mock_state)

    def test_on_failure_logs_error(self) -> None:
        from minivess.observability.prefect_hooks import create_task_timing_hooks

        _, on_fail = create_task_timing_hooks()

        mock_task = MagicMock()
        mock_task.name = "failed_task"
        mock_task_run = MagicMock()
        mock_task_run.start_time = None
        mock_state = MagicMock()
        mock_state.result.return_value = ValueError("test error")

        # Should not raise
        on_fail(mock_task, mock_task_run, mock_state)

    def test_handles_missing_task_name(self) -> None:
        from minivess.observability.prefect_hooks import create_task_timing_hooks

        on_complete, _ = create_task_timing_hooks()

        mock_task = MagicMock(spec=[])  # No name attribute
        mock_task_run = MagicMock()
        mock_task_run.start_time = None
        mock_state = MagicMock()

        # Should not raise, should use "unknown"
        on_complete(mock_task, mock_task_run, mock_state)
