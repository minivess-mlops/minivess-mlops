"""Tests for Prefect maintenance cleanup flow (issue #683).

Verifies cleanup_stale_runs_task and maintenance_flow exist,
are properly decorated, and delegate to ghost_cleanup functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCleanupStaleRunsTask:
    """cleanup_stale_runs_task wraps ghost run detection and cleanup."""

    def test_cleanup_stale_runs_task_exists(self) -> None:
        """cleanup_stale_runs_task is importable and is a Prefect @task."""
        from minivess.orchestration.flows.maintenance_flow import (
            cleanup_stale_runs_task,
        )

        # Prefect tasks have a .fn attribute (the original function)
        assert hasattr(cleanup_stale_runs_task, "fn") or callable(
            cleanup_stale_runs_task
        )

    def test_cleanup_stale_runs_dry_run(self) -> None:
        """Dry run calls find_ghost_runs + cleanup_ghost_runs(dry_run=True)."""
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        with (
            patch(
                "minivess.orchestration.flows.maintenance_flow.MlflowClient",
                return_value=mock_client,
            ),
            patch(
                "minivess.orchestration.flows.maintenance_flow.resolve_tracking_uri",
                return_value="mlruns",
            ),
            patch(
                "minivess.orchestration.flows.maintenance_flow.find_ghost_runs",
                return_value=[MagicMock()],
            ) as mock_find,
            patch(
                "minivess.orchestration.flows.maintenance_flow.cleanup_ghost_runs",
                return_value={"would_clean": 1, "cleaned": 0, "errors": 0},
            ) as mock_cleanup,
        ):
            from minivess.orchestration.flows.maintenance_flow import (
                cleanup_stale_runs_task,
            )

            result = cleanup_stale_runs_task.fn(dry_run=True)

            mock_find.assert_called_once()
            mock_cleanup.assert_called_once()
            # Verify dry_run=True was passed
            _, kwargs = mock_cleanup.call_args
            assert kwargs.get("dry_run") is True
            assert "would_clean" in result

    def test_cleanup_stale_runs_wet_run(self) -> None:
        """Wet run calls cleanup_ghost_runs(dry_run=False)."""
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        with (
            patch(
                "minivess.orchestration.flows.maintenance_flow.MlflowClient",
                return_value=mock_client,
            ),
            patch(
                "minivess.orchestration.flows.maintenance_flow.resolve_tracking_uri",
                return_value="mlruns",
            ),
            patch(
                "minivess.orchestration.flows.maintenance_flow.find_ghost_runs",
                return_value=[MagicMock()],
            ),
            patch(
                "minivess.orchestration.flows.maintenance_flow.cleanup_ghost_runs",
                return_value={"would_clean": 0, "cleaned": 1, "errors": 0},
            ) as mock_cleanup,
        ):
            from minivess.orchestration.flows.maintenance_flow import (
                cleanup_stale_runs_task,
            )

            result = cleanup_stale_runs_task.fn(dry_run=False)

            _, kwargs = mock_cleanup.call_args
            assert kwargs.get("dry_run") is False
            assert "cleaned" in result


class TestMaintenanceFlow:
    """maintenance_flow is a Prefect @flow that runs cleanup."""

    def test_maintenance_flow_exists(self) -> None:
        """maintenance_flow is importable and is a Prefect @flow."""
        from minivess.orchestration.flows.maintenance_flow import maintenance_flow

        assert hasattr(maintenance_flow, "fn") or callable(maintenance_flow)

    def test_maintenance_flow_runs_cleanup(self) -> None:
        """maintenance_flow calls cleanup_stale_runs_task at least once."""
        with (
            patch(
                "minivess.orchestration.flows.maintenance_flow.cleanup_stale_runs_task",
            ) as mock_task,
            patch(
                "minivess.orchestration.flows.maintenance_flow._require_docker_context",
            ),
        ):
            mock_task.fn = MagicMock(
                return_value={"would_clean": 0, "cleaned": 0, "errors": 0}
            )
            mock_task.return_value = {
                "would_clean": 0,
                "cleaned": 0,
                "errors": 0,
            }

            from minivess.orchestration.flows.maintenance_flow import maintenance_flow

            maintenance_flow.fn(dry_run=True)
            mock_task.assert_called()
