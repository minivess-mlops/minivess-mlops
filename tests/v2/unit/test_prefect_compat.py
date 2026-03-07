from __future__ import annotations


class TestOrchestrationPackageExports:
    """Verify that minivess.orchestration re-exports the real Prefect symbols."""

    def test_exports_flow(self):
        from prefect import flow as real_flow

        from minivess.orchestration import flow

        assert flow is real_flow

    def test_exports_task(self):
        from prefect import task as real_task

        from minivess.orchestration import task

        assert task is real_task

    def test_exports_get_run_logger(self):
        from prefect import get_run_logger as real_logger

        from minivess.orchestration import get_run_logger

        assert get_run_logger is real_logger

    def test_prefect_available_is_true(self):
        from minivess.orchestration import PREFECT_AVAILABLE

        assert PREFECT_AVAILABLE is True

    def test_exports_get_work_pool(self):
        from minivess.orchestration import get_work_pool
        from minivess.orchestration.deployments import (
            get_work_pool as real_get_work_pool,
        )

        assert get_work_pool is real_get_work_pool
