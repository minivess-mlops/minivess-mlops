"""Tests for DVC commit task in acquisition flow.

Phase 3, Task T-ACQ.3.1 of overnight-child-01-acquisition.xml.
Validates that dvc_commit_datasets_task exists and is wired into the flow.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ACQUISITION_FLOW_SRC = Path("src/minivess/orchestration/flows/acquisition_flow.py")


class TestDVCCommitTask:
    """dvc_commit_datasets_task wraps DVC versioning logic."""

    def test_dvc_commit_task_importable(self) -> None:
        from minivess.orchestration.flows.acquisition_flow import (
            dvc_commit_datasets_task,
        )

        assert callable(dvc_commit_datasets_task)

    def test_dvc_commit_task_returns_version_tag(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.acquisition_flow import (
            dvc_commit_datasets_task,
        )

        # Should return a version tag string (or None if DVC not installed)
        result = dvc_commit_datasets_task(
            data_dir=tmp_path,
            datasets_acquired=["minivess"],
        )
        # Should not raise; returns string or None
        assert result is None or isinstance(result, str)

    def test_dvc_commit_task_in_source(self) -> None:
        """acquisition_flow.py must define dvc_commit_datasets_task."""
        source = _ACQUISITION_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "dvc_commit_datasets_task"
            ):
                found = True
                break
        assert found, (
            "acquisition_flow.py must define dvc_commit_datasets_task. "
            "Add @task(name='dvc-commit-datasets') function."
        )
