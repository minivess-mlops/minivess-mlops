"""Tests for data_flow MLflow tag contract — Issue #556.

Verifies:
- run_data_flow() writes splits_path tag on the MLflow run
- run_data_flow() writes flow_status=FLOW_COMPLETE (not 'completed')
- run_data_flow() uses resolve_experiment_name("minivess_data") for experiment
- DataFlowResult.splits_path is not None after a successful run

Plan: docs/planning/prefect-flow-connectivity-execution-plan.xml Phase 1 (T1.1)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest


def _make_fake_pairs(n: int = 5) -> list[dict]:
    return [{"image": f"/img{i}.nii", "label": f"/lbl{i}.nii"} for i in range(n)]


class TestDataFlowWritesSplitsPathTag:
    def test_splits_path_tag_written_to_mlflow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_data_flow must write splits_path tag on the MLflow run."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.data_flow import run_data_flow

        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with (
            patch(
                "minivess.orchestration.flows.data_flow.discover_data_task",
                return_value=_make_fake_pairs(5),
            ),
            patch(
                "minivess.orchestration.flows.data_flow.dvc_pull_task",
                side_effect=Exception("dvc not installed"),
            ),
        ):
            result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

        assert result.splits_path is not None, "DataFlowResult.splits_path must be set"

        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        data_exp = next((e for e in experiments if "minivess_data" in e.name), None)
        assert data_exp is not None, "minivess_data experiment must exist in MLflow"

        runs = client.search_runs([data_exp.experiment_id])
        assert len(runs) >= 1, "At least one run must exist in data experiment"
        run = runs[0]
        assert "splits_path" in run.data.tags, (
            "data_flow must write splits_path tag so train_flow can discover splits"
        )
        assert run.data.tags["splits_path"] == str(result.splits_path)

    def test_flow_status_is_FLOW_COMPLETE(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_data_flow must write flow_status=FLOW_COMPLETE (not 'completed')."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.data_flow import run_data_flow

        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with (
            patch(
                "minivess.orchestration.flows.data_flow.discover_data_task",
                return_value=_make_fake_pairs(5),
            ),
            patch(
                "minivess.orchestration.flows.data_flow.dvc_pull_task",
                side_effect=Exception("dvc not installed"),
            ),
        ):
            run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        data_exp = next((e for e in experiments if "minivess_data" in e.name), None)
        assert data_exp is not None

        runs = client.search_runs([data_exp.experiment_id])
        assert len(runs) >= 1
        run = runs[0]
        assert run.data.tags.get("flow_status") == "FLOW_COMPLETE", (
            f"flow_status must be 'FLOW_COMPLETE', got '{run.data.tags.get('flow_status')}'"
        )


class TestDataFlowNoHardcodedExperimentName:
    def test_data_flow_imports_resolve_experiment_name(self) -> None:
        """data_flow must import resolve_experiment_name from constants."""
        import ast
        import inspect

        from minivess.orchestration.flows import data_flow as df_module

        source = inspect.getsource(df_module)
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "constants" in node.module
            ):
                names = [alias.name for alias in node.names]
                if "resolve_experiment_name" in names:
                    found = True
                    break
        assert found, (
            "data_flow.py must import resolve_experiment_name from "
            "minivess.orchestration.constants"
        )

    def test_data_flow_has_no_hardcoded_minivess_data(self) -> None:
        """data_flow source must not contain the bare 'minivess_data' string constant.

        The experiment name must be set via resolve_experiment_name(EXPERIMENT_DATA)
        so that MINIVESS_DEBUG_SUFFIX is respected.
        """
        import ast
        import inspect

        from minivess.orchestration.flows import data_flow as df_module

        source = inspect.getsource(df_module)
        tree = ast.parse(source)

        hardcoded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "minivess_data"
        ]
        assert len(hardcoded) == 0, (
            f"Found {len(hardcoded)} hardcoded 'minivess_data' constant(s) — "
            f"use resolve_experiment_name(EXPERIMENT_DATA) instead.\n"
            f"Line numbers: {[n.lineno for n in hardcoded]}"
        )
