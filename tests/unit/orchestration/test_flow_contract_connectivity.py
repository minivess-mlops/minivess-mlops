"""Tests for FlowContract inter-flow connectivity — Issue #555.

Verifies:
- resolve_experiment_name() appends MINIVESS_DEBUG_SUFFIX from env
- FlowContract._debug_suffix is read from env at construction time
- FlowContract._resolve_experiment() applies the suffix
- FlowContract.log_flow_completion() writes flow_status=FLOW_COMPLETE (not 'completed')
- FlowContract.log_flow_completion() accepts checkpoint_dir kwarg and tags it
- FlowContract.find_fold_checkpoints() returns fold info from parent-run tags

Plan: docs/planning/prefect-flow-connectivity-execution-plan.xml Phase 0 (T0.1)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestResolveExperimentName:
    def test_no_suffix_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MINIVESS_DEBUG_SUFFIX", raising=False)
        from minivess.orchestration.constants import resolve_experiment_name

        assert resolve_experiment_name("minivess_training") == "minivess_training"

    def test_debug_suffix_appended_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.constants import resolve_experiment_name

        assert resolve_experiment_name("minivess_training") == "minivess_training_DEBUG"

    def test_custom_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_CI")
        from minivess.orchestration.constants import resolve_experiment_name

        assert resolve_experiment_name("minivess_data") == "minivess_data_CI"

    def test_empty_string_suffix_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "")
        from minivess.orchestration.constants import resolve_experiment_name

        assert resolve_experiment_name("minivess_analysis") == "minivess_analysis"


class TestFlowContractDebugSuffix:
    def test_debug_suffix_empty_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MINIVESS_DEBUG_SUFFIX", raising=False)
        from minivess.orchestration.flow_contract import FlowContract

        fc = FlowContract(tracking_uri=str(tmp_path / "mlruns"))
        assert fc._debug_suffix == ""

    def test_debug_suffix_set_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.flow_contract import FlowContract

        fc = FlowContract(tracking_uri=str(tmp_path / "mlruns"))
        assert fc._debug_suffix == "_DEBUG"

    def test_resolve_experiment_uses_suffix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_DEBUG")
        from minivess.orchestration.flow_contract import FlowContract

        fc = FlowContract(tracking_uri=str(tmp_path / "mlruns"))
        assert fc._resolve_experiment("minivess_training") == "minivess_training_DEBUG"

    def test_resolve_experiment_no_suffix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MINIVESS_DEBUG_SUFFIX", raising=False)
        from minivess.orchestration.flow_contract import FlowContract

        fc = FlowContract(tracking_uri=str(tmp_path / "mlruns"))
        assert fc._resolve_experiment("minivess_training") == "minivess_training"


class TestLogFlowCompletionWritesFLOW_COMPLETE:
    def test_log_completion_writes_FLOW_COMPLETE_tag(self, tmp_path: Path) -> None:
        """log_flow_completion must set flow_status=FLOW_COMPLETE (not 'completed')."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_exp_complete")
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        fc = FlowContract(tracking_uri=tracking_uri)
        fc.log_flow_completion(flow_name="test-flow", run_id=run_id)

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert run_data.data.tags["flow_status"] == "FLOW_COMPLETE", (
            f"Expected 'FLOW_COMPLETE', got '{run_data.data.tags.get('flow_status')}'"
        )

    def test_log_completion_with_checkpoint_dir(self, tmp_path: Path) -> None:
        """log_flow_completion must tag checkpoint_dir when provided."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_exp_ckpt")
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        ckpt_dir = tmp_path / "checkpoints" / "fold_0"
        fc = FlowContract(tracking_uri=tracking_uri)
        fc.log_flow_completion(
            flow_name="training-flow",
            run_id=run_id,
            checkpoint_dir=ckpt_dir,
        )

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert run_data.data.tags["checkpoint_dir"] == str(ckpt_dir)
        assert run_data.data.tags["flow_status"] == "FLOW_COMPLETE"

    def test_log_completion_without_checkpoint_dir_does_not_tag_it(
        self, tmp_path: Path
    ) -> None:
        """When checkpoint_dir is not given, no checkpoint_dir tag is written."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_exp_no_ckpt")
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        fc = FlowContract(tracking_uri=tracking_uri)
        fc.log_flow_completion(flow_name="data-flow", run_id=run_id)

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert "checkpoint_dir" not in run_data.data.tags


class TestFindFoldCheckpoints:
    def test_find_fold_checkpoints_returns_empty_when_no_runs(
        self, tmp_path: Path
    ) -> None:
        from minivess.orchestration.flow_contract import FlowContract

        fc = FlowContract(tracking_uri=str(tmp_path / "mlruns"))
        result = fc.find_fold_checkpoints(parent_run_id="nonexistent_id")
        assert result == []

    def test_find_fold_checkpoints_from_parent_run_tags(self, tmp_path: Path) -> None:
        """find_fold_checkpoints reads checkpoint_dir_fold_N tags from parent run."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("minivess_training")

        ckpt_dir_0 = tmp_path / "checkpoints" / "fold_0"
        ckpt_dir_1 = tmp_path / "checkpoints" / "fold_1"
        ckpt_dir_0.mkdir(parents=True)
        ckpt_dir_1.mkdir(parents=True)

        with mlflow.start_run(
            tags={
                "flow_name": "training-flow",
                "checkpoint_dir_fold_0": str(ckpt_dir_0),
                "checkpoint_dir_fold_1": str(ckpt_dir_1),
                "n_folds": "2",
            }
        ) as run:
            parent_run_id = run.info.run_id

        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_terminated(parent_run_id, status="FINISHED")

        fc = FlowContract(tracking_uri=tracking_uri)
        results = fc.find_fold_checkpoints(parent_run_id=parent_run_id)

        assert len(results) == 2
        dirs = {r["checkpoint_dir"] for r in results}
        assert ckpt_dir_0 in dirs
        assert ckpt_dir_1 in dirs
        assert all("fold_id" in r for r in results)
        assert all("run_id" in r for r in results)

    def test_find_fold_checkpoints_returns_correct_fold_ids(
        self, tmp_path: Path
    ) -> None:
        """find_fold_checkpoints must return correct fold_id values."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flow_contract import FlowContract

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("minivess_training_foldids")

        ckpt_dir_0 = tmp_path / "ckpt0"
        ckpt_dir_2 = tmp_path / "ckpt2"
        ckpt_dir_0.mkdir()
        ckpt_dir_2.mkdir()

        with mlflow.start_run(
            tags={
                "flow_name": "training-flow",
                "checkpoint_dir_fold_0": str(ckpt_dir_0),
                "checkpoint_dir_fold_2": str(ckpt_dir_2),
                "n_folds": "3",
            }
        ) as run:
            parent_run_id = run.info.run_id

        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_terminated(parent_run_id, status="FINISHED")

        fc = FlowContract(tracking_uri=tracking_uri)
        results = fc.find_fold_checkpoints(parent_run_id=parent_run_id)

        fold_ids = {r["fold_id"] for r in results}
        assert 0 in fold_ids
        assert 2 in fold_ids
