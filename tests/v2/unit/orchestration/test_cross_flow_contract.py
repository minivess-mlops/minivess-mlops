"""Cross-flow FlowContract integration tests — verify write/read cycle.

The critical gap: log_flow_completion() and find_upstream_run() are NEVER tested
together. If a tag name changes in one but not the other, the entire 5-flow pipeline
silently breaks — biostatistics gets no upstream runs, post_training finds no
checkpoints, and analysis produces empty comparison tables.

These tests use file-based MLflow (tmp_path / "mlruns"), zero infrastructure.

Issue: integration-test-double-check.md Domain 4, Phase 1 (H4+H6).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from minivess.orchestration.flow_contract import FlowContract


@pytest.fixture()
def mlflow_dir(tmp_path: Path) -> Path:
    """Create an isolated MLflow tracking directory."""
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    return tracking_dir


@pytest.fixture()
def contract(mlflow_dir: Path) -> FlowContract:
    """FlowContract against isolated file-based MLflow."""
    uri = str(mlflow_dir)
    return FlowContract(tracking_uri=uri)


@pytest.fixture()
def client(mlflow_dir: Path) -> MlflowClient:
    """Raw MlflowClient for setup / verification."""
    return MlflowClient(tracking_uri=str(mlflow_dir))


def _create_finished_run(
    client: MlflowClient,
    experiment_name: str,
    tags: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Helper: create a FINISHED MLflow run with given tags and metrics."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)
    run_id = run.info.run_id

    if tags:
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    if metrics:
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)

    client.set_terminated(run_id, status="FINISHED")
    return run_id


# ---------------------------------------------------------------------------
# Test: log_flow_completion() → find_upstream_run() roundtrip
# ---------------------------------------------------------------------------


class TestFlowCompletionRoundtrip:
    """Verify that what log_flow_completion writes can be read by find_upstream_run."""

    def test_log_and_find_basic(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """log_flow_completion → find_upstream_run returns the same run."""
        experiment_name = "test-experiment"
        run_id = _create_finished_run(
            client, experiment_name, tags={"some_tag": "some_value"}
        )

        # Simulate: training flow logs completion
        contract.log_flow_completion(
            flow_name="training-flow",
            run_id=run_id,
        )

        # Simulate: analysis flow discovers upstream training run
        result = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="training-flow",
        )

        assert result is not None, "find_upstream_run should find the run"
        assert result["run_id"] == run_id
        assert result["tags"]["flow_name"] == "training-flow"

    def test_find_filters_by_flow_name(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """find_upstream_run must filter by flow_name, not just return most recent."""
        experiment_name = "test-filter"

        # Create training run FIRST
        train_id = _create_finished_run(client, experiment_name)
        contract.log_flow_completion(flow_name="training-flow", run_id=train_id)

        # Create post-training run SECOND (more recent)
        post_id = _create_finished_run(client, experiment_name)
        contract.log_flow_completion(flow_name="post-training-flow", run_id=post_id)

        # Finding training-flow should return train_id, not post_id
        result = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="training-flow",
        )
        assert result is not None
        assert result["run_id"] == train_id, (
            f"Expected training run {train_id}, got {result['run_id']}. "
            "find_upstream_run must filter by flow_name tag (#586)."
        )

    def test_find_returns_none_for_missing_flow(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """find_upstream_run returns None when no run matches the flow name."""
        experiment_name = "test-missing"
        run_id = _create_finished_run(client, experiment_name)
        contract.log_flow_completion(flow_name="training-flow", run_id=run_id)

        result = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="nonexistent-flow",
        )
        assert result is None

    def test_find_returns_none_for_missing_experiment(
        self, contract: FlowContract
    ) -> None:
        """find_upstream_run returns None for nonexistent experiment."""
        result = contract.find_upstream_run(
            experiment_name="does-not-exist",
            upstream_flow="training-flow",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test: fold checkpoint discovery roundtrip
# ---------------------------------------------------------------------------


class TestFoldCheckpointRoundtrip:
    """Verify checkpoint_dir_fold_N tags written by training can be read by post-training."""

    def test_write_and_read_fold_checkpoints(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """Tags written as checkpoint_dir_fold_N should be discoverable."""
        experiment_name = "test-checkpoints"
        run_id = _create_finished_run(client, experiment_name)

        # Simulate: training flow writes per-fold checkpoint tags
        client.set_tag(run_id, "checkpoint_dir_fold_0", "/app/checkpoints/fold_0")
        client.set_tag(run_id, "checkpoint_dir_fold_1", "/app/checkpoints/fold_1")
        client.set_tag(run_id, "checkpoint_dir_fold_2", "/app/checkpoints/fold_2")

        # Simulate: post-training flow discovers fold checkpoints
        folds = contract.find_fold_checkpoints(parent_run_id=run_id)

        assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"
        assert folds[0]["fold_id"] == 0
        assert folds[1]["fold_id"] == 1
        assert folds[2]["fold_id"] == 2
        assert folds[0]["checkpoint_dir"] == Path("/app/checkpoints/fold_0")

    def test_no_checkpoint_tags_returns_empty(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """Runs without checkpoint tags should return empty list."""
        run_id = _create_finished_run(client, "test-no-ckpts")
        folds = contract.find_fold_checkpoints(parent_run_id=run_id)
        assert folds == []


# ---------------------------------------------------------------------------
# Test: full pipeline chain (train → post_training → analysis)
# ---------------------------------------------------------------------------


class TestFullPipelineChain:
    """Simulate the 3-flow chain: train → post_training → analysis."""

    def test_three_flow_chain(
        self, contract: FlowContract, client: MlflowClient
    ) -> None:
        """Each downstream flow can discover its upstream flow's run."""
        experiment_name = "test-pipeline-chain"

        # Step 1: Training flow completes
        train_id = _create_finished_run(
            client,
            experiment_name,
            metrics={"val/dice": 0.85, "val/cldice": 0.72},
        )
        contract.log_flow_completion(
            flow_name="training-flow",
            run_id=train_id,
            checkpoint_dir=Path("/app/checkpoints"),
        )
        client.set_tag(train_id, "checkpoint_dir_fold_0", "/app/checkpoints/fold_0")

        # Step 2: Post-training discovers training run
        train_result = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="training-flow",
        )
        assert train_result is not None
        assert train_result["run_id"] == train_id

        # Post-training discovers fold checkpoints
        folds = contract.find_fold_checkpoints(parent_run_id=train_id)
        assert len(folds) == 1

        # Post-training completes
        post_id = _create_finished_run(client, experiment_name)
        contract.log_flow_completion(
            flow_name="post-training-flow",
            run_id=post_id,
        )

        # Step 3: Analysis discovers training run (not post-training)
        upstream = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="training-flow",
        )
        assert upstream is not None
        assert upstream["run_id"] == train_id, (
            "Analysis should find training-flow, not post-training-flow"
        )

        # Analysis can also find post-training if needed
        post_upstream = contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow="post-training-flow",
        )
        assert post_upstream is not None
        assert post_upstream["run_id"] == post_id


# ---------------------------------------------------------------------------
# Test: debug suffix experiment resolution
# ---------------------------------------------------------------------------


class TestDebugSuffixResolution:
    """FlowContract must respect MINIVESS_DEBUG_SUFFIX for experiment names."""

    def test_debug_suffix_appended(
        self, mlflow_dir: Path, client: MlflowClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With debug suffix, experiment name is base + suffix."""
        monkeypatch.setenv("MINIVESS_DEBUG_SUFFIX", "_debug_test")
        contract = FlowContract(tracking_uri=str(mlflow_dir))

        resolved = contract._resolve_experiment("my-experiment")
        assert resolved == "my-experiment_debug_test"

    def test_no_debug_suffix(
        self, mlflow_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without debug suffix, experiment name is unchanged."""
        monkeypatch.delenv("MINIVESS_DEBUG_SUFFIX", raising=False)
        contract = FlowContract(tracking_uri=str(mlflow_dir))

        resolved = contract._resolve_experiment("my-experiment")
        assert resolved == "my-experiment"


# ---------------------------------------------------------------------------
# Test: tag key consistency (the most dangerous drift vector)
# ---------------------------------------------------------------------------


class TestTagKeyConsistency:
    """Verify the tag keys used by log_flow_completion match find_upstream_run filters."""

    def test_flow_name_tag_key(self) -> None:
        """log_flow_completion must write 'flow_name' — the key find_upstream_run filters on."""
        # This is a contract test: if someone renames the tag, this catches it.
        # The actual key is hardcoded in flow_contract.py.
        source_path = Path("src/minivess/orchestration/flow_contract.py")
        source = source_path.read_text(encoding="utf-8")

        # Verify log_flow_completion writes "flow_name"
        assert '"flow_name"' in source or "'flow_name'" in source, (
            "flow_contract.py must contain the string 'flow_name' as a tag key"
        )

    def test_flow_complete_sentinel_exists(self) -> None:
        """_FLOW_COMPLETE_TAG sentinel must exist."""
        from minivess.orchestration.flow_contract import _FLOW_COMPLETE_TAG

        assert _FLOW_COMPLETE_TAG == "FLOW_COMPLETE"
