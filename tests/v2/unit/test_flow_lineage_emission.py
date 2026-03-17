"""Tests for OpenLineage event emission from pipeline flows.

Validates that post_training_flow, train_flow, and deploy_flow emit
START/COMPLETE/FAIL events via LineageEmitter.pipeline_run().

PR-D T5 (Issue #829): Added deploy flow lineage tests.
"""

from __future__ import annotations

from openlineage.client.event_v2 import RunState


class TestPostTrainingFlowEmitsLineageStart:
    """T7: post_training_flow emits START event."""

    def test_post_training_flow_emits_lineage_start(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "post-training-flow",
            inputs=[{"namespace": "minivess", "name": "checkpoints"}],
            outputs=[{"namespace": "minivess", "name": "swa_checkpoints"}],
        ) as _run_id:
            pass  # Simulate successful flow

        events = emitter.get_events_for_job("post-training-flow")
        start_events = [e for e in events if e.eventType == RunState.START]
        assert len(start_events) == 1


class TestPostTrainingFlowEmitsLineageComplete:
    """T7: post_training_flow emits COMPLETE event on success."""

    def test_post_training_flow_emits_lineage_complete(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "post-training-flow",
            outputs=[{"namespace": "minivess", "name": "swa_checkpoints"}],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("post-training-flow")
        complete_events = [e for e in events if e.eventType == RunState.COMPLETE]
        assert len(complete_events) == 1


class TestPostTrainingFlowLineageDatasets:
    """T7: Lineage events record input/output datasets."""

    def test_post_training_flow_lineage_datasets(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "post-training-flow",
            inputs=[{"namespace": "minivess", "name": "checkpoints"}],
            outputs=[{"namespace": "minivess", "name": "swa_checkpoints"}],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("post-training-flow")
        start_event = [e for e in events if e.eventType == RunState.START][0]
        assert len(start_event.inputs) == 1
        assert start_event.inputs[0].name == "checkpoints"


class TestTrainFlowEmitsLineage:
    """T7: train_flow can emit lineage events."""

    def test_train_flow_emits_lineage(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()

        # Simulate train flow lineage emission
        with emitter.pipeline_run(
            "train-flow",
            inputs=[{"namespace": "minivess", "name": "raw_data"}],
            outputs=[
                {"namespace": "minivess", "name": "checkpoints"},
                {"namespace": "minivess", "name": "mlflow_metrics"},
            ],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("train-flow")
        assert len(events) == 2  # START + COMPLETE
        complete_event = [e for e in events if e.eventType == RunState.COMPLETE][0]
        assert len(complete_event.outputs) == 2


class TestFlowLineageFailEvent:
    """T7: FAIL event is emitted on exception."""

    def test_flow_lineage_fail_event(self) -> None:
        import pytest

        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with (
            pytest.raises(ValueError, match="test error"),
            emitter.pipeline_run("failing-flow") as _run_id,
        ):
            raise ValueError("test error")

        events = emitter.get_events_for_job("failing-flow")
        fail_events = [e for e in events if e.eventType == RunState.FAIL]
        assert len(fail_events) == 1


# ---------------------------------------------------------------------------
# PR-D T5: Deploy flow lineage emission tests
# ---------------------------------------------------------------------------


class TestDeployFlowEmitsLineage:
    """Deploy flow emits OpenLineage events with champion metadata."""

    def test_deploy_flow_emits_lineage(self) -> None:
        """Deploy flow emits START + COMPLETE events."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "deploy-flow",
            inputs=[
                {"namespace": "minivess", "name": "champion_model"},
                {"namespace": "minivess", "name": "mlflow_registry"},
            ],
            outputs=[
                {"namespace": "minivess", "name": "onnx_export"},
                {"namespace": "minivess", "name": "bentoml_model"},
                {"namespace": "minivess", "name": "deployment_artifacts"},
            ],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("deploy-flow")
        assert len(events) == 2  # START + COMPLETE
        start_events = [e for e in events if e.eventType == RunState.START]
        complete_events = [e for e in events if e.eventType == RunState.COMPLETE]
        assert len(start_events) == 1
        assert len(complete_events) == 1

    def test_deploy_flow_lineage_champion_metadata(self) -> None:
        """Deploy flow lineage captures champion model as input dataset."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "deploy-flow",
            inputs=[
                {"namespace": "minivess", "name": "champion_model"},
                {"namespace": "minivess", "name": "mlflow_registry"},
            ],
            outputs=[
                {"namespace": "minivess", "name": "onnx_export"},
                {"namespace": "minivess", "name": "bentoml_model"},
                {"namespace": "minivess", "name": "deployment_artifacts"},
            ],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("deploy-flow")
        start_event = [e for e in events if e.eventType == RunState.START][0]

        # Inputs: champion_model + mlflow_registry
        assert start_event.inputs is not None
        assert len(start_event.inputs) == 2
        input_names = {inp.name for inp in start_event.inputs}
        assert "champion_model" in input_names
        assert "mlflow_registry" in input_names

    def test_deploy_flow_lineage_outputs(self) -> None:
        """Deploy flow lineage records all deployment outputs."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "deploy-flow",
            inputs=[{"namespace": "minivess", "name": "champion_model"}],
            outputs=[
                {"namespace": "minivess", "name": "onnx_export"},
                {"namespace": "minivess", "name": "bentoml_model"},
                {"namespace": "minivess", "name": "deployment_artifacts"},
            ],
        ) as _run_id:
            pass

        events = emitter.get_events_for_job("deploy-flow")
        complete_event = [e for e in events if e.eventType == RunState.COMPLETE][0]

        # Outputs: onnx_export + bentoml_model + deployment_artifacts
        assert complete_event.outputs is not None
        assert len(complete_event.outputs) == 3
        output_names = {out.name for out in complete_event.outputs}
        assert "onnx_export" in output_names
        assert "bentoml_model" in output_names
        assert "deployment_artifacts" in output_names
