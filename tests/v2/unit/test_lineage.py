"""Tests for OpenLineage/Marquez data lineage tracking (Issue #44)."""

from __future__ import annotations

import datetime

import pytest
from openlineage.client.event_v2 import RunState

# ---------------------------------------------------------------------------
# T1: LineageEmitter creation
# ---------------------------------------------------------------------------


class TestLineageEmitter:
    """Test LineageEmitter wrapper around OpenLineage client."""

    def test_emitter_creates_with_defaults(self) -> None:
        """LineageEmitter should create with default namespace."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        assert emitter.namespace == "minivess"

    def test_emitter_creates_with_custom_namespace(self) -> None:
        """LineageEmitter should accept a custom namespace."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter(namespace="custom_ns")
        assert emitter.namespace == "custom_ns"

    def test_emitter_creates_with_url(self) -> None:
        """LineageEmitter should accept a Marquez API URL."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter(url="http://localhost:5002")
        assert emitter.url == "http://localhost:5002"

    def test_emitter_noop_without_url(self) -> None:
        """LineageEmitter without URL should work in no-op mode."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        # Should not raise — events are collected but not sent
        emitter.emit_start("test_job")


# ---------------------------------------------------------------------------
# T2: Job events
# ---------------------------------------------------------------------------


class TestJobEvents:
    """Test OpenLineage job event emission."""

    def test_emit_start_creates_event(self) -> None:
        """emit_start should create a START RunEvent."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start("preprocess")
        assert event.eventType == RunState.START
        assert event.job.name == "preprocess"
        assert event.job.namespace == "minivess"

    def test_emit_complete_creates_event(self) -> None:
        """emit_complete should create a COMPLETE RunEvent."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        start_event = emitter.emit_start("train")
        run_id = start_event.run.runId
        event = emitter.emit_complete("train", run_id=run_id)
        assert event.eventType == RunState.COMPLETE
        assert event.run.runId == run_id

    def test_emit_fail_creates_event(self) -> None:
        """emit_fail should create a FAIL RunEvent."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        start_event = emitter.emit_start("validate")
        run_id = start_event.run.runId
        event = emitter.emit_fail("validate", run_id=run_id)
        assert event.eventType == RunState.FAIL

    def test_events_have_valid_timestamps(self) -> None:
        """Events should have ISO-format UTC timestamps."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start("test_job")
        # Should be parseable as ISO datetime
        parsed = datetime.datetime.fromisoformat(event.eventTime)
        assert parsed.tzinfo is not None  # Must be timezone-aware


# ---------------------------------------------------------------------------
# T3: Dataset facets
# ---------------------------------------------------------------------------


class TestDatasetEvents:
    """Test dataset input/output tracking in lineage events."""

    def test_emit_with_input_datasets(self) -> None:
        """Events should include input datasets."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start(
            "preprocess",
            inputs=[{"namespace": "file", "name": "/data/raw/minivess"}],
        )
        assert len(event.inputs) == 1
        assert event.inputs[0].name == "/data/raw/minivess"

    def test_emit_with_output_datasets(self) -> None:
        """Events should include output datasets."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        start = emitter.emit_start("preprocess")
        event = emitter.emit_complete(
            "preprocess",
            run_id=start.run.runId,
            outputs=[{"namespace": "file", "name": "/data/processed/minivess"}],
        )
        assert len(event.outputs) == 1
        assert event.outputs[0].name == "/data/processed/minivess"

    def test_emit_with_multiple_datasets(self) -> None:
        """Events should support multiple input and output datasets."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start(
            "train",
            inputs=[
                {"namespace": "file", "name": "/data/processed/imagesTr"},
                {"namespace": "file", "name": "/data/processed/labelsTr"},
            ],
        )
        assert len(event.inputs) == 2


# ---------------------------------------------------------------------------
# T4: Pipeline run context manager
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Test pipeline_run context manager for full lineage traces."""

    def test_pipeline_run_emits_start_and_complete(self) -> None:
        """pipeline_run should emit START on enter and COMPLETE on exit."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run("full_pipeline") as run_id:
            assert run_id is not None
            assert isinstance(run_id, str)

        # Check events were collected
        assert len(emitter.events) >= 2
        assert emitter.events[0].eventType == RunState.START
        assert emitter.events[-1].eventType == RunState.COMPLETE

    def test_pipeline_run_emits_fail_on_exception(self) -> None:
        """pipeline_run should emit FAIL if an exception occurs."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with (
            pytest.raises(ValueError, match="test error"),
            emitter.pipeline_run("failing_pipeline"),
        ):
            msg = "test error"
            raise ValueError(msg)

        assert emitter.events[-1].eventType == RunState.FAIL

    def test_pipeline_run_preserves_run_id(self) -> None:
        """All events within a pipeline_run should share the same run_id."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run("consistent_run") as run_id:
            emitter.emit_start("substep", parent_run_id=run_id)

        run_ids = {
            e.run.runId for e in emitter.events if e.job.name == "consistent_run"
        }
        assert len(run_ids) == 1


# ---------------------------------------------------------------------------
# T5: Event history and querying
# ---------------------------------------------------------------------------


class TestEventHistory:
    """Test event collection and querying."""

    def test_events_collected_in_order(self) -> None:
        """Events should be collected in emission order."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        emitter.emit_start("step1")
        emitter.emit_start("step2")
        emitter.emit_start("step3")

        assert len(emitter.events) == 3
        assert [e.job.name for e in emitter.events] == ["step1", "step2", "step3"]

    def test_get_events_for_job(self) -> None:
        """Should be able to filter events by job name."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        emitter.emit_start("preprocess")
        emitter.emit_start("train")
        emitter.emit_start("preprocess")

        preprocess_events = emitter.get_events_for_job("preprocess")
        assert len(preprocess_events) == 2

    def test_producer_field_set(self) -> None:
        """Events should have producer set to minivess-mlops."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start("test")
        assert event.producer == "minivess-mlops"
