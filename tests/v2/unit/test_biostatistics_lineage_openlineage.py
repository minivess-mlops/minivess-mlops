"""Tests for OpenLineage emission from biostatistics flow.

Validates that the biostatistics flow can emit OpenLineage events
alongside its own lineage manifest.
"""

from __future__ import annotations

from openlineage.client.event_v2 import RunState


class TestBiostatisticsFlowEmitsLineage:
    """T9: biostatistics flow emits OpenLineage events."""

    def test_biostatistics_flow_emits_lineage(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "biostatistics-flow",
            inputs=[
                {"namespace": "minivess", "name": "mlflow_runs"},
                {"namespace": "minivess", "name": "duckdb"},
            ],
            outputs=[
                {"namespace": "minivess", "name": "figures"},
                {"namespace": "minivess", "name": "tables"},
                {"namespace": "minivess", "name": "lineage_manifest"},
            ],
        ) as _run_id:
            pass  # Simulate successful flow

        events = emitter.get_events_for_job("biostatistics-flow")
        assert len(events) == 2  # START + COMPLETE
        start = [e for e in events if e.eventType == RunState.START][0]
        complete = [e for e in events if e.eventType == RunState.COMPLETE][0]
        assert len(start.inputs) == 2
        assert len(complete.outputs) == 3


class TestBiostatisticsLineageOpenLineageBridge:
    """T9: Lineage manifest bridges to OpenLineage standard."""

    def test_biostatistics_lineage_openlineage_bridge(self) -> None:
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()

        # Emit events for the biostatistics flow
        start_event = emitter.emit_start(
            "biostatistics-flow",
            inputs=[{"namespace": "minivess", "name": "mlflow_experiments"}],
            outputs=[{"namespace": "minivess", "name": "anova_results"}],
        )
        run_id = start_event.run.runId

        emitter.emit_complete(
            "biostatistics-flow",
            run_id=run_id,
            outputs=[{"namespace": "minivess", "name": "anova_results"}],
        )

        # Verify events are accessible
        events = emitter.get_events_for_job("biostatistics-flow")
        assert len(events) == 2
        assert events[0].job.name == "biostatistics-flow"
        assert events[0].job.namespace == "minivess"
