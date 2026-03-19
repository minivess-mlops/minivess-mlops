"""Tests for OpenLineage v2 emission with HTTP transport.

PR-2 T2.5: LineageEmitter must send valid OpenLineage v2 events when
configured with a Marquez URL (via HttpTransport). Mock-tested only --
full Marquez deployment deferred to separate issue.

References:
  - Issue #799: OpenLineage/Marquez integration
  - docs/planning/pre-full-gcp-housekeeping-and-qa.xml PR id="2" T2.5
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock


class TestLineageEmitterHttpTransport:
    """LineageEmitter configures HttpTransport when MARQUEZ_URL is provided."""

    def test_emitter_creates_client_with_url(self) -> None:
        """When url is provided, LineageEmitter attempts to create an OpenLineageClient.

        Since no actual Marquez server is running, the constructor catches
        the connection error and falls back to local-only. We test that
        providing a URL does NOT raise.
        """
        from minivess.observability.lineage import LineageEmitter

        # Constructor should not raise even with unreachable URL
        emitter = LineageEmitter(url="http://localhost:59999")
        # Client may or may not be set depending on whether the transport
        # can be constructed. The key guarantee: no exception raised.
        assert emitter.url == "http://localhost:59999"

    def test_emitter_no_client_without_url(self) -> None:
        """Without url, LineageEmitter operates in local-only mode."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        assert emitter._client is None

    def test_emitter_emits_to_client(self) -> None:
        """When client exists, emit() calls client.emit()."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        emitter._client = MagicMock()

        emitter.emit_start("test-job")
        emitter._client.emit.assert_called_once()

    def test_emitter_graceful_on_client_failure(self) -> None:
        """If client.emit() raises, event is still stored locally."""
        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        emitter._client = MagicMock()
        emitter._client.emit.side_effect = ConnectionError("mock error")

        # Should not raise
        event = emitter.emit_start("test-job")
        assert event is not None
        assert len(emitter.events) == 1


class TestLineageEmitterFromEnv:
    """LineageEmitter.from_env() reads MARQUEZ_URL from environment."""

    def test_from_env_reads_marquez_url(self) -> None:
        """from_env() constructs emitter with MARQUEZ_URL."""
        from minivess.observability.lineage import LineageEmitter

        old_val = os.environ.get("MARQUEZ_URL")
        os.environ["MARQUEZ_URL"] = "http://localhost:5002"
        try:
            emitter = LineageEmitter.from_env()
            assert emitter.url == "http://localhost:5002"
        finally:
            if old_val is not None:
                os.environ["MARQUEZ_URL"] = old_val
            else:
                os.environ.pop("MARQUEZ_URL", None)

    def test_from_env_local_mode_without_env_var(self) -> None:
        """from_env() operates in local-only mode when MARQUEZ_URL is unset."""
        from minivess.observability.lineage import LineageEmitter

        old_val = os.environ.pop("MARQUEZ_URL", None)
        try:
            emitter = LineageEmitter.from_env()
            assert emitter.url is None
            assert emitter._client is None
        finally:
            if old_val is not None:
                os.environ["MARQUEZ_URL"] = old_val

    def test_from_env_empty_string_is_local_mode(self) -> None:
        """from_env() treats empty MARQUEZ_URL as local-only mode."""
        from minivess.observability.lineage import LineageEmitter

        old_val = os.environ.get("MARQUEZ_URL")
        os.environ["MARQUEZ_URL"] = ""
        try:
            emitter = LineageEmitter.from_env()
            assert emitter.url is None
            assert emitter._client is None
        finally:
            if old_val is not None:
                os.environ["MARQUEZ_URL"] = old_val
            else:
                os.environ.pop("MARQUEZ_URL", None)


class TestOpenLineageV2EventSchema:
    """Emitted events must conform to OpenLineage v2 schema."""

    def test_start_event_has_required_fields(self) -> None:
        from openlineage.client.event_v2 import RunState

        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        event = emitter.emit_start(
            "test-job",
            inputs=[{"namespace": "minivess", "name": "raw_data"}],
            outputs=[{"namespace": "minivess", "name": "checkpoints"}],
        )

        assert event.eventType == RunState.START
        assert event.run is not None
        assert event.run.runId is not None
        assert event.job is not None
        assert event.job.name == "test-job"
        assert event.job.namespace == "minivess"
        assert event.eventTime is not None
        assert event.producer == "minivess-mlops"

    def test_complete_event_has_required_fields(self) -> None:
        from openlineage.client.event_v2 import RunState

        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        start = emitter.emit_start("test-job")
        event = emitter.emit_complete("test-job", run_id=start.run.runId)

        assert event.eventType == RunState.COMPLETE
        assert event.run.runId == start.run.runId

    def test_fail_event_has_required_fields(self) -> None:
        from openlineage.client.event_v2 import RunState

        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        start = emitter.emit_start("test-job")
        event = emitter.emit_fail("test-job", run_id=start.run.runId)

        assert event.eventType == RunState.FAIL
        assert event.run.runId == start.run.runId

    def test_pipeline_run_context_manager_emits_start_and_complete(self) -> None:
        from openlineage.client.event_v2 import RunState

        from minivess.observability.lineage import LineageEmitter

        emitter = LineageEmitter()
        with emitter.pipeline_run(
            "test-flow",
            inputs=[{"namespace": "minivess", "name": "input_data"}],
            outputs=[{"namespace": "minivess", "name": "output_data"}],
        ) as run_id:
            assert run_id is not None

        events = emitter.get_events_for_job("test-flow")
        assert len(events) == 2
        assert events[0].eventType == RunState.START
        assert events[1].eventType == RunState.COMPLETE


class TestMarquezUrlInEnvExample:
    """MARQUEZ_URL must be in .env.example (Rule #22: single-source config)."""

    def test_marquez_url_in_env_example(self) -> None:
        from pathlib import Path

        env_example = Path(".env.example")
        assert env_example.exists(), ".env.example must exist"
        content = env_example.read_text(encoding="utf-8")
        assert "MARQUEZ_URL=" in content, (
            "MARQUEZ_URL must be defined in .env.example (Rule #22)"
        )
