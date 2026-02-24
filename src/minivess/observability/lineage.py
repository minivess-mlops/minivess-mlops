"""OpenLineage/Marquez data lineage tracking.

Emits OpenLineage events at pipeline stage boundaries for
IEC 62304 traceability. Falls back to local event collection
when Marquez is not available.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from openlineage.client.event_v2 import (
    InputDataset,
    Job,
    OutputDataset,
    Run,
    RunEvent,
    RunState,
)
from openlineage.client.uuid import generate_new_uuid

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

_PRODUCER = "minivess-mlops"


class LineageEmitter:
    """OpenLineage event emitter for pipeline lineage tracking.

    Collects OpenLineage RunEvents locally. When a Marquez URL is
    provided, events are also sent to the Marquez API.

    Parameters
    ----------
    namespace:
        OpenLineage namespace (default: ``"minivess"``).
    url:
        Marquez API URL. If ``None``, operates in local-only mode.
    """

    def __init__(
        self,
        namespace: str = "minivess",
        url: str | None = None,
    ) -> None:
        self.namespace = namespace
        self.url = url
        self.events: list[RunEvent] = []
        self._client = None

        if url is not None:
            try:
                from openlineage.client import OpenLineageClient
                from openlineage.client.transport.http import (
                    HttpConfig,
                    HttpTransport,
                )

                transport = HttpTransport(HttpConfig(url=url))
                self._client = OpenLineageClient(transport=transport)
            except Exception:
                logger.warning(
                    "Could not connect to Marquez at %s; operating in local-only mode",
                    url,
                )

    def _now_iso(self) -> str:
        """Return current UTC time in ISO format."""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    def _build_datasets(
        self,
        datasets: list[dict[str, str]] | None,
        *,
        is_input: bool,
    ) -> list[InputDataset] | list[OutputDataset]:
        """Convert dataset dicts to OpenLineage dataset objects."""
        if not datasets:
            return []
        cls = InputDataset if is_input else OutputDataset
        return [cls(namespace=d["namespace"], name=d["name"]) for d in datasets]

    def _emit(self, event: RunEvent) -> RunEvent:
        """Store event locally and optionally send to Marquez."""
        self.events.append(event)
        if self._client is not None:
            try:
                self._client.emit(event)
            except Exception:
                logger.warning("Failed to emit event to Marquez")
        return event

    def emit_start(
        self,
        job_name: str,
        *,
        run_id: str | None = None,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
        parent_run_id: str | None = None,
    ) -> RunEvent:
        """Emit a START event for a pipeline job.

        Parameters
        ----------
        job_name:
            Name of the pipeline stage.
        run_id:
            Optional run UUID. Generated if not provided.
        inputs:
            Input datasets as ``[{"namespace": ..., "name": ...}]``.
        outputs:
            Output datasets as ``[{"namespace": ..., "name": ...}]``.
        parent_run_id:
            Optional parent run ID for nested pipelines.

        Returns
        -------
        The emitted RunEvent.
        """
        if run_id is None:
            run_id = str(generate_new_uuid())

        event = RunEvent(
            eventType=RunState.START,
            eventTime=self._now_iso(),
            run=Run(runId=run_id),
            job=Job(namespace=self.namespace, name=job_name),
            inputs=self._build_datasets(inputs, is_input=True),
            outputs=self._build_datasets(outputs, is_input=False),
            producer=_PRODUCER,
        )
        return self._emit(event)

    def emit_complete(
        self,
        job_name: str,
        *,
        run_id: str,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> RunEvent:
        """Emit a COMPLETE event for a pipeline job.

        Parameters
        ----------
        job_name:
            Name of the pipeline stage.
        run_id:
            Run UUID from the corresponding START event.
        inputs:
            Input datasets.
        outputs:
            Output datasets.

        Returns
        -------
        The emitted RunEvent.
        """
        event = RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=self._now_iso(),
            run=Run(runId=run_id),
            job=Job(namespace=self.namespace, name=job_name),
            inputs=self._build_datasets(inputs, is_input=True),
            outputs=self._build_datasets(outputs, is_input=False),
            producer=_PRODUCER,
        )
        return self._emit(event)

    def emit_fail(
        self,
        job_name: str,
        *,
        run_id: str,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> RunEvent:
        """Emit a FAIL event for a pipeline job.

        Parameters
        ----------
        job_name:
            Name of the pipeline stage.
        run_id:
            Run UUID from the corresponding START event.
        inputs:
            Input datasets.
        outputs:
            Output datasets.

        Returns
        -------
        The emitted RunEvent.
        """
        event = RunEvent(
            eventType=RunState.FAIL,
            eventTime=self._now_iso(),
            run=Run(runId=run_id),
            job=Job(namespace=self.namespace, name=job_name),
            inputs=self._build_datasets(inputs, is_input=True),
            outputs=self._build_datasets(outputs, is_input=False),
            producer=_PRODUCER,
        )
        return self._emit(event)

    @contextlib.contextmanager
    def pipeline_run(
        self,
        job_name: str,
        *,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Context manager for a pipeline run with automatic START/COMPLETE/FAIL.

        Parameters
        ----------
        job_name:
            Name of the pipeline stage.
        inputs:
            Input datasets.
        outputs:
            Output datasets.

        Yields
        ------
        str
            The run UUID for this pipeline execution.
        """
        start_event = self.emit_start(job_name, inputs=inputs)
        run_id = start_event.run.runId
        try:
            yield run_id
            self.emit_complete(job_name, run_id=run_id, outputs=outputs)
        except Exception:
            self.emit_fail(job_name, run_id=run_id)
            raise

    def get_events_for_job(self, job_name: str) -> list[RunEvent]:
        """Filter collected events by job name.

        Parameters
        ----------
        job_name:
            Job name to filter by.

        Returns
        -------
        List of matching RunEvents.
        """
        return [e for e in self.events if e.job.name == job_name]
