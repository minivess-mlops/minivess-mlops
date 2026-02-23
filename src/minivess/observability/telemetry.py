from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class TelemetryProvider:
    """OpenTelemetry instrumentation for ML pipeline observability."""

    def __init__(
        self,
        service_name: str = "minivess-mlops",
        *,
        endpoint: str = "http://localhost:4317",
    ) -> None:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        self.tracer = trace.get_tracer(service_name)
        logger.info("OTel initialized: service=%s, endpoint=%s", service_name, endpoint)

    @contextmanager
    def span(
        self, name: str, *, attributes: dict[str, Any] | None = None
    ) -> Generator[Any, None, None]:
        """Create a traced span."""
        with self.tracer.start_as_current_span(name, attributes=attributes or {}) as s:
            yield s

    def record_metric(
        self, name: str, value: float, *, attributes: dict[str, Any] | None = None
    ) -> None:
        """Record a metric value (simplified)."""
        logger.debug("Metric: %s=%f attrs=%s", name, value, attributes)
