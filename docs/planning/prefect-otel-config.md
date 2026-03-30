# prefect-opentelemetry Integration

**Phase 10 deliverable** — Install package and configure Prefect OTel export.

Reference: [prefect-opentelemetry on PyPI](https://pypi.org/project/prefect-opentelemetry/)

## Installation

```bash
uv add prefect-opentelemetry --optional observability
```

This adds automatic OTel span creation for every Prefect flow and task,
replacing the need for manual on_completion/on_failure hooks (Phase 5).

## Prefect Server Configuration

Add to the Prefect server service in `docker-compose.yml`:

```yaml
  prefect:
    environment:
      # ... existing env ...
      OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT:-http://minivess-grafana-lgtm:4318}/v1/traces
      OTEL_SERVICE_NAME: prefect-server
```

## .env.example Addition

```bash
# ── Prefect OpenTelemetry ────────────────────────────────
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://minivess-grafana-lgtm:4318/v1/traces
OTEL_SERVICE_NAME=minivess-training
```

## What This Provides

With `prefect-opentelemetry` installed:
- Every `@flow` invocation creates an OTel trace
- Every `@task` invocation creates an OTel span (child of the flow trace)
- Spans include: task name, duration, status, error info
- Traces are exported to the LGTM Collector → stored in Tempo → visible in Grafana
- The manual Phase 5 hooks become optional (defense-in-depth)

## Relationship to Phase 5 Custom Hooks

Phase 5 (`prefect_hooks.py`) provides custom hooks that work WITHOUT the
`prefect-opentelemetry` package. With the package installed, the custom
hooks are still useful for:
- Structured JSONL event emission (Phase 3)
- Heartbeat updates (Phase 2/4)
- Domain-specific logging (epoch progress, GPU metrics)

The two approaches are complementary, not competing.
