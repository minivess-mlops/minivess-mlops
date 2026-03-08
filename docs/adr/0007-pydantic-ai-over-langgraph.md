# ADR 0007: Pydantic AI over LangGraph for Agent Orchestration

## Status

Accepted (2026-03-08)

## Context

Issue #341 called for LangGraph micro-orchestration inside Prefect flows. During
implementation, we evaluated two approaches:

1. **LangGraph StateGraph** — Complex multi-step reasoning with its own checkpointer
2. **Pydantic AI + PrefectAgent** — Official Prefect integration, each LLM call is a
   retryable/cacheable Prefect task

Our agents are single-decision-point tasks (experiment summary, drift triage, figure
narration) — not multi-step reasoning graphs.

## Decision

Use **Pydantic AI with PrefectAgent wrapper** for all agent decision points.
Move existing LangGraph code to `_deprecated/`.

## Rationale

| Criterion | Pydantic AI + PrefectAgent | LangGraph |
|-----------|---------------------------|-----------|
| Prefect integration | Native (official) | Manual bridging |
| Retries/caching | Per-LLM-call via TaskConfig | Graph-level only |
| Type safety | Pydantic output models | TypedDict state |
| CI testing | TestModel (zero API calls) | Requires mocking |
| Observability | OTEL instrumentation | Requires LangSmith |
| Dependency footprint | pydantic-ai-slim (no fastmcp conflict) | langgraph + litellm |
| Complexity | Single decorator | StateGraph + edges + compile |

Key factors:
- **PrefectAgent** makes each `agent.run()` a nested Prefect flow with full visibility
- **TestModel** enables CI testing with zero LLM API costs
- **pydantic-ai-slim** avoids the `platformdirs` conflict with whylogs
- Our decision points are single-step, not multi-step graphs

## Consequences

- LangGraph remains available as `_deprecated/` (tests use `importorskip`)
- All 3 agent decision points wire into Prefect flows with deterministic fallback
- Agent tracing via Langfuse OTEL exporter (optional)
- Future complex reasoning tasks could still use LangGraph if needed
