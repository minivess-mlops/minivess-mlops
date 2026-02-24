# DiLLS Agent Diagnostics — Implementation Plan (Issue #16)

## Current State
- Langfuse tracing exists in `agents/tracing.py` (basic trace start/end)
- TelemetryProvider with OTel spans in `observability/telemetry.py`
- No layered summary diagnostics for agent interactions

## Architecture

### New Module: `src/minivess/observability/agent_diagnostics.py`
- **AgentInteraction** — Dataclass capturing a single agent step (node, input, output, latency, tokens)
- **SessionSummary** — Dataclass aggregating interactions into a session
- **AgentDiagnostics** — DiLLS-style layered diagnostics:
  - `record_interaction()` — Log a single agent step
  - `summarize_session()` — Conversation-level summary
  - `summarize_aggregate()` — Cross-session aggregate statistics
  - `to_markdown()` — Human-readable diagnostic report

## Test Plan
- `tests/v2/unit/test_agent_diagnostics.py` (~12 tests)
  - TestAgentInteraction: construction, metadata, timing
  - TestSessionSummary: aggregation, step count, total latency
  - TestAgentDiagnostics: record, summarize session, aggregate, markdown
