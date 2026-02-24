# LangGraph Agent Orchestration Plan

**Issue**: #40 — Implement LangGraph agent orchestration
**Date**: 2026-02-24
**Status**: Draft → Implementation

---

## 1. Current State

`src/minivess/agents/` has scaffolding but no actual LangGraph execution:

| Module | Status | Content |
|--------|--------|---------|
| `graph.py` | Config-only stubs | AgentState dataclass, 2 config dicts (training_graph, evaluation_graph) |
| `evaluation.py` | Config-only stubs | EvalResult, EvalSuite, 2 suite builders |
| `__init__.py` | Exports | 8 public symbols |

**Dependencies**: langgraph ≥0.4, langfuse ≥2.56, litellm ≥1.56, braintrust
≥0.0.180 — all in pyproject.toml `[agents]` optional deps.

**Tests**: Zero agent tests exist.

---

## 2. Design Decisions

### Deterministic first, LLM second

The training orchestrator is a **deterministic state graph** — no LLM reasoning
needed to route data→train→evaluate→register. This follows the PRD's L2→L3
autonomy taxonomy (Luo et al., 2026). The experiment comparison agent uses
LLM for summarization, tested with mocked responses.

### LiteLLM for provider flexibility

All LLM calls go through LiteLLM, allowing Anthropic/OpenAI/Ollama backends.
Tests mock at the LiteLLM `completion()` level.

### Langfuse tracing

Wrap graph execution with Langfuse traces. Tests verify trace creation without
requiring a running Langfuse server (mock the client).

---

## 3. Implementation Plan

### T1: Typed state + actual LangGraph training graph

Rewrite `graph.py`:
- `TrainingState(TypedDict)` — proper LangGraph state type
- `build_training_graph()` → returns compiled `StateGraph`
- Node functions: `prepare_data_node`, `train_node`, `evaluate_node`,
  `register_node`, `notify_node`
- Conditional edge: evaluate → register (if metrics pass) or notify (if fail)

Tests:
- `test_training_graph_compiles` — graph builds without error
- `test_training_graph_runs_to_completion` — full graph run with mocked pipeline
- `test_training_graph_skips_register_on_bad_metrics` — conditional routing

### T2: Experiment comparison agent

Create `src/minivess/agents/comparison.py`:
- `ComparisonState(TypedDict)` — experiment_name, query, summary, runs_data
- `build_comparison_graph()` → compiled StateGraph
- Nodes: `fetch_runs`, `analyse_runs`, `summarise` (LLM call via LiteLLM)

Tests:
- `test_comparison_graph_compiles` — builds ok
- `test_comparison_fetches_runs` — fetch_runs node populates state
- `test_comparison_summarise_mocked_llm` — LLM summary with mocked response

### T3: LiteLLM provider wrapper

Create `src/minivess/agents/llm.py`:
- `call_llm(prompt, *, model, temperature) -> str`
  - Thin wrapper around litellm.completion()
  - Default model from Dynaconf `agent_provider`
- `call_llm_structured(prompt, *, model, response_format) -> dict`
  - For structured outputs

Tests:
- `test_call_llm_returns_string` — mocked completion returns text
- `test_call_llm_uses_configured_model` — reads from config
- `test_call_llm_structured_returns_dict` — structured output

### T4: Langfuse tracing integration

Create `src/minivess/agents/tracing.py`:
- `traced_graph_run(graph, state, *, trace_name) -> state`
  - Wraps graph.invoke() with Langfuse trace
  - Logs each node as a Langfuse span

Tests:
- `test_traced_run_returns_state` — returns result state
- `test_traced_run_creates_trace` — trace object created (mocked client)

### T5: Braintrust evaluation suites (test existing)

Test the existing evaluation.py:
- `test_segmentation_eval_suite_scorers` — has expected scorers
- `test_agent_eval_suite_scorers` — has expected scorers
- `test_eval_suite_add_scorer` — deduplication works
- `test_eval_suite_to_config` — serialisation

---

## 4. Execution Order

```
T1 (training graph) → T2 (comparison agent) → T3 (LiteLLM) → T4 (tracing) → T5 (eval tests)
```
