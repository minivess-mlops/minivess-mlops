# Agents — Pydantic AI + PrefectAgent

## Architecture (ADR 0007)

Agent orchestration via Pydantic AI with PrefectAgent wrappers.
LiteLLM provides multi-provider LLM access (Anthropic, OpenAI, Ollama).

## Key Files

| File | Purpose |
|------|---------|
| `prefect_agent.py` | PrefectAgent wrapper for Pydantic AI agents |
| `_deprecated/` | Legacy agent implementations (being migrated) |

## LLM Stack

- **Provider**: LiteLLM (multi-provider abstraction)
- **Tracing**: Langfuse (self-hosted, Docker Compose)
- **Evaluation**: Braintrust AutoEvals
- **Framework**: Pydantic AI (not raw LangChain)

## Key Rules

- All LLM calls go through LiteLLM — never direct API calls
- Langfuse tracing is mandatory for all agent interactions
- Agent definitions are in this directory; flow integration is in orchestration/
- `langgraph` + `litellm` are in the `agents` optional dependency group
