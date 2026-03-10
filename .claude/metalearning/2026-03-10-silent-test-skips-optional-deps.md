# Silent Test Skips from Optional Dependencies

**Date**: 2026-03-10
**Severity**: HIGH — 126 tests silently skipping = 126 tests that don't exist
**Root Cause**: PRD-selected tools placed in `[project.optional-dependencies]` groups,
but dev workflow used plain `uv sync` which doesn't install optional groups.

## The Problem

pyproject.toml had 6 optional groups (eval, quality, dev, agents, sam, serving)
containing PRD-selected tools: pandera, hypothesis, great_expectations, whylogs,
deepchecks, captum, shap, quantus, netcal, gradio, cleanlab, pydantic-ai, langfuse,
braintrust.

Tests used `pytest.importorskip()` to gracefully skip when tools were missing.
This is correct behavior for *truly optional* features — but these tools were
**selected in the PRD as required capabilities**. The "graceful skip" became a
silent lie: tests appeared to pass when they were actually never running.

## Additional Bugs Found

1. **langgraph imported but never declared in pyproject.toml** — code in
   `src/minivess/agents/_deprecated/` imported langgraph, tests guarded with
   importorskip, but langgraph was NOT in any dependency group. Added to [agents].

2. **litellm same situation** — imported in deprecated agents code, never declared.
   Added to [agents].

3. **Wrong mock path** — `test_agents.py::test_traced_run_creates_trace` patched
   `minivess.agents.tracing._get_langfuse_client` but the function lives in
   `minivess.agents._deprecated.tracing`. This was invisible because langgraph
   not being installed caused the entire test class to skip before reaching it.

## The Fix

1. Added langgraph + litellm to `[project.optional-dependencies].agents`
2. Ran `uv sync --all-extras` — installs ALL optional groups
3. Fixed mock path in test_agents.py
4. Updated CLAUDE.md Rule #1: `uv sync --all-extras` is REQUIRED for dev
5. Updated Quick Commands: `uv sync` → `uv sync --all-extras`

## Result

- Before: 4010 passed, **28 skipped** (packages), many more silently deselected
- After: **4136 passed, 2 skipped** (only hardware-specific: NVIDIA CTK config)

## Anti-Pattern: Graceful Degradation as Silent Failure

`pytest.importorskip()` is the right tool for *truly optional* integrations that
only some users need. It is the WRONG tool for tools the project has committed to
using. When a tool is selected in the PRD and has tests written for it, it MUST be
installed in the dev environment. "Graceful skip" for committed tools = lying to
yourself about test coverage.

## Rule

**If a tool has tests, it's not optional.** Either:
- Install it (`--all-extras` or move to main deps)
- Or delete the tests

There is no third option where tests exist but never run.
