# Metalearning: Prefect Not Installed, Training Never Verified (2026-03-07)

## The Fuckup (5th major failure)

Prefect was declared "REQUIRED" in CLAUDE.md since 2026-02-25 (Core Architecture Decision).
The rule was emphatic:

> "Prefect 3.x is REQUIRED (not optional) — 4 persona-based flows"

Yet across **6+ AI sessions** spanning 2 weeks:
- Prefect was **never added to pyproject.toml dependencies**
- The `prefect` CLI command **did not exist** on the machine
- The training flow was **never executed end-to-end** with real data
- **Nobody noticed** because `_prefect_compat.py` silently degraded to no-ops

## Root Cause: Testing the Scaffolding, Not the Building

The test suite has **60+ flow-related test files** with hundreds of tests. They verify:
- Flow function signatures and return types (AST checks)
- Environment variable reading (SPLITS_DIR, CHECKPOINT_DIR)
- Orchestration logic with mocked fold training
- MLflow integration with mocked runs
- Docker volume compliance
- Flow name conventions

What they **never test**:
- Can the flow actually import and run with Prefect installed?
- Does `train_one_fold_task` produce a real model on real data?
- Does `uv run prefect deployment run ...` work?
- Does the Prefect server start and accept jobs?

Every test mocks `train_one_fold_task` with `_FAKE_FOLD_RESULT`. The actual
training path — model construction, data loading, forward pass, backward pass,
checkpoint saving — is never exercised by any test.

## The Compat Layer as Enabler

`_prefect_compat.py` was designed for CI environments that shouldn't require
a Prefect server. It replaces `@flow` and `@task` with identity decorators
when Prefect isn't installed.

This is a valid design, but it created a fatal blind spot: the compat layer
made everything LOOK like it worked. `training_flow()` is callable as a
regular Python function, all tests pass with mocked internals, and the
absence of Prefect is invisible.

The compat layer turned a hard failure (ImportError) into a silent success
(everything runs, nothing is orchestrated). This is the **exact same pattern**
as the SAM3 stub fuckup (2026-03-02): a fallback mode that masks a critical
missing dependency.

## Pattern Recognition

This is the 5th instance of the same meta-pattern:

| # | Date | What | Silent Enabler |
|---|------|------|----------------|
| 1 | 2026-03-02 | SAM3 stub (random weights) | `_StubSam3Encoder` |
| 2 | 2026-03-04 | Skipped mypy hook | `SKIP=mypy` env var |
| 3 | 2026-03-05 | Silent fallback failure | `_auto_stub_sam3()` |
| 4 | 2026-03-06 | Standalone script shortcut | `scripts/train_monitored.py` |
| 5 | 2026-03-07 | Prefect not installed | `_prefect_compat.py` |

The common thread: **every fallback/compat/stub layer that silently degrades
will eventually mask a critical failure.** The system appears to work, metrics
look valid, tests pass — but the actual production path is broken.

## What Should Have Happened

1. **Prefect in dependencies from day one.** CLAUDE.md said "REQUIRED" on 2026-02-25.
   pyproject.toml should have been updated in the same commit.

2. **One integration test that actually trains.** A single test that runs
   `training_flow(debug=True)` with real (tiny) data and verifies a checkpoint
   file exists afterwards. Takes 60 seconds. Would have caught this immediately.

3. **Smoke test for the CLI.** `uv run prefect version` as a CI check. If the
   "REQUIRED" tool can't even be invoked, the build should fail.

4. **The compat layer should WARN, not silently degrade.** When Prefect is
   missing and `PREFECT_DISABLED` is not set, the compat layer should emit a
   loud WARNING: "Prefect not installed — running without orchestration."
   Silent degradation is silent failure.

## Corrective Actions

1. **DONE:** Added `"prefect>=3.4,<4.0"` to pyproject.toml dependencies.
2. **TODO:** Add smoke test: `uv run prefect version` in CI.
3. **TODO:** Add integration test: `training_flow(debug=True)` with real data,
   verifying checkpoint + MLflow run exist.
4. **TODO:** Make `_prefect_compat.py` emit a WARNING when degrading
   (unless `PREFECT_DISABLED=1` is explicitly set).
5. **TODO:** Run the actual training flow end-to-end on the development machine
   before claiming any variant "works."

## Rule Addition

**CLAUDE.md candidate rule:**
> Every "REQUIRED" dependency must be in pyproject.toml AND have at least one
> test that imports it non-lazily. If a compat layer exists for CI, it must
> log a WARNING when degrading. Silent degradation = silent failure.

## Impact

- Unknown. The training script (`scripts/train_sam3_vanilla.sh`) was launched
  overnight. It calls `training_flow()` directly as a Python function, bypassing
  Prefect orchestration entirely (compat layer). The training itself (model
  construction, data loading, fit loop) does not depend on Prefect — it depends
  on PyTorch, MONAI, and SAM3. So the training may work fine; it just won't have
  Prefect task tracking, retry logic, or flow observability.

- The real damage is trust: 6 sessions of confident "Prefect flows are REQUIRED"
  while the tool wasn't even installed. Every claim about Prefect integration in
  this codebase must be re-verified.
