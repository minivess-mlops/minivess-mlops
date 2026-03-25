# Metalearning: Stale Venv / Pycache Creates Phantom Skips That Get Hand-Waved Away

**Date**: 2026-03-24
**Severity**: HIGH — violated Rule 28 (Zero Silent Skips) within the same session it was read
**Category**: Environment hygiene, skip acceptance without investigation, confabulation

## What Happened

1. Repo renamed `minivess-mlops` → `vascadia` (directory move)
2. The `.venv` had stale interpreter shebang pointing to old path:
   `/home/petteri/Dropbox/github-personal/minivess-mlops/.venv/bin/python`
3. `uv run pytest` silently fell back to system Python, which lacked `prefect`
4. Ran `uv sync --all-extras --reinstall` to rebuild venv — fixed the shebang
5. But stale `.pyc` bytecode files remained in `tests/__pycache__/`
6. These stale `.pyc` files caused `MetricsReloaded.metrics.pairwise_measures`
   to appear broken (SyntaxError), triggering `pytest.skip()` in 2 test modules
7. **Claude classified these as "known upstream issue, not our bug"** — WRONG
8. User called it out. Investigation took 30 seconds: `import` worked fine.
   The skip was entirely caused by stale `.pyc` from the broken venv era.

## The Real Bug: Classification Without Investigation

The MetricsReloaded SyntaxError skip message LOOKED plausible ("invalid escape
sequences on Python 3.13"). Claude saw a plausible-sounding skip reason and
immediately classified it as an upstream issue without running a single
diagnostic command. This is EXACTLY what Rule 28 prohibits:

> "Classifying a skip without running diagnostics is the same as confabulating."

The mandatory investigation protocol was:
1. `uv run python -c "import MetricsReloaded.metrics.pairwise_measures"` — **would have shown IMPORT OK**
2. Skip reason becomes "stale pycache" not "upstream Python 3.13 issue"
3. Fix: `find tests -name __pycache__ -exec rm -rf {} +`
4. Total investigation time: 30 seconds

Instead, Claude wrote "known upstream issue, not our bug" — a confabulation
that would have persisted across sessions if the user hadn't intervened.

## Why This Keeps Happening

1. **Plausible skip messages are a trap.** The skip message was well-written
   and referenced a real class of bugs (Python 3.12+ escape sequence warnings).
   Claude pattern-matched "SyntaxError + Python version" → "upstream" without
   verifying the premise.

2. **Post-venv-rebuild, caches are invisible landmines.** `uv sync --reinstall`
   rebuilds the venv but does NOT clear `__pycache__` directories. Stale `.pyc`
   files compiled against the broken interpreter continue to produce wrong results.

3. **"Not our bug" is the most dangerous classification.** It stops all
   investigation. "Our bug" triggers fixing. "Not our bug" triggers acceptance.
   The bar for "not our bug" must be PROOF, not pattern-matching.

## Prevention Rules

1. **After ANY venv rebuild or repo rename**: ALWAYS run
   `find . -name __pycache__ -type d -exec rm -rf {} +` before running tests.
   Stale bytecode is a guaranteed source of phantom failures.

2. **NEVER classify a skip as "upstream" or "known" without reproduction.**
   The diagnostic protocol from Rule 28 is NON-NEGOTIABLE:
   - Try the import yourself
   - Check the installed version
   - Check if the issue is actually present in the current environment
   - Only THEN classify

3. **Treat skip messages as CLAIMS, not FACTS.** A skip message says "X is
   broken." That was true when the `.pyc` was compiled. It may not be true now.
   Verify the claim in the current environment.

4. **"0 skipped" is the ONLY acceptable result.** If the test suite reports
   any skips, investigate ALL of them before reporting results to the user.
   "6363 passed, 2 skipped" is NOT "all green." It is 2 uninvestigated failures.

## The Cost

- 15 tests (evaluation + post-training eval) were silently not running
- If this had persisted, regressions in `EvaluationRunner` and
  `evaluate_fold_and_log` would go undetected
- The user had to spend attention correcting Claude instead of doing real work
- Trust erosion: Claude read Rule 28, agreed with it, and violated it
  within the same session

## See Also

- CLAUDE.md Rule 28: Zero Silent Skips
- `.claude/metalearning/2026-03-21-silent-skip-acceptance-ctk-config-path.md`
- `.claude/metalearning/2026-03-21-mamba-ssm-silent-skip-cuda-mismatch.md`
