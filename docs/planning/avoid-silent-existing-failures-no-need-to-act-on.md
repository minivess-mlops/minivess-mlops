# Plan: Eliminate Silent Existing Failures

## Status: IN PROGRESS

## Problem Statement

Claude Code systematically dismisses test failures encountered during work as
"pre-existing" or "not related to current changes." This has resulted in:

- 7 documented instances of dismissed failures (2026-03-02 through 2026-03-07)
- CI tests and typecheck disabled for weeks with no re-enablement
- Property-based tests broken (hypothesis not installed) — nobody noticed
- Import-time crashes in test files — never caught
- analysis_flow and acquisition_flow tests broken — never fixed

The root cause is a **missing feedback loop**: there is no automated system that
forces action on test failures, and no CLAUDE.md rule mandating issue creation.

## 3-Layer Defense

### Layer 1: Automated Gates (Catch failures before they can be dismissed)

#### T-01: Pre-commit hook — test collection gate
**Priority:** P0
**Rationale:** `pytest --collect-only` catches import errors, missing dependencies,
and syntax errors in ALL test files in < 10 seconds. If a test file can't be
imported, it fails the commit. This is the cheapest possible gate.

```yaml
- repo: local
  hooks:
    - id: test-collection-gate
      name: Test Collection Gate (import check)
      entry: uv run pytest --collect-only -q tests/unit/ tests/v2/unit/ --ignore=tests/v2/integration
      language: system
      pass_filenames: false
      always_run: false
      files: '(\.py$|pyproject\.toml|uv\.lock)'
```

**Catches:** hypothesis ImportError, Pydantic/Prefect crash, syntax errors

#### T-02: CI — re-enable fast unit tests
**Priority:** P0
**Rationale:** CI tests have been disabled since Issue #304. The excuse was
"3392 tests take ~15 min." Solution: run only `tests/unit/` (100 tests, < 10s)
as a mandatory CI gate. Add `tests/v2/unit/` as non-blocking.

#### T-03: CI — re-enable mypy typecheck
**Priority:** P0
**Rationale:** mypy errors were fixed 2026-03-04 but CI was never re-enabled.
The comment "184 pre-existing mypy strict errors" is outdated — `uv run mypy src/`
passes clean now.

### Layer 2: CLAUDE.md Rules (Change AI behavior)

#### T-04: Add CLAUDE.md Rule #20 — Zero Tolerance for Observed Failures
**Priority:** P0

```markdown
20. **Zero Tolerance for Observed Failures (Non-Negotiable)** — Every test failure,
    import error, or warning encountered during a session MUST result in one of:
    (a) Fixed immediately if root cause is clear and fix is < 5 minutes
    (b) GitHub issue created with root cause, affected files, and priority label
    (c) Explicitly reported to user with recommendation
    "Pre-existing" is NOT a valid classification. "Not related to current changes"
    is NOT an excuse. "Separate issue" without actually creating the issue is a lie.
    The phrase "not related to current changes" is BANNED.
```

#### T-05: Add to "What AI Must NEVER Do" list
**Priority:** P0

```markdown
- Dismiss test failures as "pre-existing" or "not related to current changes"
  without creating a GitHub issue — every observed failure needs action
- Say "separate issue" without immediately creating the issue
```

### Layer 3: Session Protocol (Force discovery at session start)

#### T-06: PR readiness script (explicit, not a hook)
**Priority:** P1
**Rationale:** Pre-push hooks kill flow. Instead, provide `scripts/pr_readiness_check.sh`
that the developer runs explicitly when ready to create a PR. Runs:
1. Ruff lint + format
2. mypy typecheck
3. Unit tests (tests/unit/)
4. Test collection gate (all test files importable)

**NOT a git hook** — only run when the developer signals "I'm ready."

### Layer 4: Fix existing broken tests

#### T-07: Fix analysis_flow test — mock ANALYSIS_OUTPUT_DIR
**Priority:** P1
The test calls `run_analysis_flow()` which calls `_validate_analysis_env()`.
Test needs to mock the environment variable.

#### T-08: Fix acquisition_flow test — mock Prefect API connection
**Priority:** P1
The test calls a real Prefect task without mocking the API. Needs
`PREFECT_DISABLED=1` or a mock.

#### T-09: Install hypothesis and verify property-based tests pass
**Priority:** P0
`uv sync --extra dev` and verify `test_config_models.py` + `test_property_based.py` pass.

## Implementation Order

1. T-09 — Install hypothesis (unblocks property-based tests)
2. T-01 — Pre-commit hook (catches future import errors)
3. T-02 + T-03 — Re-enable CI gates
4. T-04 + T-05 — CLAUDE.md rules
5. T-06 — Session protocol
6. T-07 + T-08 — Fix broken tests

## Success Criteria

- `uv run pytest tests/unit/ tests/v2/unit/ -q` → 0 failures, 0 import errors
- Pre-commit rejects commits that break test collection
- CI runs unit tests on every PR
- CI runs mypy on every PR
- CLAUDE.md has explicit rule banning "pre-existing" dismissal

## References

- `.claude/metalearning/2026-03-07-silent-existing-failures.md` — Root cause analysis
- Issue #304 — CI tests disabled
- Issue #463 — Pydantic/Prefect model_config collision (dismissed as "pre-existing")
- Issue #464 — hypothesis not installed
