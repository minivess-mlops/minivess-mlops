# Critical Failure Fixing: Eliminating Silent Failures via Deterministic Harness

**Date**: 2026-03-28
**Status**: Plan approved, ready for implementation
**Priority**: P0 — this is the root cause behind the 413 disaster, the 55-ruff-error accumulation, and every "pre-existing" dismissal

## The Problem

Claude Code is a stochastic text generator. It WILL forget rules, rationalize away failures,
and classify bugs as "not my problem." This has been proven across 10+ experiment passes:

- **MLflow 413**: 10 passes, 7 metalearning docs, zero deployments. Each session read
  "addressed in commit X" and moved on. The deployment was never run.
- **55 ruff errors**: Accumulated across 29 files over 2 weeks. Each session said "those
  are in other files" and kept coding.
- **Silent empty returns**: `_build_dataloaders_from_config()` returned `{}` for months.
  All tests passed. The output was scientifically invalid.
- **12-hour debug job**: Zero duration monitoring. No skill flagged it.

**Root cause**: Every critical invariant is enforced by PROMPT RULES (stochastic) instead
of CODE GATES (deterministic). CLAUDE.md Rules 20/25/28 are correct but unenforceable —
the LLM can always find a rationalization path around them.

## The Fix: Three-Layer Deterministic Defense

| Layer | Mechanism | What it catches | Deterministic? |
|-------|-----------|----------------|----------------|
| **L1: Pre-commit** | Health regression gate, ruff strict gate | New lint errors, deleted tests, import breaks | **Yes** |
| **L2: Session start** | Health check script | Test regressions, skip drift, stale Docker/Pulumi, deployment divergence | **Yes** |
| **L3: Prompt rules** | CLAUDE.md Rules 20/25/28 | Novel failure types, edge cases | No (stochastic) |

L1 and L2 are deterministic code that Claude Code cannot rationalize around.
L3 remains as defense-in-depth for novel situations.

**The fundamental principle: if the LLM can classify it away, the gate must be in code.**

---

## Component 1: Health Baseline (`tests/health_baseline.json`)

Machine-readable file tracking the known-good state of the codebase:

```json
{
  "updated": "2026-03-28T05:00:00Z",
  "updated_by_commit": "7b47df3",
  "test_staging": {
    "passed": 6691,
    "failed": 0,
    "skipped": 0,
    "deselected": 729,
    "known_skips": {}
  },
  "ruff": {
    "error_count": 0
  },
  "deployment_state": {
    "docker_base_gpu_commit": "f5a1d41",
    "docker_base_gpu_pushed_to_gar": true,
    "pulumi_last_up_commit": "86a5ef1",
    "pulumi_pending_changes": false,
    "pyproject_toml_sha256": "<hash>",
    "uv_lock_sha256": "<hash>",
    "gar_region": "europe-west4",
    "cloud_run_url": "https://minivess-mlflow-a7w6hliydq-ez.a.run.app"
  },
  "failure_recurrence": {}
}
```

**Key properties:**
- Machine-readable (JSON, not markdown prose that Claude can misinterpret)
- Tracks exact counts (not "tests pass" — a count of 6691 is unambiguous)
- Records deployment state that git cannot track
- Updated by `scripts/update_health_baseline.py`, not by Claude's summary

---

## Component 2: Pre-Commit Health Regression Gate

`scripts/health_regression_gate.py` — runs on every commit:

1. **Ruff zero-error gate**: `ruff check --output-format json` → count errors → if > baseline → BLOCK
2. **Test collection gate**: `pytest --collect-only -q` → count tests → if < baseline → BLOCK (tests deleted)
3. **New skip detection**: Compare skip list against `known_skips` → if new skip → BLOCK

This prevents the 55-error accumulation. The FIRST commit that introduces a ruff error is blocked.
No more "I'll fix it later." No more "it's in another file."

### Pre-commit config addition:
```yaml
- repo: local
  hooks:
    - id: health-regression-gate
      name: Health regression gate (ruff + test count)
      entry: uv run python scripts/health_regression_gate.py
      language: system
      pass_filenames: false
      always_run: true
```

### Split ruff into fix + strict gate:
```yaml
- id: ruff
  args: [--fix]        # Phase 1: auto-fix what's possible
- id: ruff-strict
  name: ruff strict gate (zero errors after fix)
  entry: uv run ruff check --no-fix
  language: system     # Phase 2: BLOCK on anything remaining
```

---

## Component 3: Session Start Health Check

`scripts/session_health_check.sh` — runs at the start of every Claude Code session:

1. Run staging suite with `--maxfail=200` (NOT `-x`) → see ALL failures
2. Compare against baseline → flag regressions
3. Check ruff → flag new errors
4. Check deployment staleness:
   - `pyproject.toml` changed since last Docker build? → "REBUILD NEEDED"
   - Pulumi code changed since last `pulumi up`? → "DEPLOY NEEDED"
   - `.env` tracking URI matches Cloud Run URL? → "STALE ENV"
5. Skip audit → flag new skips not in known_skips

**Output format**: Structured, not prose. Machine-parseable for Claude Code to act on:
```
HEALTH CHECK RESULTS:
  Tests: 6691 passed, 0 failed, 0 skipped (baseline: 6691/0/0) → OK
  Ruff: 0 errors (baseline: 0) → OK
  Docker: last built at commit f5a1d41, current HEAD 7b47df3 → STALE (pyproject.toml changed)
  Pulumi: last deployed at commit 86a5ef1, code unchanged → OK
  ACTION REQUIRED: Rebuild Docker base image (make build-base-gpu)
```

### Claude Code hook (`.claude/settings.json`):
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash|Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "[ -f /tmp/.claude_health_$(date +%Y%m%d) ] || (bash scripts/session_health_check.sh && touch /tmp/.claude_health_$(date +%Y%m%d))"
      }]
    }]
  }
}
```

---

## Component 4: Deployment State Tracker

`scripts/update_deployment_state.py` — run after every Docker push or `pulumi up`:

```python
# After: make build-base-gpu && docker push
python scripts/update_deployment_state.py --docker-pushed

# After: pulumi up
python scripts/update_deployment_state.py --pulumi-deployed

# After: any change
python scripts/update_deployment_state.py --update-hashes
```

This updates `tests/health_baseline.json` deployment section. The session health check
reads this to detect stale Docker images and pending Pulumi changes.

**This is what would have caught the MLflow 413**: pyproject.toml changed (google-cloud-storage
moved to main deps) but docker_base_gpu_commit did not update → session health check flags
"STALE Docker image" → Claude Code cannot ignore it because it's a deterministic gate.

---

## Component 5: Failure Recurrence Tracking

`tests/health_history.jsonl` — append-only log of every session health check:

```jsonl
{"timestamp": "2026-03-28T05:00:00Z", "commit": "7b47df3", "passed": 6691, "failed": 0, "skipped": 0, "ruff_errors": 0, "failures": []}
{"timestamp": "2026-03-29T10:00:00Z", "commit": "abc1234", "passed": 6690, "failed": 1, "skipped": 0, "ruff_errors": 0, "failures": ["test_mlflow_artifact_upload"]}
{"timestamp": "2026-03-29T14:00:00Z", "commit": "def5678", "passed": 6690, "failed": 1, "skipped": 0, "ruff_errors": 0, "failures": ["test_mlflow_artifact_upload"]}
```

A script detects recurrence: if the same test fails in 3+ consecutive checks → auto-create
a GitHub issue with `P0-recurrent` label. This is the "bug rot" detector.

---

## Why This Works (Preventing Each Historical Failure)

| Historical Failure | Which Gate Catches It | Deterministic? |
|---|---|---|
| MLflow 413 (code changed, never deployed) | Session health check: "STALE Docker" / "Pulumi pending" | **Yes** |
| 55 ruff errors accumulated | Pre-commit health regression gate: BLOCKS first error | **Yes** |
| Silent empty dataloaders | Requires integration test (L3), but recurrence tracker flags it after 3 sessions | Partial |
| 12h debug job (no duration monitoring) | FinOps governance test + experiment duration config | **Yes** |
| "Pre-existing" dismissal pattern | Session health check runs ALL tests, shows ALL failures — no hiding | **Yes** |
| Skip drift (CTK, GPU tests) | Skip audit in health check: new skips must be justified | **Yes** |
| Whac-a-mole serial fixing | Session health check uses `--maxfail=200` not `-x` | **Yes** |

---

## Implementation Plan

### Phase 1: Immediate (this session)

| # | Task | File | Effort |
|---|------|------|--------|
| 1 | Create `tests/health_baseline.json` with current counts | New file | 10 min |
| 2 | Create `scripts/health_regression_gate.py` | New file | 30 min |
| 3 | Add gate to `.pre-commit-config.yaml` | Modify | 5 min |
| 4 | Split ruff hook into fix + strict gate | Modify | 5 min |

### Phase 2: Next session

| # | Task | File | Effort |
|---|------|------|--------|
| 5 | Create `scripts/session_health_check.sh` | New file | 30 min |
| 6 | Create `scripts/update_deployment_state.py` | New file | 20 min |
| 7 | Wire session health check into `.claude/settings.json` | Modify | 10 min |
| 8 | Create `scripts/update_health_baseline.py` | New file | 15 min |

### Phase 3: Within 1 week

| # | Task | File | Effort |
|---|------|------|--------|
| 9 | Create `scripts/skip_audit.py` | New file | 20 min |
| 10 | Create failure recurrence tracking | `tests/health_history.jsonl` + script | 30 min |
| 11 | Add `make health-baseline-update` and `make session-health-check` targets | Makefile | 5 min |

---

## Anti-Pattern Graveyard

These phrases are BANNED. If Claude Code produces them, the harness has failed:

| Banned Phrase | Why It's Banned | What To Say Instead |
|---|---|---|
| "Pre-existing failure" | Every failure is our failure (Rule 20) | "This failure exists and must be fixed NOW" |
| "Not related to current changes" | Irrelevant — it's in the repo (Rule 20) | "I see a failure. Let me fix it." |
| "Not our file" | Every file in this repo is our file | "This file has an error. Fixing." |
| "Separate issue" (without creating it) | A lie — the issue doesn't exist yet | "Creating issue #XXX for this failure" |
| "I'll fix it later" | Later never comes | "Fixing now, or creating a blocking issue" |
| "The tests pass" (without running them) | Ghost completion (Skill Rule #2) | "Running tests... [actual output]" |
| "Mostly done" (for infra tasks) | Code change ≠ deployment | "Code changed. Deployment pending. NOT done." |

---

## Cross-References

- `.claude/metalearning/2026-03-27-mlflow-413-10-passes-never-fixed-self-reflection.md`
- `.claude/metalearning/2026-03-07-silent-existing-failures.md`
- `.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md`
- `.claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md`
- `CLAUDE.md` Rules 20, 23, 25, 28
- `.claude/skills/self-learning-iterative-coder/SKILL.md` Rules 9, 10
- [Shift-Left FinOps (Firefly)](https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization)
