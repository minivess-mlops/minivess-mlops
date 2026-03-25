# Cold-Start Prompt: SkyPilot Resilience Fixes (10 Gaps + job_recovery)

Branch: `test/run-debug-gcp-5th-pass`

## CONTEXT

The factorial launch script (`scripts/run_factorial.sh`) has ZERO resilience.
It must be fire-and-forget: user runs one command, comes back in 1 hour or 1 week,
results are there. Currently: script dies on network blips, no retry, no resume,
no idempotency. 6 distinct SkyPilot implementation errors discovered across 2 sessions.

## MANDATORY: READ SKYPILOT DOCS BEFORE ANY IMPLEMENTATION

Claude Code has a documented trust deficit with SkyPilot (6 errors in 2 sessions).
**BEFORE writing ANY code**, read and cite these docs:

- Config reference: https://docs.skypilot.co/en/stable/reference/config.html
- Config sources: https://docs.skypilot.co/en/stable/reference/config-sources.html
- YAML spec: https://docs.skypilot.co/en/stable/reference/yaml-spec.html
- Managed jobs: https://docs.skypilot.co/en/stable/examples/managed-jobs.html
- Auto-failover: https://docs.skypilot.co/en/stable/examples/auto-failover.html
- Job recovery: search for `job_recovery` in the YAML spec page

**Verify EVERY SkyPilot change** with `sky.Task.from_yaml()` before committing.
See: `.claude/metalearning/2026-03-24-skypilot-implementation-trust-deficit.md`

## WHAT'S DONE

- Phase 1 of post-run fix plan: 4 CRITICAL silent failures fixed (779e0ed)
- 5-layer YAML contract enforcement: 70 tests, pre-commit hook, preflight (9c5f696)
- A100 removed from all SkyPilot YAMLs (0c1d516)
- sync_sky_config.py conflict fix (802d2e7)
- Phase 2-5 TDD tests WRITTEN (RED) but GREEN not yet implemented (beac465)
- Retry plan documented: docs/planning/retries-for-skypilot-spots-and-autoresume-plan.md
- Metalearning: 6 SkyPilot errors documented

## WHAT'S REMAINING (priority order)

### P0: Enable job_recovery (WRONG ban test)

The test `test_no_job_recovery_field` and `test_no_job_recovery_anywhere` in
`tests/v2/unit/deployment/test_skypilot_yamls.py` BAN `job_recovery` claiming
it was "removed in SkyPilot v1.0". **THIS IS WRONG.** The field exists and
works in our installed SkyPilot v1.0.0.dev20260314.

Fix:
1. DELETE or INVERT the ban tests
2. ADD `job_recovery` to `train_factorial.yaml`:
   ```yaml
   resources:
     job_recovery:
       max_restarts_on_errors: 3
   ```
3. VERIFY with `sky.Task.from_yaml()` that the field parses correctly
4. Update YAML contract to ALLOW job_recovery field

### P0: 10 Resilience Fixes for run_factorial.sh

From reviewer agent analysis (exact fixes in report):

| # | Gap | Fix | Priority |
|---|-----|-----|----------|
| 1 | No retry on launch failure | Retry loop with exponential backoff (2s, 4s, 8s) | P0 |
| 2 | No idempotency | Check `sky jobs queue` for existing condition name before submitting | P0 |
| 3 | No resume on crash | `--resume` flag, parse job log for already-submitted conditions | P0 |
| 4 | Lost subshell exit codes | Track PIDs explicitly, check after wait | P1 |
| 5 | Log write race condition | Per-condition temp files, merge after all complete | P1 |
| 6 | No signal handling | `trap cleanup EXIT INT TERM`, kill background jobs | P0 |
| 7 | Permissive Python parsing | Check exit codes, fail fast on parse errors | P1 |
| 8 | Zero-shot no exception handling | Add try/except with traceback in heredoc | P1 |
| 9 | Insufficient preflight | Docker image, quota, HF_TOKEN checks | P2 |
| 10 | Ambiguous exit codes | Return 0/1/2 for success/total-fail/partial-fail | P1 |

### P1: Phases 2-5 of Post-Run Fix Plan

Tests written (RED) in beac465, GREEN implementation pending:
- Phase 2: Wire parallel_submissions into run_factorial.sh from config
- Phase 3: Metric key standardization (MetricKeys constants)
- Phase 4: Cross-flow contracts (checkpoint names, experiment names)
- Phase 5: Error path coverage (bare except: pass → logged)

Plan: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml`

### P1: SkyPilot any_of Spot/On-Demand Fallback

Add to `train_factorial.yaml` (REQUIRES USER AUTHORIZATION for on-demand costs):
```yaml
resources:
  accelerators: L4:1
  any_of:
    - use_spot: true    # Try spot first ($0.22/hr)
    - use_spot: false   # Fall back to on-demand ($0.70/hr) if spot unavailable
```

**DO NOT ADD WITHOUT ASKING USER.** On-demand is 3.2x more expensive.

## GCP JOB STATUS

Jobs 3-7+ submitted (all L4 spot, PENDING). Launch script running autonomously.
Check: `uv run sky jobs queue`

## KEY FILES

```
scripts/run_factorial.sh                    — THE file to fix (10 resilience gaps)
deployment/skypilot/train_factorial.yaml    — needs job_recovery added
tests/v2/unit/deployment/test_skypilot_yamls.py — has WRONG ban tests for job_recovery
configs/cloud/yaml_contract.yaml            — needs job_recovery in allowed keys
.sky.yaml                                   — project-level SkyPilot config (correct)
docs/planning/retries-for-skypilot-spots-and-autoresume-plan.md — architecture plan
.claude/metalearning/2026-03-24-skypilot-implementation-trust-deficit.md — 6 errors
```

## HOW TO EXECUTE

```bash
# 1. Read SkyPilot docs FIRST (mandatory)
# 2. Fix job_recovery ban test + add field to YAML
# 3. Implement 10 resilience fixes with /self-learning-iterative-coder
# 4. Run make test-staging (0 skipped, 0 failed)
# 5. Run make test-prod (0 skipped, 0 failed)
# 6. Rebuild Docker + push to GAR (if train_flow.py changed)
# 7. Relaunch factorial with resilient script
```

## TEST RESULTS

- Staging: **5980 passed, 0 skipped, 0 failed** (before Phase 2-5 GREEN)
- Prod: needs re-verification after resilience fixes
