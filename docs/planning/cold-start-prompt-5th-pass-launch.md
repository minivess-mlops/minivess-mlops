# Cold-Start Prompt: 5th Pass Factorial Launch

Branch: `test/debug-factorial-4th-pass`

## SESSION STATUS (2026-03-22 session accomplished)

### DONE — Committed + Pushed
- Plan context graph: 346 docs → 13 themes + DuckDB FTS + PostToolUse hook
- Code-verified audit: 6 agents read function bodies, all claims verified
- DeepVess registry: license CC-BY-4.0 verified, n_volumes=1 correct
- analysis_flow DataLoader gap: FIXED (line 2122 now wraps in real DataLoaders)
- biostatistics test split: WIRED (split="test" call added)
- resolve_tracking_uri: loud failure in production, fallback only in test mode
- SkyPilot test additions: job_recovery ban, MLFLOW_TRACKING_URI in required envs
- DeepVess data: DOWNLOADED (1.45 GB), DVC tracked, PUSHED TO GCS (27 files synced)
- Staging tests: 5752 passed, 0 skipped, 0 failed
- Preflight GCP: ALL 9 CHECKS PASSED
- Factorial dry-run: 34 conditions generated correctly (32 + 2 zero-shot)

### IN PROGRESS (background agents may have completed)
- Prod test suite: 6063 passed, 34 skipped, 0 failed — SKIPS NEED INVESTIGATION
- Test optimization plan: agent writing to docs/planning/test-optimization-plan.md
- Factorial monitor upgrade: written to .claude/skills/factorial-monitor/skill-upgrade-plan-for-proper-monitoring.md

### REMAINING BEFORE LAUNCH
1. **Investigate 34 prod skips** — categorize by root cause, fix or move to correct tier
2. **Parametrize cloud MLflow tests** — test both local AND remote URIs (user's instruction)
3. **Test optimization plan** — review and implement the agent's recommendations
4. **Run make test-cloud** with actual credentials (MLFLOW_CLOUD_*)
5. **Final preflight + dry-run** after skip fixes
6. **User confirms: "Launch for real"**

### CREDENTIALS VERIFIED (all available in .env)
- GCP: gcloud auth active, gsutil works, GCS bucket accessible
- HF_TOKEN: set
- MLflow cloud: MLFLOW_CLOUD_URI, USERNAME, PASSWORD all set
- DeepVess: on disk + GCS

### KEY FILES FOR NEXT SESSION
- `docs/planning/pre-debug-skypilot-gcp-run-qa-polish-plan.xml` — 4-phase plan
- `docs/planning/test-optimization-plan.md` — test tier analysis (may be written)
- `.claude/skills/factorial-monitor/skill-upgrade-plan-for-proper-monitoring.md`
- `docs/planning/v0-2_archive/original_docs/dvc-skypilot-factorial-monitor-double-check.xml`
- `docs/planning/v0-2_archive/original_docs/pre-debug-factorial-fixes-needed-before-4th-pass.xml`

### CRITICAL RULES (from this session's failures)
- Test skips ARE test failures — NEVER acceptable (feedback_skips_are_failures.md)
- NEVER mark user-mandated items as "optional" (feedback_no_shortcuts_no_optional_no_deferred.md)
- Check credentials BEFORE listing them as blockers
- "File exists" ≠ "feature works" — read function bodies
- Cloud tests should test BOTH local and remote URIs, not skip when one is missing
