# Cold-Start Prompt: 6th Pass Continuation — Phases 2-5

Branch: `test/run-debug-gcp-5th-pass`

## SESSION STATUS (2026-03-23/24 session accomplished)

### DONE — Committed + Pushed

**Infrastructure Performance Audit (all 4 phases):**
- Config architecture: controller + infrastructure blocks in cloud YAMLs
- SWAG scaling: `min(configured, max(2, max_epochs // 5))` (25 min → ~5 min)
- Perf metrics: 6 perf/* keys in MetricKeys + SWAG timing instrumented
- Project-level `.sky.yaml` with `infra: gcp/europe-west1` controller
- Preflight check #10: controller cloud matches job cloud
- sync_sky_config.py: generates ~/.sky/config.yaml from repo config
- 32 new config validation tests

**Phase 1 of Post-Run Fix Plan (4 CRITICAL bugs):**
- Post-training plugins: RAISE on failure (was: return empty model_paths)
- External DataLoader: RAISE on failure (was: fallback to raw pairs)
- tracking_uri: removed "mlruns" fallback (Rule #22 violation)
- builder.py: `eval_fold2_dsc` → `eval/fold2/dsc` (underscore format NEVER matched)
- 6 new tests + 3 existing tests updated
- Staging: 5886 passed, 0 skipped, 0 failed

**Experiment Harness Skill v0.2.0:**
- Literature-informed (8 papers: Bi, Li, Zhou, Shen, Ye, Anthropic)
- 5-phase workflow: GENERATE → VALIDATE → EXECUTE → COMPOUND → REFLECT
- Deep exploration protocol (protocols/deep-exploration.md)
- Issue #912 (harness meta-skill), #913 (launch bottleneck), #914 (on-demand fallback)

**6th Pass Live Report:**
- 5 cloud observations captured (controller 12x faster, quota issue, spot queuing)
- 37 findings from 4 reviewer agents (deep exploration during spot queuing)
- 8 new test opportunities identified

**Metalearning docs written:**
- `2026-03-23-launched-known-failing-jobs-wasting-credits.md`
- `2026-03-23-skypilot-controller-on-wrong-cloud.md`
- `2026-03-23-no-quota-preflight-check-wasted-30-min.md`
- `2026-03-23-bypassed-xml-harness-ad-hoc-monitoring.md`
- `2026-03-23-vram-skip-still-in-prod-tier-5th-violation.md`

### IN PROGRESS — GCP Jobs

- Jobs 1-2 PENDING (L4+A100 spot unavailable, queuing ~8+ hours)
- Launch script (`b7j1m7xlr`) running autonomously
- Controller: GCP europe-west1-b (UP, n4-standard-4)
- Check: `uv run sky jobs queue`

### REMAINING — Post-Run Fix Plan Phases 2-5

Execute: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml`

**Phase 2: Config-to-Code Wiring (P1, ~45 min)**
- Task 2.1: Wire parallel_submissions + rate_limit from cloud config into run_factorial.sh
  - Read infrastructure.cloud_config from factorial YAML
  - Load configs/cloud/{name}.yaml
  - Extract PARALLEL_SUBMISSIONS and RATE_LIMIT_SECONDS
  - Replace hardcoded `sleep 5` with config value
  - Implement parallel launch loop (background jobs + wait -n)
  - Call sync_sky_config.py at script start
  - Test: test_infra_params_wired.py

**Phase 3: Metric Key Standardization (P1, ~30 min)**
- Task 3.1: Add EVAL_FOLD_PREFIX and EVAL_TEST_PREFIX to MetricKeys
- Task 3.2: Migrate all flows to MetricKeys constants (no raw eval/ strings)
  - AST check that no flow file contains raw "eval/" or "eval_fold" strings
  - Fix: evaluation_runner.py, biostatistics_duckdb.py
  - Test: test_metric_key_consistency.py

**Phase 4: Cross-Flow Contracts (P1, ~30 min)**
- Task 4.1: Canonical checkpoint filename in constants.py
  - CHECKPOINT_BEST_FILENAME = "best_val_loss.pth"
  - Remove "best.ckpt" from post_training_flow.py:87 fallback list
- Task 4.2: Remove ALL hardcoded "minivess_training" experiment names
  - deploy_flow.py:346+355, dashboard_flow.py:470, any others
  - Replace with resolve_experiment_name(EXPERIMENT_TRAINING)
  - Test: test_cross_flow_contracts.py

**Phase 5: Error Path Coverage (P2, ~20 min)**
- Task 5.1: Test error paths (MLflow creation, metadata collection, setup timing)
  - Fix bare `except: pass` at train_flow.py:904-905
  - Test: test_error_path_coverage.py

### KEY FILES

- `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml` — TDD plan
- `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass.md` — live report
- `.claude/skills/experiment-harness/SKILL.md` — harness v0.2.0
- `.claude/skills/experiment-harness/protocols/deep-exploration.md` — deep exploration protocol
- `.sky.yaml` — project-level SkyPilot config (GCP europe-west1 controller)

### HOW TO RESUME

```bash
# 1. Check GCP job status
uv run sky jobs queue

# 2. Run Phases 2-5 via self-learning TDD skill:
# /self-learning-iterative-coder docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml
# (Phase 1 already DONE — start from Phase 2 Task 2.1)

# 3. After Phases 2-5: rebuild Docker + push to GAR
make build-base-gpu  # with INSTALL_MAMBA=1
docker tag + push to GAR

# 4. Full test verification
make test-staging  # must be 0 skipped, 0 failed
make test-prod     # must be 0 skipped, 0 failed

# 5. If GCP jobs completed: update 6th pass report with results
# Run /experiment-harness Phase 4 (COMPOUND) + Phase 5 (REFLECT)
```

### TEST RESULTS BEFORE THIS SESSION PAUSED

- Staging: **5886 passed, 0 skipped, 0 failed**
- Prod: **6152 passed, 0 skipped, 0 failed** (last run before Phase 1 fixes)

### CRITICAL RULES (from this session's failures)

- ZERO skips — "acceptable skip" is BANNED (5th violation documented)
- NEVER launch with known-failing code (metalearning: launched-known-failing-jobs)
- Controller MUST match job cloud (RunPod controller for GCP = 36 min/submission)
- Pin controller region (no region = US zone-hopping from Finland)
- Report BEFORE launch (Rule H1) — not after
- Deep exploration during idle time (protocols/deep-exploration.md)
- Things must compound — every pass leaves codebase measurably better
