# Cold-Start Prompt: 8th Pass Backlog Fix — Remaining 18 Tasks

**Branch**: `test/run-debug-gcp-5th-pass`
**Date**: 2026-03-24
**Previous cold-start**: `docs/planning/cold-start-prompt-composable-regions-phase2.md`
**Plan XML**: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-8th-pass-backlog-fix-plan.xml`
**TDD state**: `.claude/skills/self-learning-iterative-coder/state/tdd-state.json`

## SITUATION

A comprehensive 39-task, 3-phase backlog fix plan is 54% complete (21/39 tasks done).
The remaining 18 tasks are Phase 2 code fixes (10 tasks) and Phase 3 validation (8 tasks).
Use `/self-learning-iterative-coder` to execute them autonomously.

## EXPERIMENT STATUS

The 8th pass debug factorial experiment is running autonomously via
`scripts/run_factorial_resilient.sh` (PID 1187022, started 5+ hours ago).
Check `sky jobs queue` for current status before starting.

**Last known state** (2026-03-24T19:09 UTC):
- **11 SUCCEEDED**: 6 DynUNet + 5 MambaVesselNet (first MambaVesselNet cloud success EVER)
- **5 ACTIVE**: more MambaVesselNet + remaining DynUNet conditions
- **34 total**: 32 training cells + 2 zero-shot baselines
- **SAM3 TopoLoRA + SAM3 Hybrid**: NOT YET SUBMITTED (come after MambaVesselNet)

MambaVesselNet confirmed working on L4 GPU: mamba-ssm compiles, 5.6-6.7 GB VRAM,
8.5 min/job. This was previously untested across all 8 passes.

## WHAT IS DONE — DO NOT REDO

### Phase 1: Infrastructure (5/6 DONE)
- [x] **1.1** (#878) Pulumi `MLFLOW_DEFAULT_ARTIFACT_ROOT` → GCS artifact store (commit 893c414)
- [x] **1.3** train_production.yaml rewritten: GAR + GCS + L4 (commit e18699b)
- [x] **1.4** train_hpo.yaml rewritten: GAR + GCS, UpCloud removed (commit e18699b)
- [x] **1.5** MLflow auth env vars removed from train_factorial.yaml (commit eab5604)
- [x] **1.6** Evidently/beartype: jaxtyping plugin disabled, evidently pinned
- [ ] **1.2** GPU quota increase to 4 — DEFERRED (manual GCP Console action, #917)

### Phase 2: Code Fixes (11/24 DONE)
- [x] **2.3** SWAG plugin: already correct (typed PluginOutput, no silent failures)
- [x] **2.4** DataLoader fallback: already fixed (raises with logger.error + re-raise)
- [x] **2.5** DVC_REMOTE minio fallback → ValueError (commit c20c093)
- [x] **2.6** CHECKPOINT_BEST_FILENAME constant used in train_flow (commit 3ce01b1)
- [x] **2.8** Calibration MetaTensor .numpy() crash fixed (commit a6b7cc9)
- [x] **2.9** Calibration verification tests: 2 tests (commit a6b7cc9)
- [x] **2.10** ALL 50+ hardcoded seed/alpha removed: 37 files (commit ec722d3)
- [x] **2.11** 7 conformal alpha=0.1 → required param (commit bd6f83c)
- [x] **2.12** Env fallbacks: critical DVC_REMOTE fixed, Docker paths acceptable
- [x] **2.14** Hardcoded experiment names: already fixed in previous session
- [x] **2.15** Nemenyi q_alpha parameterized with scipy.stats (commit 0958c67)
- [x] **2.16** Bare except blocks: all already have logger.warning(exc_info=True)
- [x] **2.22** HPO lr=1e-3 → 1e-4 to match TrainingConfig (commit 1572235)
- [x] **2.23** DVC_REMOTE: fixed with Task 2.5

### Phase 3: Validation (4/9 DONE)
- [x] **3.3** MambaVesselNet cloud validation: CONFIRMED on L4 (5 conditions SUCCEEDED)
- [x] **3.4** AST guard test: MetricKeys usage in flows (commit 4f0fefc)
- [x] **3.5** Guard test: test_no_hardcoded_alpha.py passes
- [x] **3.6** AST guard test: env fallbacks in flows (commit 6654078)
- [x] **3.8** make test-prod: 6417 passed, 0 skipped, 0 failed

### GitHub Issues Created (21 total)
#917 (GPU quota), #918 (calibration), #919 (stale YAMLs), #920 (DVC_REMOTE),
#921 (metric key mismatch), #922 (DataLoader fallback), #923 (conformal alpha),
#924 (wire unwired functions), #925 (MambaVesselNet), #926 (E2E chain),
#927 (checkpoint naming), #928 (env fallbacks), #929 (Nemenyi), #930 (flow epilogue),
#931 (MLflow setup dedup), #932 (HPO lr), #933 (clean-rename), #934 (allowed_clouds),
#935 (n_matched topology), #936 (augmentation config), #937 (VRAM profiles)

### Test Baseline
- **Staging**: 6382+ passed, 0 skipped, 0 failed (with zero-skip enforcement)
- **Prod**: 6417 passed, 0 skipped, 0 failed, 2 xfailed

## WHAT IS NOT DONE — DO THIS (18 tasks)

### Phase 2 Remaining: Code Fixes (10 tasks)

#### Task 2.1 (P0): Central MetricKeys audit — replace hardcoded metric strings (#790)
**What**: The AST guard test (Task 3.4) found 3 hardcoded metric keys in `train_flow.py`:
- Line 689: `"vram/peak_mb"` → use `MetricKeys.VRAM_PEAK_MB`
- Line 871: `"fold/n_completed"` → use `MetricKeys.FOLD_N_COMPLETED` (add if missing)
- Line 685: `"val/loss"` → use `MetricKeys.VAL_LOSS`
**Files**: `src/minivess/orchestration/flows/train_flow.py`, `src/minivess/observability/metric_keys.py`
**TDD**: After fixing, the xfail in `tests/v2/unit/observability/test_metric_keys_usage.py` for train_flow should become a PASS. Change xfail to strict fail.
**Verify**: `uv run pytest tests/v2/unit/observability/test_metric_keys_usage.py -v` → all PASS, 0 xfail.

#### Task 2.2 (P0): Fix metric key mismatch in builder.py (#921)
**What**: `builder.py` uses underscore metric key format (`eval_fold2_dsc`) but `tracking.py` logs slash format (`eval/fold2/dsc`). Builder queries never find completed runs.
**Files**: `src/minivess/pipeline/builder.py`, `src/minivess/observability/metric_keys.py`
**TDD**: Write test that verifies builder metric keys match MetricKeys constants.
**Verify**: Builder search queries find runs logged by tracking.py.

#### Task 2.7 (P1): post_training checkpoint naming (#927)
**What**: `post_training_flow.py` docstring references `best.ckpt` but actual checkpoint is `best_val_loss.pth`. Use `CHECKPOINT_BEST_FILENAME` from `orchestration/constants.py`.
**Files**: `src/minivess/orchestration/flows/post_training_flow.py`
**TDD**: Write test that post_training_flow uses constant, not hardcoded string.

#### Task 2.13 (P1): Wire 7 unwired production functions (#924)
**What**: These functions were TDD'd but never called from flow entry points:
1. `_generate_interaction_plot()` → biostatistics_flow
2. `_generate_variance_lollipop()` → biostatistics_flow
3. `_generate_instability_plot()` → biostatistics_flow
4. `_generate_anova_table()` → biostatistics_flow
5. `_generate_cost_appendix_table()` → biostatistics_flow
6. `_resolve_ensemble_strategies()` → analysis_flow
7. One more from 6th pass code review Section 2.4
**Files**: `src/minivess/pipeline/biostatistics_figures.py`, `biostatistics_tables.py`,
`src/minivess/orchestration/flows/analysis_flow.py`, `biostatistics_flow.py`
**TDD**: Integration test that these functions are called when their flow runs.
**IMPORTANT**: Read the actual function signatures BEFORE wiring. Understand what arguments they need.

#### Task 2.17 (P1): Flow epilogue dedup (#930)
**What**: ~175 lines of flow epilogue pattern (MLflow cleanup, logging, error handling)
duplicated across 5 flows. Extract to `orchestration/flow_helpers.py`.
**Files**: All 5 flow files + new `flow_helpers.py`
**TDD**: Test that the shared helper produces identical behavior to the inline version.
**CAUTION**: Large refactor. Read all 5 epilogue patterns before extracting.

#### Task 2.18 (P1): MLflow run setup dedup (#931)
**What**: ~120 lines of MLflow run setup boilerplate duplicated across 8 copies.
Extract to `observability/mlflow_helpers.py` (already has some helpers).
**Files**: Flow files + `src/minivess/observability/mlflow_helpers.py`
**TDD**: Test that shared setup produces identical MLflow run state.

#### Task 2.19 (P1): n_matched ambiguity in component_dice (#935)
**What**: `topology_metrics.py:170-181` tracks `n_matched` but never uses it.
Ambiguity in whether divisor should be `n_target` or `n_matched`.
**Files**: `src/minivess/pipeline/topology_metrics.py`
**TDD**: Write test clarifying the correct divisor. Check the original CbDice paper.

#### Task 2.20 (P1): Augmentation hyperparameters → config-driven (#936)
**What**: `augmentation.py:15-18` hardcodes rotation/flip/intensity ranges.
Should be configurable via Hydra config group.
**Files**: `src/minivess/data/augmentation.py`, `configs/augmentation/default.yaml`
**TDD**: Test that augmentation reads from config, not hardcoded values.

#### Task 2.21 (P1): VRAM overhead constants → profile YAML (#937)
**What**: `validation.py:24-32` hardcodes VRAM overhead estimates per model.
Should read from `configs/profiles/*.yaml`.
**Files**: `src/minivess/data/validation.py`, `configs/profiles/`
**TDD**: Test that VRAM overhead comes from profile YAML.

#### Task 2.24 (P1): Orphan YAML config keys
**What**: Several YAML config keys are defined but never read by any code:
- `configs/dashboard/health_thresholds.yaml` → orphan keys
- `.env.example` `DASHBOARD_REFRESH_INTERVAL_S`, `MLFLOW_ARTIFACT_BUCKET`, `GCS_*`
**Files**: Various config files
**TDD**: AST scan for config key consumption. Document or delete orphans.

### Phase 3 Remaining: Validation (8 tasks)

#### Task 3.1 (P1): Local mock E2E test (#926)
**What**: Create a test that runs the FULL chain:
train (2 epochs, synthetic data) → analysis → biostatistics
on synthetic data locally. Catches wiring issues before cloud.
**Files**: `tests/v2/integration/e2e/test_train_analysis_biostatistics_chain.py`
**TDD**: Write the test. It should use MLflow in tmp_path and mock minimal data.
**IMPORTANT**: This is an integration test — mark with `@pytest.mark.integration`.

#### Task 3.2 (P1): Cloud E2E plan
**What**: Document the cloud E2E verification steps for AFTER the debug factorial completes:
1. Query MLflow for all 34 SUCCEEDED runs
2. Trigger analysis_flow against the experiment
3. Trigger biostatistics_flow against the analysis results
4. Verify comparison tables, figures, rankings all generated
**Files**: `docs/planning/cloud-e2e-verification-plan.md`
**Output**: Documentation only — no code changes.

#### Task 3.7 (P1): Verification script
**What**: Create `scripts/verify_8th_pass_backlog.py` that programmatically checks:
- All 21 GitHub issues created (#917-937) exist and are labeled
- All DONE tasks in tdd-state.json have corresponding commits
- make test-prod passes with 0 skips
- No remaining `alpha: float = 0.05` in src/ function signatures
- No remaining `seed: int = 42` in src/ function signatures (except config models)
- No `MLFLOW_ARTIFACTS_DESTINATION` in Pulumi code
- No `MLFLOW_TRACKING_USERNAME` in SkyPilot YAMLs
- train_production.yaml and train_hpo.yaml use GAR (not GHCR)
**Files**: `scripts/verify_8th_pass_backlog.py`
**TDD**: The script itself IS the test. Run it and report results.

#### Task 3.9 (P0): Prefect SQLite flaky tests (#880)
**What**: Investigate flaky Prefect test failures caused by SQLite locking.
The conftest.py creates a temporary Prefect server per session. Under
concurrent test execution (pytest-xdist), multiple workers may contend.
**Files**: `tests/conftest.py`
**TDD**: Run `uv run pytest tests/ -n 4 --timeout=60` to reproduce.
If flaky, isolate per-worker Prefect home (already has code for this).
**IMPORTANT**: This may be a non-issue — run the concurrent tests first to check.

#### Remaining Phase 3 tasks (lower priority)
- **3.2**: Cloud E2E plan (documentation)
- **3.3**: Already DONE (MambaVesselNet confirmed)

## EXPERIMENT MONITORING

While executing the plan, periodically check experiment progress:
```bash
uv run sky jobs queue | grep -E "SUCCEEDED|RUNNING|STARTING|PENDING" | grep -v CANCELLED
```

The resilient wrapper (`scripts/run_factorial_resilient.sh`) handles everything.
DO NOT interfere with the running experiment. DO NOT cancel jobs.
DO NOT launch new jobs. Just monitor and report.

**If the wrapper process (PID 1187022) has died**: It may need restarting:
```bash
nohup bash scripts/run_factorial_resilient.sh configs/factorial/debug.yaml &
```
The `--resume` flag in the wrapper skips already-submitted conditions.

## GOTCHAS FROM THIS SESSION

1. **The TDD state file was stale** — the background agent wrote to it but its
   changes were overwritten. Always read the state file FIRST before updating.

2. **Background agents can cause merge conflicts** — the seed/alpha agent modified
   source files while the main thread also modified tests. Coordinate carefully.

3. **Tests that hardcode `alpha=0.05` or `seed=42`** are technically correct (they
   test against config defaults) but should ideally import from the config class:
   `BiostatisticsConfig().alpha` instead of literal `0.05`.

4. **The MetricKeys guard test uses xfail** for train_flow.py (3 violations).
   After Task 2.1, change it to strict failure mode.

5. **MambaVesselNet uses 2x VRAM** vs DynUNet (6.7 vs 4.0 GB with calibration).
   SAM3 will use even more (16+ GB). L4 has 24 GB — should fit.

6. **The `pulumi up` for Task 1.1 has NOT been run yet** — only the code was changed.
   The user needs to run `cd deployment/pulumi/gcp && pulumi up` to deploy the
   GCS artifact store fix. Until then, 413 errors continue for new runs.

## FILES TO READ FIRST

```
# 1. The plan (complete task list with TDD approach for each)
docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-8th-pass-backlog-fix-plan.xml

# 2. TDD state (which tasks are done, which are TODO)
.claude/skills/self-learning-iterative-coder/state/tdd-state.json

# 3. The experiment report (findings, metrics, observations)
docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-8th-pass-multi-region.md

# 4. Key source files for remaining tasks
src/minivess/observability/metric_keys.py          # MetricKeys constants (Task 2.1)
src/minivess/pipeline/builder.py                   # Metric key mismatch (Task 2.2)
src/minivess/orchestration/flows/post_training_flow.py  # Checkpoint naming (Task 2.7)
src/minivess/pipeline/biostatistics_figures.py      # Unwired functions (Task 2.13)
src/minivess/pipeline/biostatistics_tables.py       # Unwired functions (Task 2.13)
src/minivess/pipeline/topology_metrics.py           # n_matched (Task 2.19)
src/minivess/data/augmentation.py                   # Config-driven (Task 2.20)
src/minivess/data/validation.py                     # VRAM profiles (Task 2.21)

# 5. Guard tests (verify they still pass after changes)
tests/v2/unit/observability/test_metric_keys_usage.py  # MetricKeys guard
tests/v2/unit/orchestration/test_no_env_fallbacks.py   # Env fallbacks guard
tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py # Alpha guard

# 6. Metalearning (avoid repeating mistakes)
.claude/metalearning/2026-03-24-stale-venv-pycache-phantom-skips.md
.claude/metalearning/2026-03-24-skypilot-implementation-trust-deficit.md
.claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md
```

## HOW TO EXECUTE

```bash
# 1. Read state file and plan XML (30% of effort)
# 2. Check experiment status: uv run sky jobs queue
# 3. Execute remaining tasks via /self-learning-iterative-coder
#    Start with Task 2.1 (MetricKeys audit) → 2.2 → 2.7 → 2.13 → ...
# 4. After each task: commit, push, update state file
# 5. Run make test-prod periodically (target: 6417+ passed, 0 skipped)
# 6. After all tasks: run verification script (Task 3.7)
# 7. Update experiment report with final job results
# 8. Final: make test-prod → 0 failures, 0 skips
```

## EXECUTION PRIORITY (recommended order)

1. **2.1** MetricKeys audit (P0, unblocks strict guard test)
2. **2.2** Builder metric key mismatch (P0, fixes silent ensemble failure)
3. **2.7** Checkpoint naming constant (P1, quick)
4. **2.13** Wire 7 unwired functions (P1, high-value)
5. **2.19** n_matched topology ambiguity (P1, scientific correctness)
6. **2.20** Augmentation config-driven (P1, medium)
7. **2.21** VRAM profiles (P1, medium)
8. **2.17** Flow epilogue dedup (P1, large refactor)
9. **2.18** MLflow setup dedup (P1, large refactor)
10. **2.24** Orphan YAML keys (P1, cleanup)
11. **3.1** Local E2E test (P1, validation)
12. **3.7** Verification script (P1, meta-validation)
13. **3.9** Prefect SQLite flaky (P0, investigation)
14. **3.2** Cloud E2E plan (P1, documentation)
15-18. Any remaining tasks or new issues discovered during execution

## TEST RESULTS BASELINE

- **Prod**: 6417 passed, 0 skipped, 0 failed, 2 xfailed (zero-skip enforcement active)
- **Branch**: `test/run-debug-gcp-5th-pass` (commit 0958c67)
- **Zero-skip enforcement**: Active in `tests/conftest.py` — any skip = hard failure
