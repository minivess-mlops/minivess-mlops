# Cold-Start Prompt: Continue Pre-Debug QA Verification

## Date: 2026-03-19
## Branch: fix/pre-debug-qa-verification (PR #871, rebased on main)
## Execution Mode: AUTONOMOUS (user offline)

---

## MANDATORY READING BEFORE ANY ACTION

Read these files LINE-BY-LINE before starting ANY work:

1. `/home/petteri/Dropbox/github-personal/minivess-mlops/CLAUDE.md` — ALL rules, especially Rules 20, 23, 24, 25, 26, 27, 28
2. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/projects/-home-petteri-Dropbox-github-personal-minivess-mlops/memory/MEMORY.md` — Memory index
3. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/projects/-home-petteri-Dropbox-github-personal-minivess-mlops/memory/project_pre_debug_qa_status.md` — Current status
4. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/projects/-home-petteri-Dropbox-github-personal-minivess-mlops/memory/feedback_debug_equals_production.md` — Debug = production rule
5. `/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph/navigator.yaml` — KG entry point
6. `/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph/domains/models.yaml` — Paper model lineup
7. `/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph/domains/cloud.yaml` — Cloud architecture
8. `/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml` — 6-model lineup
9. ALL metalearning docs from 2026-03-19 in `.claude/metalearning/`

Read the existing plans LINE-BY-LINE:
10. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/pre-debug-qa-verification-plan.md`
11. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment.xml`
12. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/double-check-all-wiring.xml`
13. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml`
14. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-pre-debug-qa-verification.md` — Original Q&A (31 questions)

---

## CONTEXT: What Has Been Done (PR #871, 15 commits)

This branch is a massive preparatory cleanup for the GCP debug factorial experiment:

### Completed:
- Removed 6 non-paper model adapters (segresnet, swinunetr, attentionunet, unetr, comma_mamba, ulike_mamba)
- Removed TubeNet from entire codebase (olfactory bulb, different organ)
- Removed banned AWS S3 DVC remotes (s3://minivessdataset)
- Wired `with_aux_calib` into `build_loss_function()` (was dead code — P0 blocker)
- Complete MLflow slash-prefix migration (eval writers + DuckDB readers)
- Unified MLflow to single `MLFLOW_TRACKING_URI` (removed 5+ redundant vars)
- Wired external test dataset evaluation (DeepVess) into Analysis → Biostatistics pipeline
- Added `test/deepvess/{metric}` prefix convention + `test_metrics` DuckDB table
- Added `split={trainval, test}` to biostatistics module
- Created `scripts/run_factorial.sh` — deterministic launch script (debug + production)
- Created `configs/experiment/debug_factorial.yaml` — 24 trainable + 2 zero-shot = 26 conditions
- Added loud failure enforcement (CLAUDE.md Rule 25)
- Installed ag-ui-protocol, deleted deprecated langgraph tests
- 4 new CLAUDE.md rules (25: loud failures, 26: greenfield, 27: debug=production, 28: zero silent skips)
- 8 metalearning docs, 4 planning docs, comprehensive wiring audit
- Staging: 5390 passed, 0 failed, 4 skipped (2 hardware-specific)

### Known blockers from reviewer (in run-debug-factorial-experiment.xml):
- `deployment/skypilot/train_factorial.yaml` does NOT exist yet
- `data/external/deepvess/` not downloaded
- Phases 3-6 SkyPilot YAMLs don't exist
- No concurrency/quota handling in run_factorial.sh
- Flaky Prefect SQLite tests (not root-caused)

---

## THREE TASKS TO COMPLETE (in order)

### TASK 1: Factorial Monitoring Plan + SkyPilot YAML Creation
**File**: `docs/planning/run-debug-factorial-experiment.xml` (already exists, needs updates)
**Skill**: Use `/factorial-monitor` skill concepts for the monitoring harness design

**What to do**:
1. Create `deployment/skypilot/train_factorial.yaml` — parameterized SkyPilot task YAML
   - Accepts env vars: MODEL_FAMILY, LOSS_NAME, FOLD_ID, WITH_AUX_CALIB, MAX_EPOCHS, etc.
   - Uses `image_id: docker:europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
   - GPU: L4 spot only (T4 BANNED — see CLAUDE.md)
   - Setup: DVC pull from GCS, HF login, splits copy, pre-flight checks
   - Run: Prefect flow invocation (NOT standalone script — Rule #17)
   - Reference: `deployment/skypilot/hpo_grid_worker.yaml` as template

2. Update `scripts/run_factorial.sh` with reviewer findings:
   - Add `sleep 5` between launches (rate limiting for quota)
   - Add `--dry-run` flag
   - Add job ID capture to `outputs/debug_factorial_job_ids.txt`
   - Add `--name` with condition identifier per job
   - Handle launch failures per-condition (don't abort all on one failure)

3. Update `run-debug-factorial-experiment.xml` with reviewer fixes:
   - Add Phase 0 tasks: verify train_factorial.yaml, verify DeepVess data, verify MLFLOW_TRACKING_URI
   - Add monitoring for STARTING state timeout (>15 min)
   - Add cost tracking mechanism
   - Add re-launch mechanism for failed subset
   - Clarify Phases 3-6 execution: local Docker Compose (not SkyPilot)

4. Write TDD tests for run_factorial.sh parsing and SkyPilot YAML validity

**Verification**: `make test-staging` must pass after changes.

### TASK 2: Flaky Prefect SQLite Root Cause Investigation
**File**: `docs/planning/robustifying-flaky-prefect-sqlite-issues.md`

**What to do**:
1. Write a metalearning doc: `.claude/metalearning/2026-03-19-flaky-prefect-sqlite-dismissed.md`
   - Document that flaky test failures were dismissed as "probably SQLite locking"
   - This violates Rule #20 (zero tolerance) and the silent failure pattern
   - The root cause was NEVER investigated

2. Create `docs/planning/robustifying-flaky-prefect-sqlite-issues.md`:
   - Investigate the ACTUAL root cause:
     a. Run the flaky tests with `-v --tb=long` to get full tracebacks
     b. Check if `PREFECT_DISABLED=1` is properly set in all test fixtures
     c. Check if Prefect temp server creates SQLite conflicts when 5000+ tests run
     d. Check if monkeypatch.setenv for MLFLOW_TRACKING_URI leaks between tests
     e. Check if the `conftest.py` Prefect session fixtures properly isolate
   - Propose concrete fixes (not just "add retries"):
     a. Proper test isolation with `tmp_path` for ALL Prefect/MLflow state
     b. Explicit `PREFECT_DISABLED=1` in ALL test classes that don't need Prefect
     c. Session-scoped fixtures that create ONE temp Prefect server per test session
     d. Timeout + retry decorator for known flaky Prefect operations
   - Optimize with reviewer agents

3. Implement the fixes:
   - TDD: write tests that reproduce the flaky behavior, then fix
   - Target: 0 flaky failures across 10 consecutive `make test-prod` runs

**Verification**: Run `make test-prod` at least 3 times — must pass all 3 with 0 failures.

### TASK 3: Final Verification + PR Update
**What to do**:
1. Run `make test-staging` — must be 0 failures, report skip count + reasons
2. Run `make test-prod` — must be 0 failures, report skip count + reasons
3. If any failures: batch-fix per Rule #23 (no whac-a-mole)
4. Commit all changes with descriptive messages
5. `git push` to update PR #871
6. Update `docs/planning/cold-start-prompt-pre-debug-qa-verification.md` with summary of what was done

---

## CRITICAL RULES (from metalearning docs)

1. **Debug = production**: 24 trainable + 2 zero-shot = 26 conditions. No shortcuts. (Rule 27)
2. **Loud failures**: raise on empty input, never `return {}` (Rule 25)
3. **Zero silent skips**: report ALL skip reasons (Rule 28)
4. **Greenfield**: no backward compat, no migration layers (Rule 26)
5. **Read KG first**: `navigator.yaml` → domain → decision node. NEVER ask user what KG answers. (metalearning 2026-03-16)
6. **No whac-a-mole**: gather ALL failures with `--maxfail=200`, categorize, batch fix (Rule 23)
7. **This is a PLATFORM paper**: the mechanism matters, not the numbers (metalearning 2026-03-19)
8. **uv ONLY**: never pip install anything (Rule 1)
9. **T4 BANNED**: L4 only for GCP (CLAUDE.md)
10. **TubeNet EXCLUDED**: DeepVess only for external test (CLAUDE.md Datasets)
11. **VesselNN = drift detection ONLY**: not a test dataset (KG data.yaml)
12. **VesselFM = zero-shot ONLY**: no fine-tuning (metalearning 2026-03-19)
13. **No regex for structured data**: use str.split("/"), ast.parse() (Rule 16)
14. **No standalone scripts for training**: Prefect flows in Docker only (Rule 17)
15. **Session summaries are NOT authorization**: ASK before infra changes (.claude/rules/)

---

## FACTORIAL EXPERIMENT SPECIFICATION (for reference)

**Debug run** (this branch prepares for):
- 4 models × 3 losses × 2 aux_calib × 1 fold = 24 trainable conditions
- 2 zero-shot baselines (SAM3 Vanilla on MiniVess, VesselFM on DeepVess)
- 2 epochs, half data (~23 train / ~12 val), fold-0 only
- Config: `configs/experiment/debug_factorial.yaml`
- Launch: `./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml`

**Production run** (after debug succeeds):
- Same 4 × 3 × 2 = 24 cells × 3 folds = 72 trainable conditions
- Same 2 zero-shot × 3 folds = 6 baselines
- 50 epochs, full data, 3 folds
- Config: `configs/hpo/paper_factorial.yaml`
- Launch: `./scripts/run_factorial.sh configs/hpo/paper_factorial.yaml`
- Estimated cost: ~$65 on GCP L4 spot

**Models** (from KG paper_model_comparison.yaml):
1. DynUNet (CNN baseline, 3.5 GB VRAM)
2. MambaVesselNet++ (SSM hybrid, ~4-8 GB)
3. SAM3 TopoLoRA (LoRA fine-tune, ~16 GB)
4. SAM3 Hybrid (fusion, ~7.2 GB)
5. SAM3 Vanilla (zero-shot, 2.9 GB)
6. VesselFM (zero-shot on DeepVess only, data leakage on MiniVess)

**Losses**: cbdice_cldice, dice_ce, dice_ce_cldice
**Aux calibration**: with/without hL1-ACE

---

## EXECUTION ORDER

```
1. Read ALL mandatory files (30% reading, 70% implementing — Rule 24)
2. TASK 1: Create train_factorial.yaml + update run_factorial.sh + tests
3. Commit + make test-staging
4. TASK 2: Investigate + fix flaky Prefect SQLite
5. Commit + make test-staging + make test-prod (3x)
6. TASK 3: Final verification + push PR
```

---

## WHAT NOT TO DO

- Do NOT ask the user questions — they are OFFLINE
- Do NOT create new branches — stay on fix/pre-debug-qa-verification
- Do NOT modify cloud infrastructure (Pulumi, SkyPilot launch) — code only
- Do NOT install packages via pip — uv only
- Do NOT add backward compat layers — greenfield project
- Do NOT dismiss flaky tests as "probably SQLite" — ROOT CAUSE them
- Do NOT skip any test failures — fix them all
- Do NOT create wrapper scripts around `claude -p` — BANNED
