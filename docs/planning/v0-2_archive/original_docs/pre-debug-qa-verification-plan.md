# Pre-Debug QA Verification Plan

**Branch**: `fix/pre-debug-qa-verification`
**Date**: 2026-03-19
**Purpose**: Polish codebase before GCP debug run. Fix ALL config/KG/test/logging issues so the debug run focuses on infrastructure, not code bugs.
**Scope**: GCP ONLY (no RunPod)
**Prompt**: `docs/planning/cold-start-prompt-pre-debug-qa-verification.md`

---

## Debug Run Specification (What This Branch Prepares For)

The debug run is the **FULL production experiment** with 3 differences only:

| Parameter | Debug Run | Full Production |
|-----------|-----------|-----------------|
| **Epochs** | 2 | 50 |
| **Data** | Half (~23 train / ~12 val) | Full (~47 train / ~23 val) |
| **Folds** | 1 (fold-0) | 3 (seed=42) |
| **Everything else** | IDENTICAL | IDENTICAL |

### Debug Run Matrix

**Trainable factorial**: 4 models × 3 losses × 2 aux_calib = **24 conditions** on fold-0
- Models: dynunet, mambavesselnet, sam3_topolora, sam3_hybrid
- Losses: cbdice_cldice, dice_ce, dice_ce_cldice
- Aux calibration: with hL1-ACE, without hL1-ACE

**Zero-shot baselines**: SAM3 Vanilla (MiniVess fold-0), VesselFM (DeepVess/TubeNet)

**E2E flow chain**: Training → Post-Training → Analysis (Evals) → Biostatistics → Deployment

**Source of truth**: `knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml`

---

## Reviewer Agent Corrections (2026-03-19)

The following corrections were identified by a Plan reviewer agent after thorough
code inspection. They significantly reduce the plan's scope (many tasks already done)
and add one critical blocker (aux_calib wiring).

1. **P0 BLOCKER FOUND**: `with_aux_calib` NOT wired in `train_one_fold_task()` →
   `build_loss_function()`. Aux_calib factor is dead code. Must fix in GROUP B.
2. **5/7 D-group tasks already implemented**: D1 (augmentation), D2 (grad norms),
   D3 (grad_scale), D5 (val timing), D7 (checkpoint size). Only D4, D6 remain.
3. **T-C1, T-C3 already done**: trainer.py + system_info.py already migrated.
   Main gap: evaluation_runner.py + 7 DuckDB reader files.
4. **compose_experiment_config() already wired** in train_flow.py lines 1030-1064.
   T-B1 re-scoped to extract `with_aux_calib` from config_dict.
5. **Duplicates removed**: T-H1=T-A4, T-H2=T-A3, T-H3=T-B2.
6. **ModelFamily cleanup**: SAM3_LORA (legacy alias for SAM3_TOPOLORA) to be removed.
   CUSTOM kept as generic extension point. MULTITASK_DYNUNET removed.
7. **Missing task added**: SkyPilot debug YAML (`deployment/skypilot/debug_factorial.yaml`).
8. **VesselFM corrected**: zero_shot_only (no fine-tuning). KG fixed.
9. **Greenfield rule**: No backward compatibility. Delete old formats entirely.

---

## Task Groups

### GROUP A: Config System Cleanup (P0 — Blocking)

#### T-A1: Remove ALL non-paper model adapters and configs
**Why**: Reduces maintenance burden, eliminates confusion about which models are in scope.
**Files to DELETE** (~30 files):
- Adapters: `segresnet.py`, `swinunetr.py`, `attentionunet.py`, `unetr.py`, `comma.py` + `configs/model/comma_mamba.yaml`, `configs/model/ulike_mamba.yaml`
- Model profiles: `segresnet.yaml`, `swinunetr.yaml`, `attentionunet.yaml`, `unetr.yaml`, `comma_mamba.yaml`, `comma.yaml`, `ulike_mamba.yaml`
- Smoke tests: `smoke_segresnet.yaml`, `smoke_swinunetr.yaml`, `smoke_attentionunet.yaml`, `smoke_unetr.yaml`, `smoke_comma_mamba.yaml`, `smoke_ulike_mamba.yaml`
- Unit tests: `test_segresnet_adapter.py`, `test_swinunetr_adapter.py`, `test_attentionunet_adapter.py`, `test_unetr_adapter.py`, `test_comma_adapter.py`, `test_mamba_adapter.py` (ULike-Mamba parts)
**Files to MODIFY**:
- `src/minivess/adapters/__init__.py` — remove imports
- `src/minivess/adapters/model_builder.py` — remove factory functions + registry entries
- `src/minivess/config/models.py` — remove enums
- `configs/method_capabilities.yaml` — remove non-paper entries
- `configs/experiment/debug_all_models.yaml` — update models_to_test list
- `tests/v2/integration/test_biostatistics_factorial_integration.py` — update synthetic model names
- `tests/v2/integration/test_training_flow_all_models.py` — update model list
- `knowledge-graph/code-structure/adapters.yaml` — remove non-paper entries
- `knowledge-graph/decisions/L3-technology/primary_3d_model.yaml` — remove segresnet candidate
- `knowledge-graph/manuscript/claims.yaml` — remove segresnet reference
**TDD**: Write test asserting `ModelFamily` enum has exactly 6 values matching KG paper_model_comparison.

#### T-A2: Create debug factorial config (`configs/experiment/debug_factorial.yaml`)
**Why**: Reproducible debug run config that can be used a year from now.
```yaml
# @package _global_
# Debug factorial: FULL production experiment with reduced epochs/data/folds.
# 4 models × 3 losses × 2 aux_calib = 24 conditions on fold-0, 2 epochs, half data.
experiment_name: debug_factorial
max_epochs: 2
num_folds: 1
debug: true
max_train_volumes: 23
max_val_volumes: 12
with_aux_calib: false  # overridden per-condition
losses:
  - cbdice_cldice
```
**Also create**: `configs/splits/debug_half_1fold.json` — fold-0 with first 23 train / first 12 val volumes from `3fold_seed42.json`.

#### T-A3: Fix T4 ban enforcement across ALL configs
**Already done**: `gcp_spot.yaml` (T4 removed), `gcp_quotas.yaml` (T4 refs removed).
**Remaining**: Verify no T4 references in any SkyPilot YAML, Makefile, or scripts.
**TDD**: Write test asserting no config file contains "T4:1" or "T4:" as an accelerator.

#### T-A4: Fix `paper_factorial.yaml` — correct model lineup
**Already done**: Updated to dynunet, mambavesselnet, sam3_topolora, sam3_hybrid + zero-shot baselines.
**TDD**: Write `test_factorial_matches_kg.py` — verifies paper_factorial.yaml models match KG paper_model_comparison.yaml.

---

### GROUP B: Hydra Composition & Aux Calib Wiring (P0 — Blocking)

#### T-B1: Wire `with_aux_calib` from config_dict into `build_loss_function()` (P0 BLOCKER)
**Why**: `compose_experiment_config()` is ALREADY wired (train_flow.py lines 1030-1064). But `train_one_fold_task()` calls `build_loss_function(loss_name)` WITHOUT passing `with_aux_calib` or `aux_calib_weight`. The 2× aux_calib factor is DEAD CODE — all 24 conditions silently produce only 12 distinct results.
**What**:
- In `train_one_fold_task()`, extract `with_aux_calib` and `aux_calib_weight` from `config_dict`
- Pass them to `build_loss_function(loss_name, with_aux_calib=..., aux_calib_weight=...)`
- Verify the loss function wrapper is activated when `with_aux_calib=True`
**TDD**: Test that `build_loss_function("cbdice_cldice", with_aux_calib=True)` returns an `AuxCalibrationLoss` wrapper, not a bare loss.

#### T-B2: Verify debug_factorial config resolves for all 24 conditions
**Why**: The debug config must resolve through the composition pipeline for every model×loss×calib condition.
**What**:
- Parametrized test: `compose_experiment_config(experiment_name="debug_factorial", overrides=["model=sam3_topolora", "losses=[dice_ce]", "with_aux_calib=true"])` for all 24 combos
- Also test zero-shot config resolution for sam3_vanilla and vesselfm
**TDD**: Parametrized test covering all 24 + 2 conditions.

#### T-B3: Create SkyPilot debug factorial YAML
**Why**: Debug run needs a launchable SkyPilot task file for GCP.
**File**: `deployment/skypilot/debug_factorial.yaml`
**What**: Parameterized YAML accepting MODEL_FAMILY, LOSS_NAME, FOLD_ID, WITH_AUX_CALIB env vars. Uses `image_id: docker:europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`, L4 spot, T4 banned.
**TDD**: Test that YAML is valid and contains required fields.

---

### GROUP C: MLflow Slash-Prefix Migration (P0 — Blocking)

**Reviewer correction**: trainer.py (T-C1) and system_info.py + infrastructure_timing.py (T-C3)
are ALREADY migrated to slash-prefix. Removed from plan. Main remaining gap: evaluation_runner.py
writers + 6 DuckDB/biostatistics reader files.

**Greenfield rule**: No backward compatibility. Delete old underscore-prefix code. No
normalize_metric_key() layer needed — just use new format everywhere.

#### T-C1: Migrate evaluation_runner.py metric WRITERS to slash-prefix
**Why**: `evaluation_runner.py` line 268 still constructs keys as `f"eval_{ds_name}_{subset_name}_{metric}_ci_lower"`.
**Scope**: evaluation_runner.py + tracking.py `log_evaluation_results()`. Change to `eval/{ds}/{subset}/{metric}/ci95_lo`.
**TDD**: Test asserting eval metrics use slash-prefix.

#### T-C2: Migrate DuckDB/biostatistics READERS to slash-prefix
**Why**: 6+ functions in `biostatistics_duckdb.py` and `duckdb_extraction.py` are hardcoded to
`eval_fold{i}_{metric}` underscore parsing. Functions: `_is_eval_fold_metric()`,
`_parse_eval_fold_metric()`, `parse_eval_fold_metric()`, etc.
**Scope**: biostatistics_duckdb.py, duckdb_extraction.py, reproducibility_check.py,
mlruns_inspector.py, mlruns_enhancement.py (~7 files total).
**TDD**: Test asserting readers handle slash-prefix eval keys.

#### T-C3: Remove normalize_metric_key() backward compat layer
**Why**: Greenfield project. No legacy runs to preserve. Delete the normalization code
entirely — use slash-prefix only. Remove MIGRATION_MAP from metric_keys.py.
**TDD**: Test asserting MIGRATION_MAP does not exist (or is empty).

#### T-C4: AST-based integration test — all keys follow slash convention
**Scope**: Scan all source files for string literals matching metric key patterns using
`ast.parse()` + `ast.walk()` (NOT regex — Rule #16). Assert none use underscore-prefix.
**TDD**: `test_metric_key_convention.py` — convention enforcement.

---

### GROUP D: MLflow Logging Gaps (P1 — FDA-readiness)

**Reviewer correction**: 5 of 7 original tasks are ALREADY IMPLEMENTED in trainer.py:
- ~~T-D1~~ augmentation config: DONE (tracking.py line 212-219)
- ~~T-D2~~ gradient norms: DONE (trainer.py lines 313-379)
- ~~T-D3~~ optimizer grad_scale: DONE (trainer.py line 778-779)
- ~~T-D5~~ validation timing: DONE (trainer.py lines 650-675)
- ~~T-D7~~ checkpoint size: DONE (trainer.py lines 834-839)

**Remaining tasks (2 of 7):**

#### T-D1: Inference latency logging (flow-level, not per-epoch)
**What**: After training completes, run a dedicated inference benchmark on validation set.
Log `infer/latency_ms_per_volume` and `infer/throughput_volumes_per_sec` as flow-level metrics.
**TDD**: Test asserting inference latency metric exists after fit() returns.

#### T-D2: Early stopping `stopped_early` to MLflow
**What**: `MetricKeys.TRAIN_STOPPED_EARLY` constant exists. `stopped_early` is tracked in
`trainer.py` (line 608, 926, 993) and returned in result dict. But it is NEVER logged to
MLflow. Add `mlflow.log_metric("train/stopped_early", float(stopped_early))` after training loop.
**TDD**: Test asserting `train/stopped_early` metric appears in MLflow run.

#### T-D0: Verify existing logging implementations
**What**: Parametrized test asserting ALL expected slash-prefix keys appear in a mock training
run output. Covers the 5 already-implemented gaps + the 2 new ones.
**TDD**: Single parametrized test covering all expected metric/param keys.

---

### GROUP E: Flow Integration Tests (P0 — E2E Contract)

#### T-E1: Training → Post-Training contract test
**What**: Create mock training MLflow run with expected artifacts/tags. Verify post_training_flow can discover checkpoints via `find_fold_checkpoints()`.
**TDD**: Integration test using `tmp_path` MLflow.

#### T-E2: Training → Analysis (Evals) contract test
**What**: Verify analysis_flow can find upstream training runs, build ensembles, evaluate, tag champions.
**TDD**: Integration test with mock training artifacts.

#### T-E3: Analysis → Deploy contract test
**What**: Verify deploy_flow can discover champion models from MLflow tags, export ONNX, import to BentoML, generate artifacts.
**TDD**: Integration test — full local deploy pipeline.

#### T-E4: Training → Biostatistics contract test
**What**: Verify biostatistics_flow can read per-volume metrics from training runs, compute ANOVA, generate figures.
**TDD**: Integration test — mock training → real biostatistics computation.

#### T-E5: Full e2e chain test (debug_factorial config)
**What**: Parametrized test that runs: compose config → training (1 epoch, 2 volumes) → post-training → analysis → biostatistics → deploy. All in Docker with `MINIVESS_ALLOW_HOST=1` for testing.
**TDD**: Quasi-e2e test marked `@pytest.mark.slow`.

---

### GROUP F: GCP Infrastructure Verification Tests (P1)

#### T-F1: Pulumi stack health test
**What**: Non-destructive queries to verify GCP resources exist: GCS buckets, GAR repo, Cloud SQL (if enabled).
**TDD**: Cloud test with `@pytest.mark.cloud_mlflow`, auto-skip without env vars.

#### T-F2: GAR Docker image availability test
**What**: `docker manifest inspect europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
**TDD**: Cloud test verifying image exists and has expected labels.

#### T-F3: DVC data accessibility test
**What**: `dvc status -r gcs` — verify MiniVess data is accessible on GCS.
**TDD**: Cloud test with auto-skip.

#### T-F4: SkyPilot GCP connectivity test
**What**: `sky check gcp` — verify SkyPilot can access GCP.
**TDD**: Cloud test with auto-skip.

---

### GROUP G: KG & CLAUDE.md Improvements (P1)

#### T-G1: Update CLAUDE.md with model lineup quick-reference
**What**: Add explicit model lineup table under "Quick Reference" pointing to KG source:
```
## Paper Model Lineup (Source: knowledge-graph/domains/models.yaml::paper_model_comparison)
| Model | Family | Training | Paper Section |
|-------|--------|----------|---------------|
| DynUNet | CNN baseline | Full training | R3a, R3b |
| MambaVesselNet++ | SSM hybrid | Full training | R3b |
| SAM3 Vanilla | FM frozen | Zero-shot eval | R3b |
| SAM3 TopoLoRA | FM + LoRA | LoRA fine-tune | R3b |
| SAM3 Hybrid | FM + fusion | Hybrid train | R3b |
| VesselFM | FM pretrained | Zero-shot + finetune | R3c |

Before creating ANY experiment config, read paper_model_comparison.yaml.
```

#### T-G2: Add CLAUDE.md rule for config ↔ KG validation
**What**: New rule: "Before creating or modifying experiment configs that list models, cross-check against `knowledge-graph/domains/models.yaml::paper_model_comparison`. If the config disagrees with the KG, the CONFIG is wrong."

#### T-G3: Clean KG — remove non-paper model entries
**What**: Remove segresnet, swinunetr, attentionunet, unetr from KG files (adapters.yaml, claims.yaml, primary_3d_model.yaml). Keep only paper-lineup models.

#### T-G4: Update debug config CLAUDE.md section
**What**: Add debug_factorial.yaml to the debug config table in `src/minivess/config/CLAUDE.md`. Document: "debug = production minus epochs/data/folds."

---

### GROUP H: Test Harness Improvements (P1)

**Reviewer correction**: T-H1, T-H2, T-H3 are duplicates of T-A4, T-A3, T-B2 respectively. Removed.

#### T-H1: Per-volume metrics artifact format test
**What**: Verify `per_volume_metrics/` artifact format matches what biostatistics expects.
**TDD**: Schema validation test.

#### T-H2: ModelFamily enum cleanup
**What**: Clean up enum in `src/minivess/config/models.py`:
- Remove: MONAI_SEGRESNET, MONAI_SWINUNETR, MONAI_ATTENTIONUNET, MONAI_UNETR, COMMA_MAMBA, ULIKE_MAMBA, MULTITASK_DYNUNET
- Remove: SAM3_LORA (legacy alias — SAM3_TOPOLORA is the canonical name)
- Keep: DYNUNET, MAMBAVESSELNET, SAM3_VANILLA, SAM3_TOPOLORA, SAM3_HYBRID, VESSELFM
- Keep: CUSTOM (generic extension point for researchers adding their own models)
**TDD**: Assert ModelFamily has exactly 7 values (6 paper models + CUSTOM).

---

## Execution Order

```
Phase 1:  Config cleanup (A1-A4) — remove non-paper models, create debug config, fix T4
Phase 2:  Hydra + aux_calib fix (B1-B3) — wire aux_calib, verify 24+2 conditions, SkyPilot YAML
Phase 3:  Stub audit (I-AUDIT) — verify ALL components are wired, add loud failures
Phase 4:  MLflow migration (C1-C4) — slash-prefix in eval writers + DuckDB readers
Phase 5:  Logging gaps (D0-D2) — verify existing + add inference latency + stopped_early
Phase 6:  External test sets (J1-J7) — download, wire, test/ prefix, biostats split
Phase 7:  Flow integration tests (E1-E5) — e2e contract verification for all 5 flows
Phase 8:  Launch script (K1-K3) — THE CORE DELIVERABLE: run_factorial.sh + SkyPilot YAMLs
Phase 9:  KG & docs (G1-G4) — improve discoverability, update CLAUDE.md
Phase 10: GCP infra tests + test harness (F1-F4, H1-H2) — verification guards
Phase 11: Full prod test suite (`make test-prod`) — verify everything passes
```

**Dependencies**: Phase 1 → Phase 2 (composition needs correct model list + aux_calib wiring)
→ Phase 3 (audit needs working composition) → Phase 4-5 (parallel, no deps between them)
→ Phase 6 (external test sets need correct metric keys) → Phase 7 (integration tests need
all wiring complete) → Phase 8 (launch script needs all flows working) → Phase 9-10 (parallel)
→ Phase 11 (final gate — everything must pass).

---

## Success Criteria

Before merging this branch:
1. `make test-staging` passes (all tasks complete, <3 min)
2. `make test-prod` passes (full suite, ~5-10 min)
3. ALL 6 paper models have working configs through Hydra composition
4. ALL 24 factorial conditions + 2 zero-shot baselines resolve correctly
5. `with_aux_calib` is WIRED and produces `AuxCalibrationLoss` when `True`
6. MLflow logs use slash-prefix convention everywhere (no underscore keys)
7. No backward compat / normalization layers (greenfield — clean slate)
8. No T4 references in any config
9. No non-paper model adapter code in codebase
10. Debug factorial config (`debug_factorial.yaml`) verified reproducible
11. E2E contract tests pass for all 5 flows
12. `scripts/run_factorial.sh` generates correct `sky jobs launch` commands for BOTH:
    - Debug: `debug_factorial.yaml` → 26 conditions (24 trainable + 2 zero-shot)
    - Production: `paper_factorial.yaml` → 78 conditions (72 trainable + 6 zero-shot)
13. All SkyPilot YAMLs use `image_id: docker:`, L4 spot, NO T4, NO bare VM
14. VesselFM is zero-shot ONLY (no fine-tuning anywhere in configs or code)

---

### GROUP I-AUDIT: Stub vs Wired Audit (P0 — CRITICAL)

#### T-AUDIT1: Audit ALL pipeline components for stub vs wired status
**Why**: User mandated P0 audit — "double-check that EVERY FUCKING PLANNED component needed
on this debug and full production run are actually wired and not just planned as some stub!"
**What**: Systematically verify every component in the e2e chain:
1. Training flow: Hydra composition → model build → loss build (with_aux_calib!) → training loop → MLflow logging
2. Post-training flow: checkpoint discovery → SWA → calibration → conformal → MLflow artifacts
3. Analysis flow: upstream discovery → model loading → **EXTERNAL TEST EVALUATION** → ensemble → comparison → champion tagging
4. Biostatistics flow: MLflow read → DuckDB → ANOVA → pairwise → figures → **TRAIN/VAL vs TEST split**
5. Deploy flow: champion discovery → ONNX export → BentoML import → artifact generation → promotion
**For each component**: Is it (a) WIRED and functional, (b) STUB returning empty/None, or (c) NOT IMPLEMENTED?
**Stubs that CAN stay as stubs**: PostHog, Sentry (optional observability — `logger.warning()`)
**Stubs that MUST be wired**: External test evaluation, aux_calib, test metric prefix, biostatistics split
**TDD**: Integration test that runs the full e2e chain and asserts NO empty returns at any stage.
**LOUD FAILURES**: Every pipeline function must raise on empty input (CLAUDE.md Rule 25).

#### T-AUDIT2: Add loud failure enforcement to all pipeline entry points
**Why**: CLAUDE.md Rule 25 — "Loud failures, never silent discards."
**What**: Audit all `@flow` and `@task` decorated functions. If any accept empty input and
silently return empty results, add `raise ValueError()` or `logger.error()` with descriptive message.
**Scope**: All flow files in `src/minivess/orchestration/flows/`, all pipeline modules.
**Classification**:
- **Critical pipeline**: `raise ValueError("...")` — blocks execution, forces fix
- **Optional integrations** (PostHog, Sentry): `logger.warning("... not configured, skipping")`
- **Stubs awaiting implementation**: `raise NotImplementedError("...")`
**TDD**: Test that critical pipeline functions raise on empty input.

---

### GROUP J: External Test Set Evaluation (P0 — PUBLICATION BLOCKER)

**Detailed plan**: `docs/planning/test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml`

#### T-J1: Download + DVC-version DeepVess (test set)
**What**: Download DeepVess from Cornell eCommons, create `data/external/deepvess/`.
Paper verified 7 labeled multiphoton volumes (1 training + 6 independent test, all mouse brain cortex).
DVC-version with `dvc add data/external/deepvess`. Push to GCS.
TubeNet EXCLUDED: only 1 two-photon volume (olfactory bulb — different organ, not useful).
Debug AND production use ALL 7 DeepVess volumes (same test data for both).
**TDD**: Test that `discover_external_test_pairs()` finds image/label pairs.
**FIX**: Update `n_volumes` in `external_datasets.py` after download (currently wrong — Claude confabulation).

#### T-J2: Wire data_flow → analysis_flow external dataloaders
**What**: Fix `_build_dataloaders_from_config()` to actually build DataLoaderDict from
external datasets. Pass non-empty dataloaders_dict to analysis_flow.
**LOUD FAILURE**: If external data dir is missing, `raise FileNotFoundError()` not `return {}`.
**TDD**: Integration test: data_flow discovers → analysis_flow receives non-empty loaders.

#### T-J3: Implement test/ metric prefix in evaluation_runner
**What**: When evaluating external test datasets, use `test/{dataset}/{metric}` prefix instead
of `eval_{dataset}_{metric}`. Add `test/aggregate/{metric}` (volume-weighted across test sets).
**TDD**: Test asserting test/ prefix for external datasets, eval/ for train/val.

#### T-J4: Update DuckDB readers for test/ prefix
**What**: Update `biostatistics_duckdb.py` and `duckdb_extraction.py` to recognize `test/` prefix
and route to separate DuckDB table (`test_metrics` alongside `eval_metrics`).
**TDD**: Test that test/ metrics land in test_metrics table.

#### T-J5: Add split={trainval, test} to biostatistics module
**What**: Biostatistics computes ALL analyses separately for trainval and test:
```python
for split in ["trainval", "test"]:
    results[split] = compute_all_statistics(data[split])
```
All ANOVA, pairwise, effect sizes, specification curve — full analysis for both splits.
**TDD**: Test that biostatistics output has both splits with identical structure.

#### T-J6: Wire visualization DataFrames for domain gap plots
**What**: Build DataFrames for `plot_domain_gap()` and `plot_per_volume_scatter()` from
DuckDB test_metrics table. Connect to dashboard_flow.
**TDD**: Test that visualization receives correct DataFrame structure.

#### T-J7: Update KG with dataset roles
**Already done (this session)**: data.yaml updated — VesselNN = drift_detection_only.
**Remaining**: Update manuscript/results.yaml with test set requirements for R3b/R3c.

---

### GROUP K: Deterministic Launch Script (P0 — Core Deliverable)
<!-- Was GROUP I before inserting I-AUDIT and J -->

#### T-I1: Create `scripts/run_factorial.sh` — single deterministic launch script
**Why**: The CORE DELIVERABLE of this branch. One script that launches the entire factorial
experiment on GCP via SkyPilot. Same script for debug AND production — only the YAML changes.
**Usage**:
```bash
# Debug run (2 epochs, half data, 1 fold, 24+2 conditions):
./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml

# Production run (50 epochs, full data, 3 folds, 72+6 conditions):
./scripts/run_factorial.sh configs/hpo/paper_factorial.yaml
```
**What the script does**:
1. Reads the factorial YAML (models, losses, aux_calib levels, folds, epochs, data size)
2. For each condition: `sky jobs launch deployment/skypilot/train_factorial.yaml --env MODEL_FAMILY=... --env LOSS_NAME=... --env FOLD_ID=... --env WITH_AUX_CALIB=... --env-file .env -y`
3. After all training jobs: launches post-training, analysis, biostatistics, deploy flows
4. Pure `sky jobs launch` calls in a loop — NO `claude -p`, NO `screen`, NO `nohup`, NO pipe chains
5. Each SkyPilot job uses `image_id: docker:` (bare VM banned), L4 spot (T4 banned)
**BANNED** (per metalearning 2026-03-09, 2026-03-16):
- Any wrapper around `claude -p`
- `screen -dmS`, `nohup`, `--output-format stream-json | jq`
- Heartbeat/watchdog background processes
- ANY bash complexity beyond the `sky jobs launch` loop
**TDD**: Test that script is syntactically valid (`bash -n`), reads the correct YAML fields,
generates expected `sky jobs launch` commands for both debug (26 conditions) and production (78 conditions).

#### T-I2: Create `deployment/skypilot/train_factorial.yaml` — parameterized SkyPilot task
**Why**: Each `sky jobs launch` call needs a parameterized YAML.
**Env vars**: MODEL_FAMILY, LOSS_NAME, FOLD_ID, WITH_AUX_CALIB, MAX_EPOCHS, MAX_TRAIN_VOLUMES, MAX_VAL_VOLUMES, EXPERIMENT_NAME
**GPU**: L4 spot (T4 banned). Docker image from GAR.
**Setup**: DVC pull + HF login + splits copy. NO `apt-get`, NO `uv sync` (Docker image has everything).
**Run**: Prefect flow invocation (NOT standalone script — Rule #17).
**TDD**: Test that YAML is valid, contains no T4, uses `image_id: docker:`.

#### T-I3: Create parameterized SkyPilot YAMLs for post-training, analysis, biostatistics, deploy flows
**Why**: The e2e chain requires separate SkyPilot tasks for each downstream flow.
**Files**:
- `deployment/skypilot/post_training_factorial.yaml`
- `deployment/skypilot/analysis_factorial.yaml`
- `deployment/skypilot/biostatistics_factorial.yaml` (CPU, no GPU needed)
- `deployment/skypilot/deploy_factorial.yaml` (CPU, no GPU needed)
**TDD**: Test that all YAMLs are valid and reference correct Docker images.

---

## Post-Merge: Debug Run (Separate Branch)

After this branch merges to `main`:
1. Create `debug/gcp-factorial-run` branch
2. Execute with Ralph Loop Skill for infrastructure monitoring
3. Execute with self-learning TDD Skill for code fixes
4. 24 trainable conditions + 2 zero-shot baselines on GCP L4
5. Full e2e: Train → Post-Train → Evals → Biostats → Deploy
6. Verify all MLflow artifacts, metrics, params are logged correctly
7. Verify biostatistics can compute factorial ANOVA from logged data

---

## Metalearning Summary — Failures Relevant to This Plan

62 metalearning documents were reviewed line-by-line. The following failures are
directly relevant to this plan and must NOT be repeated during execution:

### Category A: Incomplete Information Gathering
- **2026-03-02**: SAM3 vs SAM2 confusion — used wrong model due to not web-searching
- **2026-03-15**: KG scope blindness — Mamba missing from KG because code was scanned, not plans
- **2026-03-17**: Model lineup ignorance — could not name 6 paper models
- **2026-03-19**: paper_factorial.yaml wrong models (THIS SESSION) — 3rd occurrence
- **Rule**: Read KG `paper_model_comparison.yaml` BEFORE any experiment config work

### Category B: Asking Humans What Tools Can Answer
- **2026-03-16**: Asked "Is Network Volume created?" instead of `sky storage ls`
- **2026-03-19**: Asked about VesselFM data availability instead of `ls data/external/` (THIS SESSION)
- **Rule**: Check with CLI tools FIRST, ask user only for DECISIONS

### Category C: Misunderstanding Architectural Mandates
- **2026-03-14**: Docker resistance — offered bare-metal alternatives
- **2026-03-14**: SkyPilot misunderstanding — treated as "RunPod launcher"
- **2026-03-14**: Poor repo vision — planned DynUNet on cloud (wastes credits)
- **2026-03-16**: Level 4 mandate negotiation — offered "lighter alternatives"
- **Rule**: NEVER downgrade non-negotiable architecture. NEVER propose shortcuts.

### Category D: Debug = Production (NEW — This Session)
- **2026-03-19**: Proposed 12 conditions instead of 24 (skipped aux_calib)
- **2026-03-19**: Proposed skipping zero-shot baselines in debug
- **Rule**: Debug = FULL production with ONLY 3 reductions: epochs, data, folds

### Category E: Backward Compatibility in Greenfield (NEW — This Session)
- **2026-03-19**: Proposed backward compat for MLflow underscore keys
- **Rule**: Greenfield project, zero production users. Delete old formats entirely.

### Category F: KG Errors Persisting
- **2026-03-19**: VesselFM `zero_shot_AND_finetuned` → should be `zero_shot_only` (FIXED)
- **2026-03-19**: paper_factorial.yaml had segresnet instead of mambavesselnet (FIXED)
- **Rule**: Verify KG against user statements. If KG contradicts user, KG has a bug.

### Meta-Pattern: "Confident Incompleteness"
All 62 documents share one root cause: taking action based on partial understanding
with high confidence. The fix: **read completely (30% reading, 70% implementing),
verify with tools, follow mandates, no shortcuts.** See CLAUDE.md Rule #24.

---

## Post-Debug: Full Production Run (Separate Branch)

After debug run succeeds:
1. 24 cells × 3 folds = 72 training runs at 50 epochs
2. Zero-shot: SAM3 Vanilla (3 folds), VesselFM (DeepVess/TubeNet)
3. Full post-training → evaluation → biostatistics → deployment
4. Estimated cost: ~$65 on GCP L4 spot
