# Cold-Start Prompt: 3rd GCP Debug Factorial Run

**Date**: 2026-03-21
**Branch**: `test/debug-factorial-run`
**Session goal**: Re-run the 384-condition factorial on GCP with all fixes applied

---

## What Was Done (2026-03-21 Session)

### Issues Closed (#885, #889, #888, #886, #887, #884)
- **#885**: Per-method MLflow runs for factorial discovery (P0 blocker)
- **#889**: EnsembleBuilder includes post-training variants
- **#888**: Zero-shot baseline discovery and tagging
- **#886**: UQ from deep ensembles (predict_with_uncertainty task)
- **#887**: UncertaintyStoragePolicy (debug vs production)
- **#884**: CTK config.toml path fix (eliminated false skip)

### Biostatistics 6-Factor Pipeline (Phases 0-3)
- SourceRun: 4 new fields (post_training_method, recalibration, ensemble_strategy, is_zero_shot)
- DuckDB: 4 new columns in runs table
- Discovery: _read_tags() added to _parse_run_dir() for Layer B+C tags
- _compose_condition_key(): 6-factor "__"-separated condition keys
- task_compute_factorial_anova: Prefect task wired into biostatistics flow
- FACTOR_NAME_MAPPING: YAML "method" → tag "post_training_method"
- 5-way and 6-way ANOVA validated with synthetic data

### Infrastructure Fixes
- CUDA 12.6 toolkit active (old 11.5 removed)
- mamba-ssm 2.3.1 compiled and importable locally
- CUDA_HOME=/usr/local/cuda-12.6 in .env.example
- duckdb-skills plugin installed for interactive analysis

### Local Smoke Test (VERIFIED)
- Post-Training flow: 4 runs created (2 losses × {none, swa})
- Run naming: dynunet__cbdice_cldice__fold0__swa (correct)
- Tags: post_training_method=swa, recalibration=none (correct)
- SWA checkpoint averaging: works end-to-end

### Test Suite
- 5598 passed, 2 skipped (mamba-ssm IS installed skip + port binding warning)
- 0 failures

---

## Remaining Before GCP Run

### Must Fix (P0)
1. **Glitch #8**: Checkpoint volume mount — SkyPilot YAML must persist
   checkpoints to GCS after training. Fix: verify `file_mounts` or
   `storage_mounts` in SkyPilot YAML maps /checkpoints → GCS bucket.

### Must Fix (P1)
2. **Glitch #9**: SAM3-TopoLoRA LoRA Conv2d bug — investigate and fix
3. **Glitch #12**: Zero-shot max_epochs=0 validation — SkyPilot YAML fix
4. **Docker image rebuild**: INSTALL_MAMBA=1 for mamba-ssm in cloud
5. **ensemble_strategy tag**: analysis flow must set this tag on evaluation
   runs (Phase 0.5 T0.5.3-T0.5.4 not yet implemented in analysis_flow.py)

### Decision Needed
- **Storage format**: Zarr chosen (see docs/planning/zarr-vs-pt-for-5d-uq-array.md)
  but not yet implemented. Continue with .pt for debug run, switch to zarr for production.

---

## GCP Run Plan

1. Fix Glitches #8, #9, #12
2. Rebuild Docker base image with INSTALL_MAMBA=1
3. Run 24 training conditions (4 models × 3 losses × 2 calib × 1 fold)
4. Verify checkpoints persist to GCS
5. Download artifacts: make sync-cloud-artifacts
6. Run Layers B+C+D locally (post-training + analysis + biostatistics)
7. Verify full 384-condition factorial ANOVA

---

## Mandatory Files to Read

```
1. CLAUDE.md
2. docs/planning/pre-debug-factorial-local-post-analysis-biostats-final-qa-plan.xml
3. docs/planning/debug-factorial-local-post-analysis-biostats.xml
4. docs/planning/run-debug-factorial-experiment-report.md (Glitch catalog)
5. docs/planning/zarr-vs-pt-for-5d-uq-array.md (storage format decision)
6. .claude/metalearning/2026-03-21-*.md (4 metalearning docs from this session)
```

---

## Commits on test/debug-factorial-run (this session)

```
2bdb8a4 fix: RAM crash + post-training/analysis factorial pipeline verification
69c29f4 feat: per-method MLflow runs for factorial discovery (#885)
341cc57 feat: EnsembleBuilder includes post-training variants (#889)
31d5640 feat: zero-shot baselines, UQ task, storage policy, CTK logic tests
aea2475 fix: CTK config.toml path lookup — eliminate false skip (#884)
8b592cf docs: metalearning failures + zarr vs .pt decision matrix
41a0b6a fix: CUDA_HOME in .env.example + mamba-ssm installable locally
acc3180 docs: comprehensive final QA plan with 3-reviewer optimization
f1cfdf8 feat: biostatistics 6-factor pipeline — Layer B+C support (Phases 0-3)
6d34982 feat: DuckDB skills + local smoke test verified (Phases 4-5)
```
