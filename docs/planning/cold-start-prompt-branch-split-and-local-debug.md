# Cold-Start Prompt: Local 3-Flow Debug Execution

**Date**: 2026-03-21
**Current branch**: `test/local-debug-3flow-execution` (fresh from `main`)
**Session goal**: Run local 3-flow debug (Post-Training → Analysis → Biostatistics)

**STATUS**: Branch split DONE (PR #890 code + PR #891 docs merged to main).
All stale branches deleted. 5909 prod tests pass. Clean state.

---

## How to Use This Prompt

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-branch-split-and-local-debug.md"
```

---

## Part 1: Local 3-Flow Debug Execution (READY)

### Plan file:
`docs/planning/local-debug-3flow-execution-plan.xml`

### Critical reviewer findings (MUST address before execution):

1. **run_analysis_flow() signature** — requires `EvaluationConfig`, `model_config_dict`,
   `dataloaders_dict` (actual PyTorch DataLoaders), NOT a simple experiment name.
   Must construct these programmatically or use the `__main__` entry point with
   `UPSTREAM_EXPERIMENT` env var.

2. **Prefect server** — `get_run_logger()` requires Prefect flow context. Options:
   - Start Prefect server: `prefect server start` (background)
   - Use `prefect_test_harness()` in script wrapper
   - PREFECT_DISABLED is NOT a real env var (does nothing)

3. **run_biostatistics_flow()** — takes `config_path` parameter, not experiment name.
   Need `configs/biostatistics/smoke_local.yaml` with `experiment_names: ["smoke_local_evaluation"]`
   and `min_folds_per_condition: 1`.

4. **Factor name mismatch** — `_compose_condition_key()` uses `loss_function` and
   `with_aux_calib` but `BiostatisticsConfig.factor_names` defaults to `loss_name`
   and `aux_calibration`. Must reconcile via FACTOR_NAME_MAPPING.

5. **SWA plugin bug** — `model_merging.py:81` and `multi_swa.py:69` hardcode
   `ckpt["state_dict"]` — needs same fallback chain as swa.py fix.

### Monitoring:
- Use `/factorial-monitor` skill for flow tracking
- Time every phase with `time.perf_counter()`
- MLflow logs fine-grained timing metrics

### Report output:
`docs/planning/run-debug-factorial-experiment-report-2nd-pass-local.md`

### Execution script:
Create `scripts/run_local_debug_3flow.sh` that:
1. Sets all env vars
2. Starts Prefect server (background, not screen/tmux)
3. Runs Post-Training flow
4. Runs Analysis flow
5. Runs Biostatistics flow
6. Kills Prefect server
7. Reports timing
NO screen, NO tmux, NO nohup. Direct execution.
See: `.claude/metalearning/2026-03-09-overnight-script-silent-freeze.md`

---

## Available Artifacts

### DynUNet training runs (100 epochs, 7 checkpoints each):
```
Experiment: dynunet_loss_variation_v2 (id=843896622863223169)
  01d904c61b: dynunet + cbdice_cldice
  4b2451ac0a: dynunet + dice_ce_cldice
  3a9f361520: dynunet + cbdice
  af4adc1599: dynunet + dice_ce
Tags backfilled: model_family, flow_name, fold_id, with_aux_calib, loss_function
Checkpoint format: {model_state_dict: {44 tensors}, optimizer_state_dict, ...}
```

### MiniVess data (local):
```
data/raw/minivess/imagesTr/: 70 volumes (.nii.gz)
data/raw/minivess/labelsTr/: 70 volumes (.nii.gz)
```

### Test suite state:
- 5598 passed, 2 skipped (mamba IS installed skip + port binding warning)
- CUDA 12.6 active, mamba-ssm 2.3.1 installed

---

## Files to Read Before Starting

```
1. CLAUDE.md
2. docs/planning/cold-start-prompt-branch-split-and-local-debug.md (THIS FILE)
3. docs/planning/local-debug-3flow-execution-plan.xml
4. docs/planning/pre-debug-factorial-local-post-analysis-biostats-final-qa-plan.xml
5. .claude/metalearning/2026-03-09-overnight-script-silent-freeze.md
6. .claude/metalearning/2026-03-16-overnight-runner-script-freeze-v2.md
7. .claude/skills/factorial-monitor/ (SKILL.md)
8. src/minivess/orchestration/flows/analysis_flow.py (run_analysis_flow signature)
9. src/minivess/orchestration/flows/biostatistics_flow.py (run_biostatistics_flow signature)
```

---

## What NOT to Do

- Do NOT use screen, tmux, or nohup for the execution script
- Do NOT guess function signatures — READ the actual code
- Do NOT assume PREFECT_DISABLED=1 works — it's not a real env var
- Do NOT skip the branch split — 39 commits on one branch is unmanageable
- Do NOT merge without running make test-prod first
- Do NOT create PRs targeting prod — always target main
