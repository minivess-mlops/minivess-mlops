# Cold-Start Prompt: SWAG + Calibration Metrics Implementation

**Date**: 2026-03-21
**Current branch**: `test/local-debug-3flow-execution`
**Session goal**: Execute `calibration-and-swag-implementation-plan.xml` via `/self-learning-iterative-coder`

---

## How to Use This Prompt

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-swag-calibration-implementation.md"
```

---

## STATUS: Phases A0-A1 ALREADY DONE — Start at Phase A2

**Discovery**: A prior session on this branch already implemented:
- Phase A0: SWA → checkpoint averaging rename (all ~18 files)
- Phase A1: SWAG plugin (swag.py + swag plugin + config + registration)
Pre-commit hooks caught lint/mypy issues in the SWAG code — fixed and pushed.

**Start execution at Phase A2 (SWAG tests) then Phase B0 (calibration metrics).**

Previous session completed:
- 2nd pass local 3-flow debug (7 bugs found/fixed, all flows passed)
- SWA mislabeling discovered → metalearning doc + CLAUDE.md Rule #30
- SWAG selected as publication-gate post-training method (Issue #892)
- Comprehensive calibration metrics report (20 metrics identified)
- clDice skeletonization verification (Lee 1994 algorithm identical)
- Multi-metric checkpoint ensemble report with Mustafa et al. findings
- 6 speculative P2 science issues created (#894-#899)
- 3rd pass debug plan created
- MLflow epoch-level metrics confirmed working (no TensorBoard)
- All docs committed and pushed

**Test suite**: 5598 passed, 2 skipped, 0 failed (make test-staging)

---

## What to Execute

### Plan file:
`docs/planning/calibration-and-swag-implementation-plan.xml`

### Execution order (6 phases):

**Phase A0: Rename SWA → Checkpoint Averaging (~18 files)**
Pure rename, no logic changes. Tasks A0.1-A0.6:
- A0.1: `SWAPlugin` → `CheckpointAveragingPlugin` (rename file + class)
- A0.2: `MultiSWAPlugin` → `SubsampledEnsemblePlugin` (rename file + class)
- A0.3: `uniform_swa()` → `uniform_checkpoint_average()` in model_soup.py
- A0.4: Update `PostTrainingConfig` fields (swa → checkpoint_averaging)
- A0.5: Update `post_training_flow.py` orchestration references
- A0.6: Update configs/factorial/*.yaml, tests, config keys

**Phase A1: Implement SWAG Plugin**
Real SWAG (Maddox et al. 2019) using `torch.optim.swa_utils`:
- A1.1: SWAGConfig pydantic model
- A1.2: SWAGModel wrapper (AveragedModel + second-moment tracking)
- A1.3: SWAGPlugin (post-training plugin with resumed training)
- A1.4: Register in post_training_flow.py
- A1.5: Analysis flow integration (load SWAG, sample K models)
- A1.6: Update factorial configs

**Phase A2: SWAG Tests (TDD)**
- A2.1-A2.3: Unit tests for SWAGModel, SWAGPlugin, integration test

**Phase B0: Calibration Metrics Module (ALL 20 metrics)**
- B0.1: Create `calibration_metrics.py` with Tier 1 (fast) + Tier 2 (comprehensive)
- B0.2: Tests for ALL metrics

**Phase B1: Training Loop Integration (Tier 1 metrics)**
- B1.1: Add ECE, pECE, BA-ECE + 6 more fast metrics to validation loop
- B1.2: Track in MultiMetricTracker (logged per-epoch via existing MLflow mechanism)
- B1.3: Tests

**Phase B2: Analysis Flow Integration (ALL metrics)**
- B2.1: Add ALL 20 calibration metrics to UnifiedEvaluationRunner
- B2.2: Calibration shift computation (MiniVess vs DeepVess)
- B2.3: Reliability diagrams
- B2.4: Tests

### After implementation → Run 3rd pass debug:
`docs/planning/run-debug-factorial-experiment-report-3rd-pass-plan.xml`

---

## Three Co-Primary Calibration Metrics (ANOVA response variables)

| Metric | Dimension | Why |
|--------|-----------|-----|
| **ECE** | Global | Standard baseline, comparable to all literature |
| **pECE** | Class-aware | Vessel-specific FP overconfidence penalty (Li et al. 2025) |
| **BA-ECE** | Spatial | Calibration WHERE it matters — vessel boundaries (Zeevi et al. 2025) |

All 20 metrics computed, but these 3 are co-primary for the factorial ANOVA.
All 3 are fast enough for per-epoch evaluation in the training loop.

**Auto-discovery**: The factorial YAML `biostatistics.calibration_co_primary_metrics`
field feeds into `BiostatisticsConfig` which feeds into the ANOVA. This must be
implemented as part of Phase B.

---

## Knowledge Graph Updates Needed (Part of Phase B)

1. **CREATE** `knowledge-graph/decisions/L3-technology/calibration_metrics.yaml`
   - Status: resolved
   - Decision: ECE + pECE + BA-ECE as co-primary
   - All 20 computed but 3 are ANOVA response variables

2. **UPDATE** factorial YAMLs with `calibration_co_primary_metrics` section

3. **UPDATE** `BiostatisticsConfig` to auto-discover from factorial YAML

---

## Key References (read before implementing)

```
1. docs/planning/calibration-and-swag-implementation-plan.xml (THE PLAN)
2. docs/planning/run-debug-factorial-experiment-report-3rd-pass-plan.xml (debug plan)
3. docs/planning/segmentation-calibration-metrics-and-losses-sdc-pece-ace-mce-ece-etc.md
4. docs/planning/swa-swag-multi-swa-swag-pswa-lawa-model-soup-checkpoint-averaging-report.md
5. docs/planning/mlflow-vs-tensorboard-status-report-for-epoch-level-metrics.md
6. .claude/metalearning/2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md
7. CLAUDE.md (Rule #30: Algorithms Must Match Their Literature Definition)
8. src/minivess/pipeline/post_training_plugins/swa.py (TO BE RENAMED)
9. src/minivess/pipeline/post_training_plugins/multi_swa.py (TO BE RENAMED)
10. src/minivess/config/post_training_config.py (PostTrainingConfig structure)
11. src/minivess/orchestration/flows/post_training_flow.py (plugin registry)
12. src/minivess/pipeline/metrics.py (SegmentationMetrics — add calibration here)
13. src/minivess/config/biostatistics_config.py (add calibration_co_primary_metrics)
```

---

## Library References for Implementation

| Library | Use | Install |
|---------|-----|---------|
| `torch.optim.swa_utils` | SWAG core: AveragedModel, SWALR, update_bn | Built into PyTorch |
| `torchmetrics` | ECE, MCE, RMSCE | Already installed |
| [SDC-Loss repo](https://github.com/EagleAdelaide/SDC-Loss) | pECE, CECE | Fetch metrics.py |
| [Avg-Calib-Losses](https://github.com/cai4cai/Average-Calibration-Losses) | ACE, hL1-ACE | Fetch + MONAI handlers |
| `scipy.ndimage` | distance_transform_edt for BA-ECE | Already installed |
| `properscoring` | CRPS | `uv add properscoring` |

---

## What NOT to Do

- Do NOT use `skeletonize_3d` — use `skeletonize` (see clDice report)
- Do NOT call anything "SWA" that isn't Izmailov et al. 2018 (Rule #30)
- Do NOT hardcode calibration metric names — auto-derive from factorial YAML
- Do NOT skip per-epoch logging — ALL metrics must be logged with step=epoch+1
- Do NOT implement Tier 2 metrics in the training loop — only Tier 1 (fast)
- Do NOT create checkpoint averaging factorial factor — removed from publication design
- Do NOT skip tests — TDD mandatory via self-learning-iterative-coder
