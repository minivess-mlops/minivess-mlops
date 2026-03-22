# Cold-Start Prompt: Session Continuation from 2026-03-20

## To run:
```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-session-continuation-2026-03-20.md
Branch: test/debug-factorial-run" --dangerously-skip-permissions
```

---

## CONTEXT: What was accomplished in the previous session (2026-03-20)

This was a 12+ hour session that ran the first debug factorial experiment on GCP,
found 12 infrastructure glitches, fixed them all via TDD + 3 code review rounds,
and started implementing the biostatistics pipeline upgrades.

### Branch: `test/debug-factorial-run` (from main after PR #871)

### Key commits (newest first):
```
a570989 feat: biostatistics N-way factorial ANOVA + co-primary metrics
9ee12eb docs: biostatistics double-check XML + preregistration research
97926f2 docs: biostatistics flow debug double-check XML
1aa4754 fix: code review round 3 nit + PROD test suite verified
1ff82d0 fix: code review round 2 — 1 CRITICAL + 2 MEDIUM issues
7486cbc fix: 3 CRITICAL + 4 HIGH/MEDIUM issues from code review round 1
8afe302 fix: resolve 4 open glitches (#8 #9 #10 #12)
13dad6b fix: prevent MLflow from creating relative dirs in repo root
```

### Test suite status (as of last commit):
- `make test-staging`: **5441 passed, 3 skipped, 0 failed** (4:40)
- `make test-prod`: **5739 passed, 37 skipped, 0 failed** (5:50)

### Docker image:
- Rebuilt with `INSTALL_MAMBA=1` (mamba-ssm 2.3.1 compiled)
- Pushed to GAR: `europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
- SHA: `sha256:115bfb78dcdfaf6b53add59ab55daf85f2dcc846d9ac81d0a04606d9be847376`

---

## OPEN ISSUES (P0 = must fix NOW)

| # | Priority | Title | Status |
|---|----------|-------|--------|
| **#878** | **P0** | MLflow checkpoint upload 413 on Cloud Run | **FIXED in code** (train_factorial.yaml uses ${MLFLOW_TRACKING_URI} + GCS file_mounts). Needs verification on cloud. |
| **#880** | **P0** | Flaky Prefect SQLite test failures | **NOT FIXED** — 4 module-level PREFECT_DISABLED mutations need removal |
| #879 | P2 | Implement LoRAConv2d | Created, deferred |
| #875 | P1 | Docker Compose services for prod tests | Open |

---

## WHAT NEEDS TO BE DONE (in priority order)

### 1. FIX: Prefect SQLite flakiness (#880) — 30 min

Remove 4 module-level `os.environ["PREFECT_DISABLED"] = "1"` mutations from test files.
Replace with `monkeypatch.setenv` in fixtures.

**Files to fix** (from metalearning `.claude/metalearning/2026-03-19-flaky-prefect-sqlite-dismissed.md`):
1. `tests/v2/unit/test_analysis_flow.py:15`
2. `tests/v2/unit/test_acquisition_flow.py:14`
3. `tests/v2/integration/test_analysis_flow_integration.py:27`
4. `tests/v2/unit/pipeline/test_trainer_nan_handling.py:77`

**Also**: Add pre-commit hook banning `os.environ["PREFECT_DISABLED"]` in test files.
Plan: `docs/planning/robustifying-flaky-prefect-sqlite-issues.md`

### 2. CONTINUE: Biostatistics flow phases 2-6

**XML plan**: `docs/planning/biostatistic-flow-debug-double-check.xml`
**Skill**: `/self-learning-iterative-coder`

**Phase 0-1**: DONE (commit `a570989`)
- SourceRun has model_family + with_aux_calib
- DuckDB schema has with_aux_calib column
- N-way ANOVA accepts factor_names parameter
- BiostatisticsConfig has co_primary_metrics (clDice + MASD)
- 13 new tests, all passing

**Phase 2**: PARTIALLY DONE
- compute_factorial_anova() upgraded to N-way ✓
- Still needs: K=1 debug fallback (per-volume replication instead of fold random effect)

**Phase 3**: NOT STARTED — Specification curve analysis
- New file: `src/minivess/pipeline/biostatistics_specification_curve.py`
- ALL researcher degrees of freedom (metric, model, loss, calib, fold subset, threshold, aggregation)
- Permutation test for null
- Reference: Simonsohn et al. (2020)

**Phase 4**: NOT STARTED — Rank stability + derived metrics
- Kendall's tau between metric rankings (clDice vs DSC rank inversion IS a finding)
- Calibration metrics (Brier, O/E, IPA)
- Bayesian analysis upgrades

**Phase 5**: NOT STARTED — TRIPOD+AI preregistration mapping
- Create: `docs/planning/preregistration-tripod-mapping.md`
- Map each statistical test to TRIPOD+AI items

**Phase 6**: NOT STARTED — Integration test with synthetic data

### 3. NEW: Analysis Flow double-check

**User's verbatim request** (saved for context):
> After this, let's create a similar double-check for Analysis Flow that should be
> able to access all the models in MLflow, serve them with MLflow so that the Analysis
> can run the inference using the served models for or the dataset (in our current
> situation we only have trainval minivess and test deepcess but obviously this need
> to be as flexible as well so that in the future, the users can add whatever datasets
> that they went). The Analysis should be able to create all the new mlruns from
> different ensembling permutations (that are obviously factors now for Biostatistics
> flow). Is it clear now in the kg what our ensemble permutations are? Do you know
> what the output should be to MLflow from Analysis Flow as we should create new
> experiment and not use the same name as we used in Post-training and Modeling Flow?
> So we get a lot more unique mlruns from Analysis Flow that are then used as input
> for the Biostatistic Flow! Analysis creates all the uncertainty quantification stats
> from deep ensembles as well in addition to the averaged mlflow metrics from the
> ensembling as we obviously average the created voxel probability masks and then
> compute the metrics for those rather than trying to average the clDice metrics of
> all the submodels. The Analysis Flow should know how to correctly use the custom
> Mlflow Class created that loops through the invididual submodels in the ensemble
> (you could create a P2 issue that could do "parallel for" loop if we had 16 GPUs,
> we could do parallel loop, and examine how
> https://www.nvidia.com/en-us/technologies/multi-instance-gpu/ works in this case
> (NVIDIA Multi-Instance GPU, MIG) as it should allow more efficient processing, but
> these are more performance optimization and document these in the P2 Issue so that
> in the future we can implement them if needed).

**Deliverable**: `docs/planning/analysis-flow-debug-double-check.xml`
- Read KG THOROUGHLY first (navigator → domains → decisions → manuscript)
- Ask user questions (up to 10) to align on scope
- Key topics: ensemble permutations, model serving, UQ from deep ensembles,
  per-volume probability mask averaging, separate MLflow experiment name

### 4. LATER: 2nd pass debug factorial run

**When**: After all code fixes are verified
**Script**: `scripts/run_factorial_2nd_pass.sh` (15 conditions: 14 failed + 1 probe)
**Prerequisite**: All fixes committed + Docker image pushed + `make test-staging` green
**Plan**: `docs/planning/run-debug-factorial-experiment-2nd-pass.md`

### 5. LATER: 3rd pass (full 26 conditions)

After 2nd pass 15/15 SUCCEEDED → run all 26 conditions as final validation.

---

## CRITICAL CONTEXT (from Knowledge Graph — DO NOT SKIP)

### Primary Metrics (MetricsReloaded)
- **co-primary**: clDice + MASD (equal weight, Holm-Bonferroni correction)
- **FOIL**: DSC (Dice) — included to demonstrate misleading rankings for tubular structures.
  The rank inversion between DSC and clDice IS a paper finding.
- **secondary**: HD95, ASSD, NSD, BE_0, BE_1, junction_F1 (BH-FDR correction)

### Factorial Design
- **Factors**: model_family[4] × loss_name[3] × aux_calibration[2] = 24 cells
- **Random effect**: fold_id (K=3 production, K=1 debug)
- **Zero-shot baselines**: sam3_vanilla (MiniVess), vesselfm (DeepVess ONLY — data leakage on MiniVess)
- **Production runs**: 72 (24 × 3 folds)
- **Debug runs**: 24 (24 × 1 fold)

### Platform Framing (Nature Protocols)
- This is a PLATFORM paper, NOT SOTA segmentation paper
- The contribution IS the platform capability
- External test evaluation demonstrates PLATFORM's ability to handle arbitrary datasets
- Model comparison demonstrates ModelAdapter ABC handles diverse architectures

### Data
- **MiniVess**: 70 volumes, 3-fold CV, seed=42 (NO held-out test set — all cross-validated)
- **DeepVess**: ~7 volumes, external test ONLY
- **VesselNN**: drift detection ONLY (data leakage, same PI)
- **TubeNet**: EXCLUDED (wrong organ)
- **Per-volume data ESSENTIAL**: N=23 volumes per fold, NOT per-fold aggregates

### Two-Provider Architecture
- **GCP**: staging/prod (Cloud Run MLflow, GCS data, L4 GPUs via SkyPilot)
- **RunPod**: dev only (RTX 4090)
- Docker is the execution model. SkyPilot = intercloud broker.

---

## MANDATORY FILES TO READ BEFORE ANY ACTION

### KG (READ ALL — context amnesia caused bad questions last time):
1. `knowledge-graph/navigator.yaml` → route to domains
2. `knowledge-graph/domains/*.yaml` — ALL domain files
3. `knowledge-graph/decisions/L3-technology/primary_metrics.yaml` — metric hierarchy
4. `knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml` — 6-model lineup
5. `knowledge-graph/decisions/L3-technology/dataset_strategy.yaml` — data splits
6. `knowledge-graph/decisions/L2-architecture/ensemble_strategy.yaml` — ensemble design

### Plans (current work):
7. `docs/planning/biostatistic-flow-debug-double-check.xml` — biostatistics plan (phases 2-6 remaining)
8. `docs/planning/run-debug-factorial-experiment-report.md` — 12-glitch report from 1st pass
9. `docs/planning/run-debug-factorial-experiment-2nd-pass.md` — 2nd pass cold-start
10. `docs/planning/test-suite-improvement-report.md` — model-specific testing strategy
11. `docs/planning/robustifying-flaky-prefect-sqlite-issues.md` — Prefect fix plan

### Source (biostatistics — currently being modified):
12. `src/minivess/pipeline/biostatistics_statistics.py` — stats engine (N-way ANOVA done)
13. `src/minivess/pipeline/biostatistics_types.py` — result types (SourceRun updated)
14. `src/minivess/pipeline/biostatistics_duckdb.py` — DuckDB schema (updated)
15. `src/minivess/pipeline/biostatistics_discovery.py` — run discovery (updated)
16. `src/minivess/config/biostatistics_config.py` — config (co_primary_metrics added)

### Source (training — fixed in this session):
17. `src/minivess/orchestration/flows/train_flow.py` — training flow (6 factorial args, zero-shot, fold_id fix)
18. `src/minivess/adapters/sam3_topolora.py` — LoRA fix (skip Conv2d)
19. `src/minivess/pipeline/resume_discovery.py` — fingerprint includes with_aux_calib
20. `deployment/skypilot/train_factorial.yaml` — SkyPilot (MLflow URI, file_mounts, no region)

### Memory:
21. `.claude/projects/-home-petteri-Dropbox-github-personal-minivess-mlops/memory/MEMORY.md`
22. `project_biostatistics_decisions.md` — all Q&A decisions
23. `project_debug_run_decisions.md` — debug run decisions
24. `feedback_read_kg_before_questions.md` — ALWAYS read KG before asking design questions

---

## USER PREFERENCES (from this session)

1. **Debug run = pipeline validation, NOT production results**. Production goes to
   `sci-llm-writer/manuscripts/vasculature-mlops`.
2. **Factors auto-derived from experiment YAML** — never hardcode.
3. **Comprehensive > fast** — full researcher degrees of freedom in spec curve.
4. **ALL derived metrics in biostatistics flow** — rank stability, spec curve, calibration.
5. **TRIPOD+AI mapping included** — de facto preregistration.
6. **Per-volume data essential** — per-fold aggregates insufficient.
7. **Both trainval AND test splits analyzed** separately.
8. **Model-specific tests conditionally triggered** when adapter code changes.
9. **Save all Q&A verbatim** in XML appendix for repeatability.
10. **ALWAYS read full KG before asking design questions** — context amnesia = garbage.

---

## ANTI-PATTERNS TO AVOID (learned from this session)

1. **DO NOT suggest Dice as primary metric** — it's a FOIL metric. clDice + MASD are co-primary.
2. **DO NOT ask questions before reading the KG** — navigator → domains → decisions → ALL.
3. **DO NOT hardcode factor names** — auto-derive from experiment YAML `factors` dict.
4. **DO NOT celebrate partial successes** — 413 checkpoint failure is CRITICAL even if training succeeded.
5. **DO NOT fix tests serially** — gather ALL, categorize, batch fix.
6. **DO NOT dismiss Prefect SQLite flakiness** — it's P0 #880.
7. **DO NOT use relative MLflow URIs** — `_ensure_absolute_file_uri()` guard exists.
8. **DO NOT assume 2-way ANOVA** — the design has 3 factors.

---

## VERIFICATION COMMANDS (run before any cloud launch)

```bash
# Verify tests pass
make test-staging    # Target: <5 min, 0 failures

# Verify Docker image
docker manifest inspect europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest

# Verify mamba-ssm in image
docker run --rm minivess-base:latest python -c "import mamba_ssm; print(mamba_ssm.__version__)"

# Verify GCP access
uv run sky check gcp

# Dry-run 2nd pass
./scripts/run_factorial_2nd_pass.sh --dry-run
```
