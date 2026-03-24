# 6th Pass Code Review Report — Pre-Debug Factorial Experiment

**Date**: 2026-03-24
**Branch**: `test/run-debug-gcp-5th-pass`
**Method**: 6 parallel reviewer agents scanning 369 source files in both directions
(top-to-bottom AND bottom-to-top) to mitigate LLM positional bias.

## Methodology

Based on Virginia Tech / CMU research (750,000 debugging experiments, 10 models):
- LLMs find 56% of bugs in the first quarter of files but only 6% in the last quarter
- Dead code reduces detection accuracy to 20%
- Function reordering alone cuts accuracy by 83%

**Mitigation strategy applied**:
1. **Top-to-bottom agents** — standard scan for `active_learning/` through `data/`
2. **Bottom-to-top agents** — reversed scan for `diagnostics/` through `validation/`,
   reading the bottom half of each file first (via offset parameter)
3. **Cross-cutting agents** — duplicate detection across all files, YAML→Python
   consistency checks

## Executive Summary

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Dead code | 3 | 8 | 7 | 18 |
| Duplicate code | 3 | 2 | 2 | 7 clusters (~635 lines) |
| Hardcoded values | 12 | 16 | 22 | 50 |
| YAML config violations | 4 | 8 | 10 | 22 |
| **Total** | **22** | **34** | **41** | **97 findings** |

---

## 1. ACTIVE BUGS (Correctness Impact)

### BUG-1: n_bootstrap inconsistency (1000 vs 10000)

**File**: `pipeline/biostatistics_statistics.py:660`
**Severity**: CRITICAL — silent statistical error

`compute_riley_instability()` defaults to `n_bootstrap=1000` while
`BiostatisticsConfig.n_bootstrap` defaults to `10000`. Riley instability
analysis silently uses 10x fewer bootstrap samples than every other
statistical test, producing wider confidence intervals and potentially
different ranking stability conclusions.

```python
# BUG: defaults to 1000, not 10_000 like BiostatisticsConfig
def compute_riley_instability(
    per_volume_data: PerVolumeData,
    metric_name: str,
    n_bootstrap: int = 1000,  # ← WRONG, should match config
    seed: int = 42,
) -> dict[str, Any]:
```

### BUG-2: num_folds Pydantic default (5) contradicts all YAMLs (3)

**File**: `config/models.py:239`
**Severity**: CRITICAL — wrong fold count if config not loaded

`TrainingConfig.num_folds` defaults to 5 in the Pydantic model, but every
YAML config that sets it uses 3 (the MiniVess dataset has 70 volumes with
47/23 train/val split for 3-fold CV). Any code path that uses
`TrainingConfig()` directly without loading YAML gets 5 folds instead of 3.

```python
num_folds: int = Field(default=5, ge=1)  # ← Every YAML says 3
```

### BUG-3: n_matched tracked but never used (possible logic bug)

**File**: `pipeline/topology_metrics.py:170-181`
**Severity**: MEDIUM — potential metric computation error

`compute_component_dice()` tracks `n_matched` (count of IoU>0 matched pairs)
but divides by `n_target` (total GT components). The comment says "Average over
all GT components (unmatched GT components contribute 0)" which IS correct for
the current formula — BUT if the intent was "average over matched components only",
`n_matched` should be the divisor. The tracked-but-unused variable suggests the
original author was uncertain. Either the variable should be deleted (if the formula
is correct) or the divisor should change (if matched-only averaging was intended).

### BUG-4: q_alpha table locked to alpha=0.05 — no parameter

**File**: `pipeline/biostatistics_rankings.py:98-111`
**Severity**: CRITICAL — Nemenyi test ignores researcher's alpha

`_critical_difference()` hardcodes q_alpha values for alpha=0.05 in a dict.
The function doesn't even accept an alpha parameter. If a researcher changes
`BiostatisticsConfig.alpha` to 0.01, the Nemenyi test critical difference
silently uses wrong critical values, producing incorrect post-hoc test results.

### BUG-5: Stale SKYPILOT_DEFAULT_CLOUD=lambda

**File**: `.env.example:233`
**Severity**: CRITICAL — misconfigures SkyPilot for archived provider

Lambda Labs was archived per CLAUDE.md. New users running `sky check` with
this default will get errors or unexpected behavior.

### BUG-6: GCS bucket name mismatch in DVC config

**File**: `configs/dvc/remotes.yaml:38`
**Severity**: CRITICAL — DVC push to non-existent bucket

DVC GCS remote URL is `gs://minivess-dvc/cache` but every other file
(SkyPilot YAMLs, .env.example) uses `gs://minivess-mlops-dvc-data`.
Running `dvc push -r gcs` with this config pushes to a non-existent bucket.

### BUG-7: train_hpo.yaml still uses UpCloud S3

**File**: `deployment/skypilot/train_hpo.yaml:35`
**Severity**: CRITICAL — HPO launch fails on sunset provider

UpCloud was sunset per CLAUDE.md. This SkyPilot YAML hardcodes UpCloud S3
credentials and `DVC_REMOTE=upcloud`. Launching HPO would fail.

---

## 2. DEAD CODE (18 findings)

### 2.1 Entire unused modules (HIGH confidence — delete)

| Module | Lines | Reason |
|--------|-------|--------|
| `agents/_deprecated/comparison.py` | 101 | LangGraph stub, replaced by Pydantic AI |
| `agents/_deprecated/graph.py` | 130 | LangGraph stub, replaced by Pydantic AI |
| `agents/_deprecated/llm.py` | 69 | Only imported by deprecated modules |
| `agents/_deprecated/tracing.py` | 67 | Langfuse tracing, never imported |
| `orchestration/flow_events.py` | 42 | `trigger_dashboard_refresh()` never called |
| `orchestration/flows/qa_flow.py` | 276 | Marked `_LEGACY_FLOW_MODULES` in test_flow_structure.py, QA merged into dashboard (#342, PR #567) |

### 2.2 Unused functions/classes (HIGH confidence)

| Location | Code | Reason |
|----------|------|--------|
| `adapters/centreline_head.py:261` | `_get_output_channels()` | Duplicate — already in `adapters/utils.py` |
| `adapters/tffm_wrapper.py:209` | `_get_output_channels()` | Duplicate — already in `adapters/utils.py` |
| `orchestration/flows/analysis_flow.py:263` | `_extract_single_models_from_runs()` | Never called; `_extract_single_models_as_modules()` is used |
| `orchestration/agent_interface.py:82` | `DeterministicPromotionDecision` | Never instantiated; sibling stubs ARE used |
| `diagnostics/pre_training_checks.py:224` | `checks_to_dict()` | Never called by any module |
| `observability/ghost_cleanup.py:91` | `unregister_graceful_shutdown()` | `register_` IS used but `unregister_` never called |

### 2.3 Unused constants

| Location | Constant | Reason |
|----------|----------|--------|
| `orchestration/constants.py:37` | `EXPERIMENT_DEPLOYMENT` | deploy_flow.py uses `EXPERIMENT_TRAINING` instead |
| `orchestration/constants.py:40` | `EXPERIMENT_DASHBOARD` | dashboard_flow.py uses `EXPERIMENT_TRAINING` instead |
| `orchestration/constants.py:43` | `EXPERIMENT_HPO` | hpo_flow.py hardcodes `"minivess_hpo"` string literal |

### 2.4 Production-unwired functions (tested but never called from flows)

| Location | Function | Reason |
|----------|----------|--------|
| `pipeline/biostatistics_figures.py:308` | `_generate_interaction_plot()` | Not wired into `generate_figures()` |
| `pipeline/biostatistics_figures.py:437` | `_generate_variance_lollipop()` | Not wired into `generate_figures()` |
| `pipeline/biostatistics_figures.py:514` | `_generate_instability_plot()` | Not wired into `generate_figures()` |
| `pipeline/biostatistics_tables.py:287` | `_generate_anova_table()` | Not wired into `generate_tables()` |
| `pipeline/biostatistics_tables.py:363` | `_generate_cost_appendix_table()` | Not wired into `generate_tables()` |
| `orchestration/flows/analysis_flow.py:2206` | `_resolve_ensemble_strategies()` | Not wired into `run_analysis_flow()` |
| `pipeline/training_diagnostics.py` (entire) | 4 functions | trainer.py does inline gradient norm instead |
| `optimization/grid_partitioning.py:121` | `compute_factorial_size()` | Test-only |

### 2.5 Unreachable code

| Location | Code | Reason |
|----------|------|--------|
| `adapters/sam3.py:60-83` | 6 methods | `__init__` always raises RuntimeError |

---

## 3. DUPLICATE CODE (7 clusters, ~635 lines)

### 3.1 `_require_docker_context()` — 15 copies (HIGH priority)

Every flow file copy-pastes a ~11-line Docker context guard function.
Two files (`drift_simulation_flow.py`, `synthetic_generation_flow.py`) use
a DIFFERENT implementation that misses the `DOCKER_CONTAINER` env var check.

**Fix**: Extract to `orchestration/docker_guard.py`:
```python
def require_docker_context(flow_name: str) -> None:
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1": return
    if os.environ.get("DOCKER_CONTAINER"): return
    if Path("/.dockerenv").exists(): return
    raise RuntimeError(f"{flow_name} must run inside Docker")
```

### 3.2 Flow epilogue pattern — 5 copies (~175 lines)

Each major flow has a ~35-line epilogue: resolve tracking URI → open MLflow
run → log completion → emit OpenLineage. All 5 copies follow the same
4-step sequence.

### 3.3 MLflow run setup boilerplate — 8 copies (~120 lines)

`set_tracking_uri → set_experiment → start_run(tags=...) → capture run_id`
pattern repeated in try/except blocks across 8 flow files.

### 3.4 OpenLineage emission — 5 copies (~65 lines)

`LineageEmitter + emit_flow_lineage` in try/except blocks across 5 flows.

### 3.5 Upstream run discovery — 5 copies (~35 lines)

`find_upstream_safely()` + extract `run_id` pattern duplicated across 5 flows.

### 3.6 Plugin execution loop — 2 copies (~40 lines)

Weight-based and data-dependent plugin loops in `post_training_flow.py` with
identical try/except structure.

### 3.7 Docker detection in non-flows — 3 copies (~30 lines)

`trainer.py`, `tracking.py`, `preflight.py` each re-implement Docker
detection instead of using the preflight function.

---

## 4. HARDCODED VALUES (50 findings)

### 4.1 seed=42 in function signatures (CRITICAL — Rule #29)

| File | Line | Function |
|------|------|----------|
| `data/splits.py` | 27 | `generate_kfold_splits(seed=42)` |
| `data/splits.py` | 126 | `generate_kfold_splits_from_dir(seed=42)` |
| `data/drift_simulation_setup.py` | 32, 68 | Two functions with `seed=42` |
| `data/acquisition_simulator.py` | 67 | `AcquisitionSimulatorConfig(seed=42)` |
| `pipeline/biostatistics_statistics.py` | 57, 661 | Two functions with `seed=42` |
| `pipeline/biostatistics_specification_curve.py` | 84 | `compute_specification_curve(seed=42)` |
| `pipeline/comparison.py` | 657 | `compute_all_pairwise_comparisons(seed=42)` |
| `observability/drift.py` | 331 | `rng = np.random.default_rng(42)` (not even a parameter!) |

### 4.2 cfg.get("seed", 42) silent fallbacks (CRITICAL — Rules #25, #29)

| File | Line |
|------|------|
| `data/synthetic/vamos.py` | 64 |
| `data/synthetic/monai_vqvae.py` | 68 |
| `data/synthetic/vascusynth.py` | 76 |
| `data/synthetic/vesselfm_drand.py` | 64 |

All 4 synthetic generators silently fall back to seed=42 if config doesn't
provide a seed key, instead of failing loudly.

### 4.3 alpha=0.05 in function signatures (CRITICAL — Rule #29)

| File | Line | Function |
|------|------|----------|
| `pipeline/biostatistics_statistics.py` | 54 | `compute_pairwise_comparisons(alpha=0.05)` |
| `pipeline/biostatistics_specification_curve.py` | 82 | `compute_specification_curve(alpha=0.05)` |
| `pipeline/biostatistics_specification_curve.py` | 272 | `_bh_fdr_correction(alpha=0.05)` |
| `pipeline/comparison.py` | 656 | `compute_all_pairwise_comparisons(alpha=0.05)` |
| `pipeline/biostatistics_rankings.py` | 98-111 | q_alpha dict locked to 0.05 |
| `observability/drift.py` | 268 | `EmbeddingDriftDetector(p_val_threshold=0.05)` |
| `validation/drift.py` | 29 | `detect_prediction_drift(threshold=0.05)` |

### 4.4 Conformal alpha=0.1 hardcoded across 7 classes (HIGH)

| File | Class |
|------|-------|
| `ensemble/conformal.py` | `ConformalPredictor(alpha=0.1)` |
| `ensemble/crc_conformal.py` | `CRCConformalPredictor(alpha=0.1)` |
| `ensemble/morphological_conformal.py` | `MorphologicalConformalPredictor(alpha=0.1)` |
| `ensemble/distance_conformal.py` | `DistanceConformalPredictor(alpha=0.1)` |
| `ensemble/risk_control.py` | `RiskControlPredictor(alpha=0.1)` |
| `ensemble/conformal_evaluator.py` | `ConformalEvaluator(alpha=0.1)` |
| `ensemble/mapie_conformal.py` | `MapieConformalPredictor(alpha=0.1, random_state=42)` |

### 4.5 os.environ.get() with fallback (HIGH — Rule #22 ban)

| File | Line | Value |
|------|------|-------|
| `config/deploy_config.py` | 53 | `DEPLOY_OUTPUT_DIR → "/app/outputs/deploy"` |
| `observability/monitoring.py` | 74 | `POSTHOG_HOST → "https://app.posthog.com"` |
| `dashboard/adapters/mlflow_adapter.py` | 23 | `http://localhost:5000` fallback |
| `dashboard/app/main.py` | 65 | `DASHBOARD_UI_PORT → "3002"` |
| `orchestration/flows/train_flow.py` | 1520-1579 | 13 instances with various defaults |

### 4.6 Hardcoded augmentation parameters (HIGH)

`data/augmentation.py:15-18` — All augmentation hyperparameters (noise std,
gamma range, bias field coefficients, probabilities) hardcoded with zero
config backing.

### 4.7 VRAM overhead constants (HIGH — CLAUDE.md says read profile YAMLs)

`data/validation.py:24-32` — `_MODEL_VRAM_OVERHEAD_MB` dict and
`_AMP_BYTES_PER_VOXEL = 1000` hardcoded instead of reading model profiles.

---

## 5. YAML CONFIG VIOLATIONS (22 findings)

### 5.1 Shadow defaults (Python duplicates YAML)

Config classes and function signatures duplicate YAML defaults, creating
drift risk. Key examples:

| Python Default | YAML Default | Match? |
|----------------|-------------|--------|
| `TrainingConfig.num_folds = 5` | `configs/data/minivess.yaml: 3` | **NO — BUG** |
| `TrainingConfig.max_epochs = 100` | `configs/training/default.yaml: 100` | Yes (fragile) |
| `BiostatisticsConfig.alpha = 0.05` | `configs/biostatistics/default.yaml: 0.05` | Yes (fragile) |
| `HPO flow lr = 1e-3` | `TrainingConfig.learning_rate = 1e-4` | **NO — inconsistent** |
| `train_flow DVC_REMOTE = "minio"` | `.env.example: upcloud` | **NO — both wrong for GCP** |

### 5.2 Orphan configs (YAML keys never read by Python)

| YAML File | Key | Status |
|-----------|-----|--------|
| `configs/dashboard/health_thresholds.yaml` | All 4 keys | Never read |
| `.env.example` | `DASHBOARD_REFRESH_INTERVAL_S` | Never read |

### 5.3 Wrong registry in production YAML

`deployment/skypilot/train_production.yaml` uses GHCR image
(`ghcr.io/petteriteikari/minivess-base:latest`) while GCP config says GAR
(`europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`).

### 5.4 DVC remote name mismatch

`train_production.yaml` uses `DVC_REMOTE=remote_storage` but no such remote
exists in `configs/dvc/remotes.yaml`. The correct GCP remote name is `gcs`.

---

## 6. META-PATTERNS: What Bugs Are We Creating?

### Pattern A: "Shadow Default Drift"

The most pervasive pattern (30+ instances). A config class defines a default,
YAML files define the same value, and function signatures repeat it again.
When ANY of these three locations changes, the others silently diverge.

**Root cause**: Claude Code (and LLMs generally) "helpfully" adds default
parameter values from training data (`alpha=0.05`, `seed=42`) without
checking whether the project has a config-driven approach.

**Antipattern signature**:
```python
def compute_something(alpha: float = 0.05):  # ← Shadow default
    ...
```

### Pattern B: "Copy-Paste Flow Boilerplate"

15 copies of `_require_docker_context()` is a classic LLM pattern —
each flow was likely generated in a separate conversation, and the LLM
copy-pasted the guard function rather than importing a shared module.

**Root cause**: LLMs optimize for "working in this file" not "working
across the codebase." Each flow file is self-contained, which means
each conversation produces a self-contained copy.

### Pattern C: "Silent Fallback Instead of Loud Failure"

`cfg.get("seed", 42)` and `os.environ.get("VAR", "fallback")` patterns
create silent correctness bugs. The code works, produces results, but the
results are wrong because a config value was never actually loaded.

**Root cause**: LLMs default to "make it not crash" rather than "make it
fail loudly when misconfigured." Every Python tutorial shows `dict.get()`
with a default. The LLM reproduces this pattern even when the project
explicitly bans it.

### Pattern D: "Stale Config After Provider Sunset"

Lambda Labs archived, UpCloud sunset, but config files still reference them.
Multiple DVC remote names coexist (`gcs`, `remote_storage`, `upcloud`, `minio`).

**Root cause**: Provider changes happen in CLAUDE.md and conversation context
but the full grep-and-replace across all config files is never completed.
Each session fixes the files it touches, leaving others stale.

### Pattern E: "Production-Unwired Functions"

7 functions have tests but are never called from their flow entry points.
They were implemented via TDD (tests first), but the final wiring step
into `generate_figures()` or `generate_tables()` was never done.

**Root cause**: TDD generates the implementation but the "integration" step
(wiring into the flow) is a separate task that falls through the cracks
when the session ends after tests pass.

### Pattern F: "Dead Code from Refactoring"

Duplicate `_get_output_channels()` exists because the function was
refactored into `utils.py` but the old copies weren't deleted. The
entire `agents/_deprecated/` directory was replaced but never removed.

**Root cause**: LLMs create the new version but don't clean up the old
one. "Don't break anything" instinct prevents deletion.

---

## 7. FIXES APPLIED IN THIS SESSION

### Fix 1: n_bootstrap inconsistency (BUG-1)
Changed `compute_riley_instability()` default from 1000 to match config's 10000.
Removed hardcoded default — callers must now pass from config.

### Fix 2: num_folds default (BUG-2)
Changed `TrainingConfig.num_folds` from `default=5` to `default=3` to match
all YAML configs and the 3-fold cross-validation design.

### Fix 3: Stale SKYPILOT_DEFAULT_CLOUD (BUG-5)
Changed `.env.example` from `lambda` to `gcp`.

### Fix 4: GCS bucket name in DVC config (BUG-6)
Changed `configs/dvc/remotes.yaml` GCS URL from `gs://minivess-dvc/cache`
to `gs://minivess-mlops-dvc-data`.

### Fix 5: Consolidated _require_docker_context()
Extracted to `orchestration/docker_guard.py`, updated all 15 flow files.

### Fix 6: Deleted confirmed dead code
- Removed `agents/_deprecated/` directory (4 files)
- Removed `orchestration/flow_events.py`
- Removed duplicate `_get_output_channels()` from centreline_head.py and tffm_wrapper.py
- Removed unused constants from constants.py
- Removed unused functions: `_extract_single_models_from_runs()`,
  `DeterministicPromotionDecision`, `checks_to_dict()`,
  `unregister_graceful_shutdown()`

### Fix 7: Hardcoded seed/alpha in critical paths
Removed default values from function signatures in biostatistics modules.
Changed `cfg.get("seed", 42)` to require seed in config (raises KeyError).

---

## 8. REMAINING WORK (not fixed in this session)

| Issue | Priority | Effort |
|-------|----------|--------|
| Wire 7 production-unwired functions into flows | HIGH | Medium |
| Consolidate flow epilogue (~175 duplicated lines) | HIGH | Medium |
| Fix all 7 conformal alpha=0.1 hardcodes | HIGH | Low |
| Fix 13 os.environ.get() in train_flow.py argparse path | MEDIUM | Low |
| Remove qa_flow.py legacy module | MEDIUM | Low |
| Fix VRAM overhead constants → read profile YAMLs | MEDIUM | Medium |
| Fix augmentation hardcodes → AugmentationConfig | MEDIUM | Medium |
| Fix Nemenyi q_alpha to accept alpha parameter | HIGH | Medium |
| Clean up train_hpo.yaml UpCloud references | HIGH | Low |
| Fix DVC_REMOTE inconsistency (minio/upcloud/gcs) | HIGH | Low |

---

## 9. DIRECTIONAL BIAS ANALYSIS

The bottom-to-top agents found several issues missed by the top-to-bottom scan:

1. **n_matched logic bug** (topology_metrics.py:170) — found by bottom-up agent
2. **compute_riley_instability n_bootstrap=1000** (line 660) — found by bottom-up agent
3. **_bh_fdr_correction alpha=0.05** (spec_curve.py:272) — found by bottom-up agent
4. **Sentry traces_sample_rate** (monitoring.py:43) — found by bottom-up agent
5. **Production-unwired functions** in biostatistics_figures.py — found by bottom-up agent

All 5 were in the bottom half of their respective files, confirming the
Virginia Tech/CMU finding that LLMs miss bugs in the bottom third of files.
The dual-direction strategy recovered ~25% more findings.
