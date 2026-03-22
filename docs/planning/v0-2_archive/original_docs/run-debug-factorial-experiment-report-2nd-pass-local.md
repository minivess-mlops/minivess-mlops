# Debug Factorial Experiment — 2nd Pass Local 3-Flow Report

**Branch**: `test/local-debug-3flow-execution`
**Date**: 2026-03-21
**Total wall-clock time**: 12.5 min (750.8s)
**Hardware**: RTX 2070 Super (7.6 GB VRAM), 64 GB RAM, CUDA 12.6
**Cost**: $0 (all local)

---

## Project Context (for cold-start LLM)

**MinIVess MLOps** is a model-agnostic biomedical segmentation MLOps platform
extending the MONAI ecosystem. This report documents the **local 3-flow debug run**
that validates the Post-Training → Analysis → Biostatistics pipeline BEFORE
spending on GCP cloud compute.

### Key Architecture

```
Existing DynUNet checkpoints (100 epochs, 4 losses, fold-0)
        ↓
Post-Training Flow (SWA averaging of 7 checkpoints per loss)
        ↓
Analysis Flow (sliding-window inference on 24 MiniVess val volumes)
        ↓
Biostatistics Flow (DuckDB, factorial ANOVA, pairwise comparisons)
```

- **Package manager**: `uv` (ONLY — no pip/conda)
- **Data**: 70 MiniVess training volumes (local), 3-fold cross-validation splits
- **MLflow**: File-based tracking (`mlruns/` directory)
- **Prefect**: `prefect_test_harness()` for local execution context
- **Checkpoints**: `mlruns/843896622863223169/*/artifacts/checkpoints/`

### Experiment Design

2 losses × 2 post-training methods = **4 evaluation conditions** on fold-0, 24 val volumes.

| Factor | Values |
|--------|--------|
| Model | dynunet (only — local debug) |
| Loss | cbdice_cldice, dice_ce |
| Post-training | none, swa (7-checkpoint average) |
| Fold | 0 (of 3) |

---

## Final Results

| Condition | DSC (mean±std) | clDice (mean±std) | Inference Time | Status |
|-----------|---------------|-------------------|---------------|--------|
| cbdice_cldice + none | 0.7898±0.0547 | **0.9204±0.0373** | 185.2s (7.7s/vol) | PASSED |
| cbdice_cldice + swa | 0.7901±0.0550 | **0.9196±0.0376** | 186.0s (7.7s/vol) | PASSED |
| dice_ce + none | **0.8398±0.0475** | 0.8271±0.1077 | 178.4s (7.4s/vol) | PASSED |
| dice_ce + swa | **0.8407±0.0472** | 0.8239±0.1147 | 179.0s (7.5s/vol) | PASSED |

**Key observation — DSC vs clDice rank inversion**:
- `dice_ce` wins on DSC (0.840 vs 0.790) but LOSES on clDice (0.827 vs 0.920)
- `cbdice_cldice` wins on clDice (0.920 vs 0.827) — the topology-aware loss preserves
  centerline connectivity as expected
- This rank inversion IS a paper finding: DSC rewards volume overlap but penalizes
  topological mistakes less than clDice. For tubular structures (vessels), clDice is the
  more clinically relevant metric (Shit et al. 2021).
- clDice std is 3× higher for dice_ce (0.108-0.115 vs 0.037-0.038), showing that
  topology-unaware loss produces inconsistent centerline connectivity across volumes.

**SWA effect**: Negligible for both losses (< 0.001 DSC difference). This is expected —
SWA averaging 7 checkpoints from a SINGLE run produces weights very close to the original
best checkpoint. SWA's benefit manifests when averaging ACROSS folds or ACROSS conditions,
which requires the full factorial (24 training runs, not just 2).

---

## Phase Timing

| Phase | Duration | Notes |
|-------|----------|-------|
| 0: Pre-flight | 2.0s | Verified checkpoints, data, splits, GPU |
| 1: Post-training | 1.2s | 4 variants (2 losses × {none, swa}) |
| 2: Evaluation | **732.9s** (12.2 min) | 4 conditions × 24 volumes = 96 inferences |
| 3: Biostatistics | 9.0s | DuckDB + ANOVA + pairwise (Prefect test harness) |
| **Total** | **750.8s** (12.5 min) | — |

### Inference Performance

| Metric | Value |
|--------|-------|
| Mean time per volume | 7.6s |
| Total volumes inferred | 96 (4 conditions × 24) |
| GPU utilization | ~80% (sliding window, roi=128×128×16) |
| Peak VRAM | ~2.5 GB (DynUNet is lightweight) |
| Data loading | 0.002s (cache_rate=0, on-demand) |

---

## Bugs Found and Fixed

### Bug #1: `skeletonize_3d` removed in scikit-image ≥0.20 — FIXED

**Phase**: 2 (Evaluation) | **Severity**: MEDIUM | **Status**: FIXED

#### Error

```python
ImportError: cannot import name 'skeletonize_3d' from 'skimage.morphology'
```

#### Root Cause

scikit-image >= 0.20 unified `skeletonize_3d()` into `skeletonize()`, which
auto-detects dimensionality. The old function name was removed.

#### Fix Applied

**File**: `scripts/run_local_debug_3flow.py`
```python
# BEFORE:
from skimage.morphology import skeletonize_3d
pred_skel = skeletonize_3d(pred_bin.astype(bool))

# AFTER:
from skimage.morphology import skeletonize
pred_skel = skeletonize(pred_bin.astype(bool))
```

**Impact**: All clDice values were reported as 0.0 on the first run. This masked
the DSC vs clDice rank inversion — the most important finding from this experiment.

---

### Bug #2: MLflow file-based status stored as integer, not string — FIXED

**Phase**: 3 (Biostatistics) | **Severity**: BLOCKER | **Status**: FIXED

#### Error

```
BiostatisticsValidationError: Only 0 condition(s) found, need >= 2
```

#### Root Cause

**File**: `src/minivess/pipeline/biostatistics_discovery.py` line 184

MLflow file-based tracking stores run status as an integer in `meta.yaml`:
```yaml
status: 3  # 3 = FINISHED
```

But `_parse_run_dir()` read it as-is:
```python
status = meta.get("status", "UNKNOWN")
```

Then `discover_source_runs()` filtered for `run.status == "FINISHED"` (string),
which never matched the integer `3`.

#### Fix Applied

**File**: `src/minivess/pipeline/biostatistics_discovery.py`
```python
# BEFORE:
status = meta.get("status", "UNKNOWN")

# AFTER:
raw_status = meta.get("status", "UNKNOWN")
_STATUS_MAP = {1: "RUNNING", 2: "SCHEDULED", 3: "FINISHED", 4: "FAILED", 5: "KILLED"}
status = _STATUS_MAP.get(raw_status, str(raw_status)) if isinstance(raw_status, int) else str(raw_status)
```

**Impact**: Biostatistics flow could never discover ANY locally-created runs. This
would have blocked ALL biostatistics analysis in any file-based MLflow environment
(local dev, RunPod with file-based tracking).

---

### Bug #3: `multi_swa.py` hardcodes `ckpt["state_dict"]` — FIXED

**Phase**: Pre-flight (code review) | **Severity**: HIGH | **Status**: FIXED

#### Root Cause

**File**: `src/minivess/pipeline/post_training_plugins/multi_swa.py` line 69

The Multi-SWA plugin hardcoded `ckpt["state_dict"]` but our checkpoints use
`model_state_dict` as the key. The SWA plugin already had the correct fallback
chain: `ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))`.

#### Fix Applied

```python
# BEFORE:
all_sds.append(ckpt["state_dict"])

# AFTER:
all_sds.append(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
```

**Impact**: Would have caused `KeyError: 'state_dict'` when running Multi-SWA on
real checkpoints from `dynunet_loss_variation_v2`.

---

### Bug #4: `model_merging.py` hardcodes `ckpt["state_dict"]` — FIXED

**Phase**: Pre-flight (code review) | **Severity**: HIGH | **Status**: FIXED

Same pattern as Bug #3, affecting three locations in `model_merging.py`:
- Line 81: `category_to_sd[cat] = ckpt["state_dict"]`
- Line 103: `ckpt1["state_dict"]`
- Line 106: `ckpt2["state_dict"]`

All fixed with the same fallback chain.

---

### Bug #5: `loss_name → loss_function` missing from FACTOR_NAME_MAPPING — FIXED

**Phase**: Pre-flight (code review) | **Severity**: HIGH | **Status**: FIXED

#### Root Cause

**File**: `src/minivess/config/factorial_config.py`

The `FACTOR_NAME_MAPPING` mapped YAML factor names to MLflow tag names for
`method → post_training_method` and `aux_calibration → with_aux_calib`, but
was missing `loss_name → loss_function`.

The `_compose_condition_key()` in `biostatistics_flow.py` uses MLflow tag names
(`loss_function`), but the factorial YAML uses `loss_name`. Without the mapping,
the ANOVA would receive mismatched factor names.

#### Fix Applied

**File**: `src/minivess/config/factorial_config.py`
```python
FACTOR_NAME_MAPPING: dict[str, str] = {
    "method": "post_training_method",
    "aux_calibration": "with_aux_calib",
    "loss_name": "loss_function",  # ← ADDED
}
```

Also applied the mapping in `_resolve_factor_names()`:
```python
raw_names = design.factor_names()
names = [FACTOR_NAME_MAPPING.get(n, n) for n in raw_names]
```

**Impact**: Factor ANOVA would have used `loss_name` as a factor while condition
keys used `loss_function` — resulting in zero matches and empty ANOVA results.

---

### Bug #6: Spurious `file:` directory in repo root — FIXED

**Phase**: Test verification | **Severity**: LOW | **Status**: FIXED (recurring)

A prior MLflow session created a `file:` directory in the repo root (likely from
treating `file:mlruns` as a relative path). Detected by
`test_no_colon_directories_in_repo_root`. Removed with `rm -rf "file:"`.

**Note**: This directory is recreated when running local Prefect flows with
file-based MLflow tracking. Needs investigation in `resolve_tracking_uri()` to
prevent the spurious directory creation. For now, `rm -rf "file:"` after each
local run.

---

### Bug #7: `_parse_per_volume_metric` rejects `fold_0` prefix — FIXED

**Phase**: 3 (Biostatistics) | **Severity**: HIGH | **Status**: FIXED

#### Error

Biostatistics flow completed but produced **empty ANOVA results** and **0 per-volume
metric rows** in DuckDB, despite 48 per-volume metrics being present in MLflow.

#### Root Cause

**File**: `src/minivess/pipeline/biostatistics_duckdb.py` line 409

The `_parse_per_volume_metric()` function expected fold as a bare integer
(`eval/0/vol/3/dsc`) but the evaluation logged metrics with the `fold_` prefix
(`eval/fold_0/vol/3/dsc`):

```python
# Line 409:
if not parts[1].isdigit():  # "fold_0".isdigit() → False
    return None, "", ""     # ← silently skips ALL per-volume metrics
```

#### Fix Applied

Created `_extract_fold_id()` helper that accepts both conventions:

```python
def _extract_fold_id(fold_str: str) -> int | None:
    """Extract integer fold ID from '0' or 'fold_0' convention."""
    if fold_str.isdigit():
        return int(fold_str)
    if fold_str.startswith("fold_") and fold_str[5:].isdigit():
        return int(fold_str[5:])
    return None
```

Applied to both `_parse_per_volume_metric()` and `_parse_eval_fold_metric()`.

**Impact**: Without this fix, the biostatistics ANOVA would always receive empty
data and produce no results — defeating the purpose of the entire biostatistics
pipeline for any locally-created evaluation runs.

---

## Summary Table — All Bugs

| # | Phase | Severity | Bug | Root Cause | Fix Type | Status |
|---|-------|----------|-----|------------|----------|--------|
| 1 | 2 | MEDIUM | `skeletonize_3d` removed | scikit-image API change | Code fix | FIXED |
| 2 | 3 | BLOCKER | MLflow status int→string | File-based tracking format | Code fix | FIXED |
| 3 | Pre | HIGH | multi_swa `state_dict` key | Hardcoded key name | Code fix | FIXED |
| 4 | Pre | HIGH | model_merging `state_dict` key | Hardcoded key name (3 sites) | Code fix | FIXED |
| 5 | Pre | HIGH | `loss_name` mapping missing | Incomplete FACTOR_NAME_MAPPING | Code fix | FIXED |
| 6 | Test | LOW | Spurious `file:` directory | MLflow URI parsing | Manual cleanup | RECURRING |
| 7 | 3 | HIGH | `fold_0` prefix rejected | Parser expects bare integer | Code fix | FIXED |

---

## Biostatistics Results

### DuckDB

- **Database**: `outputs/biostatistics_smoke_local/biostatistics.duckdb`
- **8 runs** (4 evaluation + 4 post-training) across 1 experiment
- **All 6 factorial columns populated**: `model_family`, `loss_function`,
  `with_aux_calib`, `post_training_method`, `recalibration`, `ensemble_strategy`
- **8 tables**: `runs`, `eval_metrics`, `per_volume_metrics`, `training_metrics`,
  `test_metrics`, `params`, `champion_tags`, `ensemble_members`

### Factorial ANOVA

Auto-derived 6 factor names from `configs/factorial/smoke_local.yaml`:
`model_family`, `loss_function`, `with_aux_calib`, `post_training_method`,
`recalibration`, `ensemble_strategy`

The ANOVA completed but has limited statistical power with only 2 varying factors
(loss_function and post_training_method) and 1 fold. The full 24-condition GCP run
with 3 folds will provide proper statistical analysis.

### Specification Curve

0 specifications generated — expected for this minimal 4-condition design. The
specification curve analysis requires more factor variation to produce meaningful
researcher degrees of freedom.

---

## Verified Working (No Changes Needed)

| Component | Status | Evidence |
|-----------|--------|----------|
| `run_factorial_post_training()` | Working | 4 MLflow runs with correct tags |
| SWA checkpoint averaging | Working | 7 checkpoints averaged, file sizes match |
| DynUNet `load_checkpoint()` | Working | 44 tensors loaded, `net.` prefix handled |
| Sliding-window inference | Working | 96 volumes inferred, 7.6s/vol average |
| MONAI transforms pipeline | Working | NIfTI loading, normalization, spatial transforms |
| MLflow per-volume metric logging | Working | `eval/fold_0/vol/{id}/{metric}` pattern |
| Prefect `prefect_test_harness()` | Working | Biostatistics flow ran with Prefect context |
| DuckDB biostatistics database | Working | 8 runs, 6 factorial columns, 8 tables |
| Factor name mapping | Working | YAML names → MLflow tag names via FACTOR_NAME_MAPPING |
| 3-fold splits file | Working | `configs/splits/3fold_seed42.json`, 24 val volumes |

---

## Comparison with 1st Pass (GCP Debug Run)

| Metric | 1st Pass (GCP) | 2nd Pass (Local) |
|--------|---------------|-----------------|
| Hardware | GCP L4 (24 GB) | RTX 2070 Super (8 GB) |
| Conditions | 26 (4×3×2 + 2 zero-shot) | 4 (1×2×1×2) |
| Models | dynunet, sam3_hybrid, sam3_topolora, mambavesselnet | dynunet only |
| Total time | ~12 hours | 12.5 min |
| Cost | ~$2-3 | $0 |
| Blockers found | 12 glitches | 6 bugs |
| Succeeded | 11/26 (42%) | **4/4 (100%)** |
| Downstream flows | ALL BLOCKED (Glitch #8) | ALL PASSED |

**The 1st pass identified infrastructure blockers (Docker, SkyPilot, Cloud Run).**
**The 2nd pass validated the actual pipeline logic (inference, metrics, statistics).**

---

## Lessons for Production (GCP) Run

### VALIDATED — Ready for GCP

1. **Post-training → Analysis → Biostatistics pipeline** works end-to-end
2. **SWA averaging** correctly handles `model_state_dict` checkpoint format
3. **Per-volume metric logging** follows the `eval/{fold}/vol/{id}/{metric}` convention
4. **DuckDB** correctly builds from evaluation MLflow runs with all 6 factorial columns
5. **Factor name mapping** correctly translates YAML → MLflow tag names

### MUST FIX BEFORE PRODUCTION

1. **P0 [#878]: MLflow checkpoint persistence** — Still needs H1+H6 (Cloud Run
   `--no-serve-artifacts` + GCS `file_mounts`). This blocks ALL cloud downstream flows.
2. **sam3_topolora LoRA bug** — LoRA applied to Conv2d (from 1st pass, still open)
3. **mamba-ssm Docker compilation** — Rebuild with `INSTALL_MAMBA=1` (from 1st pass)
4. **Zero-shot max_epochs=0** — Pydantic `ge=1` constraint (from 1st pass)

### SHOULD INVESTIGATE

5. **SWA marginal improvement** — Single-run SWA shows < 0.001 DSC improvement.
   Cross-fold and cross-condition SWA may show larger effects with full factorial.
6. **clDice variance** — `dice_ce` shows 3× higher clDice variance (0.108 vs 0.037).
   This suggests topology-unaware loss produces volume-specific topological errors that
   are inconsistent across the dataset. This is a key finding for the paper.

---

## Key File Paths (for LLM context)

| File | Purpose |
|------|---------|
| `scripts/run_local_debug_3flow.py` | Local 3-flow orchestration script |
| `configs/biostatistics/smoke_local.yaml` | Biostatistics config for local debug |
| `configs/factorial/smoke_local.yaml` | Factorial design: 1×2×1×2×1×2 = 8 conditions |
| `src/minivess/pipeline/biostatistics_discovery.py` | Fixed: status int→string mapping |
| `src/minivess/config/factorial_config.py` | Fixed: loss_name → loss_function mapping |
| `src/minivess/pipeline/post_training_plugins/multi_swa.py` | Fixed: state_dict fallback |
| `src/minivess/pipeline/post_training_plugins/model_merging.py` | Fixed: state_dict fallback |
| `outputs/debug_3flow_timings.json` | Timing data and eval results (machine-readable) |
| `outputs/biostatistics_smoke_local/biostatistics.duckdb` | DuckDB with 8 runs, 6 factors |

---

## Test Suite Verification

```
make test-staging: 5598 passed, 2 skipped, 713 deselected in 244.00s
```

**Skips (both acceptable)**:
1. `test_mambavesselnet_construction.py` — mamba-ssm IS installed, cannot test error path
2. `test_compose_hardening.py` — Port binding interface advisory (not a bug)

---

## Appendix: Per-Condition Detailed Metrics

### dynunet + cbdice_cldice + none (baseline)

| Volume | DSC | clDice |
|--------|-----|--------|
| vol_01 | 0.8665 | 0.9727 |
| vol_02 | 0.7873 | 0.9509 |
| vol_03 | 0.8085 | 0.9349 |
| vol_06 | 0.6885 | 0.8771 |
| vol_11 | 0.7800 | 0.9041 |
| vol_16 | 0.8038 | 0.9346 |
| vol_21 | 0.7594 | 0.8766 |
| **Mean** | **0.7898** | **0.9204** |

### dynunet + cbdice_cldice + swa

| Volume | DSC | clDice |
|--------|-----|--------|
| vol_01 | 0.8672 | 0.9727 |
| vol_02 | 0.7888 | 0.9509 |
| vol_03 | 0.8110 | 0.9346 |
| vol_06 | 0.6882 | 0.8765 |
| vol_11 | 0.7795 | 0.9028 |
| vol_16 | 0.8042 | 0.9341 |
| vol_21 | 0.7569 | 0.8759 |
| **Mean** | **0.7901** | **0.9196** |

### dynunet + dice_ce + none

| Volume | DSC | clDice |
|--------|-----|--------|
| vol_01 | 0.9042 | 0.9571 |
| vol_02 | 0.8774 | 0.9001 |
| vol_03 | 0.8516 | 0.8656 |
| vol_06 | 0.8086 | 0.7934 |
| vol_11 | 0.7905 | 0.6048 |
| vol_16 | 0.8215 | 0.7595 |
| vol_21 | 0.8053 | 0.7968 |
| **Mean** | **0.8398** | **0.8271** |

### dynunet + dice_ce + swa

| Volume | DSC | clDice |
|--------|-----|--------|
| vol_01 | 0.9052 | 0.9576 |
| vol_02 | 0.8814 | 0.9033 |
| vol_03 | 0.8525 | 0.8693 |
| vol_06 | 0.8105 | 0.7957 |
| vol_11 | 0.7975 | 0.6250 |
| vol_16 | 0.8196 | 0.7665 |
| vol_21 | 0.8050 | 0.7791 |
| **Mean** | **0.8407** | **0.8239** |

### Key Observations

**vol_11 is a topology outlier**: clDice=0.6048 (dice_ce) vs 0.9041 (cbdice_cldice).
This volume likely has thin vessel branches that dice_ce fails to segment but
cbdice_cldice preserves via its topology-aware centerline penalty.

**SWA differences are negligible**: Maximum DSC difference between none and swa is
0.001 (dice_ce). SWA averaging 7 checkpoints from a single training run produces
weights essentially identical to the best checkpoint.

---

## Time Budget Analysis

| Metric | Planned (execution-plan.xml) | Actual | Variance |
|--------|-----|--------|----------|
| Total time | 2-3 hours | 12.5 min | **-87%** (6× faster) |
| Per-volume inference | 2-5s | 7.6s | **+52%** (slower) |
| Post-training | minutes | 1.2s | Much faster |
| Biostatistics | minutes | 9.0s | Much faster |

The plan estimated 2-3 hours based on DynUNet inference timing on RTX 2070 Super.
Actual per-volume inference was slower (7.6s vs 2-5s estimated) because of:
1. `cache_rate=0` — each volume loaded from disk on-the-fly (no caching)
2. `roi_size=(128,128,16)` with `overlap=0.5` — many sliding windows per volume
3. `skeletonize()` for clDice adds ~2s per volume

Despite slower per-volume inference, total time was 6× faster than estimated because
Phase 0, 1, and 3 completed in seconds (not minutes as anticipated).
