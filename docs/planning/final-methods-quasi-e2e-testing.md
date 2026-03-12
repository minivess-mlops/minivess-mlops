---
title: "Final Methods Quasi-E2E Testing Plan"
status: planned
created: ""
---

# Final Methods Quasi-E2E Testing Plan

> **Branch:** `test/final-quasi-e2e-testing`
> **Goal:** Dynamically discover ALL implemented methods, build conditional execution DAG,
> run debug-mode training (1 epoch, 4 volumes) through ALL 7 Prefect flows, hunt for bugs.
> **Focus:** Training mechanics correctness, NOT model performance.

---

## Table of Contents

1. [Design Decisions (User-Confirmed)](#1-design-decisions)
2. [Method Catalog (Auto-Discovered)](#2-method-catalog)
3. [Capability Schema (Exceptions-Based)](#3-capability-schema)
4. [Harmonized Serving Output Schema](#4-harmonized-serving-output-schema)
5. [Combinatorial Test Variants](#5-combinatorial-test-variants)
6. [Debug Config: test_practical_combos.yaml](#6-debug-config)
7. [Conditional Execution DAG](#7-conditional-execution-dag)
8. [Dynamic Discovery Infrastructure](#8-dynamic-discovery-infrastructure)
9. [Guardrails & Enforcement](#9-guardrails-and-enforcement)
10. [Implementation Phases (TDD)](#10-implementation-phases)
11. [Bash Script: run_quasi_e2e.sh](#11-bash-script)
12. [Verification Checklist](#12-verification-checklist)

---

## 1. Design Decisions

All confirmed by user (2026-03-04):

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Missing UQ keys | `Optional[T] = None` | Flat schema, BentoML-compatible |
| Combinatorial reduction | `allpairspy` pairwise + `filter_func` | Reduces ~720 to ~20 combos with coverage |
| Capability model | **Exceptions-based** (all-work-by-default) | Only declare incompatibilities |
| Cross-arch ensembling | **Real data, no mocks** | Even 1-epoch models produce ensembleable outputs |
| Debug training | 1 epoch, 2 train + 2 val volumes | Mechanics testing, not performance |
| External test data | Real external datasets (subset) | Catch real data glitches |
| Automation priority | **Total** — auto-discovery, pre-commit, tests | DevEx: new methods auto-discovered |

---

## 2. Method Catalog

### 2.1 Models (10 implemented / 13 enum values)

| ModelFamily | Adapter | Implemented | Notes |
|-------------|---------|:-----------:|-------|
| `dynunet` | DynUNetAdapter | YES | Primary model |
| `segresnet` | SegResNetAdapter | YES | MONAI built-in |
| `swinunetr` | SwinUNETRAdapter | YES | MONAI built-in |
| `vista3d` | Vista3dAdapter | YES | MONAI built-in |
| `vesselfm` | VesselFMAdapter | YES | Pretrained, data leakage warning |
| `comma_mamba` | CommaAdapter | YES | Mamba-based |
| `ulike_mamba` | MambaAdapter | YES | Mamba variant |
| `sam3_vanilla` | Sam3VanillaAdapter | YES | SAM3 baseline |
| `sam3_topolora` | Sam3TopoLoraAdapter | YES | SAM3 + LoRA |
| `sam3_hybrid` | Sam3HybridAdapter | YES | SAM3 + DynUNet fusion |
| `sam3_lora` | — | NO | Enum only, no adapter |
| `multitask_dynunet` | — | NO | Enum only, applied via wrapper |
| `custom` | — | NO | Placeholder enum |

**For testing:** 10 implemented models. `sam3_lora`, `multitask_dynunet`, `custom` excluded
(no adapter). Wrapper-based variants (TFFM, multitask) tested separately.

**Source:** `src/minivess/config/models.py:10-25`, `src/minivess/adapters/model_builder.py:58-144`

### 2.2 Loss Functions (22 total)

| Loss Name | Tier | Library | Notes |
|-----------|------|---------|-------|
| `dice_ce` | LIBRARY | MONAI DiceCELoss | Baseline |
| `dice` | LIBRARY | MONAI DiceLoss | |
| `focal` | LIBRARY | MONAI FocalLoss | |
| `cldice` | LIBRARY | MONAI SoftclDiceLoss | Wrapped |
| `dice_ce_cldice` | LIBRARY-COMPOUND | 0.5*DiceCE + 0.5*clDice | |
| `cbdice_cldice` | LIBRARY-COMPOUND | 0.5*cbDice + 0.5*dice_ce_cldice | **Default** |
| `skeleton_recall` | HYBRID | skimage + custom | |
| `cape` | HYBRID | skimage + scipy | |
| `betti_matching` | HYBRID | gudhi + gradient proxy | |
| `cb_dice` | EXPERIMENTAL | Custom inverse-freq | |
| `cbdice` | EXPERIMENTAL | Vendored | |
| `centerline_ce` | EXPERIMENTAL | Vendored | |
| `warp` | EXPERIMENTAL | Vendored (CoLeTra) | |
| `topo` | EXPERIMENTAL | Vendored (CoLeTra) | |
| `betti` | EXPERIMENTAL | Custom fragmentation proxy | |
| `full_topo` | EXPERIMENTAL | DiceCE + clDice + Betti | |
| `graph_topology` | EXPERIMENTAL | cbdice + skeleton + CAPE | |
| `toposeg` | EXPERIMENTAL | P2, discrete Morse proxy | |
| `spw` | EXPERIMENTAL | P2, steerable pyramid proxy | |

**Source:** `src/minivess/pipeline/loss_functions.py:418-543`

### 2.3 Metrics (19 registered)

| Metric | Direction | Category |
|--------|-----------|----------|
| `dsc` | maximize | Voxel overlap |
| `centreline_dsc` | maximize | Voxel overlap |
| `measured_masd` | minimize | Surface distance |
| `val_compound_masd_cldice` | maximize | Compound (BROKEN — range collapse) |
| `val_compound_nsd_cldice` | maximize | Compound (preferred) |
| `val_loss` | minimize | Per-epoch validation |
| `val_dice` | maximize | Per-epoch validation |
| `val_cldice` | maximize | Per-epoch validation |
| `val_masd` | minimize | Per-epoch validation |
| `val_f1_foreground` | maximize | Per-epoch validation |
| `nsd` | maximize | Topology (expensive) |
| `hd95` | minimize | Topology (expensive) |
| `ccdice` | maximize | Topology (expensive) |
| `betti_error_beta0` | minimize | Topology (expensive) |
| `junction_f1` | maximize | Topology (expensive) |
| `apls` | maximize | Graph-based |
| `skeleton_recall_metric` | maximize | Graph-based |
| `bdr` | maximize | Graph-based |
| `murray_compliance` | maximize | P2 experimental |
| `sdc_confidence` | maximize | Deployment quality gate |

**Source:** `configs/metric_registry.yaml`

### 2.4 Ensemble Methods

**Combination strategies (7):** mean, majority_vote, weighted, greedy_soup, swag,
ties_dare, learned_stacking

**Checkpoint selection (4):** per_loss_single_best, all_loss_single_best,
per_loss_all_best, all_loss_all_best

**Source:** `src/minivess/config/models.py:28-37`, `src/minivess/config/evaluation_config.py:11-26`

### 2.5 Post-Training Plugins (6)

| Plugin | Requires Cal Data | Category |
|--------|:-----------------:|----------|
| `swa` | No | Weight averaging |
| `multi_swa` | No | Weight averaging |
| `model_merging` | No | Weight-space |
| `calibration` | Yes | Probabilistic |
| `crc_conformal` | Yes | Conformal |
| `conseco_fp_control` | Yes | Conformal |

**Source:** `src/minivess/config/post_training_config.py:169-220`

### 2.6 Deployment Methods (3)

| Method | Config Key | Always Run |
|--------|-----------|:----------:|
| ONNX export | `onnx_opset` | YES |
| BentoML import | `bento_service_name` | YES |
| MONAI Deploy MAP | `monai_deploy_enabled` | NO (opt-in) |

**Source:** `src/minivess/config/deploy_config.py`

### 2.7 UQ / Conformal Predictors (7)

| Predictor | Type | Post-Hoc | Needs Cal Data |
|-----------|------|:--------:|:--------------:|
| ConformalPredictor | Split conformal | Yes | Yes |
| MorphologicalConformalPredictor | Dilation/erosion bands | Yes | Yes |
| DistanceTransformConformalPredictor | EDT FNR control | Yes | Yes |
| RiskControllingPredictor | LTT framework | Yes | Yes |
| CRCPredictor | Conformalized risk | Yes | Yes |
| DeepEnsemblePredictor | Ensemble variance | Yes | No (needs ensemble) |
| MCDropoutPredictor | MC dropout | Yes | No (needs model) |

**Source:** `src/minivess/ensemble/`

### 2.8 Serving Output Modes (4)

binary, probabilities, full, uq

**Source:** `src/minivess/serving/api_models.py:19-25`

---

## 3. Capability Schema (Exceptions-Based)

### Design Principle

> "All losses work everywhere by default, and exceptions are declared."

Instead of defining a valid-combinations allowlist (N x M matrix that must be maintained),
we define an **exclusions list**. Any model-loss pair NOT in the exclusions list is valid.

### Schema: `configs/method_capabilities.yaml`

```yaml
# Method Capability Schema — Exceptions-Based
# Default: ALL losses work with ALL models unless excluded here.
# Default: ALL post-hoc methods are model-agnostic unless excluded here.
#
# To add a new method, implement it and optionally declare exclusions.
# If no exclusions, the method works with everything (zero config).

version: "1.0"

# Models that are implemented and testable via build_adapter()
implemented_models:
  - dynunet
  - segresnet
  - swinunetr
  - vista3d
  - vesselfm
  - comma_mamba
  - ulike_mamba
  - sam3_vanilla
  - sam3_topolora
  - sam3_hybrid

# Loss exclusions: {loss_name: [model_families that CANNOT use this loss]}
# Empty dict = all losses work with all models (current state)
loss_exclusions: {}
  # Example future exclusion:
  # some_lora_specific_loss: [dynunet, segresnet]  # only works with LoRA models

# Post-training plugin exclusions: {plugin_name: [model_families excluded]}
post_training_exclusions: {}

# Ensemble exclusions: {strategy: [conditions where it fails]}
# Note: swag requires training-time collection (NOT post-hoc, excluded entirely)
ensemble_exclusions:
  swag: ["*"]  # Not implementable post-hoc
  learned_stacking: ["*"]  # Requires held-out data + training (future)

# Deployment exclusions: {method: [model_families excluded]}
deployment_exclusions:
  monai_deploy: ["*"]  # Disabled by default, opt-in only

# Per-model default loss (for practical variant — one loss per model)
model_default_loss:
  dynunet: cbdice_cldice
  segresnet: cbdice_cldice
  swinunetr: cbdice_cldice
  vista3d: dice_ce
  vesselfm: cbdice_cldice
  comma_mamba: cbdice_cldice
  ulike_mamba: cbdice_cldice
  sam3_vanilla: dice_ce
  sam3_topolora: cbdice_cldice
  sam3_hybrid: cbdice_cldice

# Per-model additional losses to test (beyond default)
# For practical variant: test default + these. For brute force: test ALL.
model_extra_losses:
  dynunet:
    - dice_ce
    - dice_ce_cldice
    - skeleton_recall
    - graph_topology
  sam3_vanilla:
    - cbdice_cldice
  sam3_topolora:
    - dice_ce_cldice

# Models excluded from quasi-e2e (not implemented in build_adapter)
not_implemented:
  - sam3_lora       # enum only, no adapter
  - multitask_dynunet  # applied via wrapper, not standalone
  - custom          # placeholder
```

### Auto-Discovery Functions

```python
# src/minivess/testing/capability_discovery.py

def discover_implemented_models() -> list[str]:
    """Return model names that have working build_adapter() implementations."""

def discover_all_losses() -> list[str]:
    """Return all loss names from build_loss_function() dispatch."""

def discover_all_metrics() -> list[str]:
    """Return all metric names from MetricRegistry YAML."""

def discover_post_training_plugins() -> list[str]:
    """Return all registered plugin names from PluginRegistry."""

def discover_ensemble_strategies() -> list[str]:
    """Return all EnsembleStrategy values minus excluded ones."""

def discover_deployment_methods() -> list[str]:
    """Return enabled deployment methods from DeployConfig defaults."""

def load_capability_schema() -> CapabilitySchema:
    """Load configs/method_capabilities.yaml into validated Pydantic model."""

def get_valid_losses_for_model(model: str) -> list[str]:
    """All losses EXCEPT those in loss_exclusions[loss] containing model."""

def get_valid_post_training_for_model(model: str) -> list[str]:
    """All plugins EXCEPT those in post_training_exclusions containing model."""

def build_full_combinations() -> list[TestCombination]:
    """Brute-force: every valid (model, loss) pair."""

def build_practical_combinations() -> list[TestCombination]:
    """Reduced: model_default_loss + model_extra_losses + allpairspy pairwise."""
```

---

## 4. Harmonized Serving Output Schema

### Design: Flat `Optional[T] = None`

Every model (deterministic or probabilistic) returns the same schema. Missing fields
are `None`. This works with both MLflow `ColSpec(required=False)` and BentoML (which
does NOT support discriminated unions).

```python
# src/minivess/serving/harmonized_output.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class HarmonizedSegmentationOutput:
    """Standardized output schema for ALL segmentation methods.

    Shared by all models, ensemble methods, and deployment APIs.
    Defined in ONE place — this file is the single source of truth.

    Deterministic models: binary_mask + probabilities (UQ fields = None)
    Probabilistic models: all fields populated
    Ensemble models: all fields populated (UQ from ensemble variance)
    Conformal models: all fields + prediction_set + coverage
    """

    # --- Always present (all methods) ---
    binary_mask: np.ndarray                      # (D, H, W) uint8
    probabilities: np.ndarray                    # (D, H, W) float32, [0, 1]
    volume_id: str                               # e.g., "mv01"
    model_name: str                              # e.g., "dynunet_cbdice_cldice"

    # --- Uncertainty Quantification (None for deterministic) ---
    uncertainty_map: Optional[np.ndarray] = None          # (D, H, W) total uncertainty
    aleatoric_uncertainty: Optional[np.ndarray] = None    # (D, H, W) data noise
    epistemic_uncertainty: Optional[np.ndarray] = None    # (D, H, W) model uncertainty
    mutual_information: Optional[np.ndarray] = None       # (D, H, W) MI decomposition

    # --- Conformal Prediction (None for non-conformal) ---
    prediction_set: Optional[np.ndarray] = None           # (D, H, W) conformal set
    coverage_guarantee: Optional[float] = None            # e.g., 0.9
    conformal_alpha: Optional[float] = None               # e.g., 0.1

    # --- Calibration (None for uncalibrated) ---
    calibrated_probabilities: Optional[np.ndarray] = None # (D, H, W) post-calibration
    ece_before: Optional[float] = None                    # ECE before calibration
    ece_after: Optional[float] = None                     # ECE after calibration

    # --- Ensemble metadata (None for single models) ---
    n_ensemble_members: Optional[int] = None
    ensemble_strategy: Optional[str] = None               # e.g., "mean"
    member_model_names: Optional[list[str]] = None

    # --- Topology metrics (optional, computed post-hoc) ---
    topology_metrics: Optional[dict[str, float]] = None   # e.g., {"cldice": 0.91}

    # --- Metadata ---
    extra: dict[str, object] = field(default_factory=dict)  # escape hatch
```

### Validation

```python
def validate_output(output: HarmonizedSegmentationOutput) -> list[str]:
    """Validate schema invariants. Returns list of errors (empty = valid)."""
    errors = []
    if output.binary_mask.dtype != np.uint8:
        errors.append(f"binary_mask dtype {output.binary_mask.dtype}, expected uint8")
    if output.probabilities.min() < 0 or output.probabilities.max() > 1:
        errors.append("probabilities outside [0, 1]")
    if output.binary_mask.shape != output.probabilities.shape:
        errors.append("binary_mask and probabilities shape mismatch")
    # Conformal consistency
    if output.prediction_set is not None and output.coverage_guarantee is None:
        errors.append("prediction_set without coverage_guarantee")
    return errors
```

---

## 5. Combinatorial Test Variants

### Variant 1: Full Brute Force

Every valid (model, loss) pair, followed by ALL post-hoc methods on each.

```
10 models x 22 losses = 220 training runs (minus exclusions)
  x 6 post-training plugins = 1,320 post-training evaluations
  x 5 ensemble strategies = 6,600 ensemble evaluations
  x 3 deployment methods = 19,800 deployment tests

Total: ~20,000+ individual operations
```

**Not practical for debugging.** Saved as `test_brute_force_combos.yaml` for future
full validation (e.g., before paper submission, overnight GPU cluster run).

### Variant 2: Practical (Debug-Mode)

Pairwise coverage via `allpairspy` with intelligent reduction:

1. **Training:** Each model with its default loss + extra losses = ~20 training runs
2. **Post-training:** ALL 6 plugins run on 2 representative models (dynunet, sam3_vanilla)
3. **Ensemble:** 4 checkpoint selection strategies on trained models
4. **Deployment:** ONNX + BentoML on best champion per category
5. **External eval:** 2 external test datasets with real data

**Reduction logic:**
- Post-hoc methods are model-agnostic → run on 2 models, not all 10
- Ensemble needs >=2 training runs → run on dynunet (4 losses) and sam3_vanilla (2 losses)
- Deployment is model-agnostic → test on 1 ONNX export per champion category

**Estimated:** ~20 training runs (1 epoch each) + ~12 post-training + 4 ensembles + 3 deploys = ~40 operations. Runtime: ~30-60 min on single GPU.

---

## 6. Debug Config: `test_practical_combos.yaml`

```yaml
# configs/experiments/test_practical_combos.yaml
#
# Debug-mode quasi-E2E testing. Overrides defaults for fast mechanics validation.
# Focus: catch bugs in architecture, NOT evaluate model performance.
#
# Usage:
#   bash scripts/run_quasi_e2e.sh --config configs/experiments/test_practical_combos.yaml

quasi_e2e:
  variant: practical  # "practical" | "brute_force"
  capability_schema: configs/method_capabilities.yaml

# --- Training overrides (debug mode) ---
training:
  max_epochs: 1
  seed: 42
  checkpoint_strategy: lightweight  # minimal checkpoint saving

# --- Data subset (2 train + 2 val from minivess) ---
data:
  dataset: minivess
  data_dir: data/raw/minivess
  subset_mode: debug  # use volume_ids below instead of k-fold
  train_volume_ids: [mv01, mv03]  # 2 train volumes (small Z-range)
  val_volume_ids: [mv05, mv07]    # 2 val volumes
  n_folds: 1  # single fold for debug (no cross-validation)

# --- External test datasets ---
external_test:
  enabled: true
  datasets:
    - name: tubenet_2pm
      data_dir: data/external/tubenet_2pm
      subset_volume_ids: null  # only 1 volume, use all
    - name: vesselnn
      data_dir: data/external/vesselnn
      subset_volume_ids: [vesselnn_01, vesselnn_02]  # 2 of 12 volumes

# --- Model-loss combinations (practical variant) ---
# Each model trains with its default loss. Extra losses where informative.
model_runs:
  # Primary model — test 4 losses
  - model: dynunet
    losses: [cbdice_cldice, dice_ce, dice_ce_cldice, skeleton_recall]

  # SAM3 variants — test 1-2 losses each
  - model: sam3_vanilla
    losses: [dice_ce, cbdice_cldice]

  - model: sam3_topolora
    losses: [cbdice_cldice]

  - model: sam3_hybrid
    losses: [cbdice_cldice]

  # Other MONAI models — default loss only
  - model: segresnet
    losses: [cbdice_cldice]

  - model: swinunetr
    losses: [cbdice_cldice]

  - model: vista3d
    losses: [dice_ce]

  # Mamba variants — default loss only
  - model: comma_mamba
    losses: [cbdice_cldice]

  - model: ulike_mamba
    losses: [cbdice_cldice]

  # VesselFM — pretrained fine-tune
  - model: vesselfm
    losses: [cbdice_cldice]

# --- Post-training plugins ---
# Model-agnostic: run on 2 representative models only
post_training:
  representative_models: [dynunet, sam3_vanilla]
  plugins:
    swa:
      enabled: true
      per_loss: true
    multi_swa:
      enabled: true
      n_models: 2  # reduced for debug
      subsample_fraction: 0.5
    model_merging:
      enabled: true
      method: slerp
      t: 0.5
    calibration:
      enabled: true
      methods: [global_temperature, isotonic_regression]
      calibration_fraction: 0.3
    crc_conformal:
      enabled: true
      alpha: 0.1
    conseco_fp_control:
      enabled: true
      tolerance: 0.05
      shrink_method: erosion

# --- Ensemble strategies ---
ensemble:
  strategies:
    - per_loss_single_best   # needs >=2 losses per model
    - all_loss_single_best   # needs >=2 models
  combination_methods: [mean, majority_vote]

# --- Deployment ---
deployment:
  onnx_export: true
  onnx_opset: 17
  bentoml_import: true
  monai_deploy: false  # opt-in only

# --- Analysis flow ---
analysis:
  include_post_training: true
  bootstrap_n_resamples: 100  # reduced for debug (default 10000)
  confidence_level: 0.95

# --- MLflow ---
mlflow:
  experiment_name: quasi_e2e_debug
  tracking_uri: mlruns
```

### Total Operations (Practical Variant)

| Phase | Count | Detail |
|-------|------:|--------|
| Training runs | 15 | 10 models x 1-4 losses, 1 epoch each |
| Post-training plugins | 12 | 6 plugins x 2 representative models |
| Ensemble evaluations | 4 | 2 strategies x 2 combination methods |
| Deployment tests | 6 | 3 champions x (ONNX + BentoML) |
| External eval | 6 | 3 champions x 2 external datasets |
| **Total** | **~43** | |

---

## 7. Conditional Execution DAG

### Decision Tree Structure

```
START
  |
  v
[1. DISCOVER MODELS] ─── query build_adapter() registry
  |
  v
[2. DISCOVER LOSSES] ─── query build_loss_function() dispatch
  |
  v
[3. LOAD CAPABILITIES] ─ load method_capabilities.yaml
  |
  v
[4. GENERATE COMBOS] ── practical or brute_force
  |
  v
  ┌─────────────────────────────────────────┐
  │  FOR EACH (model, loss) combination:    │
  │                                         │
  │  [5. TRAIN 1 EPOCH] ──> MLflow run      │
  │       │                                 │
  │       ├── checkpoint saved?             │
  │       │   YES ──> continue              │
  │       │   NO  ──> log error, skip model │
  │       │                                 │
  │       v                                 │
  │  [6. VALIDATE] ── run val metrics       │
  │       │                                 │
  │       v                                 │
  │  [7. LOG TO MLFLOW] ── params, metrics  │
  └─────────────────────────────────────────┘
  |
  v
[8. POST-TRAINING PLUGINS] ─── on representative models only
  │
  ├── IF model in representative_models:
  │     ├── Weight-based (parallel): SWA, Multi-SWA, Model Merging
  │     ├── Collect calibration data
  │     └── Data-dependent (parallel): Calibration, CRC, ConSeCo
  │
  └── ELSE: skip (model-agnostic, no need to repeat)
  |
  v
[9. ANALYSIS FLOW]
  │
  ├── Discover ALL training + post-training runs from MLflow
  ├── Build ensembles (strategies from config)
  │     ├── per_loss_single_best: needs >=2 losses for same model
  │     └── all_loss_single_best: needs >=2 models
  ├── Evaluate on external test datasets
  ├── Champion tagging (balanced, topology, overlap)
  └── Comparison tables + bootstrap CI
  |
  v
[10. DEPLOY FLOW]
  │
  ├── FOR EACH champion category:
  │     ├── ONNX export + validation
  │     ├── BentoML import
  │     └── Generate deployment artifacts
  └── Verify serving schema matches HarmonizedSegmentationOutput
  |
  v
[11. DASHBOARD FLOW] ─── best-effort
  │
  ├── Paper figures (PNG + SVG)
  ├── Parquet export
  └── DuckDB analytics
  |
  v
[12. QA FLOW] ─── best-effort
  │
  ├── MLflow data integrity checks
  ├── Ghost run cleanup
  └── Parameter validation
  |
  v
[13. AGGREGATE RESULTS]
  │
  ├── Write quasi_e2e_results.json (machine-readable)
  ├── Write quasi_e2e_report.md (human-readable)
  └── Exit code: 0 if all core flows succeeded
```

### Conditional Logic

```python
# Pseudocode for the decision tree

for model, loss in combinations:
    # CONDITION 1: Model must be implemented
    if model not in discover_implemented_models():
        skip(f"{model} not implemented")
        continue

    # CONDITION 2: Loss must not be excluded for this model
    if loss not in get_valid_losses_for_model(model):
        skip(f"{loss} excluded for {model}")
        continue

    # CONDITION 3: Train
    run_id = train(model, loss, epochs=1, volumes=4)

    # CONDITION 4: Post-training only on representative models
    if model in config.post_training.representative_models:
        for plugin in discover_post_training_plugins():
            if model not in get_excluded_models_for_plugin(plugin):
                run_plugin(plugin, run_id)

# CONDITION 5: Ensemble needs >=2 runs
training_runs = discover_mlflow_runs(experiment_name)
if len(training_runs) >= 2:
    for strategy in config.ensemble.strategies:
        build_ensemble(strategy, training_runs)

# CONDITION 6: Deploy only champions
champions = tag_champions(training_runs)
for champion in champions:
    export_onnx(champion)
    import_bentoml(champion)
```

---

## 8. Dynamic Discovery Infrastructure

### Core Module: `src/minivess/testing/capability_discovery.py`

This is the deterministic `.py` file that dynamically queries the repo for all
implemented methods and generates the combinations YAML.

```python
"""Dynamic discovery of all implemented methods for quasi-E2E testing.

This module is the SINGLE ENTRY POINT for discovering what the repo can do.
It queries enums, registries, factories, and capability schemas to produce
a deterministic list of test combinations.

Usage:
    python -m minivess.testing.capability_discovery --variant practical \
        --output configs/experiments/generated_combos.yaml
"""
```

**Key functions:**

| Function | Returns | Source |
|----------|---------|--------|
| `discover_implemented_models()` | `list[str]` | Tries `build_adapter()` for each ModelFamily |
| `discover_all_losses()` | `list[str]` | Introspects `build_loss_function()` dispatch |
| `discover_all_metrics()` | `list[str]` | Loads MetricRegistry from YAML |
| `discover_post_training_plugins()` | `list[str]` | Queries PluginRegistry |
| `discover_ensemble_strategies()` | `list[str]` | EnsembleStrategy enum minus excluded |
| `discover_deployment_methods()` | `list[str]` | DeployConfig defaults |
| `load_capability_schema()` | `CapabilitySchema` | Parses method_capabilities.yaml |
| `build_full_combinations()` | `list[TestCombination]` | All valid pairs |
| `build_practical_combinations()` | `list[TestCombination]` | Pairwise-reduced |
| `generate_combos_yaml()` | `Path` | Writes YAML file for reproducibility |

### Pydantic Models

```python
class CapabilitySchema(BaseModel):
    version: str
    implemented_models: list[str]
    loss_exclusions: dict[str, list[str]]
    post_training_exclusions: dict[str, list[str]]
    ensemble_exclusions: dict[str, list[str]]
    deployment_exclusions: dict[str, list[str]]
    model_default_loss: dict[str, str]
    model_extra_losses: dict[str, list[str]]
    not_implemented: list[str]

class TestCombination(BaseModel):
    model: str
    loss: str
    post_training_plugins: list[str]  # which plugins to run after
    ensemble_eligible: bool           # can participate in ensemble
    deploy: bool                      # should test deployment

class QuasiE2EPlan(BaseModel):
    variant: Literal["practical", "brute_force"]
    combinations: list[TestCombination]
    n_training_runs: int
    n_post_training: int
    n_ensemble: int
    n_deploy: int
    estimated_runtime_minutes: float
```

### Auto-Generation

```bash
# Generate the practical combos YAML from current repo state
uv run python -m minivess.testing.capability_discovery \
    --variant practical \
    --config configs/experiments/test_practical_combos.yaml \
    --output configs/experiments/generated_practical_combos.yaml

# Generate brute-force combos (for reference)
uv run python -m minivess.testing.capability_discovery \
    --variant brute_force \
    --output configs/experiments/generated_brute_force_combos.yaml
```

---

## 9. Guardrails & Enforcement

### 9.1 CLAUDE.md Additions

Add to Critical Rules:

```
16. **Method Capability Schema (Non-Negotiable)** — Every new model, loss, metric,
    post-training plugin, or ensemble strategy MUST be discoverable by the dynamic
    capability system. This means:
    a) Models: Add to ModelFamily enum AND implement in build_adapter()
    b) Losses: Add to build_loss_function() dispatch with tier classification
    c) Metrics: Add to configs/metric_registry.yaml
    d) Plugins: Implement PostTrainingPlugin protocol AND register in PluginRegistry
    e) If the method has incompatibilities, declare them in method_capabilities.yaml
    f) Run `uv run pytest tests/v2/unit/test_capability_discovery.py -x` to verify
```

### 9.2 Pre-Commit Hook

```yaml
# .pre-commit-config.yaml — new hook
- repo: local
  hooks:
    - id: capability-schema-check
      name: Method Capability Schema Consistency
      entry: uv run python -m minivess.testing.capability_discovery --check
      language: system
      files: '(config/models\.py|loss_functions\.py|metric_registry\.yaml|post_training_config\.py|method_capabilities\.yaml)'
      pass_filenames: false
      always_run: false
```

The `--check` flag validates:
1. Every ModelFamily enum value is either in `implemented_models` or `not_implemented`
2. Every loss in `build_loss_function()` is either in `loss_exclusions` or works universally
3. Every metric in `metric_registry.yaml` is loadable
4. Every plugin in PluginRegistry has a config in PostTrainingConfig

### 9.3 Test Enforcement

```python
# tests/v2/unit/test_capability_discovery.py

class TestCapabilityConsistency:
    """Every method in the repo must be discoverable."""

    def test_all_model_families_accounted_for(self):
        """Every ModelFamily enum has an adapter OR is in not_implemented."""

    def test_all_losses_buildable(self):
        """Every loss name in build_loss_function() actually builds."""

    def test_all_metrics_loadable(self):
        """Every metric in registry YAML loads without error."""

    def test_all_plugins_registered(self):
        """Every plugin config has a matching PluginRegistry entry."""

    def test_exclusions_reference_valid_methods(self):
        """Exclusion entries reference real models/losses/plugins."""

    def test_default_losses_are_valid(self):
        """model_default_loss references implemented losses."""

    def test_discovery_deterministic(self):
        """Two runs produce identical combinations."""

    def test_practical_covers_all_models(self):
        """Practical variant tests every implemented model at least once."""

    def test_practical_covers_all_loss_tiers(self):
        """Practical variant tests at least one loss from each tier."""
```

---

## 10. Implementation Phases (TDD)

### Phase 0: Capability Schema + Discovery Functions (~12 tests)

**Files:**
- CREATE `configs/method_capabilities.yaml`
- CREATE `src/minivess/testing/__init__.py`
- CREATE `src/minivess/testing/capability_discovery.py`
- CREATE `tests/v2/unit/test_capability_discovery.py`

**Tests:**
1. `test_discover_implemented_models` — returns list of model names
2. `test_discover_all_losses` — returns 22 loss names
3. `test_discover_all_metrics` — returns 19 metric names
4. `test_discover_post_training_plugins` — returns 6 plugin names
5. `test_discover_ensemble_strategies` — returns valid strategies minus excluded
6. `test_load_capability_schema` — parses YAML into Pydantic model
7. `test_get_valid_losses_for_model` — all losses (no current exclusions)
8. `test_exclusion_filtering` — with synthetic exclusions
9. `test_all_model_families_accounted_for` — consistency check
10. `test_all_losses_buildable` — every loss name actually builds
11. `test_discovery_deterministic` — two runs identical
12. `test_capability_schema_version` — version field present

### Phase 1: Harmonized Output Schema (~8 tests)

**Files:**
- CREATE `src/minivess/serving/harmonized_output.py`
- CREATE `tests/v2/unit/test_harmonized_output.py`

**Tests:**
1. `test_deterministic_output_valid` — binary_mask + probabilities, UQ = None
2. `test_probabilistic_output_valid` — all UQ fields populated
3. `test_ensemble_output_valid` — n_ensemble_members populated
4. `test_conformal_output_valid` — prediction_set + coverage
5. `test_validate_output_catches_dtype_error` — wrong dtype flagged
6. `test_validate_output_catches_shape_mismatch` — shape mismatch flagged
7. `test_validate_output_catches_probability_range` — out of [0,1] flagged
8. `test_validate_output_catches_conformal_inconsistency` — set without coverage

### Phase 2: Combination Generators (~10 tests)

**Files:**
- MODIFY `src/minivess/testing/capability_discovery.py` — add combination generators

**Tests:**
1. `test_build_full_combinations_count` — all valid pairs
2. `test_build_practical_combinations_covers_all_models` — every model at least once
3. `test_build_practical_combinations_covers_all_tiers` — LIBRARY + COMPOUND + HYBRID + EXPERIMENTAL
4. `test_practical_fewer_than_full` — practical < full count
5. `test_allpairspy_reduction` — pairwise coverage property
6. `test_filter_func_excludes_invalid` — exclusions respected
7. `test_generate_combos_yaml` — writes valid YAML
8. `test_generated_yaml_reproducible` — same seed = same YAML
9. `test_post_training_only_on_representative` — plugins skipped for non-representative
10. `test_ensemble_needs_minimum_runs` — >=2 runs required

### Phase 3: Debug Data Config + Loader (~8 tests)

**Files:**
- CREATE `src/minivess/testing/debug_data_config.py`
- CREATE `tests/v2/unit/test_debug_data_config.py`

**Tests:**
1. `test_debug_subset_selects_volumes` — 2 train + 2 val from list
2. `test_debug_subset_single_fold` — no cross-validation
3. `test_external_dataset_subset` — vesselnn subset = 2 volumes
4. `test_external_dataset_full` — tubenet_2pm = 1 volume (all)
5. `test_debug_config_loads_from_yaml` — test_practical_combos.yaml parseable
6. `test_debug_config_overrides_defaults` — 1 epoch overrides 100
7. `test_volume_ids_exist` — referenced volume IDs are valid
8. `test_data_paths_resolve` — data_dir paths exist or graceful error

### Phase 4: Test Runner (pytest integration) (~10 tests)

**Files:**
- CREATE `tests/v2/quasi_e2e/test_all_permutations.py`
- CREATE `tests/v2/quasi_e2e/conftest.py`

**Tests (pytest-driven, not in test count — these ARE the quasi-e2e tests):**
- Dynamic parametrization via `pytest_generate_tests` hook
- Each (model, loss) combination = one test case
- Subtests for post-training, ensemble, deployment within each
- Failure isolation: one model failing doesn't block others

```python
# tests/v2/quasi_e2e/conftest.py

def pytest_generate_tests(metafunc):
    """Dynamically parametrize from capability discovery."""
    if "model_loss_combo" in metafunc.fixturenames:
        combos = build_practical_combinations()
        metafunc.parametrize(
            "model_loss_combo",
            combos,
            ids=[f"{c.model}_{c.loss}" for c in combos],
        )
```

**Meta-tests (verify the test infrastructure):**
1. `test_parametrize_generates_correct_count`
2. `test_parametrize_covers_all_models`
3. `test_conftest_loads_without_error`
4. `test_combo_ids_are_unique`
5. `test_subtests_isolate_failures`
6. `test_results_logged_to_json`
7. `test_mlflow_experiment_created`
8. `test_artifacts_written_to_output_dir`
9. `test_external_datasets_evaluated`
10. `test_report_generated`

### Phase 5: Pre-Commit Hook + CLAUDE.md (~6 tests)

**Files:**
- MODIFY `.pre-commit-config.yaml` — add capability-schema-check hook
- MODIFY `CLAUDE.md` — add Critical Rule #16
- CREATE `tests/v2/unit/test_capability_precommit.py`

**Tests:**
1. `test_check_mode_passes_on_consistent_schema` — no errors
2. `test_check_mode_catches_missing_model` — model in enum but not in schema
3. `test_check_mode_catches_orphan_exclusion` — exclusion for nonexistent method
4. `test_check_mode_catches_missing_default_loss` — model without default loss
5. `test_check_output_format` — machine-parseable errors
6. `test_check_mode_exit_code` — 0 on success, 1 on failure

### Phase 6: Bash Script + Integration (~4 tests)

**Files:**
- CREATE `scripts/run_quasi_e2e.sh`
- CREATE `configs/experiments/test_practical_combos.yaml` (from section 6 above)
- CREATE `tests/v2/unit/test_quasi_e2e_script.py`

**Tests:**
1. `test_script_exists_and_executable`
2. `test_script_accepts_config_flag`
3. `test_script_dry_run_mode`
4. `test_results_json_schema`

---

## 11. Bash Script: `run_quasi_e2e.sh`

```bash
#!/usr/bin/env bash
# scripts/run_quasi_e2e.sh — Reproducible quasi-E2E testing of ALL implemented methods
#
# Usage:
#   bash scripts/run_quasi_e2e.sh                              # practical variant
#   bash scripts/run_quasi_e2e.sh --variant brute_force        # full combinatorial
#   bash scripts/run_quasi_e2e.sh --config custom_combos.yaml  # custom config
#   bash scripts/run_quasi_e2e.sh --dry-run                    # show plan, don't run
#
# Output:
#   outputs/quasi_e2e/quasi_e2e_results.json  — machine-readable results
#   outputs/quasi_e2e/quasi_e2e_report.md     — human-readable report
#   mlruns/quasi_e2e_debug/                   — MLflow experiment data

set -euo pipefail

VARIANT="${1:---variant}"
CONFIG="${CONFIG:-configs/experiments/test_practical_combos.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/quasi_e2e}"
DRY_RUN="${DRY_RUN:-false}"

echo "=== MinIVess Quasi-E2E Testing ==="
echo "Variant: ${VARIANT}"
echo "Config:  ${CONFIG}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

# Step 0: Verify environment
echo "[0/7] Verifying environment..."
uv run python -c "
from minivess.testing.capability_discovery import discover_implemented_models, discover_all_losses
models = discover_implemented_models()
losses = discover_all_losses()
print(f'  Models: {len(models)} implemented')
print(f'  Losses: {len(losses)} available')
"

# Step 1: Generate combinations YAML
echo "[1/7] Generating test combinations..."
uv run python -m minivess.testing.capability_discovery \
    --variant practical \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}/generated_combos.yaml"

if [ "${DRY_RUN}" = "true" ]; then
    echo "DRY RUN — printing plan and exiting"
    cat "${OUTPUT_DIR}/generated_combos.yaml"
    exit 0
fi

# Step 2: Run training (all model-loss combos, 1 epoch each)
echo "[2/7] Training all model-loss combinations (1 epoch)..."
uv run python -m minivess.testing.run_quasi_e2e_training \
    --combos "${OUTPUT_DIR}/generated_combos.yaml" \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# Step 3: Post-training plugins (on representative models)
echo "[3/7] Running post-training plugins..."
PREFECT_DISABLED=1 uv run python -m minivess.testing.run_quasi_e2e_post_training \
    --combos "${OUTPUT_DIR}/generated_combos.yaml" \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/post_training.log"

# Step 4: Analysis flow (ensemble + eval)
echo "[4/7] Running analysis flow..."
PREFECT_DISABLED=1 uv run python -m minivess.testing.run_quasi_e2e_analysis \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/analysis.log"

# Step 5: Deploy flow (ONNX + BentoML)
echo "[5/7] Running deploy flow..."
PREFECT_DISABLED=1 uv run python -m minivess.testing.run_quasi_e2e_deploy \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/deploy.log"

# Step 6: Dashboard + QA (best-effort)
echo "[6/7] Running dashboard + QA flows (best-effort)..."
PREFECT_DISABLED=1 uv run python -m minivess.testing.run_quasi_e2e_dashboard \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/dashboard.log" || true

# Step 7: Aggregate results
echo "[7/7] Aggregating results..."
uv run python -m minivess.testing.aggregate_quasi_e2e \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}"

echo ""
echo "=== Quasi-E2E Testing Complete ==="
echo "Results: ${OUTPUT_DIR}/quasi_e2e_results.json"
echo "Report:  ${OUTPUT_DIR}/quasi_e2e_report.md"
```

---

## 12. Verification Checklist

### After Implementation

```bash
# 1. Unit tests (capability discovery + schema)
uv run pytest tests/v2/unit/test_capability_discovery.py -x -q
uv run pytest tests/v2/unit/test_harmonized_output.py -x -q
uv run pytest tests/v2/unit/test_debug_data_config.py -x -q
uv run pytest tests/v2/unit/test_capability_precommit.py -x -q

# 2. Lint + typecheck
uv run ruff check src/ tests/ && uv run mypy src/

# 3. Pre-commit hook works
uv run python -m minivess.testing.capability_discovery --check

# 4. Discovery smoke test
uv run python -c "
from minivess.testing.capability_discovery import (
    discover_implemented_models,
    discover_all_losses,
    build_practical_combinations,
)
models = discover_implemented_models()
losses = discover_all_losses()
combos = build_practical_combinations()
print(f'Models: {len(models)}, Losses: {len(losses)}, Combos: {len(combos)}')
"

# 5. YAML generation
uv run python -m minivess.testing.capability_discovery \
    --variant practical --output /tmp/test_combos.yaml
cat /tmp/test_combos.yaml

# 6. Dry run
bash scripts/run_quasi_e2e.sh --dry-run

# 7. Full test suite
uv run pytest tests/ -x -q
```

### Before Paper Submission (Future)

```bash
# Full brute-force run with real training (100 epochs)
# Requires GPU cluster, estimated 24-48 hours
bash scripts/run_quasi_e2e.sh --variant brute_force \
    --config configs/experiments/test_brute_force_full.yaml

# Golden dataset creation for ensemble evaluation
uv run python scripts/create_golden_ensemble_dataset.py
```

---

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `configs/method_capabilities.yaml` | CREATE | 0 |
| `src/minivess/testing/__init__.py` | CREATE | 0 |
| `src/minivess/testing/capability_discovery.py` | CREATE | 0 |
| `tests/v2/unit/test_capability_discovery.py` | CREATE | 0 |
| `src/minivess/serving/harmonized_output.py` | CREATE | 1 |
| `tests/v2/unit/test_harmonized_output.py` | CREATE | 1 |
| `src/minivess/testing/debug_data_config.py` | CREATE | 3 |
| `tests/v2/unit/test_debug_data_config.py` | CREATE | 3 |
| `tests/v2/quasi_e2e/conftest.py` | CREATE | 4 |
| `tests/v2/quasi_e2e/test_all_permutations.py` | CREATE | 4 |
| `.pre-commit-config.yaml` | MODIFY | 5 |
| `CLAUDE.md` | MODIFY | 5 |
| `tests/v2/unit/test_capability_precommit.py` | CREATE | 5 |
| `scripts/run_quasi_e2e.sh` | CREATE | 6 |
| `configs/experiments/test_practical_combos.yaml` | CREATE | 6 |
| `tests/v2/unit/test_quasi_e2e_script.py` | CREATE | 6 |

### Test Count: ~58 new tests across 7 test files

---

## GitHub Issues

| # | Title | Phase | Labels |
|---|-------|-------|--------|
| 1 | Capability schema + dynamic discovery functions | 0 | `quasi-e2e`, `testing` |
| 2 | Harmonized serving output schema (Optional[T]=None) | 1 | `quasi-e2e`, `schema` |
| 3 | Combination generators (full + practical via allpairspy) | 2 | `quasi-e2e`, `testing` |
| 4 | Debug data config (2+2 volumes, 1 epoch, external test) | 3 | `quasi-e2e`, `data` |
| 5 | pytest quasi-E2E test runner (dynamic parametrization) | 4 | `quasi-e2e`, `testing` |
| 6 | Pre-commit capability schema check + CLAUDE.md rule | 5 | `quasi-e2e`, `guardrails` |
| 7 | run_quasi_e2e.sh script + practical combos YAML | 6 | `quasi-e2e`, `scripts` |

---

## References

- **allpairspy:** https://github.com/thombashi/allpairspy — pairwise combinatorial testing
- **pytest subtests:** Built-in since pytest 9.0 (PEP 678 style)
- **PostTrainCalibration:** https://github.com/AxelJanRousseau/PostTrainCalibration
- **ConSeCo:** https://github.com/deel-ai-papers/conseco
- **User prompt:** `docs/planning/final-methods-quasi-e2e-testing-prompt.md`
