# Experiment Pipeline Plan — DynUNet Loss Variation Study

> **Created:** 2026-02-25
> **Branch:** `feat/experiment-pipeline-variation`
> **Dataset:** MiniVess (70 volumes, EBRAINS, ~939 MB)
> **Model:** DynUNet (MONAI)
> **Variable:** Loss function (compound losses for vascular segmentation)
> **CV:** 3-fold cross-validation (configurable, hardcoded splits for reproducibility)
> **Tracking:** MLflow (local) with full artifact versioning

---

## 1. Design Decisions

### 1.1 No Optuna — Hydra Composable Configs

Optuna is overkill for this study. The independent variable is the **loss function**,
not continuous hyperparameters. All training parameters (LR, scheduler, epochs) are held
constant at vanilla defaults. Hydra-zen's composable config system handles the experiment
matrix via `--multirun` or explicit config overrides.

### 1.2 Compute Profiles (Progressive Disclosure UX)

Six ready-made profiles for `--compute-profile` argument:

| Profile | Batch Size | Patch Size | Workers | AMP | Grad Accum | Target Hardware |
|---------|-----------|-----------|---------|-----|-----------|-----------------|
| `cpu` | 1 | 64×64×16 | 2 | No | 4 | CPU 16-128 GB |
| `gpu_low` | 2 | 96×96×24 | 4 | Yes | 2 | RTX 2070 Super (8 GB) |
| `gpu_high` | 4 | 128×128×32 | 8 | Yes | 1 | RTX 4090 (24 GB) |
| `dgx_spark` | 8 | 128×128×48 | 12 | Yes | 1 | NVIDIA DGX Spark |
| `cloud_single` | 8 | 128×128×64 | 16 | Yes | 1 | A100/H100 (40-80 GB) |
| `cloud_multi` | 32 | 128×128×64 | 16 | Yes | 1 | 8×A100/H100 (DDP) |

Usage: `uv run python scripts/train.py compute=gpu_low loss=cbdice`

### 1.3 Cross-Validation Strategy

- **3-fold CV** with deterministic, hardcoded splits (seed=42)
- Splits stored as JSON file under `configs/splits/` for exact reproducibility
- Each fold produces a model artifact in MLflow → enables ensembling later
- Splits are configurable: `--num-folds 5` or `--split-file custom.json`

### 1.4 Training Parameters (Vanilla Defaults — Held Constant)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | DynUNet standard |
| Learning rate | 1e-3 | MONAI default |
| LR scheduler | Cosine annealing with warmup | Already in trainer |
| Warmup epochs | 5 | Standard |
| Max epochs | 100 | Configurable |
| Early stopping patience | 15 | On val Dice |
| Gradient clipping | 1.0 | Already in trainer |
| Weight decay | 1e-5 | Standard |
| Seed | 42 | Global via set_global_seed() |

---

## 2. Metrics (MetricsReloaded as Ground Truth)

### 2.1 Metrics Selected by MetricsReloaded Toolkit

| Metric | Category | Source | When Computed |
|--------|----------|--------|---------------|
| **Center Line Dice (clDice)** | Overlap (primary) | MetricsReloaded `centreline_dsc()` | Validation + test |
| **DSC** | Overlap (complement) | MetricsReloaded `dsc()` | Training + validation + test |
| **MASD** | Boundary (primary) | MetricsReloaded `measured_masd()` | Validation + test |

### 2.2 Application-Specific Metrics

| Metric | Source | When Computed |
|--------|--------|---------------|
| **clDice** (soft, differentiable) | MONAI `SoftclDiceLoss` (as metric proxy) | Training (fast) |
| **HD95** | MetricsReloaded `measured_hausdorff_distance_perc()` | Test only (expensive) |
| **NSD** | MetricsReloaded `normalised_surface_distance()` | Test only |

### 2.3 Metric Computation Strategy

```
Training loop (every epoch, GPU):
├── TorchMetrics DiceScore       → fast monitoring
├── TorchMetrics BinaryF1Score   → fast monitoring
└── Loss value                   → training signal

Validation (every epoch, CPU post-inference):
├── MetricsReloaded clDice       → exact, skeleton-based
├── MetricsReloaded DSC          → exact
├── MetricsReloaded MASD         → exact, boundary-based
└── Bootstrap CI (per-volume)    → statistical confidence

Test set evaluation (end of training):
├── All validation metrics
├── MetricsReloaded HD95         → expensive, test-only
├── MetricsReloaded NSD          → expensive, test-only
└── Per-fold aggregation with CIs
```

---

## 3. Loss Functions — Experiment Grid

### 3.1 Already Implemented (in repo)

| ID | Loss | Description | Config Key |
|----|------|-------------|-----------|
| L1 | DiceCELoss | Dice + Cross-Entropy (baseline) | `dice_ce` |
| L2 | SoftclDiceLoss | MONAI soft clDice | `cldice` |
| L3 | SoftDiceclDiceLoss | Dice + clDice weighted | `dice_cldice` |
| L4 | VesselCompoundLoss | DiceCE + clDice blend | `vessel_compound` |
| L5 | TopologyCompoundLoss | DiceCE + clDice + Betti | `topo_compound` |
| L6 | BettiLoss | Topology via spatial gradients | `betti` |
| L7 | ClassBalancedDiceLoss | Per-class frequency weighting | `class_balanced` |
| L8 | FocalLoss | Class-imbalance via focal weighting | `focal` |

### 3.2 To Integrate (from referenced repos)

| ID | Loss | Paper | Repo | License | 3D? | Priority |
|----|------|-------|------|---------|-----|----------|
| L9 | **cbDice** | Shi 2024 | PengchengShi1220/cbDice | Apache-2.0 | Yes | **HIGH — training** |
| L10 | **Centerline CE** | Acebes 2024 | cesaracebes/centerline_CE | Apache-2.0 | Yes | MEDIUM |
| L11 | **WarpLoss** | CoLeTra (Lipman 2025) | jmlipman/CoLeTra | — | Yes | MEDIUM |
| L12 | **TopoLoss** | CoLeTra (Lipman 2025) | jmlipman/CoLeTra | — | Yes | MEDIUM |

### 3.3 Post-Processing (not losses, separate integration)

| Tool | Paper | Repo | Purpose |
|------|-------|------|---------|
| **TopoSculpt** | Hui et al. | Puzzled-Hui/TopoSculpt | Topology-aware post-processing refinement |

### 3.4 Assessed and Excluded

| Repo | Reason |
|------|--------|
| NatsuGao7/TopoUnet | 2D only, architecture modification not standalone loss |
| rmaphoh/feature-loss | 2D only, incomplete (2 commits, "TBC" citation) |
| pshavela/tubular-aware-lfm | Data synthesis, not segmentation loss |
| Morand 2025 Smooth clDice | No code shared, 2D student work |

### 3.5 First Training Run (loss subset)

For the initial validation run, use these losses:

1. **DiceCELoss** (L1) — baseline
2. **SoftDiceclDiceLoss** (L3) — MONAI's built-in Dice+clDice
3. **cbDice** (L9) — best application-specific loss (diameter-aware)

This gives 3 losses × 3 folds = **9 training runs** for the first experiment.

---

## 4. Artifact Versioning

### 4.1 MLflow Artifacts (per run)

```
mlruns/
└── experiment_id/
    └── run_id/
        ├── params/           # Hydra config flattened
        │   ├── model.name = "dynunet"
        │   ├── loss.name = "cbdice"
        │   ├── compute.profile = "gpu_low"
        │   ├── data.num_folds = 3
        │   ├── data.fold_idx = 0
        │   └── training.seed = 42
        ├── metrics/          # All metrics per epoch
        │   ├── train_loss
        │   ├── train_dice
        │   ├── val_dice
        │   ├── val_cldice    # MetricsReloaded
        │   ├── val_masd      # MetricsReloaded
        │   └── val_dsc       # MetricsReloaded
        ├── artifacts/
        │   ├── best_model.pt         # Best checkpoint (by val_dice)
        │   ├── model_config.json     # Full Hydra resolved config
        │   ├── split.json            # Train/val volume IDs for this fold
        │   ├── git_hash.txt          # Git commit hash
        │   ├── requirements.txt      # Frozen deps (uv pip freeze)
        │   └── hydra_config.yaml     # Hydra resolved config
        └── tags/
            ├── mlflow.source.git.commit
            ├── mlflow.source.name
            └── experiment_phase = "loss_variation"
```

### 4.2 DVC Data Versioning

```
data/
├── raw/minivess/           # Original EBRAINS data (DVC-tracked)
│   ├── imagesTr/           # 70 NIfTI volumes
│   ├── labelsTr/           # 70 NIfTI labels
│   └── metadata/           # 70 JSON metadata files
└── splits/
    └── 3fold_seed42.json   # Deterministic split file (git-tracked)
```

### 4.3 Git-Tracked Config

```
configs/
├── experiment/
│   └── config_store.py     # Hydra-zen experiment configs
├── compute/                # NEW: compute profiles
│   ├── cpu.yaml
│   ├── gpu_low.yaml
│   ├── gpu_high.yaml
│   ├── dgx_spark.yaml
│   ├── cloud_single.yaml
│   └── cloud_multi.yaml
├── loss/                   # NEW: loss configs
│   ├── dice_ce.yaml
│   ├── cldice.yaml
│   ├── dice_cldice.yaml
│   ├── cbdice.yaml
│   ├── vessel_compound.yaml
│   ├── topo_compound.yaml
│   └── ...
└── splits/                 # NEW: CV split definitions
    └── 3fold_seed42.json
```

### 4.4 Git Hash in MLflow

The tracking module already logs `mlflow.source.git.commit`. We additionally store:
- `git_hash.txt` as artifact (backup)
- `hydra_config.yaml` — full resolved config for exact reproduction
- `requirements.txt` — frozen package versions via `uv pip freeze`

---

## 5. Implementation Phases

### Phase 1: Metrics Integration (~15 new tests)

1. Create `src/minivess/pipeline/evaluation.py` — MetricsReloaded-based evaluation class
2. Wire MetricsReloaded `centreline_dsc()`, `measured_masd()`, `dsc()`, `measured_hausdorff_distance_perc()`, `normalised_surface_distance()`
3. Two-tier strategy: TorchMetrics (training), MetricsReloaded (validation/test)
4. Bootstrap CI computation per metric via existing `pipeline/ci.py`
5. Tests with synthetic + real NIfTI data

### Phase 2: Loss Integration (~12 new tests)

1. Integrate cbDice from PengchengShi1220/cbDice (extract loss class + helpers)
2. Integrate Centerline CE from cesaracebes/centerline_CE (~100 LOC port)
3. Integrate WarpLoss + TopoLoss from jmlipman/CoLeTra (already MONAI-based)
4. Integrate TopoSculpt as post-processing transform
5. Loss factory: `build_loss(config) → nn.Module` with all 12+ variants
6. Tests: forward pass, gradient computation, NaN-free on edge cases

### Phase 3: Compute Profiles + Training Script (~8 new tests)

1. Create 6 Hydra compute profile YAMLs
2. Create `scripts/train.py` — main training entry point
3. Create `configs/splits/3fold_seed42.json` — deterministic splits
4. Wire: data discovery → transforms → loader → trainer → MLflow → artifacts
5. Validate git hash, config, and frozen deps are logged correctly
6. `justfile` targets: `train`, `train-debug`, `train-sweep`

### Phase 4: Training Validation (~10 new tests)

1. Train tests with **synthetic data** (CI-safe, fast)
   - 1-epoch smoke test (no NaN loss, no NaN metrics)
   - Data augmentation NaN check (all transforms produce valid volumes)
   - MLflow artifact creation check
2. Train tests with **real MiniVess data** (local only, `@pytest.mark.real_data`)
   - Download dataset if not present
   - 1-epoch training with DynUNet on real data
   - Validate all MetricsReloaded metrics compute without error
   - Validate MLflow stores correct per-epoch metrics
3. CV folding validation
   - Verify no data leakage between folds
   - Verify deterministic splits match JSON file

### Phase 5: First Real Training Run

1. Download MiniVess dataset: `uv run python scripts/download_minivess.py`
2. Debug run (1 epoch): `uv run python scripts/train.py compute=gpu_low loss=dice_ce training.max_epochs=1`
3. Full 3-fold CV with 3 losses:
   ```bash
   # 9 total runs (3 losses × 3 folds)
   uv run python scripts/train.py -m \
     compute=gpu_low \
     loss=dice_ce,dice_cldice,cbdice \
     data.fold_idx=0,1,2
   ```
4. Inspect MLflow UI: `mlflow ui --backend-store-uri mlruns/`
5. Generate comparison report

---

## 6. MLflow-as-Contract for BentoML

The MLflow model registry serves as the contract between training and serving:

```
Training Pipeline                    MLflow Registry                  Serving Pipeline
─────────────────                    ───────────────                  ────────────────
train.py                             model_registry/                  BentoML service
├── trains model         ──────►     ├── stage: "None"                │
├── logs metrics                     ├── stage: "Staging"  ──────►    ├── loads model
├── stores artifacts                 ├── stage: "Production"          ├── serves API
└── registers model                  └── stage: "Archived"            └── health check
```

For this experiment phase, we only need:
- Training → MLflow logging (this plan)
- MLflow UI inspection (manual)
- Model promotion to staging/production is a future PR

Dynaconf deployment configs are NOT needed yet — the MLflow registry is the contract.

---

## 7. Non-Functional Requirements

- **Reproducibility:** Global seed (42), deterministic CV splits, git hash in MLflow
- **DVC:** Data versioned, `dvc.yaml` updated with download + preprocess stages
- **No deployment:** No BentoML, no ONNX export, no Gradio in this PR
- **GitHub Actions CI:** Separate PR — this PR focuses on local training validation
- **CHANGELOG:** Update after first successful training run

---

## 8. Success Criteria

- [ ] `uv run python scripts/train.py compute=gpu_low loss=dice_ce training.max_epochs=1` completes without error
- [ ] MLflow UI shows all expected metrics, params, and artifacts
- [ ] 3-fold CV produces 3 separate MLflow runs per loss
- [ ] MetricsReloaded metrics (clDice, MASD, DSC) compute correctly on validation set
- [ ] No NaN in any loss or metric during training
- [ ] Train tests pass with both synthetic and real data
- [ ] Git hash and frozen deps stored in every MLflow run
- [ ] All new code has tests, `uv run pytest tests/ -x -q` passes
