# Prefect Analysis & Dashboard Flow Improvement Plan

> **Scope:** Complete the analysis flow with reproducibility verification, external test
> datasets, paper-quality artifacts, interactive dashboard, and Prefect-triggered
> visualization pipeline.
>
> **Branch:** `feat/prefect-analysis-flow-completion`
>
> **Date:** 2026-03-01

---

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Phase 1: Reproducibility Verification](#2-phase-1-reproducibility-verification)
3. [Phase 2: External Test Dataset Acquisition](#3-phase-2-external-test-dataset-acquisition)
4. [Phase 3: Paper-Quality Visualization System](#4-phase-3-paper-quality-visualization-system)
5. [Phase 4: Interactive Dashboard (Observable Framework)](#5-phase-4-interactive-dashboard-observable-framework)
6. [Phase 5: Dashboard & Reporting Flow](#6-phase-5-dashboard--reporting-flow-best-effort)
7. [Visualization Catalogue](#7-visualization-catalogue)
8. [File Manifest](#8-file-manifest)
9. [TDD Execution Order](#9-tdd-execution-order)
10. [Risks and Mitigations](#10-risks-and-mitigations)

---

## 1. Overview and Objectives

### What This Plan Delivers

| Deliverable | Phase | Confidence |
|-------------|-------|------------|
| Reproducibility verification (training metrics match inference) | 1 | HIGH |
| External test dataset (DeepVess + tUbeNet 2PM, multiphoton only) | 2 | HIGH |
| Paper-quality Seaborn figures for LaTeX | 3 | HIGH |
| Interactive DuckDB-WASM dashboard on GitHub Pages | 4 | MEDIUM |
| Dashboard & Reporting Flow (best-effort, 5th Prefect flow) | 5 | HIGH |

### Architecture Overview

The project uses **5 persona-based Prefect flows** (updated from 4). The first 4
are **core** (always run, failure blocks the pipeline); the 5th is **best-effort**
(runs when resources allow, failure does not block).

```
Flow 1: Data Engineering (core)
Flow 2: Model Training (core)
    → MLflow runs (mlruns/{experiment_id}/{run_id}/)
    → Checkpoints (.pt), metrics, params, tags

Flow 3: Analysis (core) — ENHANCED
    ├── Task 1-9: Existing (load, build, log, extract-modules, evaluate,
    │             mlflow-evaluate, compare, register-champion, tag-champion, report)
    ├── Task 10: Reproducibility verification (NEW)
    └── Task 11: External test evaluation (NEW)

Flow 4: Deployment (core)

Flow 5: Dashboard & Reporting (best-effort) — NEW
    ├── Task 1: Extract DuckDB + export Parquet
    ├── Task 2: Generate paper-quality figures (Seaborn)
    ├── Task 3: Generate Evidently drift reports
    └── Task 4: Generate supplementary tables (CSV/LaTeX)
```

**Design decision:** Dashboard is a **separate 5th flow**, not part of analysis:
- Clear separation of concerns: analysis produces science, dashboard produces artifacts
- **Best-effort semantics**: dashboard failure never blocks training/analysis pipeline
- Independent retry/scheduling: can re-run dashboard without re-running analysis
- Reads from MLflow/DuckDB (same inter-flow contract as other flows)
- Uses `_prefect_compat.py` for graceful degradation when `PREFECT_DISABLED=1`

### MLflow as Contract Architecture

All flows communicate via MLflow artifacts. The dashboard reads DuckDB/Parquet
exports derived from MLflow, never accessing training state directly. This
ensures the 5-persona flow separation (data → train → analyze → deploy + dashboard)
remains clean.

---

## 2. Phase 1: Reproducibility Verification

### Objective

Verify that single-fold and CV-mean inference on the training set produces
metrics matching those logged during training, confirming deterministic
reproducibility of the entire pipeline.

### Why This Matters

Cross-validation mixes train/val splits across folds. To confirm the system is
reproducible, we must show that:
1. Loading a checkpoint and running inference on its training-time validation
   volumes produces identical metrics
2. Aggregating per-fold results reproduces the CV-mean statistics

### Implementation

#### 2.1 New Module: `src/minivess/pipeline/reproducibility_check.py`

```python
@dataclass(frozen=True)
class ReproducibilityResult:
    run_id: str
    fold_id: int
    metric_name: str
    training_value: float      # From MLflow metric log
    inference_value: float     # From fresh inference
    absolute_diff: float
    is_reproducible: bool      # |diff| < tolerance

@dataclass
class ReproducibilityReport:
    results: list[ReproducibilityResult]
    tolerance: float = 1e-5
    all_pass: bool = True
    summary: str = ""

def verify_fold_reproducibility(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    fold_id: int,
    checkpoint_path: Path,
    val_volume_ids: list[str],
    data_dir: Path,
    metrics_to_check: list[str],
    tolerance: float = 1e-5,
) -> list[ReproducibilityResult]:
    """Run inference on fold's val volumes, compare to training metrics."""

def verify_cv_mean_reproducibility(
    fold_results: dict[int, list[ReproducibilityResult]],
    cv_mean_metrics: dict[str, float],
    tolerance: float = 1e-4,
) -> ReproducibilityReport:
    """Aggregate per-fold results and compare to CV-mean."""
```

#### 2.2 New Prefect Task

```python
@task(name="verify-reproducibility")
def verify_reproducibility_task(
    runs: list[dict],
    eval_config: EvaluationConfig,
    mlruns_dir: Path,
) -> ReproducibilityReport:
    """Verify training metrics match inference metrics."""
```

Insert after `evaluate-all-models` task in `analysis_flow.py`.

#### 2.3 Metrics to Verify

| Metric | Source (Training) | Source (Inference) | Tolerance |
|--------|------------------|--------------------|-----------|
| val_dice | `metrics/val_dice` last epoch | Fresh SlidingWindowInference | 1e-5 |
| val_cldice | `metrics/val_cldice` last epoch | Fresh SlidingWindowInference | 1e-4 |
| val_masd | `metrics/val_masd` last epoch | Fresh SlidingWindowInference | 1e-3 |
| val_compound_masd_cldice | `metrics/val_compound_masd_cldice` | Computed from above | 1e-4 |

**Note:** MASD tolerance is larger because skeleton computation is sensitive to
floating-point order on different hardware.

#### 2.4 Test Plan (RED phase first)

- `TestVerifyFoldReproducibility` (5 tests): exact match, within tolerance,
  exceeds tolerance, missing metric, missing checkpoint
- `TestVerifyCvMeanReproducibility` (4 tests): all pass, partial fail, NaN
  handling, empty folds
- `TestReproducibilityReport` (3 tests): summary formatting, all_pass flag,
  tolerance propagation

---

## 3. Phase 2: External Test Dataset Acquisition

### The Problem

Cross-validation uses all 70 MiniVess volumes in train/val rotations. There is
no held-out test set. For generalization assessment, we need external data.

### Dataset Landscape (VesselFM D_real Analysis)

VesselFM (CVPR 2025, Wittmann et al.) curates 23 classes from 17 source
datasets. Only two datasets match MiniVess's modality (multiphoton / two-photon
microscopy of mouse brain vasculature):

| Dataset | VesselFM Class | Modality | Volumes | Resolution | Domain Gap |
|---------|---------------|----------|---------|------------|------------|
| **DeepVess** (Cornell) | 16 | Multi-photon microscopy (in vivo) | 1 large → ~5-8 subvols | 1.00x1.00x1.70 um | Low (same modality, Alzheimer's disease models, different microscope/lab) |
| **tUbeNet 2PM** (UCL) | 7 | Two-photon microscopy | 1 | 0.20x0.46x5.20 um | Low-moderate (higher XY resolution, different lab) |

**Excluded modalities:** Light-sheet, electron microscopy, CT, MRA, OCTA, and
all human vascular datasets are excluded. The test set must match the training
domain (multiphoton/two-photon, mouse brain microvasculature) to provide a
meaningful generalization assessment rather than a cross-modality benchmark.

### Recommended Test Set Composition (~10 volumes)

| Source | Volumes | Purpose | Download |
|--------|---------|---------|----------|
| DeepVess | 5-8 subvolumes | Same-modality generalization (different lab, Alzheimer's models) | [Cornell eCommons](https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3) |
| tUbeNet 2PM | 1 volume (or 2-3 subvolumes) | Same-modality generalization (different resolution, different lab) | [UCL Figshare](https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604) |

**Total: ~8-10 volumes from 2 source datasets, both multiphoton microscopy.**

The DeepVess volume is large enough to extract multiple non-overlapping
512x512xN subvolumes matching MiniVess dimensions. The Alzheimer's disease
context provides scientifically meaningful distribution shift: same imaging
modality, same species, but pathological vascular changes (tortuous vessels,
reduced density). This makes the generalization assessment meaningful for a
peer-reviewed manuscript.

**Note on volume count:** If ~8-10 subvolumes from 2 datasets feels
insufficient, we can supplement with VesselFM synthetic data (D_drand) generated
to mimic 2-PM appearance, aligning with the drift monitoring objective.

### Synthetic Data for Drift Simulation (Future Phase)

VesselFM's D_drand (domain randomization) pipeline generates synthetic
vessel images from corrosion cast graphs + augmented backgrounds. This aligns
with the existing plan for drift monitoring and continuous retraining:

| Component | Purpose | Feasibility |
|-----------|---------|-------------|
| D_drand | Unlimited synthetic training data | HIGH (pipeline fully described) |
| D_flow | Flow-matching generative model | MODERATE (checkpoint not public, ~3 days training) |

**Decision:** Defer D_flow/D_drand to a separate `feat/synthetic-drift-monitoring`
branch. Use real external datasets for the current analysis flow.

### Implementation

#### 3.1 New Module: `src/minivess/data/external_datasets.py`

```python
@dataclass(frozen=True)
class ExternalDatasetConfig:
    name: str                    # "deepvess", "tubenet_2pm"
    source_url: str
    modality: str
    organ: str
    species: str
    resolution_um: tuple[float, float, float]
    domain_tier: str             # "A", "B", "C"
    n_volumes: int
    license: str
    cite_ref: str

EXTERNAL_DATASETS: dict[str, ExternalDatasetConfig] = {
    "deepvess": ExternalDatasetConfig(
        name="deepvess",
        source_url="https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3",
        modality="multi-photon microscopy",
        organ="brain",
        species="mouse",
        resolution_um=(1.0, 1.0, 1.7),
        domain_tier="A",
        n_volumes=1,  # 1 large volume → extract subvolumes
        license="TBD",  # Verify on eCommons before use
        cite_ref="haft_javaherian_2019_deepvess",
    ),
    "tubenet_2pm": ExternalDatasetConfig(
        name="tubenet_2pm",
        source_url="https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604",
        modality="two-photon microscopy",
        organ="brain",
        species="mouse",
        resolution_um=(0.20, 0.46, 5.20),
        domain_tier="A",
        n_volumes=1,
        license="TBD",  # Verify on UCL Figshare before use
        cite_ref="holroyd_2025_tubenet",
    ),
}

def download_external_dataset(
    name: str,
    target_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download and prepare an external dataset."""

def discover_external_test_pairs(
    data_dir: Path,
    dataset_name: str,
) -> list[dict[str, str]]:
    """Discover image/label pairs for an external dataset."""
```

#### 3.2 New Config: `configs/external_datasets.yaml`

```yaml
# External test sets — multiphoton/two-photon microscopy ONLY
# No light-sheet, EM, CT, MRA, or human vascular datasets
external_test_sets:
  deepvess:
    enabled: true
    modality: "multi-photon microscopy"
    species: "mouse"
    organ: "brain"
    n_subvolumes: 6
    subvolume_size: [512, 512, 50]  # Match MiniVess dimensions
    download_url: "https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3"
    citation: "Haft-Javaherian et al. (2019)"
  tubenet_2pm:
    enabled: true
    modality: "two-photon microscopy"
    species: "mouse"
    organ: "brain"
    n_subvolumes: 2
    subvolume_size: [512, 512, 50]
    download_url: "https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604"
    citation: "Holroyd et al. (2025)"
```

#### 3.3 Evaluation on External Test Sets

New Prefect task `evaluate-external-test-sets` after reproducibility
verification, producing per-dataset and per-tier metrics.

#### 3.4 Test Plan (RED phase first)

- `TestExternalDatasetConfig` (3 tests): all configs valid, resolution tuples, URLs
- `TestDiscoverExternalPairs` (4 tests): MedDecathlon layout, EBRAINS layout,
  missing dir, empty dir
- `TestExternalEvaluation` (3 tests): per-dataset metrics, tier aggregation,
  domain gap analysis

---

## 4. Phase 3: Paper-Quality Visualization System

### Design Principles (from foundation-PLR)

1. **Centralized styling** — Single `plot_config.py` with Paul Tol colorblind-safe palette
2. **Preset figure dimensions** — No hardcoded figsizes; presets for single/double/matrix
3. **Multi-format export** — PNG (300 DPI), SVG (vector), EPS (LaTeX)
4. **JSON data export** — Reproducibility data alongside every figure
5. **Batch generation** — `generate_all_figures.py` master script

### Implementation

#### 4.1 New Module: `src/minivess/pipeline/viz/plot_config.py`

```python
# Paul Tol colorblind-safe palette
COLORS = {
    "dice_ce": "#332288",         # Indigo
    "cbdice": "#88CCEE",          # Cyan
    "dice_ce_cldice": "#44AA99",  # Teal
    "cbdice_cldice": "#117733",   # Green
    "champion": "#DDCC77",        # Sand
    "reference": "#CC6677",       # Rose
}

# Loss function display names
LOSS_LABELS = {
    "dice_ce": "Dice + CE",
    "cbdice": "cbDice",
    "dice_ce_cldice": "Dice + CE + clDice",
    "cbdice_cldice": "cbDice + clDice",
}

FIGURE_DIMENSIONS = {
    "single": (8, 6),
    "double": (14, 6),
    "triple": (18, 6),
    "matrix": (10, 8),
    "forest": (10, 12),
    "specification_curve": (16, 12),
}

def setup_style(context: str = "paper") -> None:
    """Apply consistent styling across all figures."""

def save_figure(
    fig: Figure,
    name: str,
    output_dir: Path | None = None,
    formats: list[str] | None = None,  # ["png", "svg", "eps"]
    data: dict | None = None,          # Reproducibility JSON
) -> Path:
    """Save figure in multiple formats with optional data export."""
```

#### 4.2 New Module: `src/minivess/pipeline/viz/figure_dimensions.py`

Preset-based figsize management (no hardcoding anywhere in viz code).

#### 4.3 New Module: `src/minivess/pipeline/viz/generate_all_figures.py`

Master script that generates all paper figures from DuckDB data:

```bash
uv run python -m minivess.pipeline.viz.generate_all_figures
uv run python -m minivess.pipeline.viz.generate_all_figures --figure loss_comparison
uv run python -m minivess.pipeline.viz.generate_all_figures --list
```

### Visualization Catalogue

See [Section 7](#7-visualization-catalogue) for the complete catalogue of all
figures with descriptions, data sources, and implementation details.

---

## 5. Phase 4: Interactive Dashboard (Observable Framework)

### Technology Stack

| Component | Role | Why |
|-----------|------|-----|
| **Observable Framework** | Static site generator for data dashboards | Built-in DuckDB-WASM, Observable Plot, GitHub Pages deploy |
| **DuckDB-WASM** | In-browser SQL queries over Parquet files | Zero server, existing DuckDB extraction code reused |
| **Observable Plot** | High-level d3.js wrapper for interactive charts | 90% of d3.js interactivity with 10% of the code |
| **GitHub Pages** | Static hosting | Free, automated via GitHub Actions |

### Architecture

```
Analysis Flow
  └── export-dashboard-artifacts task
        ├── runs.parquet
        ├── eval_metrics.parquet
        ├── training_metrics.parquet
        ├── champion_tags.parquet
        ├── evidently_report.html (drift)
        └── figures/*.png (static Seaborn figs)

Dashboard Site (observable/)
  └── src/
        ├── index.md           — Overview + champion summary
        ├── loss-comparison.md — Interactive loss comparison
        ├── fold-analysis.md   — Per-fold metric exploration
        ├── uq-analysis.md     — Uncertainty quantification
        ├── drift-monitoring.md — Evidently report embed
        └── data/              — Parquet files (auto-loaded)
```

### DuckDB-WASM Integration (Observable Framework)

```markdown
<!-- observable/src/loss-comparison.md -->

```sql id=metrics
SELECT loss_function, metric_name,
       AVG(point_estimate) as mean_value,
       STDDEV(point_estimate) as std_value
FROM eval_metrics
GROUP BY loss_function, metric_name
ORDER BY loss_function
```

```js
Plot.plot({
  marks: [
    Plot.barY(metrics, {x: "loss_function", y: "mean_value",
                         fill: "metric_name"}),
    Plot.ruleY([0])
  ],
  color: {legend: true}
})
```
```

### Parquet Export from DuckDB Extraction

Add to existing `extract_runs_to_duckdb()`:

```python
def export_dashboard_parquet(db: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    """Export DuckDB tables as Parquet files for the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for table in ["runs", "params", "eval_metrics", "training_metrics", "champion_tags"]:
        db.execute(
            f"COPY {table} TO '{output_dir / table}.parquet' (FORMAT PARQUET)"
        )
```

### GitHub Pages Deployment

```yaml
# .github/workflows/dashboard.yml
name: Deploy Dashboard
on:
  push:
    paths: ['observable/**', 'docs/dashboard/**']
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm --prefix observable install
      - run: npm --prefix observable run build
      - uses: actions/deploy-pages@v4
        with:
          path: observable/dist
```

---

## 6. Phase 5: Dashboard & Reporting Flow (Best-Effort)

### Design

The Dashboard Flow is the **5th persona-based Prefect flow**. It is
**best-effort**: failure does not block the core pipeline (data → train →
analyze → deploy). It reads from MLflow/DuckDB (same inter-flow contract).

```python
# src/minivess/orchestration/flows/dashboard_flow.py

@flow(name="minivess-dashboard", log_prints=True)
def run_dashboard_flow(
    mlruns_dir: Path,
    experiment_ids: list[str],
    output_dir: Path,
    *,
    generate_figures: bool = True,
    generate_evidently: bool = True,
    export_parquet: bool = True,
) -> dict[str, Any]:
    """Generate dashboard artifacts from completed analysis (best-effort)."""
```

### Tasks

| # | Task Name | Input | Output |
|---|-----------|-------|--------|
| 1 | `extract-duckdb-and-parquet` | mlruns_dir, experiment_ids | DuckDB + Parquet files |
| 2 | `generate-paper-figures` | DuckDB database | PNG/SVG/EPS figures |
| 3 | `generate-evidently-reports` | mlruns_dir, eval results | HTML drift reports |
| 4 | `generate-supplementary-tables` | DuckDB database | CSV/LaTeX tables |

### Implementation Module

```python
# src/minivess/pipeline/dashboard_tasks.py

def export_dashboard_parquet(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> dict[str, Path]:
    """Export DuckDB tables as Parquet files for the dashboard."""

def generate_paper_figures(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> list[Path]:
    """Generate all paper-quality figures from DuckDB data."""

def generate_supplementary_tables(
    db: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> list[Path]:
    """Generate CSV and LaTeX supplementary tables."""
```

### Triggering Strategy

**Local / CI mode** (via `_prefect_compat.py`): Direct invocation at the end
of the analysis flow. Dashboard failure is caught and logged but does not fail
the analysis flow:

```python
# In analysis_flow.py, after generate-report task:
try:
    from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow
    dashboard_result = run_dashboard_flow(
        mlruns_dir=mlruns_dir,
        experiment_ids=[experiment_id],
        output_dir=output_dir / "dashboard",
    )
except Exception:
    logger.warning("Dashboard flow failed (best-effort), continuing")
    dashboard_result = None
```

**Production mode** (Prefect server deployed): Upgrade to
`DeploymentEventTrigger` for fully event-driven decoupling:

```python
from prefect.events import DeploymentEventTrigger

dashboard_deployment = dashboard_flow.to_deployment(
    name="dashboard_deployment",
    triggers=[
        DeploymentEventTrigger(
            expect={"prefect.flow-run.Completed"},
            match_related={"prefect.resource.name": "analysis_deployment"},
        )
    ],
)
```

---

## 7. Visualization Catalogue

### Category 1: Model Performance vs Hyperparameters

| Figure | Type | Data Source | Library |
|--------|------|-------------|---------|
| **F1.1 Loss comparison box plot** | Box/violin | eval_metrics × loss_function | Seaborn + Observable Plot |
| **F1.2 Per-fold metric heatmap** | Heatmap | eval_metrics × fold_id × metric | Seaborn |
| **F1.3 Training curves** | Line plot | training_metrics × epoch | Seaborn + Observable Plot |
| **F1.4 Architecture comparison** | Grouped bar | eval_metrics × architecture | Seaborn |
| **F1.5 Metric correlation matrix** | Heatmap | eval_metrics pairwise | Seaborn |

#### F1.1 Loss Comparison Box Plot
- X-axis: loss function (4 losses)
- Y-axis: metric value (DSC, clDice, MASD, compound)
- Color: loss function (Paul Tol palette)
- Facets: one per metric (2x2 grid)
- Annotations: mean ± std, significance stars (Holm-Bonferroni corrected)
- Static: Seaborn `catplot(kind="box")`
- Interactive: Observable Plot with hover showing per-fold values

#### F1.2 Per-Fold Metric Heatmap
- Rows: loss function × fold_id
- Columns: metrics (6 tracked)
- Cell values: point estimates with CI annotations
- Color scale: RdYlGn per column (normalize within metric)
- Champion cells highlighted with gold border

#### F1.3 Training Curves
- X-axis: epoch (1-100)
- Y-axis: metric value
- Lines: one per loss function (colored)
- Bands: ±1 std across folds (alpha=0.2)
- Vertical line at best epoch per loss
- Subplots: train_loss, val_dice, val_cldice, val_compound
- Interactive: Observable Plot with scrubber for epoch selection

### Category 2: Factorial Design Analysis

| Figure | Type | Data Source | Library |
|--------|------|-------------|---------|
| **F2.1 Parallel coordinates** | Parallel coords | params + metrics | Seaborn/plotly |
| **F2.2 Specification curve** | Spec curve | all configs sorted | Seaborn |
| **F2.3 Sensitivity heatmap** | Heatmap | loss × architecture → metric | Seaborn |
| **F2.4 Critical difference diagram** | CD diagram | ranked losses | Custom (Demsar 2006) |
| **F2.5 Effect size forest plot** | Forest plot | Cohen's d per pair | Seaborn |

#### F2.1 Parallel Coordinates
- Axes: loss_function, architecture, num_folds, primary_metric
- Each line: one trained model configuration
- Color: primary metric value (continuous colormap)
- Brushing: select ranges on any axis to filter
- Pattern reference: `foundation-PLR/src/viz/parallel_coordinates_preprocessing.py`

#### F2.2 Specification Curve
- Top panel: all model configurations sorted by primary metric (descending)
- Each point: metric ± 95% CI (bootstrap)
- Bottom panel: binary specification matrix (which loss, which architecture, which fold)
- Reference line: champion model's score
- Pattern reference: `foundation-PLR/src/viz/specification_curve.py`

#### F2.4 Critical Difference Diagram
- Friedman test across losses
- Nemenyi post-hoc pairwise comparisons
- Ranked losses on X-axis (1 = best)
- Cliques (groups not significantly different) connected by horizontal bar
- Pattern reference: `foundation-PLR/src/viz/cd_diagram.py`

### Category 3: UQ Analysis

| Figure | Type | Data Source | Library |
|--------|------|-------------|---------|
| **F3.1 Uncertainty decomposition** | Stacked bar | ensemble predictions | Seaborn |
| **F3.2 Calibration curve** | Reliability diagram | soft predictions vs GT | Seaborn |
| **F3.3 Conformal set size vs coverage** | Line plot | conformal results | Seaborn |
| **F3.4 Epistemic vs aleatoric map** | 2D slice overlay | per-voxel UQ | Matplotlib |
| **F3.5 Prediction interval width** | Violin plot | per-volume CI widths | Seaborn |

#### F3.1 Uncertainty Decomposition
- Stacked bars: total = aleatoric + epistemic (per ensemble strategy)
- X-axis: ensemble strategy (4 strategies)
- Y-axis: entropy (bits)
- Color: aleatoric (blue), epistemic (orange)
- Annotation: total uncertainty value above bar

#### F3.2 Calibration Curve
- X-axis: predicted probability bins (0.0-1.0, 10 bins)
- Y-axis: observed frequency of positive class
- Diagonal line: perfect calibration
- Area between curve and diagonal: ECE
- One curve per loss function

### Category 4: MLOps / Observability / Drift

| Figure | Type | Data Source | Library |
|--------|------|-------------|---------|
| **F4.1 Metric drift over time** | Line + threshold | training_metrics × timestamp | Seaborn + Observable Plot |
| **F4.2 Data distribution shift** | Histogram overlay | intensity distributions | Seaborn |
| **F4.3 GPU/memory utilization** | Area chart | MLflow system metrics | Seaborn |
| **F4.4 Training time breakdown** | Stacked bar | per-phase timing | Seaborn |
| **F4.5 Inference latency** | Box plot | per-model latency | Seaborn |

#### F4.1 Metric Drift Over Time (Grafana/Evidently Style)
- X-axis: run timestamp (chronological)
- Y-axis: primary metric value
- Points: one per completed run
- Horizontal bands: green (>0.9), yellow (0.8-0.9), red (<0.8)
- Annotations: champion transitions
- Rolling average line (window=3)
- This mimics Grafana time-series panels as static Seaborn figures

#### F4.3 GPU/Memory Utilization
- X-axis: training epoch
- Y-axis: % utilization / absolute memory
- Subplots: GPU utilization, GPU memory, CPU utilization, system RAM
- One line per loss function
- Source: MLflow system metrics (12 metrics at 10s intervals)

### Category 5: External Generalization

| Figure | Type | Data Source | Library |
|--------|------|-------------|---------|
| **F5.1 Domain gap degradation** | Grouped bar | external metrics by tier | Seaborn |
| **F5.2 Per-volume scatter** | Scatter | DSC vs clDice per volume | Seaborn + Observable Plot |
| **F5.3 Failure case gallery** | Image grid | worst-performing volumes | Matplotlib |

#### F5.1 External Generalization Assessment
- X-axis: external dataset (DeepVess, tUbeNet 2PM)
- Y-axis: metric value (DSC, clDice, MASD, compound)
- Groups: loss function
- Reference line: MiniVess CV-mean performance
- Quantifies how much performance degrades on unseen same-modality data
- Error bars: per-subvolume variation within each external dataset

---

## 8. File Manifest

### New Files to Create

```
src/minivess/
├── pipeline/
│   ├── reproducibility_check.py      # Phase 1: reproducibility verification
│   ├── dashboard_tasks.py            # Phase 5: dashboard task implementations
│   └── viz/                          # Phase 3: visualization system
│       ├── __init__.py
│       ├── plot_config.py            # Centralized styling (Paul Tol palette)
│       ├── figure_dimensions.py      # Preset figsize management
│       ├── figure_export.py          # Multi-format export (PNG/SVG/EPS + JSON)
│       ├── loss_comparison.py        # F1.1, F1.4 — loss/architecture comparison
│       ├── training_curves.py        # F1.3 — per-epoch training curves
│       ├── fold_heatmap.py           # F1.2 — per-fold metric heatmap
│       ├── metric_correlation.py     # F1.5 — metric correlation matrix
│       ├── factorial_analysis.py     # F2.1-F2.5 — specification curve, CD diagram
│       ├── uq_plots.py              # F3.1-F3.5 — uncertainty quantification
│       ├── observability_plots.py    # F4.1-F4.5 — drift, latency, GPU
│       ├── external_generalization.py # F5.1-F5.3 — domain gap analysis
│       └── generate_all_figures.py   # Master generation script
├── data/
│   └── external_datasets.py          # Phase 2: external dataset management
├── orchestration/flows/
│   └── dashboard_flow.py             # Phase 5: Dashboard & Reporting Flow

configs/
└── external_datasets.yaml            # Phase 2: dataset download configs

observable/                            # Phase 4: Observable Framework project
├── package.json
├── observablehq.config.js
└── src/
    ├── index.md                       # Dashboard home
    ├── loss-comparison.md             # Interactive loss comparison
    ├── fold-analysis.md               # Per-fold metrics
    ├── uq-analysis.md                 # UQ visualization
    ├── drift-monitoring.md            # Evidently embed
    └── data/                          # Parquet exports (gitignored)

.github/workflows/
└── dashboard.yml                      # GitHub Pages deployment

tests/v2/unit/
├── test_reproducibility_check.py      # Phase 1 tests
├── test_external_datasets.py          # Phase 2 tests
├── test_plot_config.py                # Phase 3 tests (styling, palette, export)
├── test_loss_comparison.py            # Phase 3 tests (F1.1, F1.4)
├── test_training_curves.py            # Phase 3 tests (F1.3)
├── test_fold_heatmap.py               # Phase 3 tests (F1.2)
├── test_factorial_analysis.py         # Phase 3 tests (F2.1-F2.5)
├── test_uq_plots.py                   # Phase 3 tests (F3.1-F3.5)
├── test_observability_plots.py        # Phase 3 tests (F4.1-F4.5)
└── test_dashboard_tasks.py            # Phase 5 tests
```

### Modified Files

```
src/minivess/orchestration/flows/analysis_flow.py  # New tasks 10-11, dashboard trigger
src/minivess/pipeline/duckdb_extraction.py          # Parquet export
pyproject.toml                                      # New dependencies
```

### New Dependencies

```toml
[project.optional-dependencies]
viz = [
    "seaborn>=0.13,<1.0",
]
dashboard = [
    "evidently>=0.5,<1.0",         # Already in pyproject.toml
]
```

---

## 9. TDD Execution Order

Each phase follows RED → GREEN → FIX → VERIFY → CONVERGE → CHECKPOINT.

### Phase 1: Reproducibility Verification (7 tasks)

| # | Task | Test Count | Depends On |
|---|------|------------|------------|
| 1.1 | RED: Write `test_reproducibility_check.py` | ~12 tests | — |
| 1.2 | GREEN: Implement `reproducibility_check.py` | — | 1.1 |
| 1.3 | FIX: Targeted fixes for any failures | — | 1.2 |
| 1.4 | VERIFY: lint + typecheck + regression | — | 1.3 |
| 1.5 | Wire into analysis_flow.py as Prefect task | — | 1.4 |
| 1.6 | Integration test with real mlruns data | — | 1.5 |
| 1.7 | CONVERGE + CHECKPOINT: all green → commit | — | 1.6 |

### Phase 2: External Test Dataset (6 tasks)

| # | Task | Test Count | Depends On |
|---|------|------------|------------|
| 2.1 | RED: Write `test_external_datasets.py` | ~10 tests | — |
| 2.2 | GREEN: Implement `external_datasets.py` + config | — | 2.1 |
| 2.3 | FIX: Targeted fixes for any failures | — | 2.2 |
| 2.4 | VERIFY: lint + typecheck + regression | — | 2.3 |
| 2.5 | Wire external eval into analysis_flow.py | — | 2.4 |
| 2.6 | CONVERGE + CHECKPOINT: all green → commit | — | 2.5 |

### Phase 3: Visualization System (14 tasks)

Each module follows a RED → GREEN → FIX → VERIFY → CONVERGE inner loop.

| # | Task | Test Count | Depends On |
|---|------|------------|------------|
| 3.1 | RED: Write `test_plot_config.py` (styling, palette, save) | ~8 tests | — |
| 3.2 | GREEN: Implement `plot_config.py` + `figure_dimensions.py` + `figure_export.py` | — | 3.1 |
| 3.3 | FIX + VERIFY: lint + typecheck + targeted fixes | — | 3.2 |
| 3.4 | CONVERGE: `test_plot_config.py` all green | — | 3.3 |
| 3.5 | RED: Write `test_loss_comparison.py` (F1.1, F1.4) | ~4 tests | 3.4 |
| 3.6 | GREEN: Implement `loss_comparison.py` | — | 3.5 |
| 3.7 | RED: Write `test_training_curves.py` (F1.3) + `test_fold_heatmap.py` (F1.2) | ~5 tests | 3.6 |
| 3.8 | GREEN: Implement `training_curves.py` + `fold_heatmap.py` + `metric_correlation.py` (F1.5) | — | 3.7 |
| 3.9 | RED: Write `test_factorial_analysis.py` (F2.1-F2.5) | ~6 tests | 3.8 |
| 3.10 | GREEN: Implement `factorial_analysis.py` (spec curve, CD diagram, forest) | — | 3.9 |
| 3.11 | RED: Write `test_uq_plots.py` (F3.1-F3.5) + `test_observability_plots.py` (F4.1-F4.5) | ~8 tests | 3.10 |
| 3.12 | GREEN: Implement `uq_plots.py` + `observability_plots.py` + `external_generalization.py` | — | 3.11 |
| 3.13 | GREEN: Implement `generate_all_figures.py` (master script) | — | 3.12 |
| 3.14 | FIX + VERIFY + CHECKPOINT: full regression + commit | — | 3.13 |

### Phase 4: Observable Framework Dashboard (7 tasks)

| # | Task | Test Count | Depends On |
|---|------|------------|------------|
| 4.1 | Set up `observable/` project skeleton | — | 3.14 |
| 4.2 | RED: Write Parquet export tests | ~3 tests | 4.1 |
| 4.3 | GREEN: Implement Parquet export in `duckdb_extraction.py` | — | 4.2 |
| 4.4 | Create dashboard pages (5 pages) | — | 4.3 |
| 4.5 | Create GitHub Actions workflow | — | 4.4 |
| 4.6 | FIX + VERIFY: lint + regression | — | 4.5 |
| 4.7 | CONVERGE + CHECKPOINT: all green → commit | — | 4.6 |

### Phase 5: Dashboard & Reporting Flow (6 tasks)

| # | Task | Test Count | Depends On |
|---|------|------------|------------|
| 5.1 | RED: Write `test_dashboard_tasks.py` | ~10 tests | — |
| 5.2 | GREEN: Implement `dashboard_tasks.py` + `dashboard_flow.py` | — | 5.1 |
| 5.3 | FIX: Targeted fixes for any failures | — | 5.2 |
| 5.4 | Wire best-effort trigger into analysis_flow.py | — | 5.3 |
| 5.5 | VERIFY: lint + typecheck + full regression | — | 5.4 |
| 5.6 | CONVERGE + CHECKPOINT: all green → commit | — | 5.5 |

**Total: 40 tasks, ~75 new tests**

---

## 10. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| DeepVess download requires registration | Phase 2 blocked | LOW | Contact authors; dataset also referenced by VesselFM |
| DeepVess / tUbeNet license unclear | Phase 2 blocked | MEDIUM | Verify actual license on eCommons / UCL Figshare before download |
| Reproducibility check fails (non-deterministic CUDA ops) | Phase 1 false alarm | MEDIUM | Use per-metric tolerances; cuDNN deterministic mode; document acceptable delta |
| Observable Framework deprecated | Phase 4 rework | LOW | It's open source; Quarto is a fallback |
| DuckDB-WASM Parquet incompatibility | Phase 4 dashboard blank | LOW | Test with DuckDB-WASM version matching |
| GitHub Pages d3.js CSP restrictions | Phase 4 interactive charts broken | LOW | Observable Framework handles CSP correctly |
| Skeleton computation non-determinism | Phase 1 MASD tolerance | MEDIUM | Use 1e-3 tolerance for MASD, document |

---

## Appendix A: foundation-PLR Patterns to Adopt

| Pattern | Source File | Adaptation |
|---------|-----------|-----------|
| Centralized styling | `src/viz/plot_config.py` | Copy COLORS dict, adapt for loss functions |
| Figure dimension presets | `src/viz/figure_dimensions.py` | Same pattern, segmentation-specific presets |
| Multi-format export | `src/viz/plot_config.save_figure()` | Identical API |
| Critical difference diagram | `src/viz/cd_diagram.py` | Replace classifiers with loss functions |
| Specification curve | `src/viz/specification_curve.py` | Replace pipeline stages with segmentation config |
| Sensitivity heatmap | `src/viz/heatmap_sensitivity.py` | loss × architecture matrix |
| Forest plot | `src/viz/forest_plot.py` | Model comparison with CI |
| Parallel coordinates | `src/viz/parallel_coordinates_preprocessing.py` | Hyperparameter → metric mapping |
| JSON data export | `src/viz/figure_data.py` | Same pattern for reproducibility |
| Master generation | `src/viz/generate_all_figures.py` | Same orchestration pattern |

## Appendix B: VesselFM Data Leakage Warning

VesselFM (CVPR 2025) was trained on **all** of these datasets including MiniVess
(Class 21), DeepVess (Class 16), tUbeNet (Class 7), VesSAP (Classes 10-11), and
VesselExpress (Classes 18-20). If comparing against VesselFM as a baseline:

- **Fair:** Evaluate VesselFM on datasets NOT in its D_real (e.g., VessMAP, COCTA)
- **Unfair:** Evaluate VesselFM on any D_real member dataset (data leakage)
- **Our approach:** We do NOT compare against VesselFM. We use external datasets
  purely for our own model's generalization assessment.

## Appendix C: Prefect 5-Flow Architecture

The project uses **5 persona-based Prefect flows**. Flows 1-4 are **core**
(failure blocks the pipeline). Flow 5 is **best-effort** (failure is caught and
logged, never blocks the pipeline).

| Flow | Persona | Priority | Failure Semantics |
|------|---------|----------|-------------------|
| 1. Data Engineering | Data Engineer | Core | Blocks pipeline |
| 2. Model Training | ML Engineer | Core | Blocks pipeline |
| 3. Model Analysis | Research Scientist | Core | Blocks pipeline |
| 4. Deployment | DevOps / SRE | Core | Blocks pipeline |
| 5. Dashboard & Reporting | Visualization / Reporting | Best-effort | Logged, never blocks |

| Mode | Approach | Server Required |
|------|----------|----------------|
| Local / CI | Analysis flow calls dashboard flow with try/except (via `_prefect_compat.py`) | No |
| Production | `DeploymentEventTrigger` — fully event-driven, independent scheduling | Yes |

The `_prefect_compat.py` shim ensures all flows run as plain Python when
`PREFECT_DISABLED=1`.
