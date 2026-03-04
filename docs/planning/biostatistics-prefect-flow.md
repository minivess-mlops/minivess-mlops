# Biostatistics Prefect Flow: Experiment-Level Statistical Analysis

> **Version**: 1.0 (2026-03-04)
> **Issue**: [#340](https://github.com/minivess-mlops/minivess-mlops/issues/340)
> **Status**: Plan — awaiting implementation
> **Reference**: [foundation-PLR](https://github.com/petteriTeikworking/foundation-PLR) DuckDB + JSON sidecar patterns

## Problem Statement

The current Analysis Flow (Flow 4) operates at the **wrong hierarchical level**. Each
mlrun evaluates a single model configuration (one loss × one fold) and produces per-run
artifacts. A single analysis mlrun has no access to other mlruns — it cannot create
cross-condition comparisons.

**What the paper needs** is experiment-level analysis: statistical tables comparing
multiple mlruns (single folds, CV means, different ensembles) across experimental
conditions (loss functions, model architectures, hyperparameters). This is standard
scientific experiment design — comparing conditions against each other — and it
requires a flow that can **read multiple mlruns simultaneously**.

### Hierarchical Levels

```
Level 0: Single prediction     (one volume, one model)
Level 1: Single fold           (one train/test split, ~23 test volumes)
Level 2: Cross-validation      (3 folds averaged, ~70 volumes)
Level 3: Experiment condition   (one loss function across all folds)  ← Analysis Flow stops here
Level 4: Cross-condition        (compare losses, architectures, ensembles)  ← NEW: Biostatistics Flow
Level 5: Cross-experiment       (compare experiments, e.g., full-width vs half-width)
```

The Biostatistics Flow operates at **Levels 4–5**. It is the ONLY flow that reads
multiple mlruns and creates cross-run statistical comparisons.

## Design Principles

1. **Single responsibility**: Only this flow creates cross-run comparisons. Analysis Flow
   creates per-run evaluation artifacts. No overlap.
2. **DuckDB as the reproducibility contract**: Every biostatistics run creates a DuckDB
   database containing ALL source data needed to reproduce every figure and table.
3. **Per-figure JSON sidecars**: Each figure gets a compact JSON file with the plotting
   data, source query, statistical test results, and lineage metadata. More compact than
   DuckDB for quick inspection; always regenerable from DuckDB.
4. **Lineage to source mlruns**: Every biostatistics mlrun tags all upstream run IDs,
   experiment IDs, and artifact paths. Full provenance chain.
5. **Idempotent**: Fingerprint-based short-circuit. If source runs haven't changed, skip.
6. **Task-agnostic**: No hardcoded metric names, loss names, or experiment names in
   infrastructure code. Everything driven by config and DuckDB queries.

## Pipeline Position

```
Flow 1: Data Engineering (core)
Flow 2: Model Training (core)
Flow 3: Post-Training (core, best-effort tasks)
Flow 4: Analysis (core) — per-run evaluation, champion tagging
Flow 5: Biostatistics (best-effort) — cross-run statistical comparisons  ← NEW
Flow 6: Deployment (core)
Flow 7: Dashboard & Reporting (best-effort)
Flow 8: QA (best-effort)
```

**Classification: best-effort.** The pipeline functions without it (training, analysis,
deployment all work). However, CI can make it a hard gate for paper-submission workflows.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Biostatistics Flow (Prefect)                                         │
│                                                                       │
│  ┌─────────────────────┐    ┌──────────────────────────┐             │
│  │ task_discover_runs   │───▶│ task_validate_completeness│             │
│  └─────────────────────┘    └──────────┬───────────────┘             │
│                                        │                              │
│                              ┌─────────▼────────────┐                │
│                              │ task_build_duckdb     │                │
│                              │ (multi-experiment)    │                │
│                              └─────────┬────────────┘                │
│                                        │                              │
│              ┌─────────────────────────┼──────────────────┐          │
│              │                         │                   │          │
│    ┌─────────▼──────────┐  ┌──────────▼─────────┐ ┌──────▼────────┐│
│    │ task_pairwise_tests │  │ task_variance_decomp│ │ task_ranking  ││
│    │ (paired bootstrap,  │  │ (Friedman, ICC,     │ │ (rank-then-  ││
│    │  Holm-Bonferroni,   │  │  mixed model)       │ │  aggregate,  ││
│    │  Cliff's delta)     │  │                     │ │  CD diagram) ││
│    └─────────┬──────────┘  └──────────┬─────────┘ └──────┬────────┘│
│              │                         │                   │          │
│              └─────────────────────────┼──────────────────┘          │
│                                        │                              │
│              ┌─────────────────────────┼──────────────────┐          │
│              │                         │                   │          │
│    ┌─────────▼──────────┐  ┌──────────▼─────────┐ ┌──────▼────────┐│
│    │ task_gen_figures    │  │ task_gen_tables     │ │ task_gen_cd   ││
│    │ (box plots, forest, │  │ (LaTeX comparison,  │ │ (critical    ││
│    │  heatmaps + JSON   │  │  variance decomp,   │ │  difference  ││
│    │  sidecars)          │  │  champion summary)  │ │  diagrams)   ││
│    └─────────┬──────────┘  └──────────┬─────────┘ └──────┬────────┘│
│              │                         │                   │          │
│              └─────────────────────────┼──────────────────┘          │
│                                        │                              │
│                              ┌─────────▼────────────┐                │
│                              │ task_log_mlflow_run   │                │
│                              │ (ONE run with all     │                │
│                              │  artifacts + lineage) │                │
│                              └──────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
         │                            │                         │
         ▼                            ▼                         ▼
   MLflow Artifacts              DuckDB + Parquet          JSON Sidecars
   (figures, tables)             (full source data)        (per-figure)
```

## Prefect Tasks

### Task 1: `task_discover_source_runs`

Discover all completed runs from upstream experiments.

```python
@task(name="discover-source-runs")
def task_discover_source_runs(
    mlruns_dir: Path,
    experiment_names: list[str],  # From config, e.g., ["dynunet_loss_variation_v2"]
) -> SourceRunManifest:
    """Find all FINISHED runs across specified experiments.

    Returns:
        SourceRunManifest with run_ids, experiment_ids, and a fingerprint
        (SHA256 of sorted run_ids) for idempotency checking.
    """
```

**Output**: `SourceRunManifest` dataclass with:
- `runs: list[SourceRun]` (run_id, experiment_id, experiment_name, loss_function, fold_id, status)
- `fingerprint: str` (SHA256 hash of sorted run_ids)
- `discovered_at: datetime`

### Task 2: `task_validate_source_completeness`

Validate that source data is complete enough for statistical analysis.

```python
@task(name="validate-source-completeness")
def task_validate_source_completeness(
    manifest: SourceRunManifest,
    min_folds_per_condition: int = 3,
    min_conditions: int = 2,
) -> ValidationResult:
    """Check that all expected folds are present and no runs are incomplete.

    Validates:
    - All conditions have >= min_folds_per_condition completed folds
    - No runs in FAILED or RUNNING state
    - At least min_conditions conditions to compare
    - Per-volume metric arrays exist (not just fold-level aggregates)

    Raises:
        BiostatisticsValidationError if critical checks fail.
    """
```

**Checks**:
- All expected folds present for each loss/condition
- No FAILED/RUNNING runs included
- Minimum sample sizes for bootstrap tests (N >= 3 folds, N >= 20 volumes)
- Source experiment names actually exist

### Task 3: `task_build_biostatistics_duckdb`

Create a unified multi-experiment DuckDB database.

```python
@task(name="build-biostatistics-duckdb")
def task_build_biostatistics_duckdb(
    manifest: SourceRunManifest,
    mlruns_dir: Path,
    output_path: Path,
) -> Path:
    """Extract all source runs into a single DuckDB database.

    Schema extends existing duckdb_extraction.py with:
    - experiment_name column in runs table
    - ensemble_members table
    - biostatistics_lineage table
    """
```

**DuckDB Schema** (7 tables):

```sql
-- Existing tables (extended from duckdb_extraction.py)
CREATE TABLE runs (
    run_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,        -- NEW: multi-experiment support
    experiment_id VARCHAR,          -- NEW
    loss_function VARCHAR,
    model_family VARCHAR,
    num_folds INTEGER,
    start_time TIMESTAMP,
    status VARCHAR
);

CREATE TABLE params (
    run_id VARCHAR,
    param_name VARCHAR,
    param_value VARCHAR,
    PRIMARY KEY (run_id, param_name)
);

CREATE TABLE eval_metrics (
    run_id VARCHAR,
    fold_id INTEGER,
    metric_name VARCHAR,
    point_estimate DOUBLE,
    ci_lower DOUBLE,
    ci_upper DOUBLE,
    ci_level DOUBLE,
    PRIMARY KEY (run_id, fold_id, metric_name)
);

CREATE TABLE training_metrics (
    run_id VARCHAR,
    metric_name VARCHAR,
    last_value DOUBLE,
    PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE champion_tags (
    run_id VARCHAR,
    tag_key VARCHAR,
    tag_value VARCHAR,
    PRIMARY KEY (run_id, tag_key)
);

-- NEW tables for biostatistics
CREATE TABLE ensemble_members (
    ensemble_run_id VARCHAR,
    member_run_id VARCHAR,
    strategy VARCHAR,
    PRIMARY KEY (ensemble_run_id, member_run_id)
);

CREATE TABLE per_volume_metrics (
    run_id VARCHAR,
    fold_id INTEGER,
    volume_id VARCHAR,
    metric_name VARCHAR,
    metric_value DOUBLE,
    PRIMARY KEY (run_id, fold_id, volume_id, metric_name)
);
```

**Key design decisions**:
- Full rebuild every time (database is small: ~840 metric rows). Fingerprint-based
  short-circuit at the flow level avoids redundant runs.
- `per_volume_metrics` table is critical — this enables per-volume paired bootstrap
  tests (N=70 pooled across folds) rather than fold-level aggregates (N=3).
- Parquet export alongside DuckDB for portability.

### Task 4: `task_compute_pairwise_comparisons`

Paired statistical tests between all condition pairs.

```python
@task(name="compute-pairwise-comparisons")
def task_compute_pairwise_comparisons(
    db_path: Path,
    metrics: list[str],         # From metric_registry
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
) -> PairwiseResults:
    """Compute all pairwise comparisons between conditions.

    For each metric × condition pair:
    1. Paired bootstrap test (per-volume, pooled across folds, BCa CIs)
    2. Cohen's d (parametric effect size)
    3. Cliff's delta (non-parametric effect size)
    4. Holm-Bonferroni correction within each metric family

    Uses per_volume_metrics table (N=70 paired observations).
    """
```

**Statistical methods** (informed by biostatistics review):
- **Primary**: Per-volume paired bootstrap test (N=70 pooled across folds)
- **Effect sizes**: Cohen's d + Cliff's delta (both, because metrics vary in skewness)
- **Correction**: Holm-Bonferroni within each metric (6 tests for 4 conditions)
- **Supplementary**: BH-FDR across all tests (48 total) as sensitivity analysis

**Libraries**: Existing `paired_bootstrap_test()`, `holm_bonferroni_correction()`,
`bca_bootstrap_ci()` from `comparison.py` and `ci.py`. New: Cliff's delta (5 lines).

### Task 5: `task_compute_variance_decomposition`

Omnibus tests and variance decomposition.

```python
@task(name="compute-variance-decomposition")
def task_compute_variance_decomposition(
    db_path: Path,
    metrics: list[str],
    min_blocks: int = 3,
) -> VarianceDecompositionResults:
    """Friedman test + ICC for each metric.

    1. Friedman test (non-parametric repeated-measures ANOVA)
       - Folds as blocks, conditions as treatments
       - 3 × 4 matrix (folds × losses) per metric
    2. Nemenyi post-hoc test (after significant Friedman)
    3. ICC (intraclass correlation) for fold effect

    Caveat: K=3 folds provides very low statistical power.
    Reports power alongside p-values.
    """
```

**Libraries**: `scipy.stats.friedmanchisquare`, `scikit_posthocs.posthoc_nemenyi_friedman`,
`pingouin.intraclass_corr`.

**New dependencies**: `uv add scikit-posthocs pingouin`

### Task 6: `task_compute_rankings`

Multi-metric ranking and champion confirmation.

```python
@task(name="compute-rankings")
def task_compute_rankings(
    db_path: Path,
    pairwise_results: PairwiseResults,
    metrics: list[str],
    champion_categories: list[str],
) -> RankingResults:
    """Rank-then-aggregate across metrics (Demšar 2006).

    1. Per-metric ranking of conditions
    2. Mean rank across all metrics
    3. Critical Difference value for CD diagram
    4. Champion confirmation (does statistical evidence support the champion?)
    """
```

### Task 7: `task_generate_figures`

Generate all cross-run comparison figures with JSON sidecars.

```python
@task(name="generate-figures")
def task_generate_figures(
    db_path: Path,
    pairwise_results: PairwiseResults,
    variance_results: VarianceDecompositionResults,
    ranking_results: RankingResults,
    output_dir: Path,
) -> list[FigureArtifact]:
    """Generate publication figures + JSON sidecars.

    Each figure produces:
    - PNG + SVG image files
    - JSON sidecar with plotting data, source query, statistical tests
    """
```

**Figure catalog** (driven by config, not hardcoded):

| Figure | Type | Data Source | Statistical Content |
|--------|------|-------------|-------------------|
| Loss comparison (per metric) | Box plot / forest plot | `per_volume_metrics` | BCa CIs, pairwise p-values |
| Fold stability heatmap | Heatmap | `eval_metrics` | Per-fold × per-loss means |
| Metric correlation matrix | Heatmap | `per_volume_metrics` | Spearman ρ between metrics |
| Critical Difference diagram | CD diagram | `ranking_results` | Friedman + Nemenyi |
| Effect size matrix | Heatmap | `pairwise_results` | Cliff's delta, color-coded |
| Cross-experiment delta | Forest plot | Multi-experiment query | Between-experiment differences |
| Training curves overlay | Line plot | `training_metrics` | Per-condition learning curves |

### JSON Sidecar Schema

Every figure gets a JSON sidecar following this schema (adapted from foundation-PLR):

```json
{
  "figure_id": "fig_loss_comparison_dsc",
  "figure_title": "DSC by Loss Function (Pooled Per-Volume Bootstrap CIs)",
  "generated_at": "2026-03-04T12:00:00Z",
  "git_commit": "9eee4d7",
  "software_versions": {
    "python": "3.12.x",
    "scipy": "1.x",
    "duckdb": "1.x",
    "matplotlib": "3.x"
  },
  "source": {
    "duckdb_artifact": "biostatistics.duckdb",
    "query": "SELECT loss_function, volume_id, metric_value FROM per_volume_metrics WHERE metric_name='dsc'",
    "source_run_ids": ["abc123", "def456"],
    "experiment_name": "dynunet_loss_variation_v2"
  },
  "data_summary": {
    "n_conditions": 4,
    "n_volumes": 70,
    "conditions": ["dice_ce", "cbdice_cldice", "dice_ce_cldice", "warp_cldice"]
  },
  "data": {
    "dice_ce": {
      "values": [0.82, 0.79, ...],
      "mean": 0.824,
      "ci_lower": 0.801,
      "ci_upper": 0.847,
      "n": 70
    }
  },
  "statistical_tests": {
    "pairwise": [
      {
        "condition_a": "cbdice_cldice",
        "condition_b": "dice_ce",
        "p_value": 0.032,
        "p_value_corrected": 0.096,
        "correction": "holm-bonferroni",
        "cohens_d": -0.34,
        "cliffs_delta": -0.28,
        "significant_at_005": false
      }
    ],
    "friedman": {
      "statistic": 7.8,
      "p_value": 0.049,
      "df": 3
    }
  }
}
```

**Design decisions**:
- `data` field contains the actual plotting arrays (not just summaries) because JSON
  sidecars should be self-contained for re-plotting without DuckDB access
- `source.query` enables exact reproduction from DuckDB
- `statistical_tests` embedded so figures and stats are never separated

### Task 8: `task_generate_latex_tables`

Generate camera-ready LaTeX tables.

```python
@task(name="generate-latex-tables")
def task_generate_latex_tables(
    pairwise_results: PairwiseResults,
    variance_results: VarianceDecompositionResults,
    ranking_results: RankingResults,
    output_dir: Path,
) -> list[TableArtifact]:
    """Generate LaTeX tables for paper submission.

    Tables:
    1. Main comparison table (metrics × conditions with CIs + significance markers)
    2. Pairwise p-value matrix (per metric, with Holm-Bonferroni correction)
    3. Effect size table (Cohen's d + Cliff's delta)
    4. Variance decomposition (Friedman statistic, ICC)
    5. Champion summary (3 categories with evidence)
    """
```

### Task 9: `task_log_biostatistics_run`

Create ONE canonical mlrun with all artifacts and lineage.

```python
@task(name="log-biostatistics-run")
def task_log_biostatistics_run(
    manifest: SourceRunManifest,
    db_path: Path,
    figures: list[FigureArtifact],
    tables: list[TableArtifact],
    pairwise_results: PairwiseResults,
    variance_results: VarianceDecompositionResults,
) -> str:  # Returns biostatistics run_id
    """Log everything to a single MLflow run in 'minivess_biostatistics' experiment.

    Artifacts logged:
    - biostatistics.duckdb (the complete database)
    - biostatistics_parquet/ (Parquet exports of each table)
    - figures/ (PNG + SVG)
    - figures/sidecars/ (JSON per figure)
    - tables/ (LaTeX .tex files)
    - lineage.json (all source run IDs and experiment IDs)

    Tags:
    - upstream_fingerprint: SHA256 of source run IDs
    - upstream_experiment_names: comma-separated list
    - upstream_lineage_artifact: "lineage.json"
    - n_source_runs: count
    - n_conditions: count
    - n_metrics: count
    - biostat_generated_at: ISO 8601
    """
```

**Lineage storage**: Source run IDs stored as a `lineage.json` artifact (not MLflow tags)
to avoid the 500-character tag value limit. A tag `upstream_lineage_artifact` points to it.

```json
{
  "biostatistics_run_id": "biostat_2026-03-04_abc123",
  "generated_at": "2026-03-04T12:00:00Z",
  "fingerprint": "sha256:abcdef...",
  "source_experiments": [
    {
      "experiment_name": "dynunet_loss_variation_v2",
      "experiment_id": "843896622863223169",
      "run_ids": ["run1", "run2", "run3", "run4"]
    }
  ],
  "source_analysis_runs": [
    {
      "experiment_name": "minivess_evaluation",
      "run_ids": ["eval_run1"]
    }
  ],
  "total_source_runs": 5,
  "conditions": ["dice_ce", "cbdice_cldice", "dice_ce_cldice", "warp_cldice"],
  "metrics_analyzed": ["dsc", "hd95", "assd", "nsd", "cldice", "be_0", "be_1", "junction_f1"]
}
```

## Statistical Methodology

### Primary Analysis (must-have for paper)

| Analysis | Method | Level | N | Library |
|----------|--------|-------|---|---------|
| Point estimates + CIs | BCa bootstrap (B=10,000) | Per-volume, pooled across folds | 70 | Existing `bca_bootstrap_ci` |
| Pairwise significance | Paired bootstrap test | Per-volume, paired by volume_id | 70 | Existing `paired_bootstrap_test` |
| Multiple comparison correction | Holm-Bonferroni (within metric) | Per metric family (6 tests) | — | Existing `holm_bonferroni_correction` |
| Effect sizes | Cohen's d + Cliff's delta | Per-volume pairs | 70 | Existing d + new delta |

### Secondary Analysis (should-have)

| Analysis | Method | Level | N | Library |
|----------|--------|-------|---|---------|
| Omnibus test | Friedman test | Fold-level (3 blocks × 4 treatments) | 3 | `scipy.stats.friedmanchisquare` |
| Post-hoc | Nemenyi test | After significant Friedman | 3 | `scikit_posthocs` |
| Multi-metric ranking | Rank-then-aggregate + CD diagram | Fold-level | 3 | Existing + `scikit_posthocs` |
| Cross-metric sensitivity | BH-FDR across all 48 tests | All tests | — | `statsmodels.stats.multitest` |

### Nice-to-Have

| Analysis | Method | Level | N | Library |
|----------|--------|-------|---|---------|
| Variance decomposition | ICC (intraclass correlation) | Per-volume with fold random effect | 70 | `pingouin.intraclass_corr` |
| Calibration | ECE + reliability diagram | Per-voxel | All voxels | `netcal` |

### Critical Statistical Fixes (from review)

1. **Fix CI averaging bug**: Current `_summarise_metric()` in `comparison.py` averages
   CI endpoints across folds — this is NOT a valid confidence interval. Replace with
   pooled BCa bootstrap across all per-volume observations.

2. **Add Cliff's delta**: Non-parametric effect size, better for skewed metrics (HD95, MASD).
   ```python
   def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
       comparisons = np.sign(a[:, None] - b[None, :])
       return float(np.sum(comparisons)) / (len(a) * len(b))
   ```
   Thresholds: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large.

3. **Fix array truncation**: `comparison_report.py` silently truncates mismatched arrays.
   Should raise an error or use unpaired testing explicitly.

4. **Report per-volume AND per-fold**: Per-volume (pooled, N=70) as primary with CIs.
   Per-fold (N=3) as supplementary showing split stability.

### Power Limitations (must acknowledge in paper)

- Friedman test with K=3 folds has very low statistical power. A non-significant result
  does NOT mean no difference — it means we cannot detect it with 3 blocks.
- Bootstrap CIs from N=3 fold-level aggregates are unreliable. Always pool to N=70.
- For heavily skewed metrics (HD95), BCa CIs may undercover even with N=70.
  Report CI width as a quality indicator.

## Relationship to Existing Code

### Code to Reuse (refactor, not rewrite)

| Existing Code | Reuse In |
|--------------|----------|
| `duckdb_extraction.py` → `extract_runs_to_duckdb()` | `task_build_biostatistics_duckdb` (extend with experiment_name, ensemble_members) |
| `comparison.py` → `paired_bootstrap_test()` | `task_compute_pairwise_comparisons` (read from DuckDB instead of in-memory) |
| `comparison.py` → `holm_bonferroni_correction()` | `task_compute_pairwise_comparisons` |
| `ci.py` → `bca_bootstrap_ci()` | `task_compute_pairwise_comparisons` (pooled per-volume) |
| `champion_tagger.py` → `rank_then_aggregate()` | `task_compute_rankings` |
| `comparison_report.py` → `format_comparison_latex()` | `task_generate_latex_tables` |
| `mlruns_inspector.py` → filesystem reading functions | `task_discover_source_runs` |

### Scripts to Deprecate (replace with thin wrappers)

| Script | Replaced By |
|--------|------------|
| `scripts/export_duckdb_parquet.py` | `task_build_biostatistics_duckdb` |
| `scripts/assemble_paper_artifacts.py` | `task_log_biostatistics_run` (writes manifest) |
| `scripts/generate_real_figures.py` | `task_generate_figures` |

Scripts become thin wrappers that call the flow's tasks directly.

## New Dependencies

```bash
uv add scikit-posthocs  # Nemenyi post-hoc, CD diagrams
uv add pingouin         # ICC, mixed ANOVA (optional — statsmodels alternative)
```

## File Structure

```
src/minivess/orchestration/flows/
    biostatistics_flow.py           # @flow biostatistics_flow()

src/minivess/pipeline/
    biostatistics_duckdb.py         # DuckDB schema + multi-experiment extraction
    biostatistics_statistics.py     # Pairwise tests, Friedman, ICC, Cliff's delta
    biostatistics_figures.py        # Figure generation + JSON sidecar writing
    biostatistics_tables.py         # LaTeX table generation
    biostatistics_lineage.py        # Lineage manifest + fingerprinting

src/minivess/config/
    biostatistics_config.py         # BiostatisticsConfig pydantic model

configs/biostatistics/
    default.yaml                    # Default config (experiments, metrics, alpha)

tests/v2/unit/
    test_biostatistics_duckdb.py
    test_biostatistics_statistics.py
    test_biostatistics_figures.py
    test_biostatistics_lineage.py

tests/v2/integration/
    test_biostatistics_flow.py
```

## Implementation Order

### Phase 1: DuckDB Foundation (4 tasks)
1. `BiostatisticsConfig` pydantic model
2. Extend `duckdb_extraction.py` with `experiment_name` column and `per_volume_metrics` table
3. `task_discover_source_runs` + `task_validate_source_completeness`
4. `task_build_biostatistics_duckdb`

### Phase 2: Statistical Engine (3 tasks)
5. Fix CI averaging bug in `comparison.py`
6. Add Cliff's delta to `comparison.py`
7. `task_compute_pairwise_comparisons` (reads from DuckDB, uses existing functions)

### Phase 3: Rankings & Omnibus (2 tasks)
8. `task_compute_variance_decomposition` (Friedman + ICC)
9. `task_compute_rankings` (rank-then-aggregate + CD data)

### Phase 4: Artifacts (3 tasks)
10. `task_generate_figures` with JSON sidecars
11. `task_generate_latex_tables`
12. `task_log_biostatistics_run` (MLflow logging + lineage)

### Phase 5: Flow Assembly (2 tasks)
13. Wire all tasks into `@flow biostatistics_flow()`
14. Update `PipelineTriggerChain` in `trigger.py`

### Phase 6: Tests
15. Unit tests for each module
16. Integration test: mock mlruns → DuckDB → figures → MLflow run

## Lineage, Provenance & Audit Trail

### Why This Matters: The Gibson et al. (2026) Wake-Up Call

Gibson, White, Collins & Barnett (2026) examined two popular Kaggle health datasets
(stroke prediction, diabetes prediction) and found **catastrophic provenance failure**:

- **124 published clinical prediction model studies** built on these two datasets
- Both datasets scored **0/9** on TRIPOD+AI data provenance items
- Only **7%** of all TRIPOD+AI items (75/1,116) adequately reported across all 124 studies
- **3 models deployed in actual clinical practice**, 1 cited in a medical device patent
- Strong evidence both datasets are **simulated or fabricated** (distributional anomalies,
  duplicate rows, clinically implausible patterns)
- **90%** of studies had no ethics statement at all

**Implication for this platform**: The Biostatistics Flow creates the definitive
experiment-level artifacts that will be cited and reused. If the provenance chain
is broken here, downstream consumers inherit the same class of failure Gibson exposed.
Automated, machine-readable lineage is not optional — it is the defense against this
failure mode becoming structurally impossible rather than merely unlikely.

### TRIPOD+AI Data Provenance Mapping

The [TRIPOD+AI checklist](https://www.tripod-statement.org/) (Collins et al., 2024)
contains 27 items. At least 10 directly address data provenance, 7 address model
lineage, and 7 address reproducibility. The Biostatistics Flow must satisfy all
provenance items for the cross-run comparison artifacts it creates.

**Gibson et al.'s 9 assessed TRIPOD+AI provenance items:**

| # | Item | What MinIVess Already Tracks | Gap |
|---|------|------------------------------|-----|
| 5a | Data source (where/who) | `data_n_volumes`, `data_train_volume_ids` | Need dataset versioning hash + DOI |
| 5b | Data collection dates | Not tracked | Add collection date metadata from NIfTI |
| 6a | Study setting/location | Not tracked | Add institutional provenance |
| 7 | Preprocessing/quality checks | Patch size, transforms logged | Need DVC pipeline DAG |
| 8a | Outcome definition | Binary segmentation (implicit) | Formalize in Data Card |
| 9b | Predictor definitions | `arch_*` params logged | Good coverage |
| 11 | Missing data handling | Not applicable (volumetric) | Document explicitly |
| 16 | Dev vs eval differences | Train/val/test split logged | Good coverage |
| 20b | Participant characteristics | Volume stats in DatasetProfile | Export to artifact |

### Regulatory Context

| Regulation | Lineage Requirement | Technical Mapping |
|-----------|--------------------|--------------------|
| **EU AI Act Art. 10** | Data origin, collection process, preparation operations | OpenLineage events, DVC provenance, Data Cards |
| **EU AI Act Art. 11** | Technical documentation before market placement | Auto-generated from MLflow metadata |
| **EU AI Act Art. 72** | Post-market monitoring with traceability | Biostatistics flow lineage + drift detection |
| **IEC 62304 §5.6** | Bidirectional traceability (requirements ↔ tests) | Traceability matrices from pipeline metadata |
| **IEC 62304 §8** | Configuration management for all software items | Git SHA + DVC hash + MLflow run ID |
| **FDA PCCP** | Predetermined change control plan | Model governance MLflow tags |
| **TRIPOD+AI** | 27-item reporting checklist | Pydantic TRIPOD+AI contracts (PRD decision) |

### Tool Landscape: Decision Analysis

We evaluated 7 lineage/provenance tools against our requirements:

#### Option A: MLflow Built-in Lineage Only (RECOMMENDED for MVP)

**Mechanism**: Run tags (`upstream_*`), `mlflow.log_input()` for datasets,
artifact co-location, `lineage.json` artifact with fingerprint.

| Criterion | Assessment |
|-----------|-----------|
| Tracks mlruns → comparison | **YES** (native tags, already partially implemented) |
| Tracks figures → DuckDB queries | **Partial** (artifact co-location + logged query params) |
| IEC 62304 audit trail | **Adequate** for academic (timestamped, append-only params) |
| Integration effort | **Zero** — already using MLflow |
| Tamper evidence | **No** — params are append-only but not cryptographically signed |
| Prefect integration | **Good** — `ExperimentTracker` wraps MLflow in each flow |

**Pros**: Zero new infrastructure, leverages existing deep MLflow integration,
sufficient for peer review and TRIPOD+AI compliance.
**Cons**: No formal lineage graph, no cross-pipeline DAG visualization,
no tamper evidence.

#### Option B: MLflow + DVC (RECOMMENDED for production)

**Mechanism**: MLflow for experiment lineage + DVC for data/pipeline lineage.
`dvc.yaml` stages define the reproducible pipeline DAG. Git tag + DVC lock =
exact data snapshot. DVC hash in MLflow tags links experiments to data versions.

| Criterion | Assessment |
|-----------|-----------|
| Tracks mlruns → comparison | **YES** (MLflow tags) |
| Tracks data versions → experiments | **YES** (DVC content-addressable hashes) |
| Pipeline DAG | **YES** (`dvc dag` generates reproducible pipeline graph) |
| IEC 62304 audit trail | **Good** (Git commit + DVC hash = full snapshot) |
| Integration effort | **Low** — `uv add dvc`, define `dvc.yaml`, add remote |

**Pros**: Git-native, well-understood by ML community, good TRIPOD+AI coverage.
**Cons**: No runtime lineage (only pipeline-level DAG).

#### Option C: MLflow + OpenLineage/Marquez (RECOMMENDED for clinical deployment)

**Mechanism**: OpenLineage standard events emitted from each Prefect task.
Marquez backend stores immutable, timestamped lineage graphs. Python SDK
emits `RunEvent` objects with `Run`, `Job`, `Dataset` entities.

| Criterion | Assessment |
|-----------|-----------|
| Tracks everything | **YES** (full DAG: data → preprocess → train → evaluate → compare) |
| IEC 62304 / EU AI Act | **Strong** (immutable lineage graph, timestamped events) |
| Prefect integration | **NONE** — [OpenLineage issue #81](https://github.com/OpenLineage/OpenLineage/issues/81) open since 2021, no resolution |
| Integration effort | **High** — Marquez Docker deployment + custom SDK wrappers for every flow |
| Current state in MinIVess | Config-only (`docker-compose` has Marquez, but `openlineage-python` not in `pyproject.toml`, no events emitted) |

**Pros**: Industry standard, formal lineage graph, regulatory-grade.
**Cons**: No Prefect integration (must write custom wrappers), infrastructure overhead
disproportionate for academic MVP. PRD decision `lineage_tracking` already notes
"not-integrated" status.

#### Option D: Hash-Chained Audit Logging (FUTURE — addresses OLIF threat)

**Mechanism**: Three-layer Capture/Store/Use architecture (Ojewale et al., 2026).
Append-only, hash-chained JSONL logs. Each entry includes SHA-256 of previous entry,
creating tamper-evident chain.

**Addresses**: OLIF threat (Operational Lie Insertion Fabrication) — LLM agents
fabricating audit trail entries under operational pressure. PRD decision
`audit_trail_architecture` evaluates this.

**Assessment**: Experimental, not needed for MVP. Relevant when LLM agents (LangGraph)
make autonomous decisions in the pipeline.

#### Option E: W3C PROV Standard (ACADEMIC FRAMING)

**Mechanism**: [W3C PROV](https://www.w3.org/TR/prov-overview/) is the dominant
academic standard for provenance (43% of papers in JMIR 2024 scoping review).
Entities, Activities, Agents in RDF/JSON-LD format.

**Assessment**: The right academic citation and framing, but implementing full W3C PROV
is overkill. OpenLineage is conceptually aligned with PROV (both are entity-activity
graphs). **Cite W3C PROV in the paper, implement via OpenLineage or MLflow tags.**

#### Decision Matrix

| Tool | MVP | Production | Clinical | Effort | Already Integrated |
|------|:---:|:----------:|:--------:|:------:|:-----------------:|
| **MLflow lineage** | **YES** | YES | Partial | Zero | YES |
| **DVC** | Optional | **YES** | YES | Low | No |
| **OpenLineage+Marquez** | No | Optional | **YES** | High | Config only |
| **Hash-chained audit** | No | No | Future | High | No |
| **W3C PROV** | Cite only | Cite only | Cite + implement | — | No |
| **Kubeflow MLMD** | No | No | No | Medium | No |
| **W&B** | No | No | No | Migration | No |
| **OpenMetadata** | No | No | No | Very High | No |

### Recommendation: Three-Tier Implementation

#### Tier 1: MVP (this PR — MLflow lineage)

The Biostatistics Flow creates `lineage.json` as an MLflow artifact:

```json
{
  "schema_version": "1.0",
  "biostatistics_run_id": "biostat_2026-03-04_abc123",
  "generated_at": "2026-03-04T12:00:00Z",
  "fingerprint": "sha256:abcdef1234567890...",
  "git_commit": "9eee4d7",
  "git_branch": "main",
  "git_dirty": false,
  "environment": {
    "python": "3.12.8",
    "torch": "2.5.1",
    "monai": "1.4.0",
    "scipy": "1.14.1",
    "duckdb": "1.1.3"
  },
  "source_experiments": [
    {
      "experiment_name": "dynunet_loss_variation_v2",
      "experiment_id": "843896622863223169",
      "run_ids": ["run1", "run2", "run3", "run4"],
      "conditions": {
        "run1": {"loss": "dice_ce", "folds": [0, 1, 2]},
        "run2": {"loss": "cbdice_cldice", "folds": [0, 1, 2]}
      }
    }
  ],
  "dataset_provenance": {
    "name": "MiniVess",
    "version": "1.0",
    "n_volumes": 70,
    "source": "University of Zurich, Institute of Pharmacology and Toxicology",
    "collection_method": "Two-photon fluorescence microscopy",
    "spatial_resolution_um": "0.31-4.97",
    "ethics": "Animal experiments approved under Swiss cantonal regulations"
  },
  "tripod_ai_items": {
    "5a_data_source": "Two-photon microscopy, MiniVess public dataset",
    "5b_data_dates": "2019-2020 collection period",
    "7_preprocessing": "Native resolution, NormalizeIntensityd(nonzero=True), no resampling",
    "8a_outcome": "Binary vessel segmentation (foreground/background)",
    "12c_model_spec": "DynUNet, 4 losses x 3 folds x 100 epochs",
    "22_full_spec": "See MLflow params (17 training + arch + system params)"
  },
  "artifacts_produced": {
    "duckdb": "biostatistics.duckdb",
    "figures": ["fig_loss_comparison_dsc.png", "fig_cd_diagram.png"],
    "tables": ["comparison_table.tex", "pvalue_matrix.tex"],
    "parquet": ["runs.parquet", "eval_metrics.parquet", "per_volume_metrics.parquet"]
  },
  "statistical_methods": {
    "primary_ci": "BCa bootstrap (B=10000, pooled N=70)",
    "pairwise_test": "Paired bootstrap with Holm-Bonferroni",
    "effect_sizes": ["cohens_d", "cliffs_delta"],
    "omnibus_test": "Friedman (K=3 blocks)",
    "random_seed": 42
  }
}
```

**MLflow tags** (summary pointers, not full data):
- `upstream_fingerprint` → SHA256 of sorted source run IDs
- `upstream_experiment_names` → comma-separated list
- `upstream_lineage_artifact` → `"lineage.json"`
- `n_source_runs`, `n_conditions`, `n_metrics`
- `tripod_ai_coverage` → "9/9" (items with automated evidence)

**Per-figure JSON sidecars** include their own lineage subset:
```json
{
  "source": {
    "biostatistics_run_id": "biostat_2026-03-04_abc123",
    "duckdb_artifact": "biostatistics.duckdb",
    "query": "SELECT ...",
    "source_run_ids": ["run1", "run2"]
  }
}
```

#### Tier 2: Production (future PR — add DVC)

- Version MiniVess dataset with DVC (`dvc add data/raw/minivess/`)
- DVC hash included in `lineage.json` → `dataset_provenance.dvc_hash`
- `dvc.yaml` pipeline stages: `discover → validate → build_duckdb → statistics → figures`
- `dvc repro` reproduces the full biostatistics pipeline from data to paper artifacts

#### Tier 3: Clinical Deployment (future — add OpenLineage)

- Deploy Marquez backend (already in docker-compose, needs PostgreSQL)
- `uv add openlineage-python`
- `LineageEmitter` wraps each Prefect task with START/COMPLETE/FAIL events
- OpenLineage events stored in Marquez for IEC 62304 §8 configuration management
- Auto-generate traceability matrices from OpenLineage graph

### Card System Integration

The Biostatistics Flow should generate structured documentation cards
alongside its statistical artifacts (from appendix-cards.md taxonomy):

| Card Type | What It Documents | Generated From |
|-----------|------------------|---------------|
| **Data Card** (Pushkarna et al., 2022) | Dataset provenance, quality, fairness | `dataset_provenance` in lineage.json + DatasetProfile |
| **Model Card** (Mitchell et al., 2019) | Model behavior, intended use, limitations | MLflow params + evaluation metrics |
| **Environment Card** | Python, CUDA, container versions | `sys_*` MLflow tags |
| **Reproducibility Checklist Card** | Seeds, configs, test harness | `lineage.json` statistical_methods + git state |
| **Deployment Lineage Card** | Git SHA + DVC tag + CI/CD metadata | `lineage.json` git_commit + fingerprint |

**MVP scope**: Generate `data_card.json` and `model_card.json` as MLflow artifacts
in the biostatistics run. These are machine-readable; human-readable markdown versions
can be auto-generated for paper supplementary materials.

### Paper Framing: Provenance as Infrastructure

The paper should frame the MLOps platform as **provenance infrastructure** —
not just experiment management:

- **Experiment tracking** answers "what happened?" (parameters, metrics)
- **Provenance infrastructure** answers "why should you trust it?"
  (data origin, processing chain, environment, reproducibility proof)

Gibson et al. (2026) shows the catastrophic consequences of provenance failure.
The Biostatistics Flow demonstrates that automated provenance tracking makes this
class of failure structurally impossible, not merely unlikely. The cost of
machine-readable provenance (a few `mlflow.log_input()` calls + `lineage.json`)
is negligible compared to the cost of discovering fabricated data after 124
publications and 3 clinical deployments.

**Key citations for the paper**:
- Gibson et al. (2026) — empirical evidence of provenance failure
- Collins et al. (2024) — TRIPOD+AI checklist
- Gebru et al. (2021) — Datasheets for Datasets
- Pushkarna et al. (2022) — Data Cards
- Mitchell et al. (2019) — Model Cards
- Albertoni et al. (2023) — ML reproducibility terminology (arXiv:2302.12691)
- W3C PROV (2013) — provenance standard (cite for academic framing)
- EU AI Act (2024) — regulatory requirements

## Open Questions

1. **Naming**: "Biostatistics Flow" vs "Experiment Comparison Flow" vs "Statistical
   Reporting Flow"? The architecture reviewer noted that "biostatistics" implies
   clinical/epidemiological statistics. However, the user explicitly requested this name.

2. **Cross-experiment comparison scope**: Should the flow compare across experiments
   (e.g., full-width vs half-width DynUNet) in the MVP, or defer to Phase 2?
   Recommendation: include in MVP since the DuckDB already supports multi-experiment.

3. **R integration**: foundation-PLR uses ggplot2 for publication figures. Should we
   support R figure generation? Recommendation: Python-only for MVP (matplotlib/seaborn),
   R as optional Phase 2 addition.

4. **TRIPOD+AI automation depth**: Should the Biostatistics Flow auto-generate a
   TRIPOD+AI compliance report mapping MLflow metadata to checklist items? This is
   the `clinical_contract_schema` PRD decision (Pydantic v2 TRIPOD+AI Contracts,
   prior 0.45). Recommendation: yes for Tier 2, not MVP.

5. **Gibson et al. response in paper**: Should the paper explicitly cite Gibson et al.
   and frame the platform as a response to the provenance crisis? Recommendation:
   yes — it strengthens the contribution and is topically relevant (March 2026 preprint).

## References

- Gibson AD, White NM, Collins GS, Barnett A. (2026). Evidence of Unreliable Data and Poor Data Provenance in Clinical Prediction Model Research. medRxiv. doi:10.64898/2026.02.24.26347028.
- Collins GS, et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. doi:10.1136/bmj-2023-078378.
- Gebru T, et al. (2021). Datasheets for Datasets. Communications of the ACM. doi:10.1145/3458723.
- Pushkarna M, Zaldivar A, Kjartansson O. (2022). Data Cards: Purposeful and Transparent Dataset Documentation. FAccT. doi:10.1145/3531146.3533231.
- Mitchell M, et al. (2019). Model Cards for Model Reporting. FAccT. arXiv:1810.03993.
- Albertoni R, et al. (2023). Reproducibility of Machine Learning: Terminology, Recommendations and Open Issues. arXiv:2302.12691.
- Demšar J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.
- Nadeau C, Bengio Y. (2003). Inference for the Generalization Error. Machine Learning.
- Romano J, et al. (2006). Appropriate statistics for ordinal level data. JGME.
- Herbold S. (2020). Autorank: A Python package for automated ranking. JOSS.
- W3C. (2013). PROV-Overview. https://www.w3.org/TR/prov-overview/.
- EU AI Act. (2024). Regulation 2024/1689. Articles 10-11.
- IEC 62304:2006+A1:2015. Medical device software lifecycle processes.
- Schelter S, et al. (2017). Automatically Tracking Metadata and Provenance of ML Experiments. NeurIPS MLSys Workshop.
- Kreuzberger D, et al. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access.
- Raschka S, et al. (2022). Machine Learning in Python: Main Developments and Technology Trends. arXiv:2207.09315.
- Liu P, et al. (2021). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods. arXiv:2107.13586.
- Ojewale V, et al. (2026). Hash-Chained Audit Logging for AI Systems.
- De la Torre L. (2026). Zero-Trust Verified Logging (RAL/FCV architecture).
- foundation-PLR: DuckDB + JSON sidecar patterns (internal reference implementation).
- Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.
- Nadeau & Bengio (2003). Inference for the Generalization Error. Machine Learning.
- Romano et al. (2006). Appropriate statistics for ordinal level data. JGME.
- Herbold (2020). Autorank: A Python package for automated ranking. JOSS.
- foundation-PLR: DuckDB + JSON sidecar patterns (internal reference implementation).
