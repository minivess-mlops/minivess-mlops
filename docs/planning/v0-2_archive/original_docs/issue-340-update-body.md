---
title: "Issue 340 Update: Analysis at the Wrong Hierarchical Level"
status: reference
created: ""
---

## Problem: Analysis at the Wrong Hierarchical Level

The issue is NOT that analysis artifacts are "scattered across MLflow runs." The real problem is that the Analysis Flow (Flow 4) operates at the **wrong hierarchical level**. Each mlrun evaluates a single model configuration and has no access to other mlruns — it cannot create cross-condition comparisons.

**What the paper needs** is experiment-level analysis: statistical tables comparing multiple mlruns (single folds, CV means, ensembles) across experimental conditions (losses, architectures, hyperparameters). This is standard scientific experiment design — comparing conditions to each other.

### Hierarchical Levels

| Level | Scope | Current Owner | Example |
|-------|-------|--------------|---------|
| 0 | Single prediction | Inference | One volume, one model |
| 1 | Single fold | Analysis Flow | ~23 test volumes, one split |
| 2 | Cross-validation | Analysis Flow | 3 folds averaged |
| 3 | Experiment condition | Analysis Flow | One loss across all folds |
| **4** | **Cross-condition** | **Biostatistics Flow (NEW)** | **Compare 4 losses** |
| **5** | **Cross-experiment** | **Biostatistics Flow (NEW)** | **Full-width vs half-width** |

The Biostatistics Flow is the **ONLY** flow that reads multiple mlruns simultaneously and creates cross-run statistical comparisons.

## Solution: Biostatistics Prefect Flow (Flow 5)

A new Prefect flow between Analysis and Deploy that:

1. **Discovers** all completed runs from upstream experiments
2. **Builds a DuckDB database** containing ALL source run data (metrics, params, per-volume results)
3. **Computes cross-run statistics** (paired bootstrap, Friedman, effect sizes, multiple comparison correction)
4. **Generates figures** with per-figure JSON sidecars (compact, reproducible, self-contained)
5. **Generates LaTeX tables** for paper submission
6. **Logs ONE canonical mlrun** in `minivess_biostatistics` experiment with all artifacts + lineage to source runs

### Architecture

```
Analysis Flow (per-run)          Biostatistics Flow (cross-run)
┌─────────────────┐             ┌──────────────────────────────┐
│ Run A: dice_ce   │────┐       │                              │
│ Run B: cbdice    │────┼──────▶│  DuckDB (all source data)    │
│ Run C: cldice    │────┤       │  ├── Statistical tests       │
│ Run D: ensemble  │────┘       │  ├── Figures + JSON sidecars │
└─────────────────┘             │  ├── LaTeX tables            │
                                │  └── Lineage to all sources  │
                                └──────────────────────────────┘
                                         │
                                    ONE mlrun in
                                minivess_biostatistics
```

### DuckDB as Reproducibility Contract

Every biostatistics run creates a DuckDB database with 7 tables: `runs`, `params`, `eval_metrics`, `training_metrics`, `champion_tags`, `ensemble_members`, `per_volume_metrics`. This database contains everything needed to reproduce every figure and table.

### Per-Figure JSON Sidecars

Each figure gets a compact JSON file (pattern from [foundation-PLR](https://github.com/petteriTeikari/foundation-PLR)) containing:
- Plotting data arrays
- Source DuckDB query (exact reproduction)
- Statistical test results (p-values, effect sizes, CIs)
- Lineage (source run IDs, experiment name, git commit)

More compact than DuckDB for quick inspection; always regenerable from DuckDB.

### Lineage

Each biostatistics mlrun stores a `lineage.json` artifact with all upstream run IDs, experiment IDs, and a fingerprint (SHA256 of sorted run IDs) for idempotency.

## Prefect Tasks

| # | Task | Purpose |
|---|------|---------|
| 1 | `task_discover_source_runs` | Find all FINISHED runs from upstream experiments |
| 2 | `task_validate_source_completeness` | Assert all folds present, no FAILED runs, minimum sample sizes |
| 3 | `task_build_biostatistics_duckdb` | Multi-experiment DuckDB with per-volume metrics |
| 4 | `task_compute_pairwise_comparisons` | Paired bootstrap, Holm-Bonferroni, Cohen's d, Cliff's delta |
| 5 | `task_compute_variance_decomposition` | Friedman test, Nemenyi post-hoc, ICC |
| 6 | `task_compute_rankings` | Rank-then-aggregate, Critical Difference diagram data |
| 7 | `task_generate_figures` | Box plots, forest plots, heatmaps, CD diagrams + JSON sidecars |
| 8 | `task_generate_latex_tables` | Comparison tables, p-value matrices, effect sizes, champion summary |
| 9 | `task_log_biostatistics_run` | ONE mlrun with all artifacts + lineage.json |

## Statistical Methods

### Primary (must-have)
- **BCa bootstrap CIs** (B=10,000) — per-volume, pooled across folds (N=70)
- **Paired bootstrap tests** — per-volume, paired by volume_id
- **Holm-Bonferroni correction** — within each metric family
- **Cohen's d + Cliff's delta** — parametric + non-parametric effect sizes

### Secondary (should-have)
- **Friedman test** — non-parametric omnibus (K=3 folds as blocks)
- **Nemenyi post-hoc** — pairwise after significant Friedman
- **Critical Difference diagram** — multi-algorithm comparison visualization
- **BH-FDR** — cross-metric sensitivity analysis

### Known Issues to Fix
- CI averaging bug in `comparison.py` (averaging CI endpoints ≠ valid CI)
- Array truncation in `comparison_report.py` (silent data loss)

## Pipeline Position

```
Data → Train → Post-Train → Analysis → **Biostatistics** → Deploy → Dashboard → QA
```

Classification: **best-effort** (pipeline functions without it; CI can make it a hard gate for paper workflows).

## Plan Document

Full implementation plan with architecture diagrams, DuckDB schema, JSON sidecar schema, statistical methodology, implementation phases, and code reuse mapping:

**[docs/planning/biostatistics-prefect-flow.md](docs/planning/biostatistics-prefect-flow.md)**

## New Dependencies

```
scikit-posthocs   # Nemenyi post-hoc, CD diagrams
pingouin          # ICC, mixed ANOVA
```

## Related
- #61 — Prefect + LangGraph optimization
- foundation-PLR — Reference implementation for DuckDB + JSON sidecar patterns
