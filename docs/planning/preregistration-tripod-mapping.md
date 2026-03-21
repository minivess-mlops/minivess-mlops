# TRIPOD+AI Preregistration Mapping for MinIVess Factorial Experiment

**Date**: 2026-03-20
**Status**: De facto preregistration — maps each statistical test to TRIPOD+AI items
**Reference**: [Collins et al. (2024). "TRIPOD+AI statement." *BMJ*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/)

## Overview

This document maps the MinIVess biostatistics flow's statistical analyses to
the TRIPOD+AI 27-item checklist, creating a de facto preregistration for
the factorial experiment reported in the Nature Protocols manuscript.

---

## Mapping: TRIPOD+AI Items → Biostatistics Flow Components

### Study Design (Items 3–4)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **3a** | Healthcare context and rationale | Multiphoton microvasculature imaging; platform for heterogeneous segmentation models | `knowledge-graph/domains/data.yaml` |
| **3b** | Target population and purpose | Mouse brain cortex microvasculature (MiniVess dataset, 70 volumes) | `configs/hpo/paper_factorial.yaml` |
| **3c** | Health inequalities / subgroups | Per-volume subgroup analysis; external test on DeepVess (different lab) | `biostatistics_flow.py:_build_per_volume_data()` |
| **4** | Study objectives | 4×3×2 factorial comparison of segmentation models on topology-aware metrics | `docs/planning/biostatistic-flow-debug-double-check.xml` |

### Data Sources (Items 5–6)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **5a** | Data sources with representativeness | MiniVess (70 volumes, development), DeepVess (~7 volumes, external test) | `knowledge-graph/decisions/L3-technology/dataset_strategy.yaml` |
| **5b** | Data dates | Specified in dataset metadata (EBRAINS provenance) | `docs/datasets/README.md` |
| **6a** | Study setting | Single-center (development), cross-lab (external test) | `configs/splits/3fold_seed42.json` |

### Outcome Definition (Item 8)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **8a** | Outcome definition | Binary vascular segmentation masks; per-volume evaluation | `src/minivess/pipeline/metrics.py` |
| **8b** | Outcome assessors | Expert manual annotations (MiniVess: multiphoton lab; DeepVess: separate lab) | Dataset documentation |

### Sample Size (Item 10)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **10** | Study size rationale | 70 volumes, 3-fold CV (seed=42), 23 val volumes per fold. N=23 replication units for per-volume ANOVA. Power analysis via Riley bootstrap instability. | `compute_riley_instability()` in `biostatistics_statistics.py` |

### Statistical Methods (Item 12)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Config Reference |
|------|----------------------|------------------------|------------------|
| **12a** | Data usage | 3-fold cross-validation for MiniVess (development); held-out DeepVess (external test). Both splits analyzed separately. | `BiostatisticsConfig.splits` |
| **12c** | Model type & development | 4 model families (DynUNet, MambaVesselNet++, SAM3 TopoLoRA, SAM3 Hybrid) × 3 losses × 2 calibration settings. Config-driven factorial design. | `configs/hpo/paper_factorial.yaml` |
| **12e** | Performance measures | **Co-primary**: clDice + MASD (Holm-Bonferroni). **FOIL**: DSC (BH-FDR). **Secondary**: HD95, ASSD, NSD, BE₀, BE₁ (BH-FDR). All with bootstrap 95% CI. | `BiostatisticsConfig.co_primary_metrics` |

### Detailed Statistical Test → TRIPOD+AI Mapping

| Statistical Test | TRIPOD+AI Item | Implementation | Purpose |
|-----------------|----------------|----------------|---------|
| **N-way ANOVA** (3 factors) | 12a, 12e | `compute_factorial_anova()` | Main effects + interactions for model×loss×calib |
| **Wilcoxon signed-rank** | 12e, 23a | `compute_pairwise_comparisons()` | Per-volume pairwise comparisons |
| **Holm-Bonferroni MCC** | 12e | `holm_bonferroni_correction()` | Co-primary metric correction |
| **BH-FDR MCC** | 12e | `_bh_fdr_correction()` | Secondary/FOIL metric correction |
| **Cohen's d + Cliff's δ + VDA** | 12e, 23a | `compute_pairwise_comparisons()` | Effect size quantification |
| **Bayesian signed-rank (ROPE)** | 12e, 23a | `compute_bayesian_comparisons()` | Practical equivalence assessment |
| **Friedman + Nemenyi** | 12e | `compute_variance_decomposition()` | Non-parametric model ranking |
| **ICC(2,1)** | 23b | `_compute_icc()` | Inter-fold agreement |
| **Specification curve** | 12e, 23a | `compute_specification_curve()` | Robustness across analytical choices |
| **Kendall's tau** | 12e, 23a | `compute_rank_concordance()` | Metric rank agreement/inversion |
| **Riley bootstrap instability** | 10, 23a | `compute_riley_instability()` | Ranking stability assessment |
| **Brier score + O/E + IPA** | 12e | `compute_calibration_summary()` | Calibration assessment (aux_calib factor) |
| **Bootstrap 95% CI** | 23a | `ConfidenceInterval` dataclass | All metric CIs |

### Model Performance (Items 23)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **23a** | Performance with CIs + subgroups | Per-volume bootstrap CIs for all metrics. Per-fold and cross-fold aggregation. External test (DeepVess) reported separately. | `biostatistics_flow.py` |
| **23b** | Heterogeneity across clusters | ICC(2,1) for inter-fold agreement. Specification curve across analytical choices. | `compute_variance_decomposition()`, `compute_specification_curve()` |

### AI-Specific Items (Fairness, Uncertainty)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **7** | Data preparation consistency | Standard preprocessing via MONAI transforms. Config-driven, not data-dependent. | `configs/data/minivess.yaml` |
| **13** | Class imbalance | Vascular structures are minority class. CbDice loss addresses this. | `configs/training/default.yaml` |
| **14** | Fairness approaches | Per-volume analysis (no demographic subgroups in mouse data). Cross-lab generalization via DeepVess. | `biostatistics_flow.py` |
| **15** | Model output | Softmax probabilities (B, 2, D, H, W). Binary segmentation via thresholding. | `MiniVessSegModel.predict()` |

### Open Science (Items 18)

| Item | TRIPOD+AI Requirement | MinIVess Implementation | Code Reference |
|------|----------------------|------------------------|----------------|
| **18c** | Protocol accessibility | This document serves as de facto preregistration | This file |
| **18d** | Registration | Not formally registered; factorial design prespecified in configs | `configs/hpo/paper_factorial.yaml` |
| **18e** | Data sharing | MiniVess: EBRAINS (planned). DeepVess: public. | `docs/datasets/README.md` |
| **18f** | Code sharing | Full pipeline code in this repository (CC BY-NC license) | GitHub repository |

---

## Key Analytical Decisions (Prespecified)

1. **Co-primary metrics**: clDice + MASD (MetricsReloaded-driven, not post-hoc)
2. **FOIL metric**: DSC included to demonstrate rank inversion
3. **Multiple comparison correction**: Two-tier (Holm for co-primaries, BH-FDR for secondary)
4. **Random effect**: Fold ID (K=3 production, K=1 debug with per-volume replication)
5. **Specification curve**: All researcher degrees of freedom systematically varied
6. **Bayesian ROPE**: Per-metric width from `BiostatisticsConfig.rope_values`
7. **Alpha**: Configurable via `BiostatisticsConfig.alpha` (never hardcoded)

## Cross-References

- Factorial design: `configs/hpo/paper_factorial.yaml`
- Biostatistics config: `src/minivess/config/biostatistics_config.py`
- KG decisions: `knowledge-graph/decisions/L3-technology/primary_metrics.yaml`
- Implementation plan: `docs/planning/biostatistic-flow-debug-double-check.xml`
