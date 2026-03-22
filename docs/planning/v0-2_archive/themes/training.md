---
title: "Theme: Training — Loss Functions, Topology, Calibration, Metrics"
theme_id: training
doc_count: 26
archive_path: docs/planning/v0-2_archive/original_docs/
kg_domain: knowledge-graph/domains/training.yaml
created: "2026-03-22"
status: archived
---

# Theme: Training

Loss functions, topology-aware losses, calibration methods, metric frameworks, and
multi-metric model selection. This theme covers the full scientific journey from
MetricsReloaded-guided metric selection through compound loss design, graph-based
topology metrics, and segmentation calibration.

---

## Key Scientific Insights

### 1. Topology-Accuracy Tradeoff is Real and Measurable

The `dynunet_loss_variation_v2` experiment (4 losses x 3 folds x 100 epochs) produced
the first quantitative evidence of the topology-accuracy tradeoff on MiniVess:

- Losses with clDice component achieve 0.90+ clDice but sacrifice ~9% Dice (DSC)
- `dice_ce_cldice` achieves best topology (0.904 clDice) at -8.8% DSC cost
- Pure `dice_ce` achieves best overlap (0.843 DSC) but lower topology preservation

This directly motivated the compound metric approach and multi-metric checkpoint tracking.

### 2. MASD Is NOT Differentiable -- Only Proxy Losses Exist

Extensive research (4 docs) confirmed that the Mean Average Surface Distance recommended
by MetricsReloaded cannot be used as a training loss due to four fundamental
non-differentiability sources: hard thresholding, surface extraction, nearest-neighbor
search, and discrete cardinality. This eliminated naive compound losses combining MASD
with clDice and motivated the Hausdorff distance transform proxy approach.

### 3. No Single Package Implements All Topology Metrics

The graph-topology survey (5 docs, 491+ lines in the main survey) found that no Python
package provides APLS + Skeleton Recall + Branch Detection Rate together. The recommended
path: MONAI clDice (production-ready, 3D) + custom BDR evaluator + gudhi for Betti
numbers. APLS was explicitly rejected as 2D-only (geospatial road networks).

### 4. Range Collapse in Naive Compound Metrics

Analysis showed that with `max_masd=50.0`, actual MASD values (0.99-6.83) compress to a
0.117 range after normalization while clDice spans 0.834. The compound metric
`val_compound_masd_cldice` was effectively `~0.46 + 0.5*clDice`, with clDice dominating
~7:1. Literature (Metrics Reloaded, BraTS, KiTS23) unanimously recommends
rank-then-aggregate over weighted averaging.

### 5. clDice Algorithm Identity Across scikit-image Versions

Verification confirmed that `skimage.morphology.skeletonize()` (new) and
`skeletonize_3d()` (deprecated in 0.23, removed in 0.25) are bit-identical, both using
Lee (1994) 3D medial surface thinning. Only dtype differs (bool vs uint8), with zero
effect on clDice computation.

### 6. Multi-Metric Checkpoints Reveal Epoch Divergence

Different validation metrics peak at different epochs: Dice saturates early (bulk
geometry), clDice improves later (thin branches), MASD peaks at boundary refinement.
Saving best checkpoints per metric enables downstream ensemble experiments comparing
topology-optimized vs overlap-optimized models.

### 7. SWAG Requires Training-Time Collection -- Post-Hoc Averaging Is Not SWA

Critical metalearning: the codebase had mislabeled checkpoint averaging as "SWA"
(Izmailov et al. 2018). Real SWA requires cyclic LR + AveragedModel + `update_bn()`.
SWAG (Maddox et al. 2019) additionally requires second-moment collection during training.
This triggered a renaming effort across ~18 files.

### 8. Calibration Metrics Are a Publication Gate

Comprehensive calibration metric survey identified ECE, MCE, pECE (patch-level), ACE,
CECE, BA-ECE (boundary-aware), RMSCE, TACE, and SDC as relevant. The `with_aux_calib`
factorial factor requires these metrics in the Analysis Flow. torchmetrics, netcal,
and specialized repos (SDC-Loss, Average-Calibration-Losses) provide implementations.

---

## Architectural Decisions Made

| Decision | Winner | Evidence Doc | KG Node |
|----------|--------|-------------|---------|
| Default loss function | `cbdice_cldice` | compound-loss-implementation-plan.md | `training.loss_function` |
| Primary metrics | MetricsReloaded (clDice + MASD + DSC) | metrics-reporting-doc.md | `training.primary_metrics` |
| Topology metrics library | gudhi + skimage + MONAI | GRAPH-TOPOLOGY-METRICS-INDEX.md | `training.topology_metrics` |
| HPO engine | Optuna multi-objective | (evaluation theme) | `training.hpo_engine` |
| Augmentation | TorchIO + MONAI combined | (infrastructure) | `training.augmentation_library` |
| Calibration method | Temperature scaling + netcal (partial) | calibration-and-swag-implementation-plan.xml | `training.calibration_method` |
| Champion selection | Rank-then-aggregate (not weighted avg) | compound-loss-double-check.md | N/A |
| Boundary loss approach | MONAI HausdorffDTLoss (not Kervadec) | boundary-losses.xml | N/A |
| Compound loss pattern | Weighted sum of nn.Module losses | compound-loss-implementation-plan.md | N/A |
| SWAG vs checkpoint avg | Renamed SWA to checkpoint averaging | calibration-and-swag-implementation-plan.xml | N/A |

---

## Implementation Status

| Document | Type | Status | Key Impl Files |
|----------|------|--------|----------------|
| GRAPH-TOPOLOGY-METRICS-INDEX.md | reference | Implemented | `pipeline/topology_metrics.py` (802 lines, 10 functions) |
| boundary-loss-implementation-plan.md | plan | Superseded | Superseded by boundary-losses.xml |
| boundary-losses.xml | execution_plan | Implemented | `pipeline/loss_functions.py` (HausdorffDTLoss registered) |
| calibration-and-swag-implementation-plan.xml | execution_plan | Partial | `ensemble/swag.py` (201 lines), `ensemble/calibration.py` (80 lines) |
| calibration-shift-plan.md | plan | Implemented | `ensemble/calibration_shift.py` (238 lines) |
| clDice-implementation-double-check-report.md | reference | Verified | Confirmed algorithm identity |
| compound-loss-double-check.md | reference | Implemented | `pipeline/validation_metrics.py` (153 lines) |
| compound-loss-implementation-plan.md | plan | Implemented | `pipeline/loss_functions.py` |
| debug-training-all-losses-plan.md | plan | Implemented | All 18 losses validated; 5 bugs found and fixed |
| graph-constrained-models-plan.md | plan | Implemented | 22 issues, `vendored_losses/` (8 files, 1247 lines) |
| graph-topology-metrics-code-examples.md | reference | Implemented | Code integrated into topology_metrics.py |
| graph-topology-metrics-quick-reference.yaml | reference | Implemented | Machine-readable lookup for 6 metrics |
| graph-topology-metrics-survey.md | reference | Implemented | Survey of 6 metrics, 3D support analysis |
| graph-topology-p0-tdd-plan.xml | execution_plan | Implemented | 10 P0 issues: NSD, HD95, ccDice, Betti, junction F1 |
| graph-topology-p1-tdd-plan.xml | execution_plan | Partial | 7 P1 issues: compound loss, Betti matching, vessel graph |
| loss-and-metrics-double-check-report.md | reference | Implemented | Mid-training review, 4-loss x 3-fold results |
| loss-metric-improvement-implementation.xml | execution_plan | Implemented | MLflow artifact completeness, serving wrapper |
| metrics-reporting-doc.md | reference | Implemented | Comprehensive metric catalog for manuscript |
| multi-metric-best-checkpoint-tracking-during-training-effect-for-ensembles.md | plan | Implemented | `MultiMetricTracker` saves 6 best checkpoints per run |
| multi-metric-downstream-double-check.md | reference | Implemented | Downstream consumer audit for Analysis/Deploy flows |
| novel-loss-debugging-plan.xml | execution_plan | Implemented | Novel loss debugging for exotic topology losses |
| segmentation-calibration-metrics-and-losses-sdc-pece-ace-mce-ece-etc.md | reference | Partial | ECE/MCE in ensemble/calibration.py; pECE, BA-ECE not yet |
| topology-approaches-discussion-notes.md | reference | Implemented | Discussion notes on topology approaches |
| topology-aware-segmentation-executable-plan.xml | execution_plan | Implemented | 3-phase topology-aware segmentation implementation |
| topology-aware-segmentation-literature-research-report.md | reference | Implemented | 80+ paper survey on graph-level topology |
| topology-loss-plan.md | plan | Implemented | Topology loss integration plan |

---

## Cross-References

- **KG Domain**: `knowledge-graph/domains/training.yaml` -- 7 decision nodes, 2 metalearning entries
- **Evaluation Theme**: Factorial design, biostatistics, and ensemble strategies consume training outputs
- **Models Theme**: Model adapter pattern determines which losses are compatible
- **Observability Theme**: MLflow metric key conventions (`val/dice`, `val/cldice`) defined here
- **Manuscript Theme**: Results section R3 blocked on GPU benchmark runs using these losses
- **Key Source Files**:
  - `src/minivess/pipeline/loss_functions.py` (740 lines) -- loss factory with 18+ losses
  - `src/minivess/pipeline/topology_metrics.py` (802 lines) -- 10 topology metric functions
  - `src/minivess/pipeline/vendored_losses/` (8 files, 1247 lines) -- topology-aware losses
  - `src/minivess/ensemble/calibration.py` + `calibration_shift.py` (318 lines) -- calibration
  - `src/minivess/ensemble/swag.py` (201 lines) -- SWAG post-training

---

## Constituent Documents

1. `GRAPH-TOPOLOGY-METRICS-INDEX.md` -- Research index for 3 graph-based topology metrics (APLS, Skeleton Recall, BDR)
2. `boundary-loss-implementation-plan.md` -- Boundary Loss + GSL implementation plan (Issue #100, #101)
3. `boundary-losses.xml` -- Converged plan: only MONAI HausdorffDTLoss, Kervadec/GSL NOT PLANNED
4. `calibration-and-swag-implementation-plan.xml` -- SWAG post-training + comprehensive calibration metrics (P0 publication gate)
5. `calibration-shift-plan.md` -- Calibration-under-shift framework (Issue #19)
6. `clDice-implementation-double-check-report.md` -- Verified skeletonize() == skeletonize_3d() (Rule #30 compliance)
7. `compound-loss-double-check.md` -- Range collapse analysis of val_compound_masd_cldice metric
8. `compound-loss-implementation-plan.md` -- Compound loss design + MetricsReloaded research (4 user prompts)
9. `debug-training-all-losses-plan.md` -- Pre-PR validation of all 18 losses (5 bugs found)
10. `graph-constrained-models-plan.md` -- 22-issue execution plan for graph-level topology (Issues #112-#136)
11. `graph-topology-metrics-code-examples.md` -- Working code samples for MONAI clDice, ccDice, BDR
12. `graph-topology-metrics-quick-reference.yaml` -- Machine-readable YAML database of 6 metrics
13. `graph-topology-metrics-survey.md` -- 491-line comprehensive survey of 6 topology metrics
14. `graph-topology-p0-tdd-plan.xml` -- TDD plan for 10 P0 issues: NSD, HD95, ccDice, centreline, Betti
15. `graph-topology-p1-tdd-plan.xml` -- TDD plan for 7 P1 issues: compound loss, vessel graph, TFFM
16. `loss-and-metrics-double-check-report.md` -- Mid-training review of dynunet_loss_variation_v2 experiment
17. `loss-metric-improvement-implementation.xml` -- MLflow artifact completeness + serving wrapper
18. `metrics-reporting-doc.md` -- Complete metric catalog for manuscript (Taha-Hanbury + Decroocq 2025)
19. `multi-metric-best-checkpoint-tracking-during-training-effect-for-ensembles.md` -- Multi-metric checkpoint strategy for ensemble experiments
20. `multi-metric-downstream-double-check.md` -- Audit of downstream consumers (Analysis, Deploy, ensemble flows)
21. `novel-loss-debugging-plan.xml` -- Debugging plan for exotic topology losses
22. `segmentation-calibration-metrics-and-losses-sdc-pece-ace-mce-ece-etc.md` -- Comprehensive calibration metric survey (ECE, pECE, BA-ECE, TACE, SDC)
23. `topology-approaches-discussion-notes.md` -- Discussion notes on topology-aware approaches
24. `topology-aware-segmentation-executable-plan.xml` -- 3-phase executable plan for topology-aware segmentation
25. `topology-aware-segmentation-literature-research-report.md` -- 80+ paper survey on graph-constrained segmentation
26. `topology-loss-plan.md` -- Topology loss integration plan
