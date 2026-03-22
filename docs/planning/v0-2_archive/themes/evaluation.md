---
theme: evaluation
doc_count: 55
last_synthesized: "2026-03-22"
implementation_health: partial
kg_domains: [training, models, architecture]
---

# Theme: Evaluation — Biostatistics, Factorial Design, Ensembles, UQ, Post-Training

This theme encompasses the scientific evaluation pipeline of MinIVess MLOps: factorial
experiment design, biostatistics analysis, ensemble strategies, uncertainty quantification,
post-training methods, and the 4-pass debug factorial experiment that validated the full
pipeline end-to-end.

---

## Key Scientific Insights

### 1. The Full Factorial Is 480+ Conditions, Not 24

The most persistent misunderstanding across 8+ sessions was conflating the 24-cell training
layer (4 models x 3 losses x 2 aux_calib) with the full factorial. The complete design is
layered across compute tiers:

- **Layer A (Training, GPU):** 4 models x 3 losses x 2 aux_calib = 24 conditions per fold
- **Layer B (Post-Training, CPU):** x 2 post-training methods (none, swag)
- **Layer C (Analysis, CPU):** x 5 ensemble strategies (per_loss_single_best, all_loss_single_best, per_loss_all_best, all_loss_all_best, cv_average)

Only Layer A requires cloud GPU compute (~$65). Layers B and C run locally on CPU for free.
The factorial YAML is composable via Hydra config groups, and the Biostatistics Flow
auto-derives ALL factors from whichever factorial YAML was used -- zero hardcoded factor names.

**Source:** `intermedia-plan-synthesis-pre-debug-run.md`, `biostatistic-flow-debug-double-check.xml`

### 2. DSC vs clDice Rank Inversion Confirms Topology-Aware Loss Value

The 2nd pass local debug (DynUNet, 100 epochs, 24 val volumes) produced the key finding
that justifies the factorial design: `dice_ce` wins on Dice (0.840) but LOSES on clDice
(0.827), while `cbdice_cldice` wins on clDice (0.920) but loses on Dice (0.790). This
rank inversion demonstrates that loss function choice determines WHICH aspect of
segmentation quality is optimized -- precisely what the factorial ANOVA must quantify.

**Source:** `run-debug-factorial-experiment-report-2nd-pass-local.md`

### 3. SWAG Improves Calibration by 12-19% With 2 Extra Epochs

The 3rd pass local debug validated real SWAG (Maddox et al. 2019) -- resumed training
with SWALR + low-rank posterior approximation. Results on DynUNet:
- ECE: -12.1% (0.0506 -> 0.0445)
- MCE: -19.0% (0.3859 -> 0.3126)
- Brier: -9.6% (0.0311 -> 0.0281)

This confirmed that the post-training factor is scientifically justified. The critical
implementation note: our earlier "SWA" was mislabeled checkpoint averaging, not real SWA
(Izmailov et al. 2018). Real SWA requires cyclic LR + `torch.optim.swa_utils.AveragedModel`
+ `update_bn()`.

**Source:** `run-debug-factorial-experiment-report-3rd-pass-local.md`, `swa-swag-multi-swa-swag-pswa-lawa-model-soup-checkpoint-averaging-report.md`

### 4. Biostatistics: BCa Bootstrap on Per-Volume Metrics Is Primary Inference

The statistical methodology crystallized across multiple reports:
- **Primary inference:** BCa bootstrap on per-volume paired differences (N=70 volumes)
- **NOT fold-level:** Friedman+Nemenyi is severely underpowered for K=3 folds (minimum p=0.125)
- **Multiple comparisons:** Holm-Bonferroni for primary metric (clDice), BH-FDR for secondary
- **Bayesian supplement:** baycomp signed-rank test with ROPE for equivalence testing
- **Effect sizes:** eta-squared (partial) and omega-squared (bias-corrected) from 2-way ANOVA

The Nadeau-Bengio correction (via correctipy) is required for any parametric test on
fold-level aggregates. Grandvalet-Bengio (2004) proved this variance cannot be unbiasedly
estimated.

**Source:** `biostatistics-prefect-flow-plan.xml`, `preregistration-statistical-methods-report.md`

### 5. Preregistration: Hofman Two-Phase Template for ML Benchmarks

The preregistration report adapted Hofman et al. (2023) for ML experiments:
- **Phase A (before training):** research question, outcome variables, data construction, baseline comparisons, analysis plan
- **Phase B (before testing):** final model specifications, test set integrity confirmation, exploratory analysis labeling

This is standard practice for combating ML-specific p-hacking (adjusting metrics/splits
post-hoc), selective reporting, and HARKing. Platform for registration: OSF with DOI.

**Source:** `preregistration-statistical-methods-report.md`

### 6. External Test Evaluation Is a Publication Blocker

MiniVess train/val metrics are NOT test metrics. The generalization gap is measured
between cross-validated train/val and external test (DeepVess). VesselNN is NOT a test
dataset -- reserved for drift detection simulation only (data leakage from same PI).
Test metric prefix: `test/deepvess/{metric}` (extensible for future datasets).

**Source:** `test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml`

### 7. RAM Crash From Unmocked Biostatistics Tasks

Running biostatistics flow tests consumed 66.8 GB RAM + 17.2 GB swap, crashing the
system. Root cause: 3 of 16 tasks were not mocked, and `MagicMock()` arithmetic
behavior with `range()` created infinite loops. Fix: added 3 missing `@patch`
decorators, real `BiostatisticsConfig`, 32 GB RLIMIT_AS safety guard.

**Source:** `ram-issue-mock-data-biostatistics-duckup-report.md`

---

## Architectural Decisions Made

| Decision | Outcome | Source Doc | KG Node |
|----------|---------|-----------|---------|
| Primary inference method | BCa bootstrap on per-volume N=70 | biostatistics-prefect-flow-plan.xml | training.primary_metrics |
| Primary loss | cbdice_cldice (CbDiceClDiceLoss) | experiment-planning-and-metrics-prompt.md | training.loss_function |
| Co-primary metrics | clDice + MASD (compound: 0.5*clDice+0.5*MASD) | multi-validation-metric-tracking-plan.md | training.primary_metrics |
| Post-training methods | {none, swag} (checkpoint averaging removed, NOT real SWA) | swa-swag report | training.calibration_method |
| Ensemble strategies | 5 strategies (per_loss/all_loss x single/all_best + cv_average) | evaluation-and-ensemble-execution-plan.xml | architecture |
| UQ methods | MC Dropout, Deep Ensembles, Conformal (MAPIE) | uq-beyond-temperature-plan.md | training.calibration_method |
| HPO engine | Optuna multi-objective with MedianPruner + ASHA | optuna-hpo-plan.md | training.hpo_engine |
| Multiple comparisons | Holm-Bonferroni (primary) + BH-FDR (secondary) | biostatistics-prefect-flow-plan.xml | -- |
| Calibration loss | hL1-ACE auxiliary (Barfoot et al. 2025) | pr-c-post-training-factorial-plan.xml | training.calibration_method |
| Test datasets | DeepVess only (TubeNet excluded, VesselNN = drift only) | test-sets-external-validation.xml | data.external_datasets |
| Factorial scope (debug) | 4x3x2x2x2x4 = 384 conditions, 1 fold, 2 epochs | run-debug-factorial-experiment.xml | models.paper_model_comparison |

---

## Implementation Status

| Document | Type | Status | Key Deliverable |
|----------|------|--------|-----------------|
| advanced-ensembling-bootstrapping-report.md | research_report | reference | Bootstrap vs training ensembles distinction |
| analysis-flow-debug-double-check.xml | execution_plan | partial | Analysis Flow 6-factor factorial integration |
| biostatistic-flow-debug-double-check.xml | execution_plan | partial | Biostatistics 6-factor pipeline verification |
| biostatistics-prefect-flow-plan-double-check.xml | execution_plan | partial | 9-task Prefect flow redesign (v2, no compat layer) |
| biostatistics-prefect-flow-plan.xml | execution_plan | implemented | Original 9-task biostatistics flow (PR #398 Phase 1) |
| biostatistics-prefect-flow.md | document | implemented | 1014-line design doc for biostatistics |
| cold-start-prompt-debug-factorial-run.md | cold_start | executed | 1st GCP factorial launch prompt |
| cold-start-prompt-factorial-gaps-completion.md | cold_start | executed | Post-2nd-pass gap closure prompt |
| conformal-uq-execution-plan.xml | execution_plan | partial | 5-phase conformal UQ (vectorize + MAPIE) |
| conformal-uq-segmentation-report.md | research_report | reference | Conformal prediction for 3D segmentation |
| debug-factorial-local-post-analysis-biostats.xml | execution_plan | executed | Local 3-flow integration test plan |
| evaluation-and-ensemble-execution-plan.xml | execution_plan | implemented | Flow 3 analysis: ensemble + UQ + evaluation |
| experiment-planning-and-metrics-prompt.md | plan | implemented | Original user prompt for DynUNet + metrics |
| factorial-design-demo-experiment-plan.md | plan | implemented | Initial factorial design |
| final-methods-quasi-e2e-testing-plan.xml | execution_plan | partial | 58-test quasi-E2E with capability discovery |
| final-methods-quasi-e2e-testing-prompt.md | prompt | reference | User prompt for combinatorial testing |
| final-methods-quasi-e2e-testing.md | document | partial | Schema, reduction, capabilities design |
| generative-uq-plan.md | plan | planned | Generative UQ methods |
| hpo-implementation-background-research-report.md | research_report | reference | Optuna + ASHA background research |
| lit-report-ensemble-uncertainty.md | research_report | reference | R2: 40-paper ensemble+UQ survey |
| lit-report-ensemble-uncertainty.xml | execution_plan | executed | Lit report generation plan |
| lit-report-post-training-methods.md | research_report | reference | R1: 31-paper post-training survey |
| lit-report-post-training-methods.xml | execution_plan | executed | Lit report generation plan |
| local-debug-3flow-execution-plan.xml | execution_plan | executed | Local DynUNet 3-flow pipeline test |
| mapie-conformal-plan.md | plan | partial | MAPIE voxel-level conformal integration |
| multi-validation-metric-tracking-plan.md | plan | implemented | 6 best checkpoints per metric per run |
| optuna-hpo-plan.md | plan | planned | Optuna study/trial integration |
| post-run-debug-factorial-experiment-report-2nd-pass-local-fixes.xml | execution_plan | executed | Post-2nd-pass fix plan |
| post-training-flow-debug-double-check.xml | execution_plan | partial | Post-Training Flow factorial integration |
| post-training-plugins-and-swa-planning.md | plan | implemented | Plugin protocol + SWA/Multi-SWA/calibration DAG |
| pr-a-biostatistics-gaps-plan.xml | execution_plan | partial | 2-way ANOVA, calibration metrics, DCA |
| pr-b-evals-analysis-flow-plan.xml | execution_plan | partial | CV-average pyfunc, factorial ensembles |
| pr-c-post-training-factorial-plan.xml | execution_plan | partial | hL1-ACE aux loss, SWA in factorial |
| pre-debug-factorial-fixes-needed-before-4th-pass.xml | execution_plan | active | 4th pass pre-launch fixes |
| pre-debug-factorial-local-post-analysis-biostats-final-qa-plan.xml | execution_plan | partial | Final QA before local debug |
| prefect-flow-evaluation-and-ensemble-planning.md | plan | implemented | Original user prompt for eval flow |
| preregistration-statistical-methods-report.md | research_report | reference | Hofman 2-phase preregistration |
| ram-issue-mock-data-biostatistics-duckup-report.md | research_report | fixed | RAM crash root cause + fix |
| run-debug-factorial-experiment-2nd-pass.md | document | executed | 2nd pass cold-start (14 failed conditions) |
| run-debug-factorial-experiment-4th-pass.xml | execution_plan | failed | 4th pass GCP launch (DVC pull + monitoring failures) |
| run-debug-factorial-experiment-report-2nd-pass-fix-plan.xml | execution_plan | executed | Fix plan for 14 failures |
| run-debug-factorial-experiment-report-2nd-pass-local.md | research_report | executed | Local 3-flow results (DSC vs clDice inversion) |
| run-debug-factorial-experiment-report-3rd-pass-local.md | research_report | executed | SWAG + calibration results |
| run-debug-factorial-experiment-report-3rd-pass-plan.xml | execution_plan | executed | 3rd pass plan |
| run-debug-factorial-experiment-report-4th-pass-failure.md | research_report | documented | 5 failure modes, $6.30 wasted |
| run-debug-factorial-experiment-report-tofix-for-4th-pass.md | research_report | active | 12 glitch status (10 fixed, 1 partial, 1 Docker) |
| run-debug-factorial-experiment-report.md | research_report | documented | 1st pass: 12 glitches, 12/26 succeeded |
| run-debug-factorial-experiment.xml | execution_plan | executed | Original 7-phase GCP experiment plan |
| self-reflection-low-confidence-regions.md | document | reference | Tier 1-2 decision uncertainty audit |
| skypilot-observability-for-factorial-monitor.md | document | planned | Batch monitoring upgrade for factorial |
| swa-swag-multi-swa-swag-pswa-lawa-model-soup-checkpoint-averaging-report.md | research_report | reference | Definitive weight averaging methods survey |
| swag-dataloader-implementation.xml | execution_plan | active | Wire calibration_data to SWAGPlugin |
| test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml | execution_plan | partial | DeepVess integration into Analysis Flow |
| uq-beyond-temperature-plan.md | plan | partial | MC Dropout, Deep Ensembles, Conformal |
| user-prompt-post-analysis-biostats-local-debug.md | prompt | reference | Verbatim user prompt for 3-flow local debug |

---

## Cross-References

- **Infrastructure theme:** Docker-per-flow isolation, Prefect orchestration
- **Cloud theme:** GCP L4 spot execution, SkyPilot job monitoring, cost analysis
- **Models theme:** 6-model lineup (DynUNet, MambaVesselNet++, SAM3 x3, VesselFM)
- **Harness theme:** Cold-start prompts, factorial-monitor skill, context compounding
- **KG domains:** `training.yaml` (loss_function, calibration_method, hpo_engine), `models.yaml` (paper_model_comparison)
- **Key metalearning:** `2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md`, `2026-03-20-hardcoded-significance-level-antipattern.md`, `2026-03-21-ram-crash-biostatistics-test.md`

---

## Constituent Documents

1. `advanced-ensembling-bootstrapping-report.md`
2. `analysis-flow-debug-double-check.xml`
3. `biostatistic-flow-debug-double-check.xml`
4. `biostatistics-prefect-flow-plan-double-check.xml`
5. `biostatistics-prefect-flow-plan.xml`
6. `biostatistics-prefect-flow.md`
7. `cold-start-prompt-debug-factorial-run.md`
8. `cold-start-prompt-factorial-gaps-completion.md`
9. `conformal-uq-execution-plan.xml`
10. `conformal-uq-segmentation-report.md`
11. `debug-factorial-local-post-analysis-biostats.xml`
12. `evaluation-and-ensemble-execution-plan.xml`
13. `experiment-planning-and-metrics-prompt.md`
14. `factorial-design-demo-experiment-plan.md`
15. `final-methods-quasi-e2e-testing-plan.xml`
16. `final-methods-quasi-e2e-testing-prompt.md`
17. `final-methods-quasi-e2e-testing.md`
18. `generative-uq-plan.md`
19. `hpo-implementation-background-research-report.md`
20. `lit-report-ensemble-uncertainty.md`
21. `lit-report-ensemble-uncertainty.xml`
22. `lit-report-post-training-methods.md`
23. `lit-report-post-training-methods.xml`
24. `local-debug-3flow-execution-plan.xml`
25. `mapie-conformal-plan.md`
26. `multi-validation-metric-tracking-plan.md`
27. `optuna-hpo-plan.md`
28. `post-run-debug-factorial-experiment-report-2nd-pass-local-fixes.xml`
29. `post-training-flow-debug-double-check.xml`
30. `post-training-plugins-and-swa-planning.md`
31. `pr-a-biostatistics-gaps-plan.xml`
32. `pr-b-evals-analysis-flow-plan.xml`
33. `pr-c-post-training-factorial-plan.xml`
34. `pre-debug-factorial-fixes-needed-before-4th-pass.xml`
35. `pre-debug-factorial-local-post-analysis-biostats-final-qa-plan.xml`
36. `prefect-flow-evaluation-and-ensemble-planning.md`
37. `preregistration-statistical-methods-report.md`
38. `ram-issue-mock-data-biostatistics-duckup-report.md`
39. `run-debug-factorial-experiment-2nd-pass.md`
40. `run-debug-factorial-experiment-4th-pass.xml`
41. `run-debug-factorial-experiment-report-2nd-pass-fix-plan.xml`
42. `run-debug-factorial-experiment-report-2nd-pass-local.md`
43. `run-debug-factorial-experiment-report-3rd-pass-local.md`
44. `run-debug-factorial-experiment-report-3rd-pass-plan.xml`
45. `run-debug-factorial-experiment-report-4th-pass-failure.md`
46. `run-debug-factorial-experiment-report-tofix-for-4th-pass.md`
47. `run-debug-factorial-experiment-report.md`
48. `run-debug-factorial-experiment.xml`
49. `self-reflection-low-confidence-regions.md`
50. `skypilot-observability-for-factorial-monitor.md`
51. `swa-swag-multi-swa-swag-pswa-lawa-model-soup-checkpoint-averaging-report.md`
52. `swag-dataloader-implementation.xml`
53. `test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml`
54. `uq-beyond-temperature-plan.md`
55. `user-prompt-post-analysis-biostats-local-debug.md`
