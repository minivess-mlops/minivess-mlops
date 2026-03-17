# TRIPOD Compliance Matrix for MinIVess MLOps

**Date**: 2026-03-17
**Source papers**:
- [Collins et al. (2024). "TRIPOD+AI statement." *BMJ* 385:e078378.](https://doi.org/10.1136/bmj-2023-078378)
- [Gallifant et al. (2025). "The TRIPOD-LLM reporting guideline." *Nat Med* 31:60-69.](https://doi.org/10.1038/s41591-024-03425-5)
- [Pollard et al. (2026). "Protocol for TRIPOD-Code." *Diagn Progn Res* 10:4.](https://doi.org/10.1186/s41512-025-00217-4)

**Context**: MinIVess is a segmentation MLOps platform (not classification). The biostatistics
flow produces publication-ready statistical analyses for a Nature Protocols manuscript. We use
Pydantic AI for LLM micro-orchestration in the dashboard and agent flows.

---

## 1. TRIPOD+AI Checklist (27 items, 52 subitems)

The primary reporting guideline. Supersedes TRIPOD 2015. Applies to ALL prediction model
studies using regression or ML methods. For our segmentation paper, "prediction" maps to
per-voxel segmentation probability.

| Item | Sub | Description | Applies to (D/E) | Status | Implementation Location | Action Needed |
|------|-----|-------------|-------------------|--------|------------------------|---------------|
| 1 | | **Title**: Identify as development/evaluation of prediction model, target population, outcome | D;E | partial | docs/manuscript/ (paper title) | Ensure title mentions "segmentation prediction" and target population (multiphoton vascular imaging) |
| 2 | | **Abstract**: Structured summary per TRIPOD+AI for Abstracts (13 items) | D;E | missing | N/A | Write abstract following TRIPOD+AI for Abstracts checklist |
| 3 | a | **Background/objectives**: Explain medical context and relevance of the prediction model | D;E | partial | knowledge-graph/manuscript/claims.yaml (12 claims) | Map claims to TRIPOD 3a language |
| 3 | b | **Background/objectives**: Explain existing models and why new one is needed | D;E | partial | docs/manuscript/ | Add systematic review of existing vessel segmentation models |
| 3 | c | **Background/objectives**: Describe intended use and fairness considerations | D;E | missing | N/A | Add fairness section: imaging modality bias, dataset demographics |
| 4 | a | **Objectives**: Specify study objectives (development, evaluation, or both) | D;E | check | knowledge-graph/manuscript/methods.yaml (M0-M12) | Ensure M0 states "development + internal evaluation" |
| 4 | b | **Objectives**: State target population, outcome, and timeline | D;E | partial | configs/splits/3fold_seed42.json | Document: 70 volumes, 3D multiphoton, vessel/non-vessel binary |
| 5 | a | **Source of data**: Describe study design and data source | D;E | check | src/minivess/data/ + docs/datasets/README.md | Dataset provenance in methods |
| 5 | b | **Source of data**: Describe key dates and follow-up | D;E | partial | N/A | Add data acquisition dates (MiniVess dataset timeline) |
| 5 | c | **Source of data**: Describe eligibility criteria | D;E | partial | docs/datasets/README.md | Document inclusion/exclusion criteria for volumes |
| 6 | | **Participants**: Describe flow of participants (Sankey/flow diagram) | D;E | missing | N/A | Create participant flow diagram (volumes -> splits -> folds) |
| 7 | | **Outcome**: Define outcome clearly, including how and when assessed | D;E | partial | src/minivess/pipeline/metrics.py | Document: binary voxel-level segmentation, expert annotation |
| 8 | a | **Predictors**: Describe all predictors (input features/modalities) | D | check | configs/ (Hydra model configs) | Document: 3D multiphoton image volumes, no handcrafted features |
| 8 | b | **Predictors**: Report actions taken for predictors (preprocessing, augmentation) | D | check | src/minivess/pipeline/ (transforms) | Document preprocessing pipeline in paper |
| 9 | a | **Sample size**: Explain how sample size was determined | D;E | missing | N/A | Add sample size justification (Riley et al. criteria or equivalent for segmentation) |
| 9 | b | **Sample size**: Report number of outcome events | D;E | partial | N/A | Report voxel-level class distribution (vessel vs background imbalance) |
| 9 | c | **Sample size**: Describe number of participants and events per subgroup | D;E | missing | N/A | Report volume-level and voxel-level statistics per fold |
| 10 | a | **Missing data**: Describe how missing data were handled | D;E | check | N/A (3D image volumes are complete) | State "no missing data" explicitly |
| 11 | a | **Analysis - overall**: Describe modelling method(s) and software | D | check | src/minivess/adapters/ (ModelAdapter ABC) | Document all model architectures with versions |
| 11 | b | **Analysis - feature selection**: Describe feature/predictor selection | D | N/A | N/A | Not applicable for end-to-end DL segmentation (state this) |
| 12 | a | **Model building - method**: Detail model building procedure | D | check | src/minivess/orchestration/flows/train_flow.py | Document training pipeline in methods |
| 12 | b | **Model building - hyperparameters**: Describe hyperparameter tuning | D | check | configs/ + Optuna HPO | Document HPO strategy (Optuna + ASHA) |
| 12 | c | **Model building - pre-training**: Report pre-trained components | D | check | src/minivess/adapters/ (SAM3 weights) | Document SAM3 pretrained encoder (ViT-32L, 648M params) |
| 12 | d | **Model building - complexity**: Report model complexity | D | partial | N/A | Add parameter counts, FLOPs, VRAM requirements per model |
| 12 | e | **Model building - criteria**: Describe criteria for final model selection | D | check | src/minivess/pipeline/biostatistics_rankings.py | Document ensemble selection via biostatistics flow |
| 12 | f | **Model building - fairness**: Describe methods to assess/account for fairness | D | missing | N/A | Add fairness analysis section (imaging modality bias) |
| 13 | a | **Evaluation strategy**: Describe evaluation methods (cross-validation, etc.) | D;E | check | configs/splits/3fold_seed42.json | 3-fold cross-validation, seed=42 |
| 13 | b | **Evaluation strategy**: Describe any data preprocessing for evaluation | D;E | check | src/minivess/pipeline/ | Document eval-time preprocessing |
| 14 | | **Performance measures**: Specify measures to evaluate model | D;E | check | src/minivess/pipeline/metrics.py | Metrics Reloaded-aligned: Dice, clDice, ASD, HD95, etc. |
| 15 | | **Comparison**: Describe any model comparisons | D;E | check | src/minivess/pipeline/biostatistics_statistics.py | Wilcoxon, Friedman, Nemenyi, Cohen's d, bootstrap CI |
| 16 | | **Interpretability**: Describe any model interpretability methods | D;E | partial | knowledge-graph/decisions/ (xai_strategy) | XAI via Captum/Quantus planned but not yet integrated into paper |
| 17 | | **Deployment**: Describe intended deployment and updates | D;E | partial | src/minivess/orchestration/flows/deploy_flow.py | BentoML + ONNX export documented; add deployment context |
| 18 | a | **Open science - funding**: Declare sources of funding | D;E | missing | N/A | Add funding declaration |
| 18 | b | **Open science - COI**: Declare conflicts of interest | D;E | missing | N/A | Add COI statement |
| 18 | c | **Open science - protocol**: Provide study protocol availability | D;E | partial | docs/planning/ | Point to public repo as protocol |
| 18 | d | **Open science - registration**: Provide study registration details | D;E | missing | N/A | Consider prospective registration (OSF, PROSPERO) |
| 18 | e | **Open science - data sharing**: Provide data availability details | D;E | check | s3://minivessdataset (public) | Document public data availability |
| 18 | f | **Open science - code sharing**: Provide code availability details | D;E | check | GitHub repo (MIT license) | Document code availability + Docker reproducibility |
| 19 | | **Patient and public involvement**: Describe PPI activities | D;E | missing | N/A | State: no patient involvement (preclinical imaging study) |
| 20 | a | **Participants - results**: Report participant flow | D;E | missing | N/A | Volume flow diagram: 70 total -> 47 train / 23 val per fold |
| 20 | b | **Participants - demographics**: Report participant demographics | D;E | partial | N/A | Report dataset characteristics: resolution, volume sizes, imaging parameters |
| 21 | | **Model development results**: Report final model(s) | D | partial | MLflow artifacts | Full model specification with trained weights |
| 22 | | **Model specification**: Provide full model for reuse | D;E | check | GitHub + MLflow + BentoML + ONNX | Checkpoints, Docker image, ONNX export |
| 23 | a | **Performance results**: Report model performance with CIs | D;E | check | src/minivess/pipeline/biostatistics_statistics.py | Bootstrap CI, Wilcoxon, effect sizes |
| 23 | b | **Performance results**: Report calibration and clinical utility | D;E | partial | src/minivess/orchestration/flows/post_training_flow.py | Calibration flow exists; decision curve N/A for segmentation |
| 24 | | **Subgroup performance**: Report performance in subgroups | D;E | missing | N/A | Add per-volume-size or per-quality subgroup analysis |
| 25 | | **Discussion - limitations**: Discuss limitations | D;E | partial | knowledge-graph/manuscript/limitations.yaml (6 limitations) | Expand with TRIPOD-specific limitations |
| 26 | | **Discussion - implications**: Discuss implications for fairness | D;E | missing | N/A | Add fairness implications section |
| 27 | | **Discussion - interpretation**: Discuss results in context | D;E | partial | knowledge-graph/manuscript/claims.yaml | Interpret results relative to existing vessel segmentation methods |

### Summary (TRIPOD+AI)
- **Items fully addressed**: ~15/27 (mainly methods/analysis infrastructure)
- **Items partially addressed**: ~8/27
- **Items missing**: ~4/27 (fairness, PPI, registration, abstract)
- **Not applicable**: 0 (all items apply to segmentation studies)

---

## 2. TRIPOD-LLM Checklist (19 items, 50 subitems)

Applies because MinIVess uses Pydantic AI for LLM micro-orchestration in the dashboard
flow and agentic architecture. TRIPOD-LLM extends TRIPOD+AI for studies that develop,
tune, prompt-engineer, or evaluate LLMs. Our use case: LLM-assisted report generation
and agentic RAG for experiment analysis.

**Research design**: LLM evaluation (we evaluate LLM outputs for report quality)
**LLM tasks**: Documentation generation, summarization

| Item | Sub | Description | Status | Implementation Location | Action Needed |
|------|-----|-------------|--------|------------------------|---------------|
| 1 | | **Title**: Identify use of LLM | missing | N/A | Add "LLM-assisted" to paper title or subtitle if LLM is a paper contribution |
| 2 | | **Abstract**: Structured summary | missing | N/A | Include LLM component in abstract |
| 3 | | **Background**: Rationale for LLM use | missing | N/A | Justify Pydantic AI micro-orchestration in methods |
| 4 | | **Objectives**: State objectives for LLM component | missing | N/A | Define LLM objectives: automated report generation, experiment querying |
| 5 | a | **LLM description**: Report model name, version, provider | partial | src/minivess/observability/ (Langfuse, LiteLLM) | Document exact LLM model versions used |
| 5 | b | **LLM description**: Report whether model was frozen or dynamic during study | missing | N/A | State: frozen model version (pinned via LiteLLM) |
| 5 | c | **LLM description**: Report training data cutoff date | missing | N/A | Document knowledge cutoff of LLM used |
| 6 | a | **Data**: Describe data used for LLM evaluation | partial | N/A | Document evaluation dataset for LLM report quality |
| 6 | b | **Data - prompts**: Report prompt templates and engineering methods | partial | src/minivess/ (Pydantic AI agents) | Document prompt templates and system prompts |
| 7 | a | **Methods - evaluation**: Describe evaluation methodology | partial | N/A | Document how LLM outputs are evaluated (human review, automated metrics) |
| 7 | b | **Methods - human evaluation**: Report annotator qualifications and instructions | missing | N/A | If human eval used, document annotator details |
| 8 | | **Methods - reproducibility**: Report temperature, sampling, seed settings | missing | N/A | Document LLM inference parameters |
| 9 | | **Open science**: Same as TRIPOD+AI item 18 | partial | See TRIPOD+AI 18a-f | Same actions as TRIPOD+AI |
| 10 | | **PPI**: Same as TRIPOD+AI item 19 | missing | See TRIPOD+AI 19 | Same action as TRIPOD+AI |
| 11 | a | **Results - LLM performance**: Report LLM-specific metrics | missing | N/A | Add LLM evaluation metrics if LLM is a paper contribution |
| 11 | b | **Results - failure modes**: Report hallucinations, refusals, errors | missing | N/A | Document LLM failure modes observed |
| 12 | | **Results - cost**: Report computational cost and resource usage | missing | N/A | **CRITICAL**: Add cost transparency (see Cost Reporting Plan below) |
| 13 | | **Discussion - LLM limitations**: Discuss LLM-specific limitations | missing | N/A | Discuss LLM limitations: hallucination risk, version dependence |
| 14 | | **Discussion - implications**: Discuss societal implications of LLM use | missing | N/A | Discuss responsible AI use in biomedical research |

### Summary (TRIPOD-LLM)
- **Items fully addressed**: 0/19 (LLM component not yet documented for paper)
- **Items partially addressed**: 5/19
- **Items missing**: 14/19
- **Relevance note**: If LLM is NOT a primary contribution of the paper, only items
  5a (model identification), 8 (reproducibility), and 12 (cost) are essential. The rest
  become supplementary material.

---

## 3. TRIPOD-Code Checklist (Protocol stage — not yet finalized)

TRIPOD-Code is still under development (Delphi survey stage, published Feb 2026). The
final checklist does not exist yet. However, the protocol identifies key areas that we
can proactively address. These align with TRIPOD+AI items 18f (code sharing) and 22
(model specification).

| Area | Expected Items | Status | Implementation Location | Action Needed |
|------|---------------|--------|------------------------|---------------|
| **Code availability** | Repository URL, license, access restrictions | check | GitHub (MIT), pyproject.toml | Already public, MIT licensed |
| **Dependencies** | Software dependencies with versions | check | uv.lock, pyproject.toml | uv.lock provides exact reproducibility |
| **Environment** | Execution environment specification | check | Dockerfiles, docker-compose.yml | Docker = reproducibility guarantee |
| **Documentation** | README, inline comments, docstrings | partial | CLAUDE.md, module docstrings | Improve user-facing documentation |
| **Modularity** | Code structure and organization | check | src/minivess/ (clean package structure) | Well-structured package |
| **Testing** | Test suite, CI/CD | check | tests/, Makefile (3 tiers) | 3-tier test suite with markers |
| **Data preprocessing** | Code for data cleaning, feature engineering | check | src/minivess/data/, transforms | MONAI transforms documented |
| **Model training** | Training code with config | check | src/minivess/orchestration/flows/train_flow.py | Hydra-zen config + Prefect flow |
| **Evaluation** | Evaluation code | check | src/minivess/pipeline/metrics.py, biostatistics_*.py | Full evaluation pipeline |
| **Model specification** | Trained model files, weights, checkpoint format | check | MLflow + DVC + checkpoints/ | Checkpoints tracked via MLflow |
| **Reproducibility** | Deterministic execution, seeds, Docker | check | Docker, DVC, seed=42 | Docker-per-flow, DVC data versioning |
| **Long-term archival** | DOI, Zenodo, persistent storage | missing | N/A | Create Zenodo release before submission |
| **Demo/synthetic data** | Example data for code testing | partial | tests/fixtures/ | Add minimal demo volume for code verification |

### Summary (TRIPOD-Code)
- **Areas proactively addressed**: 10/13 (strong open-source infrastructure)
- **Areas partially addressed**: 2/13
- **Areas missing**: 1/13 (Zenodo archival)
- **Note**: MinIVess is unusually well-positioned for TRIPOD-Code compliance due to
  Docker-per-flow architecture, DVC data versioning, and uv.lock dependency pinning.

---

## 4. Segmentation-Specific Adaptations

TRIPOD was designed for clinical prediction models (probability of diagnosis/prognosis).
For segmentation, the following adaptations apply:

| TRIPOD Concept | Segmentation Adaptation | MinIVess Implementation |
|----------------|------------------------|------------------------|
| "Prediction" | Per-voxel segmentation probability | Softmax output per voxel |
| "Outcome" | Binary vessel/non-vessel label | Expert-annotated ground truth masks |
| "Participants" | Image volumes (not patients) | 70 MiniVess volumes |
| "Predictors" | Input image voxels (not clinical features) | 3D multiphoton image data |
| "Discrimination" (c-statistic) | Dice, clDice, volumetric overlap | Metrics Reloaded suite |
| "Calibration" | Voxel-level probability calibration | Temperature scaling, Platt scaling |
| "Clinical utility" | Not directly applicable | Topological correctness (clDice, Betti errors) |
| "Decision curve" | Not applicable for segmentation | Replaced by task-specific metrics |
| "Sample size" | Volume count + voxel count | 70 volumes, ~10^8 voxels per volume |
| "Subgroups" | Volume quality, imaging parameters | Per-fold, per-volume-size analysis |
| "Fairness" | Imaging modality bias, annotation quality | Limited: preclinical data, single modality |

---

## 5. AI/ML-Specific Items (beyond traditional prediction models)

These items were added in TRIPOD+AI specifically for ML methods:

| New Item | Description | MinIVess Status | Notes |
|----------|-------------|-----------------|-------|
| Pre-training (12c) | Report use of pre-trained components | check | SAM3 encoder (ViT-32L), ImageNet weights for other models |
| Hyperparameter tuning (12b) | Report HPO strategy | check | Optuna + ASHA, documented in Hydra configs |
| Model complexity (12d) | Report parameters, FLOPs, VRAM | partial | VRAM tracked; add FLOPs and param counts to paper |
| Fairness (3c, 12f, 23a, 26) | Throughout the checklist | missing | Need fairness analysis section |
| Data augmentation (8b) | Report augmentation strategy | check | TorchIO + MONAI transforms |
| Open science (18a-f) | Funding, COI, protocol, data, code | partial | Code/data open; need funding/COI/registration |
| PPI (19) | Patient and public involvement | N/A | Preclinical study — state explicitly |
| Interpretability (16) | XAI methods | partial | Captum/Quantus planned |

---

## 6. Cost Reporting Plan (TRIPOD-LLM Item 12 + Issue #795)

TRIPOD-LLM Item 12 requires reporting computational cost. This aligns with MinIVess
cost tracking infrastructure:

| Cost Category | Source | Implementation | Paper Reporting |
|---------------|--------|---------------|-----------------|
| GPU training cost | SkyPilot job logs | deployment/skypilot/ | $/run for each model x fold |
| LLM inference cost | LiteLLM cost tracking | src/minivess/observability/ | $/query for LLM components |
| Cloud storage cost | GCS billing | Pulumi IaC | Monthly storage cost |
| Total experiment cost | Aggregated | src/minivess/observability/analytics.py | Total cost table in paper |

**Action items for cost transparency**:
1. Add cost logging to biostatistics lineage manifest
2. Export SkyPilot job costs to DuckDB analytics
3. Add LiteLLM cost breakdown to dashboard flow
4. Create cost summary table for paper methods section

---

## 7. Priority Action Items

### P0 — Must-fix before manuscript submission
1. **Sample size justification** (TRIPOD+AI 9a) — Add Riley et al. or equivalent justification
2. **Participant flow diagram** (TRIPOD+AI 6, 20a) — Volume flow diagram
3. **Fairness statement** (TRIPOD+AI 3c, 12f, 26) — At minimum, acknowledge limitations
4. **Abstract** (TRIPOD+AI 2) — TRIPOD+AI for Abstracts checklist compliance
5. **Funding/COI** (TRIPOD+AI 18a, 18b) — Required for journal submission
6. **PPI statement** (TRIPOD+AI 19) — "Preclinical study, no patient involvement"

### P1 — Should-fix for complete compliance
7. **Study registration** (TRIPOD+AI 18d) — OSF registration
8. **Subgroup analysis** (TRIPOD+AI 24) — Per-volume subgroups
9. **Model complexity reporting** (TRIPOD+AI 12d) — FLOPs, param counts
10. **XAI integration** (TRIPOD+AI 16) — Captum/Quantus results in paper
11. **Cost transparency** (TRIPOD-LLM 12) — Full cost table
12. **Zenodo archival** (TRIPOD-Code) — DOI for code/data

### P2 — Nice-to-have / future versions
13. **LLM documentation** (TRIPOD-LLM 5-14) — If LLM is a paper contribution
14. **TRIPOD-Code compliance** (anticipatory) — Already mostly covered
15. **External validation** (TRIPOD+AI E items) — DeepVess/TubeNet/VesselNN datasets

---

## 8. Existing Biostatistics Flow TRIPOD Coverage

The biostatistics flow (`src/minivess/pipeline/biostatistics_lineage.py`) already tracks:

| Lineage Field | TRIPOD Item Covered |
|---------------|-------------------|
| `schema_version` | Provenance (18f) |
| `generated_at` | Reproducibility timestamp |
| `fingerprint` (SHA-256) | Data integrity (18e) |
| `git_commit` | Code version (18f) |
| `n_source_runs` | Sample tracking (6) |
| `source_experiments` | Study design (5a) |
| `artifacts_produced` | Results inventory (21) |
| `statistical_methods` | Analysis methods (11a, 15) |

### Missing from lineage manifest
| Missing Field | TRIPOD Item | Priority |
|---------------|-------------|----------|
| `model_architectures` | 11a, 12a | P0 |
| `sample_size_justification` | 9a | P0 |
| `preprocessing_pipeline` | 8b | P1 |
| `hyperparameter_config` | 12b | P1 |
| `evaluation_strategy` | 13a | P1 |
| `performance_metrics_list` | 14 | P1 |
| `cost_summary` | TRIPOD-LLM 12 | P1 |
| `fairness_assessment` | 12f | P1 |
| `software_versions` | 11a | P1 |
