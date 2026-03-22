---
title: "Theme: Data — DVC, Datasets, Drift, Synthetic, DeepVess"
theme_id: data
doc_count: 18
archive_path: docs/planning/v0-2_archive/original_docs/
kg_domain: knowledge-graph/domains/data.yaml
created: "2026-03-22"
status: archived
phantom_flag: true
phantom_docs:
  - vesselnn-dataset-implementation.xml
  - topology-real-data-e2e-plan.xml
  - synthicl-plan.md
---

# Theme: Data

DVC data versioning, dataset strategy, drift detection, synthetic vascular data
generation, data quality frameworks, and external test datasets. This theme covers
the full data lifecycle from acquisition through validation, versioning, and
drift monitoring.

**PHANTOM FLAG**: Three documents in this theme describe features that were planned
but never fully implemented, creating a gap between documentation and reality.
See the phantom analysis in the Implementation Status table below.

---

## Key Scientific Insights

### 1. Three-Dataset Strategy with Strict Role Separation

The dataset strategy is resolved and non-negotiable:

- **MiniVess** (70 volumes, multiphoton mouse brain cortex) -- primary train/val, 3-fold seed=42
- **DeepVess** (7 volumes, Cornell, multiphoton mouse brain cortex) -- external TEST only
- **VesselNN** (12 volumes, same PI as MiniVess) -- drift detection simulation ONLY (data leakage risk)
- **TubeNet** -- EXCLUDED (only 1 two-photon volume, different organ, removed 2026-03-19)

Metric prefix convention: `test/deepvess/{metric}` (extensible to `test/{newdataset}/{metric}`).
VesselNN is NOT a test dataset -- it is reserved for synthetic drift simulation.

### 2. Drift Detection Is Not a Live System Problem

MinIVess is not a deployed system receiving new data from real experiments. Drift
detection is simulated using synthetic data generation and VesselNN as a known
distribution shift. The `prompt-574-synthetic-data-drift-detection.md` captures the
comprehensive research agenda: Evidently DataDriftPreset for tabular features, kernel
MMD for embedding drift, and DeepChecks Vision for image-level validation.

### 3. Evidently Is the Resolved Drift Monitoring Tool

KG decision `data.drift_monitoring` is resolved: Evidently DataDriftPreset + kernel
MMD. Implementation exists in `pipeline/drift_detection.py` and `pipeline/embedding_drift.py`
(referenced in KG but file location may have moved to `validation/drift.py`). The
drift monitoring plan covers data drift, concept drift, and model performance drift
with Grafana dashboards for visualization.

### 4. DATA-CARE Multi-Dimensional Quality Scoring

The `data-care-plan.md` maps the DATA-CARE quality framework (6 dimensions:
Completeness, Correctness, Consistency, Uniqueness, Timeliness, Representativeness)
to MinIVess's NIfTI metadata and training metrics DataFrames. Implementation exists
in `validation/data_care.py` (352 lines) with per-dimension scoring and GateResult
integration for pipeline enforcement.

### 5. 12-Layer Data Validation Onion

The resolved `data_validation_depth` decision specifies a layered validation approach:
Pydantic (runtime types) + Pandera (DataFrame schemas) + Great Expectations (batch
quality gates). The `pr2-data-quality-pipeline-plan.md` details wiring these into
`data_flow.py` with configurable severity levels. Implementation exists in
`validation/enforcement.py` (124 lines).

### 6. Synthetic Vascular Stack Generation Is Research-Grade

Three synthetic data documents explore generation approaches:

- `synthetic-vascular-stack-generators-plan.md` -- parametric L-system generators, 3D vessel tree simulation
- `synthicl-plan.md` -- SynthICL (Synthetic In-Context Learning) integration
- `prompt-574-synthetic-data-drift-detection.md` -- comprehensive research agenda

The key insight: "super-realistic" generation (latent diffusion, normalizing flows) is
unnecessary. Simple parametric generators suffice for drift detection simulation because
the goal is controlled distribution shift, not photorealistic realism. VesselNN serves
as a real-data fallback for drift simulation.

### 7. Zarr vs .pt for 5D UQ Arrays

The `zarr-vs-pt-for-5d-uq-array.md` evaluates storage formats for uncertainty
quantification arrays with shape (N_samples, B, C, D, H, W). Zarr provides chunked,
compressed, lazy-loading access suitable for large UQ ensembles; PyTorch .pt files
are simpler but load everything into memory. Decision likely favors Zarr for production
UQ workflows where ensemble size exceeds memory.

### 8. VessQC = Vessel-Specific Quality Control

The `vessqc-plan.md` defines vessel-specific quality control metrics beyond generic
image QC: vessel density per ROI, branching pattern consistency, diameter distribution
normality, and connectivity graph completeness. Implementation exists in
`validation/vessqc.py` (228 lines).

### 9. DVC Test Suite Hardening

The `dvc-test-suite-improvement.xml` documents test suite improvements for DVC
operations: verifying local cache consistency, remote push/pull roundtrips, and
data hash reproducibility. These tests are particularly important after the
GCS migration (`gs://minivess-mlops-dvc-data`).

---

## Architectural Decisions Made

| Decision | Winner | Evidence Doc | KG Node |
|----------|--------|-------------|---------|
| Data versioning | DVC (local MinIO, cloud GCS) | dataset-use-plan.md | `data.data_versioning` |
| Data validation depth | 12-layer onion (Pydantic+Pandera+GE) | pr2-data-quality-pipeline-plan.md | `data.data_validation_depth` |
| DataFrame validation | Pandera schemas | pr2-data-quality-pipeline-plan.md | `data.dataframe_validation` |
| Data profiling | whylogs (config_only) | N/A | `data.data_profiling` |
| Label quality tool | Cleanlab candidate (config_only) | N/A | `data.label_quality_tool` |
| Lineage tracking | OpenLineage/Marquez candidate (config_only) | openlineage-plan.md (operations) | `data.lineage_tracking` |
| Data quality orchestration | Prefect gate wiring | pr2-data-quality-pipeline-plan.md | `data.data_quality_orchestration` |
| Drift monitoring | Evidently DataDriftPreset + kernel MMD | drift-monitoring-implementation-plan.xml | `data.drift_monitoring` |
| Dataset strategy | MiniVess primary, DeepVess test, VesselNN drift | dataset-use-plan.md | `data.dataset_strategy` |
| Database backend | PostgreSQL only (SQLite BANNED) | N/A | `data.database_backend` |
| DVC remote strategy | MinIO local, GCS cloud | N/A | `data.dvc_remote_strategy` |

---

## Implementation Status

| Document | Type | Status | Key Impl Files | Phantom? |
|----------|------|--------|----------------|----------|
| data-care-plan.md | plan | Implemented | `validation/data_care.py` (352 lines) | No |
| data-engineering-improvement-plan.xml | execution_plan | Partial | Data pipeline improvements planned | No |
| data-engineering-quality-etl-report.md | reference | Reference only | ETL quality landscape analysis | No |
| dataset-use-plan.md | plan | Implemented | `data/external_datasets.py` (353 lines) | No |
| drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md | plan | Partial | Evidently implemented; Grafana dashboard not wired | No |
| drift-monitoring-implementation-plan.xml | execution_plan | Implemented | `validation/drift.py` (89 lines) | No |
| drift-monitoring-plan.md | plan | Partial | Basic drift detection exists; full monitoring loop not closed | No |
| dvc-test-suite-improvement.xml | execution_plan | Implemented | DVC test hardening for GCS migration | No |
| pr2-data-quality-pipeline-plan.md | plan | Implemented | `validation/enforcement.py` (124 lines), `validation/gates.py` (108 lines) | No |
| prompt-574-synthetic-data-drift-detection.md | prompt | Reference only | Comprehensive research prompt (Issue #574) | No |
| segmentation-qc-plan.md | plan | Implemented | `validation/vessqc.py` (228 lines) | No |
| synthetic-data-qa-engineering-drifts-knowledge-agentic-systems-report.md | reference | Reference only | Academic survey: synthetic data + drift + agentic | No |
| synthetic-vascular-stack-generators-plan.md | plan | Not started | L-system generators not implemented | PHANTOM |
| synthicl-plan.md | plan | Not started | SynthICL integration not implemented | PHANTOM |
| topology-real-data-e2e-plan.xml | execution_plan | Partial | Topology metrics on real data; E2E pipeline not complete | PHANTOM |
| vesselnn-dataset-implementation.xml | execution_plan | Partial | VesselNN registered but drift simulation not wired | PHANTOM |
| vessqc-plan.md | plan | Implemented | `validation/vessqc.py` (228 lines) | No |
| zarr-vs-pt-for-5d-uq-array.md | reference | Reference only | Format comparison, no implementation decision | No |

**PHANTOM documents** describe infrastructure that exists in planning only. The
synthetic data generators (L-systems, SynthICL) were never built. VesselNN is
registered as a dataset but the drift simulation workflow using it is not wired.
The topology-real-data E2E plan depends on complete vessel graph extraction which
is only partially implemented (P1 in graph-topology plans).

---

## Cross-References

- **KG Domain**: `knowledge-graph/domains/data.yaml` -- 12 decision nodes, dataset strategy resolved
- **Training Theme**: Loss functions consume data from the 3-fold split strategy defined here
- **Operations Theme**: Drift monitoring (Evidently) is cross-referenced in both data and operations KG
- **Observability Theme**: MLflow artifacts include data provenance metadata
- **Manuscript Theme**: Dataset description required for TRIPOD+AI Item 4 (study participants)
- **Key Source Files**:
  - `src/minivess/data/external_datasets.py` (353 lines) -- dataset registry, download, splits
  - `src/minivess/validation/` (13 files, 2110 lines total):
    - `data_care.py` (352 lines) -- DATA-CARE 6-dimension quality scoring
    - `deepchecks_3d_adapter.py` (123 lines) -- DeepChecks 3D volume adapter
    - `deepchecks_vision.py` (98 lines) -- DeepChecks Vision integration
    - `drift.py` (89 lines) -- drift detection (KS/PSI)
    - `enforcement.py` (124 lines) -- quality gate enforcement in flows
    - `expectations.py` (126 lines) -- Great Expectations suites
    - `gates.py` (108 lines) -- GateResult pattern for pipeline control
    - `ge_runner.py` (162 lines) -- GE batch runner
    - `profiling.py` (118 lines) -- whylogs profiling
    - `schemas.py` (85 lines) -- Pandera DataFrame schemas
    - `vessqc.py` (228 lines) -- vessel-specific QC metrics
  - `configs/splits/3fold_seed42.json` -- deterministic 47 train / 23 val splits

---

## Constituent Documents

1. `data-care-plan.md` -- DATA-CARE 6-dimension quality framework for NIfTI metadata and metrics (Issue #11)
2. `data-engineering-improvement-plan.xml` -- XML execution plan for data pipeline improvements
3. `data-engineering-quality-etl-report.md` -- ETL quality landscape: ingestion, transformation, validation
4. `dataset-use-plan.md` -- Dataset strategy: MiniVess primary, DeepVess test, VesselNN drift
5. `drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md` -- Comprehensive drift detection: Evidently + Grafana + VesselNN + synthetic generators
6. `drift-monitoring-implementation-plan.xml` -- XML execution plan for Evidently drift monitoring
7. `drift-monitoring-plan.md` -- Drift monitoring architecture: data drift, concept drift, model drift
8. `dvc-test-suite-improvement.xml` -- DVC test hardening for GCS migration and cache consistency
9. `pr2-data-quality-pipeline-plan.md` -- PR#2 data quality pipeline: Pandera + GE + DATA-CARE gate wiring
10. `prompt-574-synthetic-data-drift-detection.md` -- Verbatim user prompt for Issue #574: synthetic data + drift + agentic science
11. `segmentation-qc-plan.md` -- Vessel-specific segmentation quality control metrics
12. `synthetic-data-qa-engineering-drifts-knowledge-agentic-systems-report.md` -- Academic survey: synthetic data generation, drift detection, data engineering for science
13. `synthetic-vascular-stack-generators-plan.md` -- Parametric L-system vessel tree generators (PHANTOM: not implemented)
14. `synthicl-plan.md` -- SynthICL integration for synthetic in-context learning (PHANTOM: not implemented)
15. `topology-real-data-e2e-plan.xml` -- E2E topology metrics on real MiniVess data (PHANTOM: partially implemented)
16. `vesselnn-dataset-implementation.xml` -- VesselNN dataset registration and drift simulation wiring (PHANTOM: partial)
17. `vessqc-plan.md` -- VessQC: vessel-specific quality control (density, branching, diameter, connectivity)
18. `zarr-vs-pt-for-5d-uq-array.md` -- Storage format evaluation for 5D UQ arrays (Zarr vs PyTorch .pt)
