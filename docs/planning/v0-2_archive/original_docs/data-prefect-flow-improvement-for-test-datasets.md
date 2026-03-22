---
title: "Data Pipeline and Prefect Flow Improvement for Test Datasets"
status: planned
created: ""
---

# Data Pipeline & Prefect Flow Improvement for Test Datasets

> **Multi-hypothesis architecture planning report**
> Closes: #150 (license verification), #151 (synthetic data for drift monitoring)
> Branch: `feat/test-dataset`
> Date: 2026-03-02

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [License Verification (#150)](#2-license-verification-150)
3. [Synthetic Data Generator for Drift Monitoring (#151)](#3-synthetic-data-generator-for-drift-monitoring-151)
4. [Prefect Data Flow Architecture Analysis](#4-prefect-data-flow-architecture-analysis)
5. [Data Proofreading Queue Architecture](#5-data-proofreading-queue-architecture)
6. [DVC Versioning & Data Contract Model](#6-dvc-versioning--data-contract-model)
7. [SDD Data Pipeline Specification](#7-sdd-data-pipeline-specification)
8. [PRD Updates Required](#8-prd-updates-required)
9. [Architecture Options Analysis](#9-architecture-options-analysis)
10. [Recommended Execution Plan](#10-recommended-execution-plan)
11. [Open Questions](#11-open-questions)

---

## 1. Executive Summary

The MinIVess MLOps data pipeline (Prefect Flow 1: Data Engineering) is the
**least developed** of the 5 persona-based flows. Flows 2-5 (Training, Analysis,
Deploy, Dashboard) are implemented with tests; Flow 1 exists only as DVC stages
(`dvc.yaml`) and a preprocessing script (`preprocess.py`). This report addresses
three interconnected needs:

1. **Test dataset acquisition** -- DeepVess and tUbeNet for cross-dataset
   generalization in the Analysis Flow
2. **Synthetic data generation** -- controlled distribution shifts for drift
   monitoring validation
3. **Data pipeline architecture** -- how raw data (real or synthetic) flows
   through quality gating into DVC-versioned training data

### Key Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| DeepVess license | **Use with attribution + author confirmation** | eCommons educational use; no explicit CC license |
| tUbeNet license | **CC-BY 4.0 confirmed** | UCL RDR explicit license |
| Proofreading queue | **Prefect `suspend_flow_run`** (not a separate MQ) | Native human-in-the-loop; no infra overhead |
| Annotation stub | **3D Slicer + nnInteractive P2 stub** | CVPR 2025 winner; prior-mask proofreading |
| Message pattern | **Prefect events + webhooks** | Replaces RabbitMQ/Redis; works offline via store-and-forward |
| DVC versioning | **Git tag per DVC version** | Cross-system lineage with MLflow |
| Data flow structure | **`@flow` for pipeline, `@task` for steps** | Existing pattern is correct |

---

## 2. License Verification (#150)

### 2.1 Dataset License Matrix

| Dataset | Source | License | Academic Use | Confirmed |
|---------|--------|---------|-------------|-----------|
| **MiniVess** | EBRAINS KG (doi:10.25493/HPBE-YHK) | **CC-BY 4.0** | Yes | Yes (Scientific Data 2023) |
| **DeepVess** | Cornell eCommons (doi:10.7298/X4FJ2F1D) | **No explicit data license** | Likely yes (educational use) | Partial -- see below |
| **tUbeNet** | UCL RDR (doi:10.5522/04/25715604.v1) | **CC-BY 4.0** | Yes | Yes (UCL RDR terms) |
| **Damseh 2PM** | Google Drive via GitHub | **No explicit data license** (code: GPL-3.0) | Likely yes | No -- contact authors |
| **vesselNN** | GitHub (petteriTeikari) | **MIT** | Yes | Yes |
| **VesselGraph** | GitHub (jocpae) | **CC-BY-NC 4.0** | Yes (non-commercial) | Yes |

### 2.2 DeepVess License Analysis

**Status: Ambiguous -- requires author confirmation.**

The DeepVess dataset is hosted on Cornell eCommons. Key findings:

- **Code license**: Apache 2.0 (GitHub: mhaft/DeepVess)
- **Data license**: Not explicitly stated on the eCommons record
- **eCommons Terms of Use**: Content is openly accessible for "educational and
  research" purposes, but derivative works require authorization
- **Paper** (PLoS ONE): CC-BY 4.0, but paper license does not automatically
  apply to dataset
- **Paper statement**: "we have made all images and expert annotations publicly
  available" -- intent is clearly open access

**Recommendation**: Use for academic research with attribution. Contact
corresponding author (Chris Schaffer, Cornell) for explicit CC-BY confirmation.
Add a `license_verified: false` flag in `ExternalDatasetConfig` until confirmed.
The risk is low: the authors explicitly published the data for reuse, and
eCommons permits educational/research use.

### 2.3 tUbeNet License Analysis

**Status: Confirmed CC-BY 4.0.**

Two distinct projects often conflated:

1. **tUbeNet** (Holroyd et al. 2025, Biology Methods and Protocols):
   - Data: UCL Research Data Repository, **CC-BY 4.0** confirmed
   - Code: **MIT License** (GitHub: natalie11/tUbeNet)
   - Note: Two-photon data used for *validation* only; training covers HREM,
     microCT, RSOM, OCTA modalities
   - The UCL RDR uses CC-BY for shared datasets

2. **Damseh 2PM DNN** (2020, BME Frontiers):
   - Data: Google Drive, **no explicit license**
   - Code: **GPL-3.0**
   - Contact authors for data license confirmation

**Recommendation for `external_datasets.py`**: Update the `tubenet_2pm` entry
with `license="CC-BY-4.0"` and `license_verified=True`. Keep `deepvess` with
`license="TBD-educational-use"` and `license_verified=False`.

### 2.4 VesselFM Data Leakage Warning (#151)

VesselFM (Wittmann et al., CVPR 2025) was trained on:
- MiniVess (Class 21)
- DeepVess (Class 16)
- tUbeNet (Class 7)

**VesselFM cannot be used as a fair baseline on any of these datasets.** Its
D_drand synthetic pipeline can generate novel data, but the generator itself was
trained with knowledge of these distributions. The synthetic data is safe for
drift monitoring validation (the goal of #151), but not for performance
benchmarking.

### 2.5 Additional Datasets Identified

| Dataset | License | Relevance |
|---------|---------|-----------|
| **vesselNN** (Teikari et al. 2016) | MIT | Precursor to MiniVess, 12 volumes, 3 pathologies |
| **VesselGraph** (Paetzold et al. 2022) | CC-BY-NC 4.0 | Graph-level vessel data, useful for topology work |

vesselNN is particularly relevant as it includes tumor vasculature and BBB
disruption pathologies not present in MiniVess. Its MIT license is the most
permissive of all options.

---

## 3. Synthetic Data Generator for Drift Monitoring (#151)

### 3.1 Current Implementation

The codebase already has two complementary synthetic generation modules:

| Module | Purpose | Status |
|--------|---------|--------|
| `data/drift_synthetic.py` | 4 controlled drift types (intensity, noise, resolution, topology) | Implemented |
| `data/domain_randomization.py` | SynthICL-style multi-parameter randomization | Implemented |
| `data/debug_dataset.py` | Minimal NIfTI fixtures for testing | Implemented |

**Gap**: These modules generate individual volumes but lack a **pipeline** that
produces a stream of synthetic acquisitions simulating real lab operations (new
2-PM experiments arriving over time).

### 3.2 Proposed: SyntheticAcquisitionSimulator

A new module that wraps existing generators into a streaming acquisition
simulator:

```python
# src/minivess/data/synthetic_acquisition.py

@dataclass
class SyntheticAcquisition:
    """Simulates a single 2-PM imaging session."""
    volume_id: str          # e.g., "synth_001"
    image: np.ndarray       # 3D volume
    label: np.ndarray       # Binary mask
    metadata: dict           # Acquisition params, drift severity, etc.
    timestamp: datetime      # Simulated acquisition time
    drift_type: str | None   # Applied drift (if any)
    drift_severity: float    # 0.0 = clean, 1.0 = maximal shift

class SyntheticAcquisitionSimulator:
    """Generates a temporal stream of synthetic 2-PM acquisitions.

    Simulates:
    - Baseline clean acquisitions (drift_severity=0)
    - Gradual instrument drift (severity increases over time)
    - Sudden distribution shifts (new staining protocol, new microscope)
    - Domain-randomized augmentations (SynthICL)
    """

    def generate_batch(
        self, n: int, *, drift_schedule: str = "gradual"
    ) -> list[SyntheticAcquisition]: ...

    def generate_stream(
        self, *, interval_seconds: float = 0.0
    ) -> Iterator[SyntheticAcquisition]: ...
```

This feeds directly into the Prefect Data Flow as if it were a real data source.

### 3.3 Integration with Drift Detection

```
SyntheticAcquisitionSimulator
    │
    ▼
Prefect Data Flow (Flow 1)
    ├── automated_qc_task()          ← Pandera + Great Expectations
    ├── extract_features_task()       ← feature_extraction.py
    ├── detect_drift_task()           ← Evidently / Alibi-Detect
    │       │
    │       ├── No drift → dvc_version_task()
    │       └── Drift detected → alert + suspend_flow_run (human review)
    │
    └── dvc_push_task()
```

### 3.4 VesselFM D_drand Integration (Future P2)

D_drand from VesselFM generates synthetic vessels from corrosion cast graphs with
domain-randomized backgrounds. This is **significantly more realistic** than our
random-walk SyntheticVesselGenerator. However:

- The D_drand checkpoint is not publicly released
- Training D_drand from scratch requires ~3 days on 8xA100
- D_flow (flow-matching generative model) is even more expensive

**Decision**: Defer D_drand integration to `feat/synthetic-drift-monitoring`.
Use existing `drift_synthetic.py` + `domain_randomization.py` for the current
implementation. Create a P2 issue for D_drand when the checkpoint is released.

---

## 4. Prefect Data Flow Architecture Analysis

### 4.1 Current State

| Flow | Status | Files |
|------|--------|-------|
| Flow 1: Data Engineering | **Not implemented as Prefect flow** | `dvc.yaml`, `data/preprocess.py` |
| Flow 2: Training | Implemented (scripts, not Prefect-decorated) | `scripts/train_monitored.py` |
| Flow 3: Analysis | Implemented | `orchestration/flows/analysis_flow.py` |
| Flow 4: Deploy | Implemented | `orchestration/deploy_flow.py` |
| Flow 5: Dashboard | Implemented | `orchestration/flows/dashboard_flow.py` |

**Problem**: Flow 1 is the foundation of the entire pipeline but exists only as
DVC stages. It has no Prefect tasks, no human-in-the-loop capability, no drift
detection integration, and no connection to the annotation platform.

### 4.2 Flow vs. Task Decision for Data Pipeline

Based on Prefect 3.x best practices (see references):

| Component | Type | Rationale |
|-----------|------|-----------|
| Data Engineering Pipeline | `@flow` | Top-level orchestration, independent run entry |
| Dataset discovery | `@task` | Discrete step, cacheable |
| Data validation (automated QC) | `@task` | Retry-able, cacheable per input hash |
| Feature extraction | `@task` | CPU-bound, parallelizable |
| Drift detection | `@task` | Side-effect (logs metrics), retry-able |
| Proofreading review | `suspend_flow_run` | Human-in-the-loop, hours/days wait |
| DVC versioning | `@task` with `transaction()` | Atomic with rollback |
| External dataset download | `@task(retries=3)` | Network-dependent |
| Synthetic generation | `@task` | Pure computation, cacheable |

**Key design decision**: The proofreading step is NOT a separate `@task` or
`@flow`. It uses Prefect's native `suspend_flow_run()` which tears down
infrastructure while waiting for human input (no cost during review).

### 4.3 Proposed Data Flow Architecture

```python
@flow(name="Data Engineering Pipeline")
async def data_flow(config: DataFlowConfig):
    """Flow 1: Data Engineering — raw data → validated DVC-versioned data."""

    # Phase 1: Ingest (real or synthetic sources)
    raw_volumes = ingest_task(config)

    # Phase 2: Automated quality checks
    qc_results = automated_qc_task(raw_volumes)
    flagged = [r for r in qc_results if r.needs_review]
    passed = [r for r in qc_results if not r.needs_review]

    # Phase 3: Human proofreading (if any flagged)
    if flagged:
        reviewed = await proofreading_gate(flagged)
        passed.extend(reviewed)

    # Phase 4: DVC version the approved data
    version_tag = dvc_version_task(passed, config.version_tag)

    return DataFlowResult(
        n_volumes=len(passed),
        version_tag=version_tag,
        drift_detected=any(r.drift_score > config.drift_threshold for r in qc_results),
    )
```

### 4.4 Two Data Ingestion Paths

```
Path A: 3rd-party datasets (DeepVess, tUbeNet, vesselNN)
    │
    ├── download_external_task(dataset_config)     # @task(retries=3)
    ├── validate_external_layout_task()              # @task
    └── No proofreading needed (trust original authors)
         │
         └── dvc_version_task()

Path B: New acquisitions (real 2-PM or synthetic)
    │
    ├── ingest_raw_task() / generate_synthetic_task()
    ├── automated_qc_task()                          # @task
    ├── extract_features_task()                      # @task
    ├── detect_drift_task()                          # @task
    │       │
    │       └── If drift: Prefect event → alert
    │
    ├── proofreading_gate()                          # suspend_flow_run
    │       │
    │       ├── Approved → continue
    │       ├── Rejected → archive + log reason
    │       └── Needs revision → route to annotation platform
    │
    └── dvc_version_task()
```

---

## 5. Data Proofreading Queue Architecture

### 5.1 Multi-Hypothesis Analysis

Five architectural options were evaluated for the proofreading queue:

#### Option A: Prefect `suspend_flow_run` (Recommended)

```
Prefect Data Flow
    │
    ├── automated_qc_task()
    │
    ├── suspend_flow_run(wait_for_input=ProofreadingDecision)
    │       │
    │       ├── Prefect UI: reviewer clicks approve/reject
    │       ├── OR: programmatic resume via Prefect API
    │       └── OR: webhook from external tool (Slicer, Label Studio)
    │
    └── dvc_version_task()
```

**Pros**:
- Zero additional infrastructure (no MQ, no Redis)
- Native Prefect UI for review forms (`RunInput` with Pydantic models)
- Infrastructure tears down during wait (cost-efficient)
- Programmatic resume via `send_input()` or REST API
- Webhook integration for external tools (3D Slicer, Label Studio)
- Full audit trail in Prefect run history

**Cons**:
- Requires Prefect Cloud/Server for `suspend` (not available in local-only mode)
- Review UI is basic (no 3D volume rendering in Prefect UI)
- Single reviewer at a time per flow run (no multi-annotator consensus natively)

#### Option B: External Message Queue (RabbitMQ/Redis)

```
Data Flow → RabbitMQ → Annotation Frontend → RabbitMQ → Data Flow
```

**Pros**:
- Decoupled producer/consumer
- Multi-consumer support (multiple reviewers)
- Works offline (local RabbitMQ)
- Language-agnostic (any frontend can consume)

**Cons**:
- Additional infrastructure to deploy and maintain
- Message serialization complexity (NIfTI paths, not raw volumes)
- No native Prefect integration (custom polling task needed)
- Requires dead-letter queue for failed reviews
- Violates Design Goal #1 (adds infra complexity for PhD researchers)

#### Option C: Prefect Events + Automations

```
Data Flow → emit_event("review.requested") → Automation → triggers Review Flow
Review Flow → emit_event("review.completed") → Automation → resumes Data Flow
```

**Pros**:
- Event-driven, loosely coupled
- Supports absence-based triggers ("no review in 48h → escalate")
- Multiple automation actions (Slack notify, email, webhook)

**Cons**:
- More complex event routing to set up
- Two separate flows to coordinate
- Event ordering not guaranteed

#### Option D: Label Studio Webhook Integration

```
Data Flow → Label Studio task queue → Label Studio UI → webhook → Data Flow resume
```

**Pros**:
- Rich annotation UI (already in docker-compose)
- Multi-annotator workflows built-in
- Agreement metrics (Cohen's kappa, Fleiss' kappa)

**Cons**:
- Label Studio is primarily 2D (limited 3D NIfTI support)
- Requires running Label Studio service (full docker profile)
- Heavy for simple approve/reject decisions

#### Option E: Hybrid (Recommended for Production)

```
Data Flow
    │
    ├── Simple QC decisions → suspend_flow_run (Prefect UI)
    │
    └── Complex annotation → Prefect webhook ← 3D Slicer + nnInteractive
```

**Pros**:
- Simple cases stay in Prefect (fast approve/reject)
- Complex cases routed to specialized tool (3D Slicer)
- Incremental adoption: start with Option A, add Slicer later

**Cons**:
- Two review paths to maintain
- Routing logic adds complexity

### 5.2 Recommendation

**Phase 1 (this PR)**: Option A -- `suspend_flow_run` only.
Simple, zero-infra, sufficient for 3rd-party dataset QA.

**Phase 2 (P2 issue)**: Option E -- add 3D Slicer webhook integration
for proofreading new acquisitions with prior-mask overlay.

### 5.3 Offline / Air-gapped Operation

For overseas flights or air-gapped environments:

```
Desktop (offline)                  Prefect Server (online)
┌─────────────────┐               ┌─────────────────────┐
│ 3D Slicer       │               │                     │
│ nnInteractive    │               │  data_flow (suspended)
│                  │               │  waiting for input  │
│ Store-and-forward│               │                     │
│ annotation queue │ ──on-connect──▶ send_input() API    │
│ (local SQLite)   │               │                     │
└─────────────────┘               └─────────────────────┘
```

The annotation frontend stores completed reviews in a local SQLite database.
When connectivity is restored, a sync agent pushes reviews to the Prefect API
via `send_input()`. This is the **store-and-forward** pattern.

**Implementation**: A lightweight Python CLI (`minivess review sync`) that reads
the local SQLite queue and calls `prefect.client.orchestration.get_client()` to
resume suspended flow runs. No additional MQ infrastructure needed.

---

## 6. DVC Versioning & Data Contract Model

### 6.1 Data Versioning Strategy

```
data/
├── minivess/          # DVC-tracked (minivess.dvc)
│   ├── imagesTr/
│   └── labelsTr/
├── external/          # DVC-tracked (external.dvc)
│   ├── deepvess/
│   │   ├── images/
│   │   └── labels/
│   ├── tubenet/
│   │   ├── images/
│   │   └── labels/
│   └── vesselnn/
│       ├── images/
│       └── labels/
└── synthetic/         # DVC-tracked (synthetic.dvc)
    ├── drift_gradual_v1/
    └── drift_sudden_v1/
```

### 6.2 Version Naming Convention

```
data-minivess-v1.0.0          # Original MiniVess (70 volumes)
data-deepvess-v1.0.0          # First DeepVess import
data-tubenet-v1.0.0           # First tUbeNet import
data-synthetic-drift-v1.0.0   # First synthetic drift batch
```

Git tags paired with DVC versions enable `git checkout data-deepvess-v1.0.0 &&
dvc checkout` to reproduce exact data state.

### 6.3 Data Contract: DVC → MLflow

The data version tag is logged as an MLflow param in the Training Flow:

```python
mlflow.log_param("data_version_tag", "data-deepvess-v1.0.0")
mlflow.set_tag("dvc_data_hash", "<md5-from-dvc-file>")
```

This creates the inter-flow contract: Analysis Flow can query MLflow for runs
trained on a specific data version.

### 6.4 3rd Party Dataset Acceptance

3rd-party datasets (DeepVess, tUbeNet, vesselNN) are accepted as-is:

```
download_external_task()
    ├── Verify checksum (SHA-256)
    ├── Validate NIfTI readability (nibabel)
    ├── Validate layout (images/ + labels/ dirs)
    ├── Profile volumes (profiler.py)
    └── DVC version (no proofreading needed)
```

New DVC versions are created when:
- Dataset authors release updates
- We discover and fix data issues (e.g., corrupted NIfTI headers)
- New volumes are added by the authors

---

## 7. SDD Data Pipeline Specification

### 7.1 Data Flow Block Diagram (IEC 62304 SDD-style)

```
┌──────────────────────────────────────────────────────────────────┐
│                    FLOW 1: DATA ENGINEERING                      │
│                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Ingest  │───▶│ Validate │───▶│ Proofread│───▶│ DVC       │  │
│  │ Sources │    │ (Auto QC)│    │ (Human)  │    │ Version   │  │
│  └─────────┘    └──────────┘    └──────────┘    └───────────┘  │
│       │              │               │                │         │
│  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌─────┴──────┐  │
│  │External │    │Pandera  │    │Prefect  │    │git tag +   │  │
│  │Download │    │GE Suite │    │suspend  │    │dvc push    │  │
│  │Synthetic│    │Profiler │    │(P2:Slicer│    │MLflow param│  │
│  │Raw 2-PM │    │Features │    │webhook) │    │            │  │
│  └─────────┘    └─────────┘    └─────────┘    └────────────┘  │
│                                                      │         │
│  ┌─────────────────────────┐                         │         │
│  │ Drift Detection         │◀────────────────────────┘         │
│  │ (Evidently + features)  │                                   │
│  └─────────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│ FLOW 2: TRAINING│          │ FLOW 3: ANALYSIS │
│ (reads DVC data)│          │ (test datasets)  │
└─────────────────┘          └──────────────────┘
         │
         ▼
┌──────────────────┐
│ FLOW 4: DEPLOY   │ ◀──── Serves model for proofreading overlay
│ (BentoML server) │
└──────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│ ANNOTATION PLATFORM STUB (P2)                      │
│                                                    │
│  3D Slicer + nnInteractive (client-server)         │
│  └── Prior mask from Deploy Flow server            │
│  └── Proofreading corrections → webhook → Flow 1   │
│                                                    │
│  Status: STUB ONLY — full implementation deferred  │
└────────────────────────────────────────────────────┘
```

### 7.2 Annotation Platform Stub

The annotation platform is a **P2 deferred item** but must exist as a stub in
the architecture so it is never forgotten:

```python
# src/minivess/data/annotation_platform_stub.py

"""Annotation platform integration stub.

This module is a PLACEHOLDER for the future 3D Slicer + nnInteractive
proofreading integration. It defines the interface that the Data Flow
uses to route volumes to human review.

PRD Decision: L3-technology/annotation-platform.decision.yaml
  → 3D Slicer + SlicerNNInteractive (0.40 prior, RECOMMENDED)

PRD Decision: L5-operations/annotation-workflow.decision.yaml
  → Slicer Proofreading (0.45 prior, RECOMMENDED)

The full implementation will:
1. Query the Deploy Flow's BentoML server for an initial segmentation mask
2. Send (volume, prior_mask) to Slicer via the nnInteractive client-server API
3. Receive corrected mask via webhook → Prefect send_input()
4. Validate corrections (Dice vs prior, topology checks)
5. Route to DVC versioning

See also:
- K-Prism (Guo et al., 2025): >30% NoC90 reduction with prior mask
- nnInteractive (Isensee et al., 2025): CVPR 2025 1st place
- VessQC (Puttmann et al., 2025): uncertainty-guided curation
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Status sentinel — checked by data_flow to decide routing
ANNOTATION_PLATFORM_AVAILABLE: bool = False


def get_prior_mask(volume_id: str, server_url: str) -> None:
    """Query Deploy Flow server for initial segmentation mask.

    NOT IMPLEMENTED — P2 deferred.
    """
    logger.warning(
        "Annotation platform not implemented. "
        "Volume %s will proceed without proofreading.",
        volume_id,
    )
    return None
```

### 7.3 Interface Contract: Data Flow ↔ Analysis Flow

The Analysis Flow (Flow 3) consumes test datasets via
`build_hierarchical_dataloaders()`. The data contract is:

```
Data Flow outputs (DVC-versioned):
    data/external/{dataset_name}/images/*.nii.gz
    data/external/{dataset_name}/labels/*.nii.gz

Analysis Flow inputs (DatasetRegistry):
    DatasetEntry(
        name="deepvess",
        data_dir=Path("data/external/deepvess"),
        layout="decathlon",
        subsets=[DatasetSubset(name="all")],
    )
```

---

## 8. PRD Updates Required

### 8.1 New Decision Node: `data_ingestion_queue` (L3-technology)

```yaml
# docs/prd/decisions/L3-technology/data-ingestion-queue.decision.yaml
node_id: data_ingestion_queue
title: "Data ingestion queue for human-in-the-loop review"
level: L3-technology
status: active
classification: recommended

options:
  - name: prefect_suspend
    prior: 0.50
    description: "Prefect suspend_flow_run + RunInput for review forms"
    implementation_status: not_started
  - name: prefect_events_webhooks
    prior: 0.25
    description: "Prefect events + webhook relay from external annotation tools"
    implementation_status: not_started
  - name: external_mq
    prior: 0.10
    description: "RabbitMQ or Redis Streams as dedicated message queue"
    implementation_status: not_started
  - name: label_studio_queue
    prior: 0.15
    description: "Label Studio's built-in task queue for multi-annotator review"
    implementation_status: config_only

conditional_on:
  - pipeline_orchestration  # Prefect is prerequisite
  - annotation_platform     # Determines external tool
  - data_management_strategy  # DVC versioning target

rationale: >
  Prefect 3.x native suspend_flow_run eliminates the need for external MQ
  infrastructure. For simple approve/reject decisions, the Prefect UI is
  sufficient. For complex 3D annotation, webhooks bridge to Slicer/napari.
  External MQ (RabbitMQ) is only justified if the annotation frontend has
  strict latency or multi-consumer requirements.
```

### 8.2 Updates to Existing Decision Nodes

| Decision | Update |
|----------|--------|
| `annotation_platform` | Add implementation_status: `stub_created` for Slicer option |
| `annotation_workflow` | Add implementation_status: `stub_created` for Slicer proofreading |
| `data_management_strategy` | Add note about external dataset DVC versioning pattern |
| `drift_detection_method` | Link to synthetic acquisition simulator for validation |

### 8.3 New Bibliography Entries

```yaml
# Additions to docs/prd/bibliography.yaml

holroyd2025tubenet:
  authors: "Holroyd et al."
  year: 2025
  title: "tUbeNet: a generalizable deep learning tool for 3D vessel segmentation"
  venue: "Biology Methods and Protocols"
  doi: "10.1093/biomethods/bpaf087"

haftjavaherian2019deepvess:
  authors: "Haft-Javaherian et al."
  year: 2019
  title: "Deep convolutional neural networks for segmenting 3D in vivo multiphoton images of vasculature"
  venue: "PLoS ONE"
  doi: "10.1371/journal.pone.0213539"

teikari2016vesselnn:
  authors: "Teikari et al."
  year: 2016
  title: "Deep Learning Convolutional Networks for Multiphoton Microscopy Vasculature Segmentation"
  venue: "arXiv"
  arxiv: "1606.02382"
```

---

## 9. Architecture Options Analysis

### 9.1 Option Matrix: Proofreading Architecture

| Criterion | A: Prefect suspend | B: External MQ | C: Events | D: Label Studio | E: Hybrid |
|-----------|-------------------|----------------|-----------|-----------------|-----------|
| **Infrastructure** | Zero (Prefect only) | +1 service (RabbitMQ) | Prefect only | +1 service (LS) | Prefect + webhook |
| **DevEx (PhD)** | Excellent | Poor (MQ config) | Good | Good (web UI) | Good |
| **Offline support** | No (needs server) | Yes (local MQ) | No | Yes (local LS) | Partial |
| **3D volume review** | No (basic forms) | N/A (transport) | No | Limited 2D | Yes (Slicer) |
| **Multi-annotator** | No | Yes | No | Yes (built-in) | Yes (via Slicer) |
| **Audit trail** | Prefect run history | Custom logging | Event stream | LS export | Both |
| **Implementation** | ~2 days | ~5 days | ~3 days | ~4 days | ~3+5 days |
| **Phase 1 fit** | Excellent | Overkill | Good | Moderate | Phase 2 |

### 9.2 Option Matrix: Synthetic Data Strategy

| Criterion | Existing drift_synthetic | + AcquisitionSimulator | + VesselFM D_drand |
|-----------|-------------------------|----------------------|-------------------|
| **Realism** | Low (simple transforms) | Medium (temporal stream) | High (learned) |
| **Diversity** | 4 drift types | 4 types + scheduling | Unlimited |
| **Cost** | Zero | Zero | ~3 days A100 training |
| **Data leakage risk** | None | None | Medium (trained on MiniVess) |
| **Implementation** | Done | ~1 day | ~2 weeks |
| **Phase 1 fit** | Yes | Yes | No (P2) |

### 9.3 Option Matrix: DVC External Dataset Pattern

| Criterion | Single .dvc file | Per-dataset .dvc | DVC pipeline stage |
|-----------|-----------------|-----------------|-------------------|
| **Granularity** | All-or-nothing pull | Per-dataset pull | Per-dataset + transform |
| **Disk usage** | All datasets | Selective | Selective |
| **Reproducibility** | Good | Better | Best (transform versioned) |
| **Implementation** | Simple | Moderate | Complex |
| **Phase 1 fit** | Yes | Yes (recommended) | Overkill |

**Recommendation**: Per-dataset `.dvc` files. Each external dataset gets its own
DVC tracking file (`data/external/deepvess.dvc`, etc.) so researchers can pull
only the datasets they need.

---

## 10. Recommended Execution Plan

### Phase 1: Test Dataset Acquisition (P0 -- this branch)

| # | Task | Files | Tests |
|---|------|-------|-------|
| 1 | Update `external_datasets.py` license fields | `external_datasets.py` | ~3 |
| 2 | Create download scripts for DeepVess + tUbeNet | `scripts/download_external.py` | ~4 |
| 3 | Create `SyntheticAcquisitionSimulator` | `data/synthetic_acquisition.py` | ~6 |
| 4 | Create Prefect Data Flow skeleton | `orchestration/flows/data_flow.py` | ~8 |
| 5 | Create annotation platform stub | `data/annotation_platform_stub.py` | ~2 |
| 6 | Register datasets in `DatasetRegistry` | `data/test_datasets.py` | ~4 |
| 7 | Per-dataset DVC tracking | `data/external/*.dvc` | ~2 |
| 8 | Integration: Data Flow → Analysis Flow | `tests/v2/integration/` | ~4 |

**Estimated: ~33 new tests, ~8 new/modified files**

### Phase 2: Proofreading Queue (P1 -- separate branch)

| # | Task |
|---|------|
| 1 | Implement `suspend_flow_run` proofreading gate in Data Flow |
| 2 | Create `ProofreadingDecision` RunInput model |
| 3 | Wire automated QC (Pandera + GE) into Data Flow tasks |
| 4 | Implement store-and-forward sync CLI (`minivess review sync`) |
| 5 | Add Prefect webhook for external annotation tools |
| 6 | Create PRD decision node: `data_ingestion_queue` |

### Phase 3: Annotation Platform (P2 -- separate branch)

| # | Task |
|---|------|
| 1 | Integrate 3D Slicer + nnInteractive (client-server) |
| 2 | Query Deploy Flow BentoML for prior mask |
| 3 | Implement webhook relay: Slicer corrections → Prefect `send_input()` |
| 4 | Multi-annotator consensus (Dice agreement threshold) |
| 5 | VessQC uncertainty-guided triage |

### Phase 4: Advanced Synthetic Data (P2 -- separate branch)

| # | Task |
|---|------|
| 1 | VesselFM D_drand pipeline integration |
| 2 | Temporal drift scheduling (gradual + sudden) |
| 3 | Multi-site simulation (different microscope parameters) |
| 4 | Synthetic → real domain gap measurement |

---

## 11. Open Questions

### 11.1 DeepVess License Confirmation

**Status**: Requires author contact. The eCommons terms support educational use,
but no explicit CC license is attached to the dataset record. Email to Chris
Schaffer (Cornell) recommended.

**Mitigation**: If DeepVess license is incompatible, vesselNN (MIT license, same
first author as MiniVess) is an immediate fallback.

### 11.2 Prefect Cloud vs. Self-Hosted for `suspend_flow_run`

`suspend_flow_run` requires a Prefect server (Cloud or self-hosted) to serialize
state. The `_prefect_compat.py` fallback (CI/local mode) cannot support suspend.

**Options**:
- Add Prefect Server to docker-compose `dev` profile (~200MB RAM)
- Use `pause_flow_run` as fallback (keeps process alive, simpler)
- Skip human review in `_prefect_compat` mode (auto-approve)

**Recommendation**: Auto-approve in compat mode, full suspend in server mode.

### 11.3 tUbeNet Two-Photon Subset

tUbeNet's training data covers multiple modalities (HREM, microCT, RSOM, OCTA).
Only the validation subset is two-photon (mouse olfactory bulb). Should we:
- Use only the two-photon validation subset? (most domain-relevant)
- Use the full multi-modality training set? (more data, cross-modality)
- Both, as separate DatasetRegistry entries?

**Recommendation**: Register as two entries: `tubenet_2pm` (validation subset
only, two-photon) and `tubenet_full` (all modalities). The Analysis Flow can
evaluate on both.

### 11.4 vesselNN as Additional Test Dataset

vesselNN (MIT license, 12 volumes, 3 pathologies) was not in the original plan
(#150/#151) but is highly relevant:
- Same first author as MiniVess (Teikari)
- Includes tumor vasculature and BBB disruption
- Most permissive license (MIT)

**Recommendation**: Add vesselNN as a third external test dataset.

---

## References

- Haft-Javaherian et al. (2019) "Deep convolutional neural networks for
  segmenting 3D in vivo multiphoton images" PLoS ONE 14(3): e0213539
- Holroyd et al. (2025) "tUbeNet: a generalizable deep learning tool for 3D
  vessel segmentation" Biology Methods and Protocols 10(1): bpaf087
- Teikari et al. (2016) "Deep Learning Convolutional Networks for Multiphoton
  Microscopy Vasculature Segmentation" arXiv:1606.02382
- Teikari et al. (2023) "A dataset of rodent cerebrovasculature from in vivo
  multiphoton fluorescence microscopy imaging" Scientific Data 10: 175
- Wittmann et al. (2025) "vesselFM: A Foundation Model for Universal 3D Blood
  Vessel Segmentation" CVPR 2025
- Terms et al. (2025) "SynthICL: Tell Me What You Want, and I'll Train a Model
  for You with Synthetic Data"
- Isensee et al. (2025) "nnInteractive" CVPR 2025 Interactive 3D Segmentation
- Guo et al. (2025) "K-Prism" prior mask reduces NoC90 >30%
- Puttmann et al. (2025) "VessQC" uncertainty-guided curation 67%→94%
- de Vente et al. (2025) "SlicerNNInteractive" client-server architecture
- Xu et al. (2025) "OAIMS" online adaptation +25.5pp OOD
- Prinster et al. (2025) "WATCH" anytime-valid sequential testing
- Paetzold et al. (2022) "VesselGraph" CC-BY-NC 4.0
- Prefect 3.x docs: pause-resume, events, webhooks, interactive workflows
