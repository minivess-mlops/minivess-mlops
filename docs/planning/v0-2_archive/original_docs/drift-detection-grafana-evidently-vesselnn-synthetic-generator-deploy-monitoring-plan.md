---
title: "Drift Detection, Grafana, Evidently, VesselNN, Synthetic Generator, Deploy Monitoring Plan"
status: planning
created: "2026-03-16"
branch: feat/observability-drift-detection
parent_branch: test/mambavesselnet
pr_strategy: one_large_pr
level4_mandate: NON-NEGOTIABLE
---

# Drift Detection + Grafana + Evidently + VesselNN + Synthetic Generators + Deploy Monitoring

## 0. User Prompt (Verbatim)

> Let's next create a a new branch feat/observability-drift-detection on top of test/mambavesselnet
> to continue working on the observability stack involving Grafana, Evidently, etc on monitoring the
> deployed model and the "deployment drift" (the data drift, data quality, etc with DeepChecks, Great
> Expectations shoud be its own PR after this then, and open a P0 Issue on this). As you should by now
> realize that as we don't have any live deployment, we are simulating the drift by adding new data
> from https://github.com/petteriTeikari/vesselNN_dataset in chunks to simulate the new evaluation
> data in real-world deployment and see how well different models perform. And don't ever forget that
> this is now a MLOps repo and academic paper, so we need to be demonstrating the MLOps Level 4
> maturity here rather than focusing which is the best model to handle the new data (we use this task
> as a a demo task to demonstrate the platform functioning) at first phase, and follow-up to that we
> should implement some synthetic generator to create new data. And as we are now using the synthetic
> generation to test the drift detection, even better if the data quality does not match so well to
> real 2-PM data as we want to see the drift detection working and how the data distribution is
> clearly changing! See these older plans for some context as well.

---

## 1. Interactive Decision Log (All Q&A Verbatim)

### Q1: VesselNN Data Access Strategy
**Question**: VesselNN data: Is the vesselNN_dataset repo already cloned locally, or should
the pipeline download it from GitHub on first use?

**Answer**: If we have not yet implemented a downloader to the ACQUISITION FLOW, this must be
then implemented so that we can automatically fetch all the datasets. This downloader should
obviously work as local downloader on anyone's laptop (remember that we are building a repo here
to be published as an academic paper and not for us per se, so there are people interested in
using this code without any access to our GCP stack so that the tools should automate the download
of all the open-source data. And provide Pulumi-type of mechanisms so that these external
researchers can automatically also upload to GCP/AWS/Azure blob or whatever they have set up,
so remember this flexibility when planning. This is not our product for our startup or our
research lab. But for a wider community!

**Decision**: Universal cloud-agnostic downloader in Acquisition Flow. Works locally on any
laptop. Supports upload to any cloud (GCP/AWS/Azure) via config.

### Q2: DVC Chunked Versioning Strategy
**Question**: What does 'chunked DVC commits' mean concretely?

**Answer**: We should add in batches of two for example, and think on how this is helpful both
for the academic paper and how the vesselNN_dataset could also function as golden dataset for a
test suite used for BentoML+Evidently. And can we vectorize the volumetric stacks to 1D vectors
and use other tools as well not designed for images and 3D volumes.

**Decision**: Batches of 2 volumes. Single DVC-tracked directory with git tags per batch
(drift-batch-1 through drift-batch-6). VesselNN serves dual purpose: drift simulation source
AND golden test dataset. Vectorize 3D volumes to 1D feature vectors for tabular drift tools.

**Referenced bibliography** (user provided 13 monitoring papers — see Section 8 for full list).

### Q3: Champion Model Selection
**Question**: Which trained model checkpoint should the 'deployed BentoML model' use?

**Answer**: This should be defined quite flexibly. This could be done all automatic so that in
"MODEL→POST-TRAINING→EVALS→BIOSTATISTIC" pipeline we have evaluated all the hyperparameter
combinations with the evals creating all the ensemble permutations and the biostatistics doing
statistical comparisons of all these different models. The Biostatistics model can then return
scenarios where there is no statistically significant difference and the user must be prompted
on what model to use. And we can have multiple champions like one champion for each model family
(one for CNN such as dynUnet, one for foundation models SAM3 or vesselfm, one for MAMBA). And
then the "champion model" is the one that we simulate on being in production, deployed via
BentoML to GCP in "prod" or "staging". Or using Runpod's Serverless Endpoint in "env" for
quick demos.

**Decision**: Data-driven champion selection from Biostatistics flow. Multiple champions per
model family (CNN, foundation, Mamba). Human-in-the-loop when no statistically significant
difference. Config-driven via YAML.

### Q4: Synthetic Generator Scope
**Question**: Which synthetic generator approach for this PR?

**Answer**: For the publication gate, we need to have some existing 3D vascular generation
finetuned with some of the 3rd party test dataset. And as our goal is to have just a decent
synthetic generator that is clearly OOD generator, we don't have to finetune long. We need to
implement the mechanisms how to finetune easily the synthetic generators via the same
Skypilot/Pulumi infra provisioning pipeline. [...] Create an open-ended multi-hypothesis
decision matrix on all these different options with pros/cons outlined. [...] it seems that
we should implement all those 3 methods and we simply define in the .yaml which ones we use
for default workflows. [...] stop resisting and being lazy and not wanting to engineer any
more SOTA-like generators and instead of collapse to some garbage solutions! [...] An
infrastructure scaffold with excellent DevEx to integrate the standalone approaches into a
system, and not to create our own synthetic generators per se. If a code exists for the
synthetic generator with finetuning recipe, should be rather straightforward to implement
right. If we need to implement from an arxiv paper with no repo to build upon, then this
task is too complex.

**Decision**: Implement ALL viable generators as adapters behind `SyntheticGeneratorAdapter` ABC
with YAML config selection. `generate_stack(method='vqvae')` API. New Prefect Flow for
synthetic generation. Integrate via SkyPilot/Pulumi for cloud finetuning.

### Q5: Grafana Dashboard Strategy
**Question**: Extend existing dashboards or create new dedicated drift monitoring dashboard?

**Decision**: New dedicated 'Drift Monitoring Timeline' dashboard. Existing 4 dashboards remain.

### Q6: RunPod vs GCP (Recurring)
**Question**: RunPod Serverless endpoint scope?

**Answer**: RunPod is ONLY used with "dev" environment. The key focus is on "staging" and
"prod" with GCP. RunPod is only the backup choice. All handled via SkyPilot and Pulumi.

**Decision**: GCP = primary (staging/prod), RunPod = dev backup. Both via SkyPilot abstraction.
See metalearning: `.claude/metalearning/2026-03-16-runpod-dev-not-primary-recurring-confusion.md`

### Q7: Active Learning Scope
**Question**: How much active learning in THIS PR?

**Decision**: Architecture only — define interfaces: `UncertaintySampler` ABC,
`AnnotationRequest` dataclass, MONAI Label adapter stub. Implementation in follow-up PR.

### Q8: P0 Issue Scope (DeepChecks + GE)
**Question**: Focused on VesselNN or full DQ pipeline?

**Decision**: Full DQ pipeline — DeepChecks Vision (2D slices from 3D), GE batch validation,
whylogs profiling, Pandera schema enforcement for ALL datasets.

### Q9: E2E Pipeline Orchestration
**Question**: New 6th flow or wire into existing?

**Decision**: New 6th Prefect flow ("Drift Simulation Flow"). Clean separation of concerns.

### Q10: Synthetic Generator Methods (see Q4 expansion)

**Decision**: Implement ALL with code repos. YAML-driven selection. New Prefect Flow for
synthetic generation with `generate_stack(method='vesselFM_drand')` API.

### Q11: Evidently→Grafana Export
**Question**: JSON or Prometheus or both?

**Decision**: Both — JSON reports to MLflow artifacts for archival, AND key drift scores
pushed to Prometheus for Grafana time-series panels.

### Q12: Whylogs Integration Depth
**Question**: Continuous, batch, or whylabs?

**Decision**: Continuous profiling — profile EVERY volume that enters the pipeline. Mergeable
profiles over time. Custom Prometheus exporter for Grafana integration.

### Q13: Evidently Docker Service
**Question**: New Docker service, embedded in BentoML, or embedded in Prefect?

**Decision**: New Docker service in docker-compose.yml with its own /metrics endpoint.
Workspace API (0.7+). Prometheus scrapes alongside BentoML.

### Q14: Alerting Mechanism (Level 4 NON-NEGOTIABLE)
**Question**: Alerting depth?

**Answer**: Level 4 is an OBLIGATORY requirement. This repo has ZERO value without Level 4.
Not a negotiation.

**Decision**: ALL THREE LAYERS — Prometheus Alertmanager (production alerts via webhook + JSONL
log) + Grafana built-in alerts (visual) + MLflow logs (audit trail). Full Level 4.
See metalearning: `.claude/metalearning/2026-03-16-level4-mandate-never-negotiate.md`

### Q15: DVC Versioning Strategy
**Question**: Separate .dvc files, tags, or pipeline stages?

**Decision**: Single directory `data/drift_simulation/` tracked by DVC. Git tags per batch
(drift-batch-1 through drift-batch-6). Pipeline checks out specific tags.

### Q16: PR Strategy
**Question**: Split into sub-PRs or one large PR?

**Decision**: One large PR. Easier to review holistically for paper coherence.

### Q17: Synthetic Generator Verification
**Question**: Verify repos before planning?

**Answer**: Verify all. Licenses don't have to be commercially permissive — academic use is fine.

**Verification Results** (see Section 4 for full decision matrix):
- vesselFM d_drand: **STRONG YES** (MONAI-native, CPU, GPL-3.0, CVPR 2025)
- MONAI VQ-VAE: **STRONG YES** (already in stack, Apache-2.0, learnable)
- VaMos/Nader: **CONDITIONAL YES** (Python, MRA domain mismatch)
- VascuSynth: **YES w/ caveats** (C++ compilation, Apache-2.0)
- svVascularize: **BONUS** (pure Python, pip-installable, Stanford)
- VesselVAE: **NO** (outputs graphs, not volumes)
- VasTSD: **NO** (no code released)

### Q18: Evidently Version
**Question**: Workspace API (0.7+) or WSGI monitor pattern?

**Decision**: Workspace API (0.7+) — current recommended pattern.

### Q19: Champion Evaluation Modes
**Question**: Supervised (with masks), unsupervised, or both?

**Decision**: Both modes — supervised (Dice/clDice degradation curves) AND unsupervised
(drift detection + uncertainty only). Config-driven switch. Shows both scenarios in paper.

### Q20: Alertmanager Notification Target
**Question**: Default notification channel?

**Decision**: Webhook + JSONL log file (configurable URL via .env). External researchers
point webhook at their Slack/Teams/email gateway. All channels (Slack, email, webhook,
PagerDuty) available via Dynaconf config.

### Q21: Docker Compose Profile
**Question**: 'monitoring' profile or new 'drift' profile?

**Decision**: Add Evidently + Alertmanager to existing 'monitoring' profile. Single
`docker compose --profile monitoring up` starts everything.

### Q22: Simulation Timeline
**Question**: Single automated run or separate batch triggers?

**Decision**: Single automated run — one flow invocation iterates through all 6 batches
sequentially. Each batch: DVC checkout → drift detect → champion eval → log. Produces
complete temporal drift curve in one run.

### Q23: Dashboard Provisioning
**Question**: JSON file or API-driven?

**Decision**: JSON file auto-load in `deployment/grafana/dashboards/drift-monitoring-timeline.json`.
Consistent with existing 4 dashboards. Version-controlled.

---

## 2. Architecture Overview

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTION & MONITORING SYSTEM                      │
│                          (MLOps Level 4)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA SOURCES                    DETECTION                    RESPONSE      │
│  ┌───────────────┐              ┌─────────────────┐          ┌───────────┐ │
│  │ VesselNN      │──┐           │ Tier 1: Evidently│          │ Grafana   │ │
│  │ (12 vols,     │  │  DVC      │ DataDriftPreset  │──┐       │ Dashboard │ │
│  │ batches of 2) │  ├──tags──→  │ (feature KS/PSI) │  │       │ Timeline  │ │
│  ├───────────────┤  │           ├─────────────────┤  │  ┌───→│ + Alerts  │ │
│  │ Synthetic Gen │  │           │ Tier 2: Alibi-   │  ├──┤   ├───────────┤ │
│  │ (vesselFM,    │──┘           │ Detect MMD       │  │  │   │ Prometheus│ │
│  │  MONAI VQVAE, │              │ (embedding drift)│──┘  │   │ Alertmgr  │ │
│  │  VaMos, etc.) │              ├─────────────────┤      │   │ → webhook │ │
│  └───────────────┘              │ whylogs          │──────┘   │ → JSONL   │ │
│                                 │ (continuous prof) │         └───────────┘ │
│  CHAMPION MODELS                └────────┬────────┘                         │
│  ┌───────────────┐                       │                                  │
│  │ BentoML ONNX  │←── evaluate ──────────┘                                 │
│  │ (per family:  │                                                          │
│  │  CNN, Found., │──→ Dice, clDice (supervised)                            │
│  │  Mamba)       │──→ Uncertainty (unsupervised)                           │
│  └───────────────┘──→ MLflow artifacts                                     │
│                                                                             │
│  ORCHESTRATION                                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Flow 6: Drift Simulation Flow (Prefect)                       │          │
│  │   for batch in DVC tags (drift-batch-1..6):                   │          │
│  │     1. checkout DVC tag → load 2 VesselNN volumes             │          │
│  │     2. extract features → whylogs profile                    │          │
│  │     3. Tier 1 drift detection (Evidently)                    │          │
│  │     4. Tier 2 embedding drift (Alibi-Detect MMD)             │          │
│  │     5. Champion evaluation (BentoML → Dice if supervised)    │          │
│  │     6. Push metrics → Prometheus → Grafana                   │          │
│  │     7. Log reports → MLflow artifacts                        │          │
│  │     8. If drift detected → Alertmanager webhook              │          │
│  │   end                                                         │          │
│  │ Flow 7: Synthetic Generation Flow (Prefect)                   │          │
│  │   generate_stack(method='vesselFM_drand', n=10, config=...)   │          │
│  └───────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Docker Compose Stack (monitoring profile)

```
┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
│  BentoML    │  │ Evidently │  │ Prometheus   │  │ Grafana      │
│  :3000      │  │ :8000     │  │ :9090        │  │ :3001        │
│  /metrics ──┤  │ /metrics ─┤  │ scrapes both ├──│ 5 dashboards │
│  /predict   │  │ workspace │  │ + alertrules │  │ (incl. drift │
│             │  │ API 0.7+  │  │              │  │  timeline)   │
└─────────────┘  └──────────┘  └──────┬───────┘  └──────────────┘
                                      │
                                ┌─────┴──────┐
                                │ Alertmanager│
                                │ :9093       │
                                │ → webhook   │
                                │ → JSONL log │
                                └─────────────┘
```

### 2.3 Data Flow

```
VesselNN GitHub repo
    │
    ▼ (universal downloader — works on any laptop)
data/drift_simulation/
    │
    ├── drift-batch-1 (tag) → vol_01, vol_02
    ├── drift-batch-2 (tag) → vol_03, vol_04
    ├── drift-batch-3 (tag) → vol_05, vol_06
    ├── drift-batch-4 (tag) → vol_07, vol_08
    ├── drift-batch-5 (tag) → vol_09, vol_10
    └── drift-batch-6 (tag) → vol_11, vol_12
    │
    ▼ (feature extraction — 9 features + vectorization to 1D)
pd.DataFrame (features per volume)
    │
    ├─→ whylogs profile (continuous, mergeable)
    ├─→ Evidently Tier 1 (DataDriftPreset on features)
    ├─→ Alibi-Detect Tier 2 (MMD on model embeddings)
    │
    ▼ (champion evaluation — BentoML ONNX endpoint)
    ├─→ Supervised: Dice, clDice (when masks available)
    ├─→ Unsupervised: MC Dropout uncertainty, Mahalanobis distance
    │
    ▼ (reporting)
    ├─→ MLflow artifacts (JSON + HTML reports)
    ├─→ Prometheus metrics (drift scores, p-values, Dice)
    ├─→ Grafana dashboard (drift timeline)
    └─→ Alertmanager (webhook + JSONL on threshold breach)
```

---

## 3. Synthetic Generator Decision Matrix

### Verification Results (2026-03-16)

| Generator | Repo | License | Training? | VRAM | Python? | Output Format | Feasibility | Effort |
|---|---|---|---|---|---|---|---|---|
| **vesselFM d_drand** | [bwittmann/vesselFM](https://github.com/bwittmann/vesselFM) | GPL-3.0 / Open RAIL++-M | Config-driven (YAML) | CPU | Yes (PyTorch+MONAI) | 128³ numpy image+mask | **STRONG YES** | ~1 day |
| **MONAI VQ-VAE** | [MONAI core](https://github.com/Project-MONAI/MONAI) `monai.networks.nets.VQVAE` | Apache-2.0 | Yes (train on patches) | ~2-4 GB | Yes | 3D patches | **STRONG YES** | ~1 day |
| **VaMos (Nader)** | [GitLab](https://gitlab.univ-nantes.fr/autrusseau-f/vamos/) | TBD | N/A (procedural) | CPU | Yes | 3D volumetric + GT | **CONDITIONAL** | ~2 days |
| **VascuSynth** | [sfu-mial/VascuSynth](https://github.com/sfu-mial/VascuSynth) | Apache-2.0 | N/A (procedural) | CPU | **No (C++)** | GXL tree + raw vol | YES w/ caveats | ~2 days |
| ~~svVascularize~~ | ~~[SimVascular/svVascularize](https://github.com/SimVascular/svVascularize)~~ | ~~TBD~~ | ~~N/A~~ | ~~CPU~~ | ~~Yes~~ | ~~Tree structures~~ | **NO** (outputs tree/CFD, not 3D volumes — same problem as VesselVAE) | — |
| **MONAI synthetic** | MONAI core `create_test_image_3d()` | Apache-2.0 | N/A | CPU | Yes | 3D vol + GT (spheres) | YES (limited) | ~0.5 day |
| ~~VesselVAE~~ | ~~LIA-DiTella/VesselVAE~~ | ~~TBD~~ | ~~Notebook~~ | ~~<1GB~~ | ~~Yes~~ | ~~Graph/mesh~~ | **NO** (needs rasterization) | — |
| ~~VasTSD~~ | ~~No code~~ | ~~N/A~~ | ~~N/A~~ | ~~N/A~~ | ~~N/A~~ | ~~N/A~~ | **NO** (no code) | — |

### Implementation Priority

1. **vesselFM d_drand** — MONAI-native, CVPR 2025, 128³ 3D volumes directly
2. **MONAI VQ-VAE** — already in stack, Apache-2.0, 3D patches (spatial_dims=3)
3. **VaMos/Nader** — Python procedural, 3D volumetric output, cerebral vasculature
4. **VascuSynth** — classic procedural, 3D volume rendering, needs C++ wrapper

**Rejected** (output tree/graph structures, NOT 3D volumes):
- ~~svVascularize~~ — outputs centerlines + radii for CFD, not volumetric images
- ~~VesselVAE~~ — outputs recursive tree graphs, needs rasterization
- ~~VasTSD~~ — no code released

### Adapter Interface

```python
class SyntheticGeneratorAdapter(ABC):
    """ABC for all synthetic vascular volume generators."""

    @abstractmethod
    def generate_stack(
        self, n_volumes: int, config: dict[str, Any]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate n synthetic (image, mask) pairs."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def requires_training(self) -> bool: ...

# Usage: generate_stack(method='vesselFM_drand', n=10)
# Config: configs/synthetic/method.yaml selects default method
```

---

## 4. Existing Code Inventory (What We Build On)

| Component | File | Status |
|---|---|---|
| Tier 1 drift (KS + Evidently) | `src/minivess/observability/drift.py` | FUNCTIONAL |
| Tier 2 embedding drift (MMD) | `src/minivess/observability/drift.py` | FUNCTIONAL |
| Feature extraction (9 features) | `src/minivess/data/feature_extraction.py` | FUNCTIONAL |
| Acquisition simulator | `src/minivess/data/acquisition_simulator.py` | FUNCTIONAL |
| Drift triage agent | `src/minivess/agents/drift_triage.py` | FUNCTIONAL |
| BentoML ONNX service | `src/minivess/serving/bento_service.py` | FUNCTIONAL |
| Grafana 4 dashboards | `deployment/grafana/dashboards/*.json` | FUNCTIONAL |
| Prometheus config | `deployment/prometheus/prometheus.yml` | FUNCTIONAL |
| Dashboard flow (Flow 5) | `src/minivess/orchestration/flows/dashboard_flow.py` | FUNCTIONAL |
| VesselNN registry entry | `src/minivess/data/external_datasets.py` | FUNCTIONAL |
| whylogs profiling | `src/minivess/validation/profiling.py` | FUNCTIONAL (basic) |
| GE runner | `src/minivess/validation/ge_runner.py` | FUNCTIONAL |
| Drift synthetic perturbations | `src/minivess/data/drift_synthetic.py` | FUNCTIONAL |
| Docker Compose (12 services) | `deployment/docker-compose.yml` | FUNCTIONAL |
| Data flow (Flow 1) | `src/minivess/orchestration/flows/data_flow.py` | FUNCTIONAL |
| MiniVess downloader | `src/minivess/data/downloader.py` | FUNCTIONAL |

---

## 5. Task Breakdown

### Phase A: Infrastructure Foundation

#### T-A1: Universal Dataset Downloader for Acquisition Flow
**Files**: `src/minivess/data/dataset_downloader.py`, `src/minivess/data/external_datasets.py`
**Tests**: `tests/v2/unit/test_dataset_downloader.py`
- `DatasetDownloader` ABC with implementations for GitHub releases, EBRAINS, direct HTTP
- Cloud upload adapter: GCS (primary), S3, Azure Blob via config
- VesselNN-specific: download from GitHub, extract TIFF stacks, convert to NIfTI
- Integration into Data Flow (Flow 1) as Prefect task
- Config: `configs/data/download.yaml` with per-dataset settings

#### T-A2: DVC Drift Simulation Setup
**Files**: `scripts/setup_drift_simulation.py`, `data/drift_simulation/.dvc`
**Tests**: `tests/v2/unit/test_drift_simulation_setup.py`
- Script to partition 12 VesselNN volumes into 6 batches of 2
- Create DVC-tracked directory `data/drift_simulation/`
- Git tags: drift-batch-1 through drift-batch-6
- Batch manifest JSON: which volumes in which batch, ordering rationale

#### T-A3: Docker Compose — Evidently + Alertmanager Services
**Files**: `deployment/docker-compose.yml`, `deployment/evidently/`, `deployment/alertmanager/`
**Tests**: `tests/v2/integration/test_monitoring_stack.py`
- Evidently service (Workspace API 0.7+) in 'monitoring' profile
- Alertmanager service with webhook + JSONL log receiver
- Alert rules in `deployment/prometheus/alert_rules.yml`
- Prometheus scrape config updated for Evidently /metrics
- `.env.example` updated with EVIDENTLY_PORT, ALERTMANAGER_WEBHOOK_URL, etc.

#### T-A4: Grafana Drift Monitoring Timeline Dashboard
**Files**: `deployment/grafana/dashboards/drift-monitoring-timeline.json`
**Tests**: `tests/v2/integration/test_grafana_dashboards.py` (update expected count to 5)
- Panels: DVC batch timeline, per-batch drift scores (Tier 1 + Tier 2), champion
  performance degradation curve (Dice/clDice), triage recommendations, whylogs
  profile comparisons, uncertainty heatmaps
- Template variables for model family, batch range, drift threshold
- Auto-provisioned alongside existing 4 dashboards

### Phase B: Drift Detection Enhancement

#### T-B1: Evidently Workspace Service Integration
**Files**: `src/minivess/observability/evidently_service.py`
**Tests**: `tests/v2/unit/test_evidently_workspace.py`
- Evidently 0.7+ Workspace API client
- Project creation, snapshot storage, report retrieval
- Prometheus metrics exporter (drift scores as Prometheus gauges)
- JSON export to MLflow artifacts
- Column mapping for our 9 image features

#### T-B2: whylogs Continuous Profiling Pipeline
**Files**: `src/minivess/validation/whylogs_profiler.py`
**Tests**: `tests/v2/unit/test_whylogs_continuous.py`
- Profile EVERY volume entering the pipeline
- Mergeable profiles over time (per-batch, per-day, cumulative)
- Reference profile from MiniVess training set (47 volumes)
- Drift detection via profile comparison (mean/std/quantile shifts)
- Prometheus exporter for key statistics

#### T-B3: Prometheus Alertmanager Integration
**Files**: `src/minivess/observability/alerting.py`, `deployment/alertmanager/alertmanager.yml`
**Tests**: `tests/v2/unit/test_alerting.py`
- Alert rules: drift p-value < threshold, Dice drop > epsilon, uncertainty spike
- Notification: configurable webhook URL (default) + JSONL audit log
- Dynaconf/env-var driven channel selection (webhook, Slack, email, PagerDuty)
- Alert history persistence for dashboard timeline

### Phase C: Champion Evaluation Pipeline

#### T-C1: Multi-Family Champion Registry
**Files**: `src/minivess/serving/champion_registry.py`
**Tests**: `tests/v2/unit/test_champion_registry.py`
- Champion model per family: CNN (DynUNet), Foundation (SAM3), Mamba
- Data-driven selection from Biostatistics flow output
- Human-in-the-loop alert when no statistically significant difference
- MLflow model registry integration with stage transitions
- Config: `configs/serving/champions.yaml`

#### T-C2: Dual-Mode Champion Evaluation
**Files**: `src/minivess/serving/champion_evaluator.py`
**Tests**: `tests/v2/unit/test_champion_evaluator.py`
- **Supervised mode**: Dice, clDice, topology metrics (when masks available)
- **Unsupervised mode**: MC Dropout uncertainty, Mahalanobis embedding distance
- Config switch: `evaluation_mode: supervised | unsupervised | both`
- BentoML client for champion inference
- Results logged to MLflow with batch metadata

#### T-C3: 3D Volume Vectorization for Tabular Tools
**Files**: `src/minivess/data/volume_vectorizer.py`
**Tests**: `tests/v2/unit/test_volume_vectorizer.py`
- Flatten 3D volumes to 1D feature vectors for tabular drift tools
- Multiple strategies: statistical features (existing 9), histogram, PCA embeddings,
  model embeddings (penultimate layer), spatial frequency (FFT)
- Enables NannyML, Deepchecks tabular, and other non-image drift tools

### Phase D: Synthetic Generation Infrastructure

#### T-D1: SyntheticGeneratorAdapter ABC + Registry
**Files**: `src/minivess/data/synthetic/base.py`, `src/minivess/data/synthetic/__init__.py`
**Tests**: `tests/v2/unit/test_synthetic_generator_base.py`
- `SyntheticGeneratorAdapter` ABC with `generate_stack()`, `name()`, `requires_training()`
- Registry pattern: `SYNTHETIC_GENERATORS` dict, config-driven selection
- Config: `configs/synthetic/method.yaml` with per-method parameters
- `generate_stack(method='vesselFM_drand', n=10)` top-level API

#### T-D2: vesselFM d_drand Adapter
**Files**: `src/minivess/data/synthetic/vesselfm_drand.py`
**Tests**: `tests/v2/unit/test_vesselfm_drand_adapter.py`
- Wrap vesselFM d_drand pipeline as `SyntheticGeneratorAdapter`
- MONAI-native transforms, CPU-based, 128³ volume+mask output
- Configurable domain randomization parameters via YAML
- GPL-3.0 license — document in NOTICE file

#### T-D3: MONAI VQ-VAE Adapter
**Files**: `src/minivess/data/synthetic/monai_vqvae.py`
**Tests**: `tests/v2/unit/test_monai_vqvae_adapter.py`
- Wrap `monai.networks.nets.VQVAE` for 3D patch generation
- Train on MiniVess + VesselNN patches (32³ or 64³)
- Codebook sampling for novel patch generation
- Stitching strategy for full-volume reconstruction
- SkyPilot YAML for cloud training

#### T-D4: VaMos Procedural Adapter
**Files**: `src/minivess/data/synthetic/vamos.py`
**Tests**: `tests/v2/unit/test_vamos_adapter.py`
- Wrap VaMos (Nader et al.) spline-based procedural generator
- Adapt cerebral MRA noise model → 2PM noise model
- CPU-based, Python-native

#### T-D5: VascuSynth Adapter (C++ wrapper)
**Files**: `src/minivess/data/synthetic/vascusynth.py`
**Tests**: `tests/v2/unit/test_vascusynth_adapter.py`
- Subprocess wrapper around compiled VascuSynth binary
- GXL tree → numpy conversion
- CMake build integration (optional, graceful degradation if not compiled)
- Docker image with pre-compiled binary for reproducibility

#### ~~T-D6: svVascularize Adapter~~ — REJECTED
svVascularize outputs tree structures (centerlines + radii) for CFD simulation, NOT 3D
volumetric images. Same problem as VesselVAE. Would require a full volumetric rendering
pipeline on top. Not viable for this PR.

#### T-D6: Synthetic Generation Prefect Flow (Flow 7)
**Files**: `src/minivess/orchestration/flows/synthetic_flow.py`
**Tests**: `tests/v2/unit/test_synthetic_flow.py`
- `run_synthetic_flow(method='vesselFM_drand', n_volumes=10, config=...)`
- Output: DVC-tracked synthetic volumes in `data/synthetic/{method}/`
- SkyPilot YAML for cloud-based generation/finetuning
- Integration with drift simulation (synthetic → drift detection)

### Phase E: Drift Simulation Flow (E2E)

#### T-E1: Drift Simulation Flow (Flow 6)
**Files**: `src/minivess/orchestration/flows/drift_simulation_flow.py`
**Tests**: `tests/v2/unit/test_drift_simulation_flow.py`
- Single automated run through all 6 VesselNN batches
- Per-batch: DVC checkout → feature extraction → whylogs profile →
  Tier 1 drift → Tier 2 drift → champion eval → log → alert
- Accumulates temporal drift curve across batches
- Configurable: n_batches, champion_family, evaluation_mode
- Docker context required (`_require_docker_context()`)

#### T-E2: Synthetic Drift Simulation Integration
**Files**: `src/minivess/orchestration/flows/drift_simulation_flow.py` (extend T-E1)
**Tests**: `tests/v2/integration/test_synthetic_drift_simulation.py`
- Add synthetic volumes from Flow 7 as additional drift source
- Interleave: VesselNN batches (natural drift) + synthetic batches (OOD drift)
- Compare drift detection sensitivity: natural vs. synthetic vs. mixed

#### T-E3: End-to-End Integration Test
**Files**: `tests/v2/integration/test_drift_e2e_pipeline.py`
**Tests**: Full pipeline integration
- Spin up monitoring Docker stack (Evidently, Prometheus, Grafana, Alertmanager)
- Run drift simulation flow with synthetic data
- Verify: drift detected → metrics in Prometheus → dashboard updated → alert fired
- Verify: MLflow artifacts contain all reports
- Verify: whylogs profiles accumulated correctly

### Phase F: Active Learning Architecture (Stubs)

#### T-F1: Active Learning Interfaces
**Files**: `src/minivess/active_learning/base.py`, `src/minivess/active_learning/__init__.py`
**Tests**: `tests/v2/unit/test_active_learning_interfaces.py`
- `UncertaintySampler` ABC: `select_samples(volumes, n) -> ranked_indices`
- `AnnotationRequest` dataclass: volume_id, uncertainty_score, source, priority
- `MONAILabelAdapter` stub: interface for MONAI Label integration
- Strategies: max_entropy, max_mc_variance, max_mahalanobis, bald

### Phase G: P0 Issue + Documentation

#### T-G1: P0 Issue — Full Data Quality Pipeline (DeepChecks + GE)
- Create GitHub issue with full scope
- DeepChecks Vision (2D slices from 3D volumes)
- Great Expectations batch validation for ALL datasets
- whylogs profiling integration
- Pandera schema enforcement
- Priority: P0

#### T-G2: Update Intent Summary
**Files**: `docs/planning/intent-summary.md`
- Add this prompt as P11 with cross-references to all planning docs
- Update themes list

---

## 6. Execution Order

```
Phase A (Infrastructure)
  T-A1 (downloader) ──┐
  T-A2 (DVC setup)  ──┼──→ Phase E (can start once A1+A2 done)
  T-A3 (Docker) ──────┤
  T-A4 (Grafana) ─────┘

Phase B (Drift Enhancement) — parallel with Phase A
  T-B1 (Evidently workspace) ──┐
  T-B2 (whylogs continuous) ───┼──→ feeds into Phase E
  T-B3 (Alertmanager) ─────────┘

Phase C (Champion Eval) — parallel with Phase B
  T-C1 (champion registry) ──┐
  T-C2 (dual-mode eval) ─────┼──→ feeds into Phase E
  T-C3 (3D vectorization) ───┘

Phase D (Synthetic Generators) — parallel with Phase C
  T-D1 (ABC + registry) ──→ T-D2..T-D5 (4 adapters, parallel) ──→ T-D6 (flow)

Phase E (E2E Integration) — after A+B+C
  T-E1 (drift sim flow) ──→ T-E2 (synthetic integration) ──→ T-E3 (e2e test)

Phase F (Active Learning Stubs) — anytime
  T-F1 (interfaces)

Phase G (Issues + Docs) — final
  T-G1 (P0 issue) + T-G2 (intent summary)
```

---

## 7. Referenced Planning Documents

| Document | Status | Relevance |
|---|---|---|
| `monitoring-research-report.md` | Reference | MLOps Level 4 architecture, 3-tier OOD, champion-challenger |
| `drift-monitoring-plan.md` | Planned → superseded by this | Original Tier 1+2 detection plan |
| `drift-monitoring-implementation-plan.xml` | 8/9 tasks done | Tasks T1-T9 for core drift detection |
| `ralph-loop-for-cloud-monitoring.md` | Planned | Ralph monitor loop for SkyPilot jobs |
| `synthetic-vascular-stack-generators-plan.md` | Reference | Literature review for synthetic generators |
| `synthetic-data-qa-engineering-drifts-knowledge-agentic-systems-report.md` | Reference | 33-ref drift + synthetic + QA literature review |
| `prompt-574-synthetic-data-drift-detection.md` | Reference | Original prompt P7 for Issue #574 |
| `cover-letter-to-sci-llm-writer-for-knowledge-graph.md` | Living | Paper context and contribution framing |

---

## 8. Bibliography (Monitoring + Drift Detection)

### Post-Deployment Monitoring (Medical Imaging)
- [Cook et al. (2026). "State of the AI: Post-Deployment Monitoring of Radiology-Focused Internally Developed AI." *Mayo Clinic Proceedings: Digital Health*.](https://doi.org/10.1016/j.mcpdig.2026.XXX) — Mayo Clinic FAST Team, 17 deployed algorithms, daily/weekly monitoring
- [Keyes et al. (2024). "Monitoring Deployed AI Systems in Health Care." Stanford/Shah lab.](https://doi.org/XXX) — Three-principle framework: integrity, performance, impact

### Drift Detection Methods
- [Kore et al. (2024). "Empirical data drift detection experiments on real-world medical imaging data." *Nature Communications* 15:1887.](https://doi.org/10.1038/s41467-024-46109-z) — Natural + synthetic drift on chest X-rays, performance monitoring ≠ drift detection
- [Muller et al. (2024). "Open-Source Drift Detection Tools in Action." *arXiv:2404.18673*.](https://arxiv.org/abs/2404.18673) — D3Bench: Evidently AI validated as best production tool
- [Zamzmi et al. (2024). "OOD Detection and Data Drift Monitoring Using SPC." *FDA CDRH Preprint*.](https://doi.org/XXX) — SPC control charts for clinical imaging, FDA perspective
- [Roschewitz et al. (2023). "Automatic correction of performance drift under acquisition shift." *Nature Communications* 14:6608.](https://doi.org/10.1038/s41467-023-42396-y) — UPA: label-free drift correction for mammography/histopath
- [Singh et al. (2025). "SHIFT: Diagnosing Heterogeneous Performance Drift." *ICML* 267:55757.](https://doi.org/XXX) — Subgroup-level performance decay, hierarchical hypothesis testing
- [Xiong et al. (2026). "ADAPT: Adversarial Drift-Aware Predictive Transfer." *arXiv:2601.11860*.](https://arxiv.org/abs/2601.11860) — DRO for durable clinical AI

### Monitoring Practices & Observability
- [Leest et al. (2025). "Monitoring and Observability of ML Systems: Current Practices and Gaps." *arXiv:2510.24142*.](https://arxiv.org/abs/2510.24142) — 77% use custom-built monitoring
- [Protschky et al. (2025). "What Gets Measured Gets Improved." *IEEE Access* 13.](https://doi.org/XXX) — 17 monitoring practices taxonomy
- [Kim et al. (2025). "Monitoring strategies for continuous evaluation of deployed clinical prediction models." *JBHI* 168.](https://doi.org/XXX) — Feedback-loop-aware monitoring
- [Schirmer et al. (2025). "Monitoring Risks in Test-Time Adaptation." *arXiv:2507.08721*.](https://arxiv.org/abs/2507.08721) — Sequential testing for TTA risk monitoring

### Integration Patterns (Web Sources)
- [BentoML Blog: "Monitoring Metrics with Prometheus and Grafana"](https://www.bentoml.com/blog/monitoring-metrics-in-bentoml-with-prometheus-and-grafana) — BentoML /metrics auto-exposition, custom Histogram/Counter/Gauge
- [BentoML Blog: "A Guide to ML Monitoring and Drift Detection"](https://www.bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection) — `bentoml.monitor()` context manager, log forwarding via Fluentbit
- [Evidently Blog: "Evidently and Grafana: ML Monitoring Live Dashboards"](https://www.evidentlyai.com/blog/evidently-and-grafana-ml-monitoring-live-dashboards) — YAML-driven monitors, Prometheus-compatible /metrics, pre-built Grafana JSON
- [giorgiaBertacchini/MLOps-ProductionPhase](https://github.com/giorgiaBertacchini/MLOps-ProductionPhase) — Full reference stack: BentoML + Evidently + Prometheus + Alertmanager + Grafana
- [Jeremy Jordan: "ML Monitoring"](https://www.jeremyjordan.me/ml-monitoring/) — Drift detection sidecar pattern, image monitoring descriptors, whylogs
- [Daily Dose of DS: "MLOps Crash Course Part 17"](https://www.dailydoseofds.com/mlops-crash-course-part-17/) — KS test, ADWIN, Evidently + Prefect integration
- [Grafana Cloud Logs](https://grafana.com/products/cloud/logs/) — Metrics-from-logs capability, LogQL
- [BentoML Blog: "Why Is ML Deployment Hard?"](https://bentoml.com/blog/why-do-people-say-its-so-hard-to-deploy-a-ml-model-to-production) — Shadow pipelines, data volatility

### Synthetic Data
- See `synthetic-vascular-stack-generators-plan.md` for full synthetic data bibliography (30+ refs)

---

## 9. Metalearning Documents Created

| File | Topic |
|---|---|
| `.claude/metalearning/2026-03-16-runpod-dev-not-primary-recurring-confusion.md` | RunPod = dev ONLY, GCP = primary |
| `.claude/metalearning/2026-03-16-infrastructure-scaffold-not-shortcuts.md` | Implement ALL methods, never collapse to "simple" |
| `.claude/metalearning/2026-03-16-level4-mandate-never-negotiate.md` | Level 4 is NON-NEGOTIABLE, never offer downgrades |

---

## 10. Key Architectural Principles (from Q&A)

1. **Level 4 is the publication gate** — repo has ZERO value without it
2. **Infrastructure scaffold** — integrate ALL viable approaches behind config-driven adapters
3. **GCP primary, RunPod backup** — both via SkyPilot, never treat as equal
4. **Universal DevEx** — works on any laptop, any cloud, for any lab
5. **Both evaluation modes** — supervised (with masks) AND unsupervised (deployment-realistic)
6. **Continuous profiling** — whylogs on EVERY volume, not just batches
7. **ALL three alert layers** — Alertmanager + Grafana + MLflow audit trail
8. **Config-driven everything** — YAML selects generator, champion, evaluation mode, alert channel
