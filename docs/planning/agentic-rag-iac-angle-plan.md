# Agentic RAG & Infrastructure-as-Code Angle Plan

**Phase 16 — Agentic Capabilities for MinIVess MLOps v2**

*Generated: 2026-02-24 | PRD Network: v1.5.0 → v1.6.0*

---

## Executive Summary

This document identifies **seven genuinely novel research angles** where agentic AI
capabilities can be integrated into the MinIVess MLOps platform. Each angle follows
the **Nikola T. Markov novelty rule**: novelty arises from synthesising distinct
fields that have not been previously combined, rather than incremental improvement
within a single domain. Concretely, each proposed direction sits at the intersection
of at least two of: (a) smart microscopy and adaptive acquisition, (b) 3D vascular
segmentation MLOps, (c) agentic AI orchestration, (d) graph-based retrieval-augmented
generation, (e) clinical evaluation methodology, and (f) time-series infrastructure.

The guiding principle is **"agentic where it earns its complexity"** — we avoid
LLM-driven automation for tasks that a well-typed Python function handles better.
Of the seven proposed angles, five require no LLM at their core mechanism; the
remaining two (vasculometric copilot, documentation agent) use LLMs for their
irreducible natural-language capabilities.

**Key finding**: The most novel contribution is the **Uncertainty-Driven Adaptive
Acquisition Agent** (Angle 1), which synthesises conformal prediction from the
segmentation pipeline with closed-loop microscopy control. No published work combines
calibrated voxel-level uncertainty from 3D segmentation models with real-time
re-scanning decisions for multiphoton vascular imaging. This constitutes a genuine
gap at the intersection of three literatures: conformal prediction (Angelopoulos
& Bates, 2023; Ye et al., 2025), smart microscopy (Hinderling et al., 2025;
Oatman et al., 2025), and vascular segmentation MLOps (Teikari, 2024).

---

## 1. The Nikola T. Markov Novelty Rule

> *"IF WE CANNOT SAY SOMETHING NOVEL, WHY WRITE THIS AT ALL?"*

The rule, articulated across the sci-llm-writer manuscript framework, defines novelty
not as novel empirical discovery alone, but as **novel synthesis from combining
distinct domain literatures**. A contribution is novel when it makes a connection
between fields A and B that practitioners of neither field would make independently.

For this plan, we operationalise the rule as a **novelty matrix**: each angle must
demonstrate that its core idea cannot be found in any single existing paper, but
emerges from combining insights from at least two disjoint literatures. Angles that
fail this test are downgraded to "engineering convenience" and excluded from the
research narrative.

### 1.1 Novelty Assessment Criteria

| Criterion | Description |
|-----------|-------------|
| **Cross-domain synthesis** | Combines 2+ literatures that do not currently cite each other |
| **Non-obvious connection** | A domain expert in field A would not discover the idea from reading only field A |
| **Falsifiable claim** | The angle produces a testable hypothesis or measurable improvement |
| **Implementation tractability** | Can be prototyped within the existing minivess-mlops stack |

---

## 2. Domain Background

### 2.1 MinIVess MLOps Platform

MinIVess MLOps v2 is a model-agnostic biomedical segmentation platform targeting 3D
vascular structures in multiphoton microscopy volumes. The platform uses:

- **Segmentation**: DynUNet, SegResNet, VISTA-3D (MONAI-native)
- **Uncertainty**: Temperature scaling (implemented), conformal prediction (planned),
  MC Dropout, deep ensembles (Lakshminarayanan et al., 2017)
- **Agent orchestration**: LangGraph (resolved-partial)
- **Experiment tracking**: MLflow + DuckDB analytics
- **Data validation**: Pydantic v2, Pandera, Great Expectations
- **Observability**: Langfuse (LLM tracing), Prometheus + Grafana (infrastructure)

The probabilistic PRD (61 nodes, 118 edges) captures technology decisions as a
Bayesian network with conditional probability tables.

### 2.2 User's Prior Work

The multiphoton vasculature segmentation pipeline (Teikari, 2024; see also
arXiv:1606.02382) defines a 5-component architecture that maps directly onto the
agentic framework proposed here:

1. **Active learning** → Acquisition agent (Angle 1)
2. **Annotation** → Data quality triage agent (Angle 4)
3. **Semi-supervised model** → Pipeline state graph (Angle 3)
4. **Proofreading** → Interactive segmentation integration (Phase 15)
5. **Biomarker extraction** → Vasculometric copilot (Angle 2)

The embedded deep learning work (Teikari et al., 2019; PMC6425531) established
the concept of on-device inference for adaptive acquisition in ophthalmology,
which directly informs Angle 1.

---

## 3. Angle 1: Uncertainty-Driven Adaptive Acquisition Agent

**Novelty score: HIGH** — Combines conformal prediction (statistics), closed-loop
microscopy (engineering), and vascular segmentation (biomedical) in a way that no
published work addresses.

### 3.1 Literature Synthesis

**Smart microscopy literature** has established four acquisition paradigms
(Hinderling et al., 2025): event-driven, outcome-driven, quality-driven, and
information-driven. Ye et al. (2023) demonstrated that pixel-wise uncertainty
from deep learning models can drive selective re-scanning, achieving up to 16x
light dose reduction in fluorescence microscopy. Ye et al. (2025) extended this
with QUTCC (Quantile Uncertainty Training with Conformal Calibration), producing
tighter nonlinear asymmetric uncertainty bounds through conformal prediction.
Oatman et al. (2025) showed that programming-free closed-loop microscopy (PyCLM)
can be implemented with modular TOML-configured threads, avoiding hard-coded
instrument coupling.

**Conformal prediction in MLOps** is advancing rapidly. ConSeMa (Mossina,
Friedrich & Dalmau, 2025; MICCAI 2025) uses morphological dilation to construct
conformalized segmentation margins with coverage guarantees. MAPIE provides the
Python implementation for split conformal prediction.

**The gap**: No published work connects conformal prediction-calibrated uncertainty
from 3D vascular segmentation models with real-time microscopy re-scanning
decisions. Ye et al. (2023; 2025) use uncertainty for 2D fluorescence with
simpler models; the vascular segmentation community uses uncertainty for
annotation triage (VessQC; Puttmann et al., 2025) but not acquisition control;
the MLOps community tracks uncertainty for drift detection but not instrument
feedback.

### 3.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Acquisition Agent Loop                     │
│                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐ │
│  │ Microscope│───→│ Segmentation  │───→│ Conformal        │ │
│  │ Control   │    │ Model         │    │ Uncertainty Map  │ │
│  │ (PyCLM)   │    │ (DynUNet)     │    │ (MAPIE/ConSeMa)  │ │
│  └─────▲─────┘    └───────────────┘    └────────┬─────────┘ │
│        │                                         │           │
│        │         ┌───────────────────┐           │           │
│        └─────────│ Re-scan Decision  │◄──────────┘           │
│                  │ (Whittle Index)   │                        │
│                  └───────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

**Components**:

1. **Segmentation model** produces voxel-level predictions on initial low-dose scan
2. **Conformal calibration** (MAPIE + ConSeMa) produces calibrated uncertainty sets
   with coverage guarantees (e.g., 1-α = 0.90)
3. **Uncertainty aggregation** identifies high-uncertainty regions (vascular branch
   points, thin vessels, boundary zones)
4. **Re-scan scheduler** uses restless multi-armed bandit formulation (Anguera Peris
   et al., 2025) with Whittle index policy to prioritise spatial locations for
   re-scanning, balancing information gain against photodamage budget
5. **PyCLM integration** sends re-scan commands to microscope via TOML-configured
   hardware abstraction layer (Oatman et al., 2025)

### 3.3 Testable Hypothesis

> Given a fixed photon budget B, the adaptive acquisition agent achieves higher
> segmentation Dice on thin vessels (diameter < 3 voxels) than uniform-dose
> scanning, by concentrating photon budget on high-uncertainty regions identified
> by conformal prediction sets.

### 3.4 Why This is Not "Agentic for the Sake of Agentic"

The core mechanism is a **feedback loop**, not an LLM. The "agent" here is a
stateful controller with a well-defined objective (minimise uncertainty subject
to photon budget), implemented as a LangGraph state machine with typed state
transitions. No natural language is needed; the value is in the
**inference-integrity feedback coupling** — a pattern where downstream inference
quality feeds back to upstream data acquisition. This feedback coupling is the
defining characteristic of agentic pipelines that justifies their complexity over
static workflows.

### 3.5 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Ye et al. (2023) | Uncertainty-driven adaptive acquisition, 16x dose reduction | Core concept |
| Ye et al. (2025) | QUTCC conformal calibration for tighter bounds | Calibration method |
| Oatman et al. (2025) | PyCLM programming-free closed-loop microscopy | Hardware integration |
| Anguera Peris et al. (2025) | Restless bandits for microscopy scheduling; 37% cumulative regret reduction in simulation, 93% more biologically relevant events captured in live imaging | Scheduling policy |
| Hinderling et al. (2025) | Smart microscopy interoperability roadmap | Standards |
| Mossina, Friedrich & Dalmau (2025) | ConSeMa conformalized segmentation margins (MICCAI 2025) | UQ method |
| Puttmann et al. (2025) | VessQC uncertainty-guided vascular curation | Domain validation |
| Pinkard et al. (2021) | Pycro-Manager: open-source microscope control (Nature Methods) | Hardware abstraction |

---

## 4. Angle 2: Vasculometric Copilot with Graph RAG

**Novelty score: MEDIUM-HIGH** — Combines graph-based retrieval (NLP/IR), vascular
morphometry (biomedical), and multi-turn consultation (HCI) in an unoccupied niche.

### 4.1 Literature Synthesis

**Medical copilot paradigm**: PathChat (Lu et al., 2024; *Nature*, 634, 466–473)
demonstrated that domain-specific vision encoders + projector + LLM achieve 78.1%
accuracy on pathology queries, with +11.4 percentage points when clinical context
is provided. This establishes the architectural blueprint: visual encoder →
projection layer → language model → multi-turn dialogue.

**Graph RAG for healthcare**: MedRAG (Xiong et al., 2024; ACL 2024 Findings)
integrates medical knowledge graphs for retrieval-augmented diagnosis. Medical
Graph RAG (Wu et al., 2024) combines knowledge graphs with LLMs for medical
question answering. PathRAG-style systems (Chen et al., 2025) apply graph pruning
to improve retrieval efficiency. None of these address 3D vascular morphometry.

**Vascular morphometry biomarkers**: Established measures include vessel diameter
distribution, tortuosity index, branching angle statistics, fractal dimension,
vessel density, and connected component analysis. These are typically computed
post-segmentation and reported as summary statistics.

**The gap**: No copilot system exists for interactive multi-turn queries about 3D
vascular morphometry. A researcher who segments a vascular network has no
conversational interface to ask: "Which vessels have tortuosity > 2 standard
deviations from the cohort mean?" or "Show me the bifurcation angles that differ
between treatment groups." This requires a domain-specific ontology graph
connecting morphometric features to anatomical context and experimental metadata.

### 4.2 Proposed Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                  Vasculometric Copilot                         │
│                                                               │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐│
│  │ Vascular   │──→│ Morphometry  │──→│ Feature Graph         ││
│  │ Segmentation│  │ Extraction   │   │ (NetworkX/neo4j-lite) ││
│  │ (DynUNet)  │   │ (VesselVio)  │   │                      ││
│  └────────────┘   └──────────────┘   └──────────┬───────────┘│
│                                                  │            │
│  ┌──────────────┐   ┌──────────────┐            │            │
│  │ LLM          │◄──│ Graph RAG    │◄───────────┘            │
│  │ (LiteLLM)    │   │ Retriever    │                         │
│  └──────┬───────┘   └──────────────┘                         │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐                                            │
│  │ Multi-turn   │                                            │
│  │ Gradio UI    │                                            │
│  └──────────────┘                                            │
└───────────────────────────────────────────────────────────────┘
```

**Components**:

1. **Morphometry extraction**: Post-segmentation feature computation (diameter,
   tortuosity, branching angles, fractal dimension) per vessel segment
2. **Feature graph**: Directed graph where nodes = vessel segments, edges =
   connectivity, node attributes = morphometric features + experiment metadata
3. **Graph RAG retriever**: Traverses feature graph to retrieve relevant vessel
   segments and their context for a natural-language query
4. **LLM reasoning**: Generates natural-language answers with statistical summaries,
   referencing specific vessel segments and morphometric values
5. **Gradio UI**: Multi-turn chat interface with 3D visualisation (VTK/PyVista)

### 4.3 Testable Hypothesis

> The vasculometric copilot enables researchers to answer morphometry queries in
> < 30 seconds that currently require manual scripting (> 10 minutes), as measured
> by a task-completion benchmark on 20 representative vascular morphometry questions.

### 4.4 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Lu et al. (2024) | PathChat multimodal copilot, *Nature* 634, 466–473 | Architectural blueprint |
| Xiong et al. (2024) | MedRAG knowledge graph retrieval, ACL 2024 Findings | Graph RAG pattern |
| Wu et al. (2024) | Medical Graph RAG | Healthcare graph retrieval |
| Chen et al. (2025) | PathRAG: graph pruning for efficient RAG | Graph retrieval efficiency |

---

## 5. Angle 3: Typed Pipeline State Graph with Clinical Contracts

**Novelty score: HIGH** — Combines machine-readable clinical evaluation standards
(regulatory), typed state machines (software engineering), and pre-completion
verification (agentic MLOps) in a way that no published work addresses.

### 5.1 Literature Synthesis

**Clinical evaluation methodology**: TRIPOD+AI (Collins et al., 2024) defines
reporting standards for AI prediction models. The clinical evaluation reproducibility
framework (Teikari, 2024, internal) argues for beyond-AUROC evaluation including
multiverse analysis, calibration curves, and subgroup analysis. These standards
exist as human-readable checklists.

**Typed pipeline state graphs**: The agentic MLOps literature identifies typed state
machines as a key pattern where pipeline failures can be localised to specific state
transitions rather than diagnosed through end-to-end debugging (cf. LangGraph
persistence and checkpointing; LangChain, 2024). A typed state graph makes the
legal transitions explicit and the failure modes enumerable.

**Pre-completion verification**: The agent evaluation literature (Anthropic, 2026;
King et al., 2026) emphasises that agentic systems need verification before
completing tasks, not just post-hoc auditing. This maps to clinical evaluation
where a model should not be promoted to production without verified compliance.

**The gap**: Clinical evaluation checklists (TRIPOD+AI, CLAIM, STARD-AI) exist
only as human-readable documents. No system encodes them as machine-readable
constraints on a pipeline state graph, enabling automated verification that a
model promotion satisfies all required evaluation criteria before deployment.

### 5.2 Proposed Architecture

```yaml
# Machine-readable clinical contract (example)
contract:
  name: "TRIPOD+AI Deployment Gate"
  required_evaluations:
    - metric: dice_score
      subgroups: [vessel_diameter_lt_3, vessel_diameter_ge_3]
      threshold: 0.65
      dataset: held_out_test
    - metric: calibration_ece
      threshold: 0.05
      requires: [conformal_prediction_calibrated]
    - metric: multiverse_range
      description: "Max Dice variance across preprocessing choices"
      threshold: 0.08
    - report: subgroup_analysis
      minimum_subgroups: 3
    - report: failure_mode_catalogue
      minimum_examples: 5
  gate_type: hard  # blocks promotion if any criterion fails
```

**Components**:

1. **Clinical contract schema**: Pydantic v2 models encoding TRIPOD+AI, CLAIM,
   and STARD-AI requirements as typed constraints
2. **Pipeline state graph**: LangGraph state machine where each node is a pipeline
   stage (preprocess → train → evaluate → calibrate → promote) with typed
   transitions carrying accumulated evidence
3. **Pre-completion verifier**: Before the `promote` transition, a verification
   agent checks all contract constraints against accumulated evidence. This is
   a deterministic check, not an LLM call.
4. **Audit trail**: Every state transition is logged to OpenLineage (Marquez)
   with the contract constraint it satisfies

### 5.3 Testable Hypothesis

> A typed pipeline state graph with clinical contracts catches significantly
> more TRIPOD+AI reporting omissions than manual review alone (target: >90%
> recall vs. <60% for manual) in a blinded comparison of N=20 model promotion
> attempts, as measured against a ground-truth checklist audit.

### 5.4 Why This is Not "Agentic for the Sake of Agentic"

The verifier is **deterministic**. It checks typed constraints against typed
evidence. No LLM is needed. The "agentic" value is in the **state graph
structure** that makes clinical evaluation requirements machine-checkable,
not in natural-language reasoning. This aligns with the finding from the agentic
resources review that "none of the top 5 most novel ideas require an LLM as
the core mechanism."

### 5.5 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Collins et al. (2024) | TRIPOD+AI reporting guidelines | Clinical standards |
| LangChain (2024) | LangGraph stateful agent persistence and checkpointing | State graph pattern |
| King et al. (2026) | Pre-completion verification for agent evals | Verification pattern |
| Anthropic (2026) | Designing AI-resistant evals | Evaluation methodology |
| Tzanis et al. (2026) | Agentic systems in radiology: regulatory and sustainability | Compliance context |

---

## 6. Angle 4: Data Quality Triage Agent

**Novelty score: MEDIUM** — Combines label quality assessment (ML), interactive
segmentation proofreading (HCI), and agentic routing (software engineering).
The individual components exist, but the end-to-end triage pipeline for 3D
vascular annotation is novel.

### 6.1 Literature Synthesis

**Label quality assessment**: Cleanlab provides confident learning for detecting
label errors (Northcutt et al., 2021). VessQC (Puttmann et al., 2025) is a napari
plugin for uncertainty-guided 3D vascular segmentation curation that improved
error detection recall from 67% to 94%.

**Interactive segmentation proofreading**: Phase 15 research established
nnInteractive (Isensee et al., 2025) as SOTA for 3D interactive segmentation,
with K-Prism (Guo et al., 2025) showing >30% click reduction when starting
from prior mask.

**Agentic data quality**: Databricks (2025) introduced data quality agents for
automated data validation in enterprise ML pipelines. The broader agentic MLOps
community has identified data quality as a key agent use case, but none of this
work addresses 3D medical image annotation specifically.

**The gap**: No system combines automatic quality scoring (Cleanlab/VessQC) with
automatic routing to the appropriate correction pathway (auto-fix vs. human
proofread vs. discard) for 3D vascular annotations.

### 6.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Data Quality Triage Agent                     │
│                                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Segmentation │───→│ Quality Score │───→│ Triage Logic │ │
│  │ + Uncertainty│    │ (Cleanlab +   │    │              │ │
│  │ Map          │    │  VessQC +     │    │ Route to:    │ │
│  └──────────────┘    │  topology)    │    │ ┌──────────┐ │ │
│                      └───────────────┘    │ │ Auto-fix  │ │ │
│                                           │ │ (morph.)  │ │ │
│                                           │ ├──────────┤ │ │
│                                           │ │ Proofread │ │ │
│                                           │ │ (Slicer)  │ │ │
│                                           │ ├──────────┤ │ │
│                                           │ │ Discard + │ │ │
│                                           │ │ re-scan   │ │ │
│                                           │ └──────────┘ │ │
│                                           └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Triage routing rules** (deterministic, not LLM):

| Quality Score | Topology Check | Route |
|--------------|---------------|-------|
| High (>0.9) | Pass | Accept as-is |
| Medium (0.7-0.9) | Pass | Auto-fix (morphological operations) |
| Medium (0.7-0.9) | Fail (disconnected) | Route to human proofreading (Slicer) |
| Low (<0.7) | Any | Discard + flag for re-acquisition |

### 6.3 Testable Hypothesis

> The triage agent reduces human proofreading time by >40% compared to reviewing
> all annotations, while maintaining >95% detection rate for topologically
> incorrect segmentations, as measured on a held-out set of 50 vascular volumes
> with known error labels.

### 6.4 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Northcutt et al. (2021) | Confident learning for label errors | Quality scoring |
| Puttmann et al. (2025) | VessQC uncertainty-guided curation | Vascular QC |
| Isensee et al. (2025) | nnInteractive: SOTA 3D interactive segmentation (CVPR 2025) | Proofreading backend |
| Guo et al. (2025) | K-Prism: prior mask reduces correction burden (NoC90 improvement) | Proofreading efficiency |

---

## 7. Angle 5: TimescaleDB Microscopy Telemetry Backend

**Novelty score: MEDIUM-HIGH** — An **unoccupied niche**. Web research found no
published implementation of time-series databases for microscopy instrument
telemetry in the biomedical imaging literature.

### 7.1 Literature Synthesis

**Time-series databases for scientific instruments**: TimescaleDB extends PostgreSQL
with hypertable partitioning, continuous aggregates, and compression for time-series
data. TigerData (Princeton Research Computing) provides a similar approach for
research data management with hierarchical metadata.

**Smart microscopy telemetry**: Hinderling et al. (2025) describe the Euro-BioImaging
Smart Microscopy Working Group (SMWG) roadmap, identifying interoperability across
instrument vendors as a critical challenge. Smart microscopy generates continuous
streams of: laser power, detector gain, stage position, temperature, humidity,
image quality metrics, and model uncertainty scores.

**Anomaly detection for instruments**: QUAL-IF-AI (automated QC for fluorescence
images) demonstrates that image quality can be monitored continuously. No system
monitors the instrument state alongside image quality in a unified telemetry
store.

**The gap**: Smart microscopy papers describe event streams but store them in ad-hoc
formats (CSV, JSON logs). No published work uses a purpose-built time-series
database for microscopy telemetry, enabling SQL-queryable anomaly detection,
continuous aggregates for drift monitoring, and correlation between instrument
state and segmentation quality.

### 7.2 Proposed Architecture

```sql
-- TimescaleDB hypertable for microscopy telemetry
CREATE TABLE microscopy_telemetry (
    time        TIMESTAMPTZ NOT NULL,
    session_id  UUID NOT NULL,
    instrument  TEXT NOT NULL,
    -- Instrument state
    laser_power_mw    DOUBLE PRECISION,
    detector_gain     DOUBLE PRECISION,
    stage_x_um        DOUBLE PRECISION,
    stage_y_um        DOUBLE PRECISION,
    stage_z_um        DOUBLE PRECISION,
    temperature_c     DOUBLE PRECISION,
    -- Image quality metrics
    snr_db            DOUBLE PRECISION,
    contrast_ratio    DOUBLE PRECISION,
    -- Segmentation quality (from model)
    mean_dice         DOUBLE PRECISION,
    mean_uncertainty  DOUBLE PRECISION,
    conformal_coverage DOUBLE PRECISION
);
SELECT create_hypertable('microscopy_telemetry', 'time');

-- Continuous aggregate: 1-minute quality summary
CREATE MATERIALIZED VIEW telemetry_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       session_id,
       AVG(mean_dice) AS avg_dice,
       AVG(mean_uncertainty) AS avg_uncertainty,
       MAX(laser_power_mw) AS max_laser_power
FROM microscopy_telemetry
GROUP BY bucket, session_id;
```

**Integration with Acquisition Agent (Angle 1)**: The telemetry backend provides
the observability layer for the adaptive acquisition loop. Conformal coverage
metrics stored in TimescaleDB enable drift detection on the acquisition agent's
calibration guarantee over time.

### 7.3 Testable Hypothesis

> TimescaleDB continuous aggregates detect laser power drift (>5% change over
> 10 minutes) within 60 seconds, enabling the acquisition agent to compensate
> before segmentation quality degrades, as measured on simulated telemetry from
> N=100 imaging sessions.

### 7.4 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Hinderling et al. (2025) | SMWG interoperability roadmap | Requirements |
| TimescaleDB (2024) | Time-series PostgreSQL extension | Implementation |
| TigerData (Princeton, 2025) | Research data management | Alternative |
| QUAL-IF-AI | Automated fluorescence QC | Quality metrics |

---

## 8. Angle 6: MCP for Laboratory Instruments

**Novelty score: MEDIUM** — The Model Context Protocol (MCP) has exploded for
software tooling but has not yet reached wet-lab instruments. Combining MCP with
the PyCLM hardware abstraction layer creates a standardised interface for LLM
agents to control microscopes.

### 8.1 Literature Synthesis

**MCP for science**: BioinfoMCP (2025) auto-generates MCP servers from
bioinformatics tool documentation. FORTHought (Adamidis, 2025) provides MCP
servers for SEM, spectroscopy, XRD, and OriginLab in physics/STEM labs. MCPmed
(2025) provides MCP servers for medical tools. These demonstrate the pattern but
none directly control multiphoton microscope hardware.

**Hardware abstraction**: PyCLM (Oatman et al., 2025) provides a programming-free
closed-loop microscopy framework with modular TOML configuration. The thread
architecture (coordinator → analysis → acquisition → hardware) can be exposed
as MCP tool endpoints.

**AILA**: Mandal et al. (2025; *Nature Communications*, 16, 9104) demonstrated an
LLM agent controlling atomic force microscopy (AFM) via function calls, achieving
autonomous experiment execution. This is the closest precedent for LLM-controlled
physical instruments, but AFM control is fundamentally different from volumetric
microscopy.

**The gap**: No MCP server exists for multiphoton microscope control. The PyCLM
thread architecture could be wrapped as MCP tools, enabling LLM agents to query
instrument state, request re-scans, and adjust acquisition parameters through a
standardised protocol.

### 8.2 Proposed MCP Tools

```json
{
  "tools": [
    {
      "name": "microscope_get_state",
      "description": "Query current microscope state (laser power, stage position, etc.)",
      "inputSchema": {
        "type": "object",
        "properties": {
          "instrument_id": {"type": "string"},
          "fields": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    {
      "name": "microscope_request_rescan",
      "description": "Request re-scan of a spatial region with specified parameters",
      "inputSchema": {
        "type": "object",
        "properties": {
          "region": {"type": "object", "properties": {
            "x_min": {"type": "number"}, "x_max": {"type": "number"},
            "y_min": {"type": "number"}, "y_max": {"type": "number"},
            "z_min": {"type": "number"}, "z_max": {"type": "number"}
          }},
          "laser_power_mw": {"type": "number"},
          "averaging": {"type": "integer"}
        }
      }
    },
    {
      "name": "microscope_get_quality_metrics",
      "description": "Retrieve image quality metrics for the latest acquisition",
      "inputSchema": {
        "type": "object",
        "properties": {
          "session_id": {"type": "string"},
          "metric_names": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  ]
}
```

### 8.3 Safety Considerations

Physical instrument control via LLM agents introduces safety concerns:

1. **Photodamage budget**: Hard constraint on total laser exposure must be enforced
   at the hardware abstraction layer, not the LLM layer
2. **Stage movement limits**: Physical limits on stage travel must be enforced
   independently of LLM decisions
3. **Human-in-the-loop**: Any irreversible operation (e.g., high-power scan on
   live tissue) requires explicit human confirmation

These constraints should be implemented as **typed guards** in the LangGraph
state machine, not as LLM prompt instructions.

### 8.4 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Mandal et al. (2025) | AILA: LLM agent for AFM, Nature Communications 16, 9104 | LLM-instrument control |
| Oatman et al. (2025) | PyCLM closed-loop microscopy framework | Hardware abstraction |
| BioinfoMCP (2025) | MCP for bioinformatics tools | MCP pattern for science |
| FORTHought (2025) | MCP for microscopy image analysis | MCP + microscopy |
| Anthropic (2025) | Model Context Protocol specification | Protocol standard |

---

## 9. Angle 7: Agentic Biostatistics with Multiverse Analysis

**Novelty score: MEDIUM** — Combines automated multiverse analysis (statistics),
clinical evaluation contracts (Angle 3), and LLM-generated statistical narratives
(agentic) for a domain where existing tools require substantial statistical expertise.

### 9.1 Literature Synthesis

**Multiverse analysis**: Steegen et al. (2016) introduced multiverse analysis —
exploring the space of reasonable analytical choices and reporting the distribution
of results. For medical imaging, this means varying preprocessing (resolution,
normalisation), model hyperparameters, evaluation metrics, and subgroup definitions.

**Beyond-AUROC evaluation**: The clinical evaluation framework (Teikari, 2024,
internal) argues that single-metric reporting (AUROC, Dice) is insufficient for
clinical claims. Required: calibration curves, subgroup analysis, failure mode
catalogues, and multiverse sensitivity.

**Agentic data science**: Recent work on LLM-driven scientific research automation
(Woodruff et al., 2026) describes Gemini-based acceleration of experimental
workflows. The agentic MLOps community has explored automated experiment analysis
but has not applied multiverse analysis to medical imaging evaluation systematically.

### 9.2 Proposed Architecture

1. **Multiverse specification**: Pydantic schema defining the space of analytical
   choices (preprocessing variants, evaluation metrics, subgroup definitions)
2. **Execution engine**: Hydra-zen multirun over the multiverse specification,
   producing a DuckDB table of results
3. **Statistical analysis**: Automated computation of result distributions,
   specification curves, and vibration-of-effects plots
4. **Narrative generation**: LLM summarises the multiverse results in clinical
   evaluation language, flagging specifications where the conclusion changes
5. **Contract verification**: Results checked against clinical contracts (Angle 3)

### 9.3 Testable Hypothesis

> Automated multiverse analysis reveals >1 preprocessing choice that reverses
> the sign of a subgroup comparison (e.g., "model A > model B for thin vessels")
> in >50% of model evaluation campaigns, as measured across N=10 evaluation
> runs with different model checkpoints.

### 9.4 Key References

| Reference | Contribution | Supports |
|-----------|-------------|----------|
| Steegen et al. (2016) | Multiverse analysis methodology | Core concept |
| Woodruff et al. (2026) | Gemini for scientific research acceleration | LLM-in-the-loop science |

---

## 10. Cross-Cutting: Inference Integrity Feedback Coupling

All seven angles share a common architectural principle: **inference integrity
feedback coupling**.
This is the pattern where downstream inference quality metrics feed back to
upstream pipeline stages:

| Feedback Path | Upstream | Downstream | Mechanism |
|--------------|----------|-----------|-----------|
| Acquisition → Segmentation | Microscope control | Segmentation quality | Re-scan high-uncertainty regions |
| Segmentation → Annotation | Model predictions | Label quality | Triage corrections to proofreading |
| Evaluation → Training | Multiverse results | Hyperparameter search | Expand search in sensitive regions |
| Calibration → Serving | Conformal coverage | Drift detection | Trigger recalibration |
| Telemetry → Acquisition | Instrument state | Acquisition parameters | Compensate for drift |

This feedback graph is **the core novelty** of treating the vascular imaging
pipeline as an agentic system: not because it uses LLMs, but because it makes
the feedback loops explicit, typed, and verifiable.

---

## 11. Novelty Assessment Matrix

| Angle | Fields Combined | Cross-cites? | Non-obvious? | Falsifiable? | Tractable? | Score |
|-------|----------------|-------------|-------------|-------------|-----------|-------|
| 1. Adaptive Acquisition | Conformal + Smart Microscopy + Vascular MLOps | No | Yes | Yes | Yes | **HIGH** |
| 2. Vasculometric Copilot | Graph RAG + Vascular Morphometry + Copilot HCI | No | Moderate | Yes | Yes | **MED-HIGH** |
| 3. Clinical Contracts | TRIPOD+AI + Typed State Machines + Pre-completion Verification | No | Yes | Yes | Yes | **HIGH** |
| 4. Data Quality Triage | Cleanlab + Interactive Seg + Agentic Routing | Partially | Moderate | Yes | Yes | **MEDIUM** |
| 5. Microscopy Telemetry | TimescaleDB + Smart Microscopy + Drift Detection | No | Yes | Yes | Yes | **MED-HIGH** |
| 6. MCP for Instruments | MCP + PyCLM + LLM Control | Partially | Moderate | Yes | Yes | **MEDIUM** |
| 7. Multiverse Biostatistics | Multiverse Analysis + Clinical Contracts + LLM Narrative | Partially | Moderate | Yes | Yes | **MEDIUM** |

**Angles 1, 3, and 5 pass the Nikola T. Markov novelty rule most strongly**: their
core ideas cannot be found in any single existing paper, and the fields they
combine do not currently cite each other.

---

## 12. Implementation Phasing

### Phase A: Foundation (Months 1-2)

1. **Clinical contract schema** (Angle 3) — Pydantic v2 models for TRIPOD+AI
2. **Pipeline state graph** (Angle 3) — LangGraph typed state machine
3. **TimescaleDB setup** (Angle 5) — Docker Compose service, hypertable schema

### Phase B: Quality Loop (Months 2-3)

4. **Data quality triage** (Angle 4) — Cleanlab + VessQC integration
5. **Proofreading routing** (Angle 4) — Integration with Phase 15 Slicer workflow
6. **Pre-completion verifier** (Angle 3) — Contract verification before promotion

### Phase C: Acquisition Intelligence (Months 3-5)

7. **Conformal calibration** — MAPIE/ConSeMa integration with DynUNet
8. **Uncertainty-driven scheduling** (Angle 1) — Restless bandit implementation
9. **PyCLM integration** (Angle 1) — Hardware abstraction layer

### Phase D: Knowledge Interface (Months 4-6)

10. **Morphometry extraction pipeline** — VesselVio or custom NetworkX graph
11. **Graph RAG retriever** (Angle 2) — Feature graph + retrieval
12. **Copilot UI** (Angle 2) — Gradio multi-turn interface

### Phase E: Observability & Narrative (Months 5-7)

13. **Telemetry pipeline** (Angle 5) — Instrument → TimescaleDB → Grafana
14. **Multiverse engine** (Angle 7) — Hydra-zen multirun → DuckDB → narrative
15. **MCP server** (Angle 6) — PyCLM → MCP tool wrapper (if hardware available)

---

## 13. PRD Integration Recommendations

### 13.1 New Decision Nodes (4 proposed)

| Node ID | Level | Title | Options |
|---------|-------|-------|---------|
| `acquisition_agent` | L3_technology | Acquisition Agent Backend | `conformal_bandit` (0.40), `uncertainty_threshold` (0.30), `fixed_schedule` (0.20), `none` (0.10) |
| `telemetry_backend` | L4_infrastructure | Microscopy Telemetry Backend | `timescaledb` (0.45), `duckdb_local` (0.30), `prometheus_only` (0.15), `none` (0.10) |
| `clinical_contract_schema` | L3_technology | Clinical Contract Schema | `pydantic_tripod` (0.45), `json_schema` (0.25), `yaml_manual` (0.20), `none` (0.10) |
| `copilot_backend` | L3_technology | Vasculometric Copilot Backend | `graph_rag` (0.35), `vector_rag` (0.30), `direct_llm` (0.20), `none` (0.15) |

### 13.2 New Edges (10 proposed)

| From | To | Influence | Rationale |
|------|-----|-----------|-----------|
| `uncertainty_quantification` | `acquisition_agent` | strong | UQ method determines acquisition agent calibration |
| `acquisition_agent` | `annotation_workflow` | moderate | Agent-acquired data feeds annotation pipeline |
| `agent_framework` | `acquisition_agent` | strong | LangGraph orchestrates acquisition loop |
| `clinical_contract_schema` | `model_promotion_strategy` | strong | Contracts gate model promotion |
| `clinical_contract_schema` | `model_governance` | strong | Contracts encode governance requirements |
| `compliance_depth` | `clinical_contract_schema` | strong | Compliance level determines contract depth |
| `telemetry_backend` | `monitoring_stack` | moderate | Telemetry feeds monitoring dashboards |
| `telemetry_backend` | `drift_detection_method` | moderate | Instrument drift detected from telemetry |
| `data_validation_tools` | `label_quality` | moderate | Validation tools inform quality assessment |
| `copilot_backend` | `llm_provider` | moderate | Copilot needs LLM for natural language |

### 13.3 Bibliography Additions (15 new entries)

New entries needed for Phase 16:

- `ye2023uncertainty` — Ye et al. (2023/2025), uncertainty-driven adaptive acquisition (*Optics Express*)
- `ye2025qutcc` — Ye et al. (2025), QUTCC conformal calibration
- `oatman2025pyclm` — Oatman et al. (2025), PyCLM closed-loop microscopy
- `angueraperis2025bandits` — Anguera Peris et al. (2025), restless bandits microscopy (arXiv:2512.14930)
- `hinderling2025smartmicroscopy` — Hinderling et al. (2025), smart microscopy survey + interoperability roadmap (*bioRxiv*)
- `lu2024pathchat` — Lu et al. (2024), PathChat multimodal copilot (*Nature* 634, 466–473)
- `xiong2024medrag` — Xiong et al. (2024), MedRAG knowledge graph retrieval (*ACL 2024 Findings*)
- `wu2024medicalgraphrag` — Wu et al. (2024), Medical Graph RAG
- `chen2025pathrag` — Chen et al. (2025), PathRAG graph pruning for efficient RAG (arXiv:2502.14902)
- `mandal2025aila` — Mandal et al. (2025), AILA LLM agent for AFM (*Nature Communications* 16, 9104)
- `steegen2016multiverse` — Steegen et al. (2016), multiverse analysis (*Perspectives on Psychological Science*)
- `northcutt2021confidentlearning` — Northcutt et al. (2021), confident learning (*JAIR*)
- `pinkard2021pycromanager` — Pinkard et al. (2021), Pycro-Manager microscope control (*Nature Methods*)
- `puttmann2025vessqc` — Puttmann et al. (2025), VessQC uncertainty-guided vascular curation (arXiv:2511.22236)
- `collins2024tripodai` — Collins et al. (2024), TRIPOD+AI reporting guidelines (*BMJ*)

### 13.4 Existing Node Updates

| Node | Change | Rationale |
|------|--------|-----------|
| `agent_framework` | Add `acquisition_control` to use cases; increase domain_applicability for vascular_segmentation from 0.75 to 0.90 | Acquisition agent is a core use case for agents in this domain |
| `uncertainty_quantification` | Add `acquisition_feedback` to description as downstream consumer of UQ | UQ now drives acquisition decisions, not just annotation triage |
| `monitoring_stack` | Add TimescaleDB as complementary telemetry source | Microscopy telemetry extends beyond standard ML monitoring |

---

## 14. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hardware access for Angle 1 | High | Blocks validation | Simulate with synthetic telemetry; use public multiphoton datasets |
| LangGraph API instability | Medium | Refactoring cost | Abstract behind interface; custom fallback (agent_framework decision) |
| Conformal prediction computational cost for real-time | Medium | Latency | Use split conformal (no re-fitting); precompute calibration sets |
| Graph RAG complexity for Angle 2 | Medium | Over-engineering | Start with vector RAG; upgrade to graph only if queries require topology |
| TimescaleDB operational overhead | Low | Infrastructure complexity | Use Docker Compose; fallback to DuckDB for local development |

---

## 15. Conclusion

The seven angles identified in this plan follow the Nikola T. Markov novelty rule
by synthesising distinct literatures — smart microscopy, conformal prediction,
graph RAG, clinical evaluation methodology, and time-series infrastructure — that
do not currently cite each other. The strongest research contributions are:

1. **Uncertainty-driven adaptive acquisition** (Angle 1): the first system to connect
   conformal prediction-calibrated uncertainty from 3D segmentation with closed-loop
   microscopy re-scanning for vascular imaging
2. **Machine-readable clinical contracts** (Angle 3): the first encoding of
   TRIPOD+AI as typed pipeline constraints with pre-completion verification
3. **Microscopy telemetry on time-series DB** (Angle 5): an unoccupied niche with
   clear engineering value

The guiding principle throughout is: **agentic where the feedback loop is the
value, not the language model**. Five of seven angles need no LLM at their core.
The remaining two (copilot, multiverse narrative) use LLMs for their irreducible
natural-language capabilities.

---

## References

- Anguera Peris, M. et al. (2025). Restless multi-armed bandits for microscopy scheduling. *arXiv:2512.14930*.
- Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction: A gentle introduction. *Found. Trends ML*, 16(4), 494–591.
- Anthropic (2026). Designing AI-resistant evals. *Anthropic Research Blog*.
- Chen, B. et al. (2025). PathRAG: Pruning graph-based retrieval augmented generation with relational paths. *arXiv:2502.14902*.
- Collins, G. S. et al. (2024). TRIPOD+AI statement: Updated reporting guidelines. *BMJ*, 385, e078378.
- Guo, W. et al. (2025). K-Prism: Knowledge-based prior mask for interactive segmentation. *MICCAI 2025*.
- Hinderling, L. et al. (2025). Smart microscopy: Current implementations and a roadmap for interoperability. *bioRxiv*, August 2025.
- Isensee, F. et al. (2025). nnInteractive: Interactive 3D segmentation. *CVPR 2025*.
- King, S. et al. (2026). Automating evals with Claude and Phoenix. *Anthropic Research Blog*.
- Lakshminarayanan, B. et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
- LangChain (2024). LangGraph: Framework for stateful multi-actor agent applications. *LangChain*.
- Lu, M. Y. et al. (2024). A multimodal generative AI copilot for human pathology. *Nature*, 634, 466–473.
- Mandal, I. et al. (2025). Evaluating large language model agents for automation of atomic force microscopy. *Nature Communications*, 16, 9104.
- Mossina, L., Friedrich, C. & Dalmau, O. (2025). ConSeMa: Conformalized segmentation margins. *MICCAI 2025*.
- Northcutt, C. G. et al. (2021). Confident learning: Estimating uncertainty in dataset labels. *JAIR*, 70, 1373–1411.
- Oatman, S. et al. (2025). PyCLM: Programming-free closed-loop microscopy. *Princeton*.
- Pinkard, H. et al. (2021). Pycro-Manager: Open-source software for customized and reproducible microscope control. *Nature Methods*, 18, 226–228.
- Puttmann, S. et al. (2025). Bridging 3D deep learning and curation for analysis and high-quality segmentation in practice. *arXiv:2511.22236*.
- Steegen, S. et al. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*, 11(5), 702–712.
- Teikari, P. (2024). Multiphoton vasculature segmentation pipeline. *Slides/Internal*.
- Teikari, P. et al. (2019). Embedded deep learning in ophthalmology. *PMC6425531*.
- Tzanis, I. et al. (2026). Agentic systems in radiology: Principles, opportunities, privacy risks, regulation, and sustainability concerns. *Diagnostic and Interventional Imaging*.
- Woodruff, D. P. et al. (2026). Accelerating scientific research with Gemini. *arXiv:2602.03837*.
- Wu, P. et al. (2024). Medical Graph RAG: Knowledge graph-enhanced LLM for healthcare QA. *Preprint*.
- Xiong, G. et al. (2024). MedRAG: Benchmarking retrieval-augmented generation for medicine. *ACL 2024 Findings*.
- Ye, Z. et al. (2023). Uncertainty-driven adaptive acquisition for microscopy. *arXiv:2310.16102*; published *Optics Express*, 33(6), 2025.
- Ye, Z. et al. (2025). QUTCC: Quantile uncertainty training with conformal calibration. *Preprint*.
