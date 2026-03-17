# NEUROVEX Literature Research Report — v3.0 (Domain Reports Integrated)

**Target journal:** Nature Protocols
**Framing:** Reproducible protocol for deploying multi-model 3D segmentation pipelines
**Generated:** 2026-03-17 (v3.0 — 17 domain reports woven in, 3 reviewer passes preserved)
**Purpose:** Comprehensive fuel for Introduction + Discussion .tex sections. Distill key insights when writing .tex.

**v2.0 → v3.0 changes:** Woven in insights from 17 domain research reports covering topology-aware segmentation, conformal prediction, ensemble methods, monitoring, data engineering, security, regulatory compliance, interactive segmentation, synthetic data, agentic systems, loss functions, and metrics. Citation count target raised from 85-110 to 120-150.

---

## 1. Introduction Themes

### 1.1 The Reproducibility Crisis in Biomedical Machine Learning

The replication crisis extends beyond psychology into computational biomedicine. Baker (2016, *Nature*) surveyed 1,576 researchers: 70% failed to reproduce others' experiments. Zhao (2026) identifies systemic forces — infrastructure fragmentation, undocumented preprocessing, hardware-dependent numerics — that make replication structurally difficult rather than merely culturally neglected.

Medical imaging compounds this. Jeon et al. (2025, *Investigative Radiology*) argue that standardization requires shared *execution environments*. Renton et al. (2024) demonstrate this with Neurodesk. Lee et al. (2025, *bioRxiv*) propose Spyglass. Gillon et al. (2025) survey open neurophysiology infrastructure.

**The FDA transparency gap amplifies the crisis:** Babic et al. (2025) report that 93.3% of FDA-authorized AI/ML medical devices do not disclose their training data sources. Seyferth et al. (2024) propose the METRIC framework (15 dimensions) for evaluating ML model transparency. Without standardized MLOps, each lab's reproducibility is a local achievement, not a community asset.

**Gap:** No existing protocol provides end-to-end reproducible 3D biomedical segmentation integrating data versioning, containerized training, multi-model evaluation, uncertainty quantification, and deployment — all replicable across heterogeneous cloud environments.

### 1.2 MLOps: From Industry to Biomedical Research

MLOps emerged to address "technical debt" in ML systems (Sculley et al. 2015). Kreuzberger et al. (2023, *IEEE Access*) formalize the architecture with maturity levels: Level 0 (manual) through Level 2 (CI/CD for ML). Amershi et al. (2019, *ICSE*) provide the foundational software engineering perspective. Zarour et al. (2025, *IST*) contribute the most comprehensive SLR. Paleyes et al. (2022, *ACM Computing Surveys*) survey 147 deployment case studies.

**Recent empirical findings strengthen the case:**
- Leest et al. (2025a,b) find 77% of ML teams have no dedicated or only custom monitoring solutions.
- Matthew (2025) demonstrates that ML Canvas strategic planning (beta=0.428, p<0.001) matters more for deployment success than tooling selection.
- Leest et al. (2025b) introduce Causal System Maps with 6 attribution patterns for drift root-cause analysis.
- Zampetti et al. (2026) and Biswas et al. (2026) characterize emerging MLOps practice patterns.

**Biomedical MLOps adoption remains nascent:**
- Cheimarios (2025): MLOps essential but rarely practiced in scientific computing.
- Testi et al. (2026, FetalMLOps): first domain-specific MLOps for fetal ultrasound.
- Krishnan et al. (2025, *JAMA Network Open*, CyclOps): clinical data shift detection.
- Sitcheu et al. (2023): MLOps for microscopy (classification only, no 3D segmentation).
- Angelo et al. (2025): hexagonal architecture (ports+adapters) for ML microservices.

**The maturity gap:** Most academic labs operate at Kreuzberger Level 0. This protocol enables a direct transition to Level 2 for any lab with Docker and a cloud account.

### 1.3 The MONAI Ecosystem and Its Missing MLOps Layer

MONAI (Cardoso et al. 2022) provides transforms, networks, losses, metrics. VISTA-3D extends to general 3D segmentation. However, MONAI does not prescribe orchestration, experiment tracking (MLflow; Zaharia et al. 2018), data versioning (DVC), or deployment. This protocol fills this gap with the ModelAdapter pattern + full lifecycle infrastructure.

**Distinction from nnU-Net:** Isensee et al. (2021, *Nature Methods*) self-configure a fixed architecture. This protocol makes the *model itself* configurable — DynUNet, SAM3, VesselFM, or MambaVesselNet via a single YAML field.

### 1.4 Vascular Segmentation: Biology and Computational Challenges

Cerebrovascular connectivity determines blood flow and neural metabolic support (Iadecola 2017, *Neuron*). The cortical vascular network has noncolumnar flow patterns (Blinder et al. 2013, *Nature Neuroscience*). Sweeney et al. (2018, *Nature Neuroscience*) link vascular dysfunction to neurodegeneration. Falcetta et al. (2026) provide broader cerebrovascular imaging context.

**Dataset:** MiniVess (Poon & Teikari 2023, *Scientific Data*): 70 rat cerebrovasculature volumes, 2-photon microscopy. External: DeepVess, TubeNet, VesselNN.

**Topology-preserving segmentation (from topology report):**
The key insight from our topology literature survey is that **representation changes outperform loss function changes** for topology preservation. Five competing approaches exist:
1. **Loss-only** (clDice, Shit et al. 2021; CbDice+clDice — our baseline: clDice=0.906)
2. **Continuous representation** — SDF + centerline dual heads (Zhao et al. 2025)
3. **Graph-based output** — predict topology graphs directly
4. **Foundation model adaptation** — SAM3 perception encoder for vessel features
5. **Topological regularization** — persistent homology constraints (Stucki et al. 2023, 2024)

**Loss function landscape (from loss report):**
- 13 losses in factory. Standard Dice+CE: DSC=0.824, clDice=0.832. CbDice+clDice: DSC=0.772, clDice=0.906 (+8.9% topology).
- Topology-accuracy tradeoff expected (textbook). Pure clDice risky — empty prediction collapse without Dice anchor.
- Boundary Loss (Kervadec et al. 2019) best patch-safe proxy for Hausdorff-like metric.
- Skeleton losses (Liu et al. 2026, Skea-Topo; Kirchhoff et al. 2024) refine topological evaluation.

**Metrics (from metrics report):**
No single metric covers topology + morphology + geometry. Three error families must be reported separately (Maier-Hein et al. 2024, Metrics Reloaded):
- **Topology:** Betti numbers (Stucki et al. 2023), persistent homology matching
- **Morphology:** clDice (centerline), cbDice (class-balanced)
- **Geometry:** DSC, NSD, HD95, MASD
Rank-then-aggregate for champion selection. Betti matching > naive counting.

### 1.5 Foundation Models in 3D Medical Imaging

SAM3 (Meta, Nov 2025) — 648M params, ViT-32L, requires BF16. Three research frontiers converge (from SAM3 report): SAM evolution, topology-preserving losses, and conformal prediction for coverage guarantees. The intersection is underexplored.

**Interactive segmentation (from interactive report):**
nnInteractive (Isensee et al. 2025) won CVPR 2025 1st place. SlicerNNInteractive enables client-server architecture (GPU on server, thin client). K-Prism proofreading paradigm: >30% click reduction from prior mask. **Critical finding:** all interactive methods degrade on small structures (~27% Dice on organs-at-risk). Topology matters 5-15x more than architecture choice for tubular structures.

### 1.6 Uncertainty Quantification for Clinical Deployment

**(New section — from conformal prediction report)**

Conformal prediction (CP) provides distribution-free, finite-sample coverage guarantees — essential for clinical deployment. Two CP strategies apply to vascular segmentation:
- **Morphological CP (ConSeMa):** dilation/erosion inner/outer contours. Naturally suited to vascular topology where boundary uncertainty varies along vessel diameter.
- **Distance-transform CP (CLS):** controls false negative rate clinically. Angelopoulos et al. (2022) establish the theoretical foundation.
- Mossina & Friedrich (2025), Tan et al. (2025), Bereska et al. (2025), Gaillochet et al. (2026) advance CP for segmentation.

**Library gap:** No production-quality CP library exists for segmentation. All implementations are bespoke. MAPIE 2026 roadmap includes image segmentation.

---

## 2. Methods Themes

### 2.1 Platform Architecture: Docker-Per-Flow Isolation

Each Prefect flow runs in its own Docker container. Flows communicate through MLflow artifacts only.

**Security hardening (from Docker security report):**
- Base image hierarchy: distroless > alpine > Ubuntu.
- Multistage Dockerfile (builder → runner). Non-root user enforcement.
- SecMLOps PTPGC framework (Zhang et al. 2026): 8 ML-specific security roles.
- STRIDE threat modeling for ML pipelines: data poisoning, membership inference, adversarial examples.
- CVE tracking via Trivy scans on all images.

### 2.2 Cloud Compute: SkyPilot as Intercloud Broker

SkyPilot (Yang et al. 2023, *NSDI*) = "Slurm for multi-cloud." Docker image_id mandatory.

**FinOps (from SkyPilot/FinOps report):**
- Spot recovery automatic (SkyPilot managed jobs).
- Cost tracking per-job via Ralph Loop JSONL.
- Controller idle cost optimization ($0.17/hr RunPod).
- Two-provider architecture: RunPod (dev, Docker Hub images) + GCP (staging/prod, GAR images).

### 2.3 Configuration-Driven Extensibility

Dual config: Hydra-zen (experiments) + Dynaconf (deployment). Lab-level + user-level overrides.

### 2.4 Multi-Model Evaluation Protocol

ModelAdapter ABC + Metrics Reloaded alignment + bootstrap CIs.

**Ensemble methods (from ensemble report):**
- M=5 sufficient for deep ensembles; strong diminishing returns beyond (Lakshminarayanan et al. 2017).
- Standard bootstrap of training data hurts ensemble diversity (Nixon et al. 2020).
- Loss-conditioned ensembles validated: 2-7% DSC improvement (Li et al. 2025).
- Model Soups (Wortsman et al. 2022): weight-space averaging post-training.
- SWA/SWAG (Izmailov et al. 2018; Maddox et al. 2019): single-model uncertainty.
- Evaluation bootstrap (B=10k) for CIs is computationally free and standard.

### 2.5 Data Engineering and Quality

**(New section — from data engineering report)**

- DVC for data versioning. 12-layer "validation onion" (Pydantic → Pandera → Great Expectations → whylogs → OpenLineage).
- FHIR-MEDS-OWL-CONNECTED vertical stack for clinical interoperability (Marfoglia et al. 2025; Marfoglia et al. 2025, 2026).
- Lakehouse medallion architecture (Bronze/Silver/Gold) for FAIR compliance.
- **Critical gap identified:** No harmonization methods exist specifically for vascular MRI (from arXiv:2507.16962v2 survey).

### 2.6 Monitoring and Observability

**(Expanded — from monitoring report)**

- 17 monitoring practices on 5-phase quality cycle (Leest et al. 2025).
- ML System Maps with 3 views: ML/Subsystem/Environment (Protschky et al. 2025).
- Drift detection: Evidently DataDriftPreset + kernel MMD + whylogs profiling.
- AgentOps CHANGE framework for agentic system monitoring (Biswas et al. 2026).

### 2.7 Experiment Tracking and Reproducibility

**(Expanded — from MLflow report)**

MLflow (Zaharia et al. 2018) with DuckDB analytics. Identified logging gaps in v2: architecture metadata, environment fingerprint, hardware state, training hyperparameters, volume IDs, commit hashes, run lifecycle events, cross-flow links. Protocol addresses these systematically.

### 2.8 Agentic Development Methodology

*(Methodology, not contribution — per R3 v2.0 correction)*

SDD with Claude Code. Living Specification Graph: 52 decision nodes with confidence labels, domain overlays, intent-expression fields. Metalearning persistence (15+ failure analyses).

**Data science agent context (from agents report):**
- L0-L5 data agent autonomy taxonomy (Luo Y. et al. 2026). NEUROVEX currently L2, targeting L3.
- TissueLab (2025): LLM orchestrator + local tool factories + active learning for microscopy.
- >90% surveyed agents lack trust/safety mechanisms (Rahman et al. 2025).
- DSGym shortcut phenomenon: 40.5% accuracy without data access (Nie et al. 2026).
- Convergence toward hybrid LLM-ML for executable programs (Zhou et al. 2026).

**Comparable agent systems:** Wang et al. (2024, *Frontiers CS*) — LLM agent survey. Qian et al. (2024, *ACL*) — ChatDev. Jimenez et al. (2024, *ICLR*) — SWE-bench. Piskala (2026) — SDD taxonomy.

---

## 3. Discussion Themes

### 3.1 MLOps Maturity as Cultural Change

Kreuzberger Level 0 → Level 2 mapping. Protocol demonstrates that Google-level MLOps maturity is within reach of academic labs using open-source tools.

### 3.2 Regulatory and Compliance Implications

**(New section — from regulatory report)**

EU AI Act (high-risk classification effective Aug 2, 2026). Compliance costs: EUR 7.5k-400k per system. Gemini 2.5 Pro achieves kappa=0.863 on compliance assessment (AIReg-Bench: Marino et al. 2025 (AIReg-Bench)).

- 10-step aerospace certification pipeline applicable to medical imaging (Lacasa et al. 2025).
- AIReg-Bench: benchmarking LLMs for AI regulation compliance (Marino et al. 2025).
- TechOps: Technical Documentation Templates for the AI Act (Lucaj et al. 2025, AAAI/ACM AIES).
- Protocol's Docker isolation, MLflow tracking, and DVC versioning provide the audit trail that regulatory frameworks require.

### 3.3 Topology-Preserving Evaluation as Platform Capability

CbDice+clDice result (clDice=0.906) is a platform capability demonstration. The 5 competing hypotheses from the topology report provide a research roadmap that the platform uniquely enables.

### 3.4 Foundation Model Democratization

SAM3/VesselFM deployment friction wrapped in YAML config. Interactive segmentation (nnInteractive) as future protocol extension.

### 3.5 Synthetic Data and Drift Simulation

**(New — from synthetic data report)**

- Drift taxonomy: covariate, prior probability, concept drift.
- Synthetic data generation for controlled drift experiments.
- Drift detection suite: Evidently + Alibi-Detect + whylogs.
- Data quality confidence gates agent autonomy (Zamzmi et al., Schwabe et al.).

### 3.6 Limitations and Honest Assessment

- Single primary dataset (MiniVess)
- VesselFM data leakage (pre-trained on MiniVess)
- No external replication yet
- Agentic metrics self-reported
- cuDNN non-determinism
- No CP library for segmentation (all implementations bespoke)
- Small structure degradation (~27% Dice) universal across interactive methods
- Regulatory compliance not yet validated end-to-end

---

## 4. Comparable Published Work (Expanded)

| Paper | Journal | Year | Similarity | Differentiation |
|-------|---------|------|-----------|-----------------|
| nnU-Net (Isensee et al.) | Nature Methods | 2021 | Self-configuring segmentation | Single architecture, no multi-model, no MLOps lifecycle |
| CellPose 2.0 | Nature Methods | 2022 | Segmentation platform | Cell-focused, 2D, GUI-centric, no topology |
| NeuroCAAS (Abe et al.) | Neuron | 2022 | Cloud neuroscience | AWS-only, no experiment tracking |
| Sitcheu et al. | --- | 2023 | MLOps for microscopy | Classification only, single cloud |
| Windhager et al. | Nature Protocols | 2023 | End-to-end workflow | Tissue imaging, not multi-model |
| TotalSegmentator | Radiology AI | 2023 | 3D organ segmentation | Pre-trained nnU-Net, not configurable platform |
| nnInteractive (Isensee et al.) | CVPR | 2025 | Interactive segmentation | Tool, not MLOps platform |
| FetalMLOps (Testi et al.) | --- | 2026 | MLOps for medical imaging | Ultrasound classification |
| CyclOps (Krishnan et al.) | JAMA Network Open | 2025 | MLOps for health | Clinical deployment, not research protocol |

---

## 5. Manuscript Section Mapping (Updated v3.0)

| Section | Key Themes | Must-Cite (min) | Length |
|---------|-----------|-----------------|--------|
| **Introduction** | Reproducibility, MLOps, MONAI, vascular biology, foundation models, UQ | 35-40 refs | 3 pp |
| **Equipment** | Hardware, software, accounts, costs | 3-5 refs | 0.5 pp |
| **Procedure** | Numbered protocol steps | 5-8 refs | 2-3 pp |
| **Methods: Platform** | Docker, SkyPilot, Prefect, MLflow, DVC, security | 12-15 refs | 2 pp |
| **Methods: Models** | Adapters, losses, metrics, topology, ensembles | 15-18 refs | 2 pp |
| **Methods: Data** | Engineering, quality, drift, synthetic | 8-10 refs | 1 pp |
| **Methods: Agentic** | SDD, LSG, metalearning, agent taxonomy | 8-10 refs | 1 pp |
| **Troubleshooting** | 20-30 known issues + solutions | 0 refs | 1-2 pp |
| **Anticipated Results** | Expected metrics, example figures | 3-5 refs | 1 pp |
| **Discussion** | Culture change, regulatory, topology, foundation models, limitations | 20-25 refs | 3 pp |
| **Total** | | ~120-150 refs | ~17-22 pp |

---

## 6. Reviewer Feedback (Preserved from v2.0)

All R1/R2/R3 corrections from v2.0 are preserved. v3.0 additions address the gaps identified by self-reflection against the 17 domain reports, specifically: topology depth, conformal prediction, ensemble methods, monitoring practices, data engineering, security hardening, regulatory compliance, interactive segmentation, and synthetic data/drift.
