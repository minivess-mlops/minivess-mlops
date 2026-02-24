# Phase 19: MLOps Practices for Vascular Imaging Pipelines

> **Date**: 2026-02-24
> **PRD Version**: 1.8.0 → 1.9.0
> **Seed Papers**: 19 (MLOps cluster from vascular-tmp)
> **Web Research**: 7 topic searches (maturity models, experiment tracking, CI/CD for ML, model registry governance, observability, SecMLOps, small-dataset MLOps)

---

## Executive Summary

This report synthesises 19 seed papers and post-January 2025 web research on MLOps practices for biomedical imaging pipelines. The MLOps field is notably under-represented in academic venues — much of the actionable knowledge comes from industry blog posts, tool documentation, and experience reports rather than peer-reviewed publications. Three structural findings dominate: (1) **77% of ML teams report using no dedicated monitoring solution or only custom-built/in-house monitoring** (Leest et al., 2025a), (2) **47% of ML projects never leave the prototype stage** (Panawala, 2025), and (3) **>70% of researchers cannot reproduce others' ML work** (Matthew, 2025). For vascular segmentation MLOps, five actionable angles emerge, the strongest being the **Causal System Maps for Distribution Shift Attribution** pattern (Leest et al., 2025b) which provides a structured methodology for diagnosing why segmentation performance degrades across imaging sites.

---

## 1. Seed Paper Synthesis

### 1.1 Architecture Evolution and Production Readiness

1. **Angelo et al. (2025)** — "Making a Pipeline Production-Ready: Challenges and Lessons Learned in the Healthcare Domain." arXiv:2506.06946v3. Documents the SPIRA project (respiratory insufficiency detection via speech analysis, deployed during COVID-19). Three architecture versions: v1 "Big Ball of Mud" (tightly coupled), v2 Modular Monolith (Strategy/Factory patterns), v3 Test-Driven Microservices (full CI/CD). Key lesson: investing in Ports and Adapters pattern pays off in maintainability and team velocity for healthcare ML.

2. **Cordeiro et al. (2025)** — "Reusability in MLOps: Leveraging Ports and Adapters to Build a Microservices Architecture for the Maritime Domain." arXiv:2512.08657v1. Ocean Guard system: a microservices architecture with components for data ingestion, processing, loading, anomaly detection, and API serving, sharing reusable ports and adapters via monorepo. Hexagonal Architecture enables swapping data sources, ML models, and storage backends without modifying business logic. Directly applicable decomposition for segmentation service design.

3. **Wu et al. (2020)** — "DDeep3M: Docker-powered deep learning for biomedical image segmentation." J. Neuroscience Methods 342. Achieves Dice >0.96 for vessel and somata segmentation across three spatial scales (MOST microscale, optical mesoscale, BraTS19 MRI macroscale). Docker containerization eliminates environment configuration issues. Early example of containerised biomedical segmentation predating formal MLOps terminology.

### 1.2 Monitoring, Observability, and Drift Attribution

4. **Leest et al. (2025a)** — "Monitoring and Observability of Machine Learning Systems: Current Practices and Gaps." arXiv:2510.24142v1. 7 focus groups, 37 participants from banking, fintech, e-commerce, industrial technology, and related sectors. **77% report using no dedicated monitoring solution or only custom-built/in-house monitoring** for ML systems. Feature drift and prediction quality monitoring are the most desired but least implemented capabilities. Proposes a descriptive model organised along system context and information type dimensions.

5. **Leest et al. (2025b)** — "Tracing Distribution Shifts with Causal System Maps." arXiv:2510.23528v1. Introduces ML System Maps: causal diagrams with three layered views for attributing distribution shifts. Three Attribution Questions: AQ1-Route (which pipeline path?), AQ2-Localize (which subsystem component?), AQ3-Externalize (which environmental factor?). Six attribution patterns (AP1.1–AP3.2) for systematic root-cause analysis. Demonstrated on two industrial case studies.

6. **Protschky et al. (2025)** — "What Gets Measured Gets Improved: Monitoring ML in Production." IEEE Access 13, 18693–18714. Identifies **17 monitoring practices** arranged on a quality management cycle with 5 phases (define, measure, assess, act, control) plus 3 cross-sectional practices. Derived from 10 interviews, 81-source multivocal literature review, and analysis of 15 monitoring tools. 385 codes organised into 31 categories.

### 1.3 MLOps Maturity, Process, and Strategy

7. **Bade (2026)** — "From Experimentation To Enterprise Reality: Why MLOps Is The Backbone Of Production AI." J. International Crisis and Risk Communication Research 9(1). Enterprise MLOps lifecycle covering data management, feature engineering, model training, deployment strategies (canary, blue-green, shadow mode), and continuous monitoring. Traces maturity evolution from ad-hoc to self-healing pipelines. Emphasises model drift (concept drift, covariate shift, data drift) as primary production threat.

8. **Panawala (2025)** — "From DevOps to MLOps: A Case Study on Adapting Continuous Software Engineering for ML Operationalization." MSc Thesis, Blekinge Institute of Technology. **47% of ML projects remain at prototype stage** and never reach production. Existing CI/CD supports deployment and artifact management but lacks data versioning, feature stores, and experiment tracking. Proposes reusable MLOps blueprint for incremental adoption (Telenor OSS division case study).

9. **Prause (2026)** — "The Machine Learning Canvas: Empirical Findings on Why Strategy Matters More Than AI Code Generation." IEEE format preprint (submitted January 2026). SEM survey of 150 data scientists identifies 4 latent success factors: Strategy, Process, Ecosystem, Support. Support→Strategy path: beta=0.432 (p<0.001). Strategy→Process: beta=0.428 (p<0.001). Process→Ecosystem: beta=0.547 (p<0.001). **Strategy (not technical tooling) is the strongest predictor of ML project success.** Over 80% of AI projects fail to deliver real business value (Ryseff et al., 2024, as cited by Prause).

10. **Matthew (2025)** — "Model Versioning and Reproducibility Challenges in Large-Scale ML Projects." Reports **>70% of researchers fail to reproduce others' ML work; >50% cannot reproduce their own**. Three pillars of ML versioning: code (Git), data (DVC, Delta Lake, lakeFS), model (MLflow Registry, W&B). Advocates for ML-specific semantic versioning (major=architecture, minor=retraining, patch=hyperparameters).

### 1.4 MLOps Frameworks and Empirical Characterisation

11. **Wozniak et al. (2025)** — "MLOps Components, Tools, Process, and Metrics: A Systematic Literature Review." IEEE Access 13. SLR of 41 publications from 2615 initial results. Key tool classes by frequency: Model Repository (22 papers), Model Orchestrator (20), CI/CD (19), Feature Store (12), Model Monitoring (11). Most mentioned tools: Kubeflow, MLflow, Docker, Kubernetes, Jenkins. **No established metrics exist for evaluating MLOps implementation effectiveness.**

12. **Zampetti et al. (2026)** — "How are MLOps Frameworks Used in Open Source Projects?" MSR 2026. Analyses 8 frameworks (BentoML, Deepchecks, Evidently, Kedro, Metaflow, MLflow, Prefect, W&B) across 969 GitHub repositories. MLflow and W&B are the most widely adopted frameworks across the studied repositories. **Frameworks are rarely used out-of-the-box; developers use APIs for custom workflows.** Only 24 projects integrate MLOps frameworks into CI/CD pipelines. Taxonomy of 14 categories, 37 sub-categories of feature requests.

### 1.5 AgentOps and Emerging Paradigms

13. **Biswas et al. (2026)** — "Architecting AgentOps Needs CHANGE." arXiv:2601.06456v1. CHANGE framework for operationalising agentic AI: Contextualise, Harmonise, Anticipate, Negotiate, Generate, Evolve. Classical ML models are stateless inference endpoints; agentic systems maintain state, invoke tools, chain reasoning — requiring fundamentally different operational patterns. Unique challenges: emergent behaviour monitoring, multi-agent coordination, tool-use governance, runtime safety boundaries.

14. **Warnett et al. (2026)** — "MLOps Pipeline Generation for Reinforcement Learning: A Low-Code Approach Using Large Language Models." University of Vienna. LLM-based automated RL pipeline generation using Pipes and Filters pattern. GPT-4o achieves 0.0 error rate after iterative prompt refinement (initial average 0.187 across 7 LLMs). Demonstrates LLMs can generate functional MLOps configurations from high-level specifications.

### 1.6 Security in MLOps

15. **Zhang et al. (2026)** — "SecMLOps: A Comprehensive Framework for Integrating Security Throughout the Machine Learning Operations Lifecycle." Empirical Software Engineering 31, 74. arXiv:2601.10848v1. Carleton/Polytechnique Montreal. PTPGC model (People, Technology, Processes, Governance, Compliance). **8 specialised security roles (R1–R8)** for ML operations. STRIDE threat analysis applied to MLOps pipelines. Attack vectors: data poisoning, adversarial examples, model extraction, membership inference. Recommended technologies: AWS KMS, TensorFlow Privacy, PySyft, SonarQube, OWASP ZAP.

### 1.7 Domain-Specific and Standards-Based Approaches

16. **Seuru et al. (2026)** — "The ODE (Overview, Data, and Execution) protocol for a standardised use of machine learning." Environmental Modelling & Software 198. Structured checklist inspired by ODD protocol for agent-based models. Three sections: Overview, Data, Execution. Aligns with FAIR principles and CRISP-DM lifecycle. Demonstrated on flood susceptibility mapping.

17. **Mira et al. (2025)** — "A Model-Driven Engineering Approach to AI-Powered Healthcare Platforms." Under review, MDPI Informatics. MILA Domain-Specific Language for declarative ML pipeline specification. Incorporates federated learning for privacy-preserving training. Uses HL7 FHIR for semantic interoperability. QUALITOP multi-centre validation across 4 European clinical centres.

18. **Khadem & Movaghar (2025)** — "From challenges to metrics: An LLM-driven DevOps recommendation system grounded in evidence-based mappings." Array 28, 100547. Systematic mapping of 378 studies into structured tuples: Challenge→Success Factor→Process→Metric. 294 challenges, 286 success factors, 35 processes, 83 metrics. Cohen's kappa = 0.82.

19. **Husseini (2026)** — "Inventory Operations (InvOps): Bringing CI/CD Principles to Inventory Management." MSc Thesis, American University of Beirut. CI/CD-inspired continuous retraining loop with Human-Stakeholder-in-the-Loop architecture. Tangential domain (supply chain) but transferable concept: treating model configurations as policies subject to CI/CD cycles.

---

## 2. Web Research: Post-January 2025 Literature

### 2.1 MLOps Maturity Models

**JMIR Healthcare MLOps Maturity (2025)** — 3-stage framework (low, partial, full); only 13/19 studied systems reached full maturity. MLOps pipelines "not well researched in medical settings."

**Zarour et al. (2025)** — SLR identifying 9 best practices, 8 challenges, 5 maturity models from 45 articles. Tertiary analysis found 33 reviews encompassing 1,397 primary studies on MLOps.

**Chalmers Empirical Guide (2025)** — 5-dimensional framework from 14-company case study. Most healthcare implementations fall under DataOps or Manual MLOps stages.

### 2.2 Experiment Tracking Evolution

**MLflow 3.x** — Architecture overhaul with `LoggedModel` as first-class entity. MLflow 3.4: OpenTelemetry Metrics Export (span-level statistics as OTel metrics), MCP server integration. MLflow 3.6: full OTLP endpoint for ingesting OpenTelemetry traces.

**CoreWeave acquired Weights & Biases for $1.7B** (May 2025) — strategic integration of GPU cloud with experiment tracking. W&B maintains platform interoperability.

**Neptune.ai** — positioned as enterprise metadata store, can monitor thousands of per-layer metrics with no lag. Self-hosted from day one.

### 2.3 CI/CD for ML Pipelines

**CML + DVC + GitHub Actions** — CML generates visual reports in pull requests with metrics/plots. Medical imaging applications exist (brain tumour detection, pneumonia). DVC benchmarks against previously deployed models before release.

**EU AI Act compliance deadline: August 2, 2026** — AI SaMD automatically high-risk under Article 6(1). Dual compliance required: MDR + AI Act. Post-market monitoring requires real-world performance and bias-drift telemetry.

### 2.4 Model Registry Governance

**Databricks Challenger-Champion** — Initial model registered as Champion; challengers A/B tested with traffic splits (75/25 typical). Canary releases (1–10% traffic), blue/green swaps, alias flips. Promotion requires human review.

**Healthcare governance gap** — 70% of hospital leaders experienced at least one AI pilot failure. Only 4.2% median IT budget allocated to AI governance. Industry reports suggest formalising MLOps substantially reduces model time-to-production.

### 2.5 Observability for ML Systems

**OpenTelemetry + MLflow integration** — MLflow handles experiment tracking but lacks runtime observability. Bidirectional linking: OTel trace ID stored as MLflow tag, MLflow run ID stored as span attribute. Enables jumping between observability dashboard and experiment page.

**GPU Fleet FinOps** — unit economics "cost per training run" and "cost per 1k inferences" as business-centric metrics. NVIDIA DCGM + Prometheus + Grafana + Kubecost as core observability stack.

### 2.6 SecMLOps

**SecMLOps (Empirical Software Engineering, 2026)** — First systematic security paradigm for MLOps. PTPGC model, 8 security roles, STRIDE threat modelling. Phase-specific threats: data poisoning, membership inference, adversarial examples, drift exploitation.

**MITRE ATLAS framework** (Patel et al., 2025) — Systematic application across MLOps phases. Red-teaming exercises and real-world incidents as evidence.

**OpenSSF MLSecOps whitepaper** (2025) — Practical guide from Dell/Ericsson. Integrates OWASP Top 10 for ML and LLMs.

### 2.7 Lightweight MLOps for Research Labs

**MLflow local-first** — modular components integrate into existing workflows without Kubernetes. Framework-agnostic. PyTorch holds >55% production share Q3 2025.

**Metaflow (Netflix)** — supports >3,000 AI/ML projects. New `Config` object and `spin` command for notebook-like iteration. Scales from laptop to production.

**ZenML** — transforms standard Python into reproducible pipelines through minimal annotations. Develop locally, deploy anywhere. Does not require Kubernetes.

---

## 3. Five Actionable Angles for MinIVess

### Angle 1: Causal System Maps for Vascular Segmentation Drift — **NOVELTY: HIGH**

**The concept**: Adapt the ML System Maps framework (Leest et al., 2025b) specifically for vascular segmentation pipelines. Build causal diagrams with three views — ML System View (segmentation pipeline), Subsystem View (preprocessing/model/postprocessing components), Environment View (scanner parameters, contrast agents, patient demographics). Apply the six attribution patterns (AP1.1–AP3.2) when Dice score drops are detected.

**Why novel**: ML System Maps exist for banking/fintech. Applying causal attribution to **medical imaging distribution shifts** — where scanner firmware updates, contrast agent brand changes, and patient population drift create interleaved causes — is novel. The three-view decomposition maps naturally: System (end-to-end segmentation), Subsystem (MONAI preprocessing → DynUNet → postprocessing), Environment (DICOM metadata, institutional protocols).

**Testable hypothesis**: Causal system maps reduce mean time-to-root-cause for segmentation performance drops from >48 hours (current ad-hoc investigation) to <4 hours with structured attribution.

**PRD integration**: Strengthen `drift_detection_method` and `drift_response` nodes with causal attribution references.

### Angle 2: Hexagonal Architecture for Segmentation Microservices — **NOVELTY: MEDIUM-HIGH**

**The concept**: Implement the Cordeiro (2025) Hexagonal Architecture pattern for the MinIVess segmentation pipeline. Five microservices: Data Ingestor (DICOM/NIfTI sources), Data Processor (harmonisation, preprocessing), Segmentation Engine (MONAI/nnU-Net adapters), Quality Gate (validation onion), and Serving API (BentoML/ONNX). Ports define contracts; adapters enable swapping implementations.

**Why novel**: Hexagonal Architecture is well-established in software engineering but rarely applied to **medical image segmentation pipelines with regulatory constraints**. The combination of swappable model adapters (already in the MinIVess `ModelAdapter` ABC) with swappable data source adapters (PACS, DVC, cloud storage) and quality gate adapters (Pandera, Deepchecks, custom validators) creates a uniquely flexible architecture for multi-site deployment.

**Testable hypothesis**: Hexagonal Architecture reduces time to integrate a new imaging site from >2 weeks (current monolithic approach) to <3 days by swapping only the Data Ingestor adapter.

**PRD integration**: Strengthen `serving_architecture` and `containerization` nodes with hexagonal patterns.

### Angle 3: 17-Practice Monitoring Checklist for Medical Imaging — **NOVELTY: MEDIUM**

**The concept**: Map Protschky et al.'s (2025) 17 monitoring practices onto the MinIVess validation onion. For each practice, specify: which layer of the validation onion implements it, which tool provides the metric (Evidently, whylogs, Prometheus, custom), and which drift response is triggered. The 5-phase quality management cycle (define-measure-assess-act-control) becomes the operational backbone.

**Why novel**: The 17 practices exist for general ML. Mapping them onto a **4-tier data quality gate** (Phase 17's metadata→statistical→embedding→batch MMD tiers) with specific medical imaging metrics (voxel spacing drift, contrast timing deviation, vessel-to-background ratio shifts) is novel integration.

**Testable hypothesis**: Implementing all 17 monitoring practices catches >90% of segmentation degradation events within 24 hours, compared to the current manual monthly review.

**PRD integration**: Strengthen `monitoring_stack`, `drift_response`, and `data_validation_tools` nodes.

### Angle 4: SecMLOps for Clinical Segmentation Pipelines — **NOVELTY: MEDIUM**

**The concept**: Apply Zhang et al.'s (2026) SecMLOps PTPGC framework to the MinIVess pipeline. Map STRIDE threats to each pipeline phase: data poisoning risks in multi-site annotation aggregation, adversarial perturbation risks in vascular contrast images, model extraction risks in served endpoints, membership inference risks from patient imaging data. Define the 8 security roles for a small research team (consolidated).

**Why novel**: SecMLOps exists generically. Applying STRIDE analysis to **vascular MRI segmentation** — where adversarial perturbations to contrast-enhanced images could cause vessel misdetection with clinical consequences — and mapping security roles to a small academic team (where one person may hold 3–4 roles) is novel adaptation.

**Testable hypothesis**: STRIDE analysis identifies >5 previously unconsidered attack vectors specific to the vascular segmentation pipeline.

**PRD integration**: Add new `security_posture` decision node or strengthen `secrets_management` with SecMLOps references.

### Angle 5: ML Semantic Versioning for Regulatory Compliance — **NOVELTY: MEDIUM-LOW**

**The concept**: Implement Matthew's (2025) ML-specific semantic versioning (major=architecture change, minor=retraining, patch=hyperparameter tuning) combined with the ODE documentation protocol (Seuru et al., 2026) and FDA PCCP guidance. Each model version in the MLflow registry carries: semantic version, ODE protocol checklist, and PCCP change type classification.

**Why lower novelty**: Semantic versioning and documentation protocols exist separately. The integration is engineering rather than research. However, the specific combination of ML semantic versioning + ODE protocol + FDA PCCP creates a novel compliance-ready model governance pattern.

**Testable hypothesis**: ML semantic versioning + ODE protocol reduces regulatory submission preparation time by >30% through automated version classification and documentation generation.

**PRD integration**: Strengthen `model_governance` and `model_promotion_strategy` nodes.

---

## 4. PRD v1.9.0 Integration Recommendations

### 4.1 Updated Existing Nodes (5)

1. **`monitoring_stack`**: Add Protschky 17-practice reference. Add Leest monitoring gap (77% no dedicated/only custom monitoring) as motivating evidence. Strengthen Prometheus+Grafana option.

2. **`drift_detection_method`**: Add Leest causal system maps reference. Add attribution patterns (AP1.1–AP3.2) as structured methodology. Connect to drift_response.

3. **`drift_response`**: Add causal attribution as response methodology. Link to Leest AQ1-AQ3 framework.

4. **`model_governance`**: Add ML semantic versioning reference (Matthew, 2025). Add ODE protocol reference (Seuru et al., 2026). Add Prause ML Canvas (strategy > tooling) as planning framework.

5. **`serving_architecture`**: Add hexagonal architecture reference (Cordeiro et al., 2025). Add DDeep3M Docker pattern (Wu et al., 2020) as precedent.

### 4.2 New Edges (4)

1. `monitoring_stack` → `drift_detection_method` (strong): Monitoring provides input signals for drift detection methods
2. `secrets_management` → `model_governance` (moderate): Security posture constrains governance workflows for clinical deployment
3. `serving_architecture` → `drift_response` (moderate): Serving architecture determines how drift responses are executed (canary rollback, model swap)
4. `model_governance` → `documentation_generation` (moderate): ML semantic versioning drives automated documentation generation

### 4.3 New Bibliography Entries (15)

| citation_key | inline_citation | venue |
|---|---|---|
| angelo2025pipeline | Angelo et al. (2025) | arXiv:2506.06946v3 |
| cordeiro2025hexagonal | Cordeiro et al. (2025) | arXiv:2512.08657v1 |
| wu2020ddeep3m | Wu et al. (2020) | J. Neuroscience Methods 342 |
| leest2025monitoring | Leest et al. (2025) | arXiv:2510.24142v1 |
| leest2025causal | Leest et al. (2025) | arXiv:2510.23528v1 |
| protschky2025monitoring | Protschky et al. (2025) | IEEE Access 13 |
| bade2026mlops | Bade (2026) | J. Int. Crisis Risk Comm. 9(1) |
| panawala2025devops | Panawala (2025) | MSc Thesis, BTH |
| prause2026canvas | Prause (2026) | IEEE format preprint |
| matthew2025versioning | Matthew (2025) | 2025 |
| wozniak2025slr | Wozniak et al. (2025) | IEEE Access 13 |
| zampetti2026msr | Zampetti et al. (2026) | MSR 2026 |
| biswas2026agentops | Biswas et al. (2026) | arXiv:2601.06456v1 |
| seuru2026ode | Seuru et al. (2026) | Env. Modelling & Software 198 |
| zhang2026secmlops | Zhang et al. (2026) | arXiv:2601.10848v1 |

---

## 5. Key References (Verified)

1. Angelo, D. E. L. et al. (2025). Making a Pipeline Production-Ready. arXiv:2506.06946v3.
2. Cordeiro, R. et al. (2025). Reusability in MLOps: Hexagonal Architecture. arXiv:2512.08657v1.
3. Wu, X. et al. (2020). DDeep3M: Docker-powered biomedical segmentation. J. Neurosci. Methods 342.
4. Leest, J. et al. (2025a). Monitoring and Observability of ML Systems. arXiv:2510.24142v1.
5. Leest, J. et al. (2025b). Tracing Distribution Shifts with Causal System Maps. arXiv:2510.23528v1.
6. Protschky, D. et al. (2025). Monitoring ML in Production. IEEE Access 13, 18693–18714.
7. Bade, S. (2026). MLOps as Backbone of Production AI. J. Int. Crisis Risk Comm. 9(1).
8. Panawala, J. (2025). From DevOps to MLOps. MSc Thesis, Blekinge Inst. Technology.
9. Prause, M. (2026). ML Canvas: Strategy Matters More Than AI Code Generation. IEEE format preprint.
10. Matthew, B. (2025). Model Versioning and Reproducibility Challenges.
11. Wozniak, A. P. et al. (2025). MLOps Components, Tools, Process, and Metrics. IEEE Access 13.
12. Zampetti, F. et al. (2026). MLOps Frameworks in Open Source. MSR 2026.
13. Biswas, S. et al. (2026). Architecting AgentOps Needs CHANGE. arXiv:2601.06456v1.
14. Warnett, S. J. et al. (2026). MLOps Pipeline Generation for RL. University of Vienna.
15. Zhang, X. et al. (2026). SecMLOps. arXiv:2601.10848v1.
16. Seuru, S. et al. (2026). The ODE Protocol. Env. Modelling & Software 198.
17. Mira, R. et al. (2025). MDE for Healthcare AI Platforms. MDPI Informatics.
18. Khadem, E. A. & Movaghar, A. (2025). LLM-Driven DevOps Recommendations. Array 28, 100547.
19. Husseini, M. A. (2026). Inventory Operations (InvOps). MSc Thesis, AUB.

---

## 6. Cross-References to Existing PRD

| Existing Node | Connection | Evidence |
|---|---|---|
| `monitoring_stack` | 17-practice monitoring framework; 77% no dedicated/custom-only monitoring | Protschky (2025), Leest (2025a) |
| `drift_detection_method` | Causal system maps with 6 attribution patterns | Leest (2025b) |
| `drift_response` | Causal attribution as response methodology (AQ1-AQ3) | Leest (2025b) |
| `serving_architecture` | Hexagonal Architecture for segmentation microservices | Cordeiro (2025), Wu (2020) |
| `model_governance` | ML semantic versioning + ODE protocol + strategy > tooling | Matthew (2025), Seuru (2026), Prause (2026) |
| `model_promotion_strategy` | Challenger-champion with canary/blue-green patterns | Bade (2026), Databricks (2025) |
| `ci_cd_platform` | CML + DVC + GitHub Actions for medical imaging CI/CD | CML docs, Panawala (2025) |
| `experiment_tracking` | MLflow 3.x LoggedModel entity; W&B $1.7B acquisition | MLflow (2025), CoreWeave (2025) |
| `secrets_management` | SecMLOps PTPGC framework; STRIDE for ML pipelines | Zhang (2026) |
| `agent_framework` | AgentOps CHANGE framework for agentic operations | Biswas (2026) |
| `pipeline_orchestration` | LLM-generated pipeline configurations (0.0 error rate) | Warnett (2026) |
| `containerization` | DDeep3M Docker pattern; Dice >0.96 for vessel segmentation | Wu (2020) |
