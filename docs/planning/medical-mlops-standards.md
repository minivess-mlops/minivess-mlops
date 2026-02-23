# Medical MLOps Standards: From Research to Clinical Deployment

> **Phase 11 Deliverable** — This document was generated as part of Phase 11 of the
> MinIVess MLOps project. It synthesizes research into medical device software standards,
> academic foundations for regulatory-compatible development, project management tooling,
> and clinical deployment strategies for ML/AI-based Software as a Medical Device (SaMD).

> **Scope**: This report is self-contained and readable without the MinIVess PRD.
> It is intended for any team building ML-based medical device software who needs
> to understand the intersection of MLOps practices and regulatory compliance.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Regulatory Landscape](#2-regulatory-landscape)
3. [Academic Foundations](#3-academic-foundations)
4. [Project Management Tool Analysis](#4-project-management-tool-analysis)
5. [MONAI Deploy for Clinical Market Entry](#5-monai-deploy-for-clinical-market-entry)
6. [Compliance Strategy for MinIVess MLOps](#6-compliance-strategy-for-minivess-mlops)
7. [Cross-Standard Traceability Matrix](#7-cross-standard-traceability-matrix)
8. [PCCP-Ready MLOps Architecture](#8-pccp-ready-mlops-architecture)
9. [Recommendations](#9-recommendations)
10. [References](#references)

---

## 1. Executive Summary

Building machine learning systems for medical applications requires navigating a
complex web of international standards, regulatory frameworks, and quality management
requirements. The traditional medical device software development paradigm, rooted
in waterfall processes and batch documentation, is fundamentally incompatible with
the iterative, experiment-driven nature of ML development. Yet the standards themselves
do not mandate waterfall — they mandate *evidence of process*, *traceability*, and
*risk management*.

This report demonstrates that modern MLOps practices — experiment tracking, data
versioning, automated testing, CI/CD pipelines — already produce most of the evidence
required for regulatory compliance. The gap is not in the technical practices but
in the *structured capture and presentation* of that evidence.

### Key Findings

1. **IEC 62304 Edition 2 (expected September 2026)** will explicitly address
   agile/DevOps workflows and AI/ML considerations, validating the approach
   of building compliance on top of good MLOps practices.

2. **The "calm compliance" paradigm** (Granlund et al., 2022) demonstrates that
   when ML experiments are tracked in MLflow, data is versioned in DVC, and tests
   are automated with pytest, compliance evidence generation can be largely automated
   through CI/CD pipelines.

3. **Traceability consumes 20-25% of project effort** in medical software development
   (Lähteenmäki et al., 2023). Automated traceability through GitHub Issues, CI/CD
   artifacts, and compliance overlays like Ketryx can reduce this to near-zero
   incremental overhead.

4. **FDA's PCCP guidance (December 2024)** enables predetermined change control
   plans for ML models, allowing pre-specified retraining and update protocols
   without requiring new regulatory submissions — a paradigm shift for ML-based
   SaMD.

5. **A phased compliance strategy** — from research MLOps through pre-compliance
   to full clinical deployment — allows teams to build regulatory readiness
   incrementally without front-loading the cost of compliance.

### Recommended Stack

| Layer | Tool | Role |
|-------|------|------|
| Code & PM | GitHub Projects V2 | Source control, issue tracking, project management |
| Compliance | Ketryx (free tier) | IEC 62304/ISO 13485 automation overlay |
| Experiments | MLflow | Experiment tracking, model registry |
| Data | DVC | Data versioning, pipeline DAGs |
| Clinical Serving | MONAI Deploy SDK | DICOM workflows, MAP packaging |
| Research Serving | BentoML + ONNX | REST APIs, Gradio demos |
| Templates | OpenRegulatory | IEC 62304/ISO 14971 document templates |

### PRD Decision Cross-References

This report informs the following PRD decisions: `compliance_depth`,
`project_management_tool`, `regulatory_documentation_tool`,
`clinical_deployment_platform`, `serving_architecture`, `model_governance`,
`documentation_standard`, `ci_cd_platform`.

---

## 2. Regulatory Landscape

The regulatory landscape for AI/ML-based medical device software spans international
standards (IEC, ISO), regional regulations (EU MDR/IVDR, FDA), and emerging
AI-specific legislation (EU AI Act). Understanding how these frameworks interact
is essential for any team building ML systems intended for clinical use.

### 2.1 IEC 62304 — Software Lifecycle

IEC 62304:2006+A1:2015 is the foundational standard for medical device software
lifecycle processes. It applies to software that *is* a medical device (SaMD) or
software that is *embedded in* a medical device (SiMD). For ML-based systems,
IEC 62304 is the primary standard governing the development process.

#### Safety Classification

IEC 62304 defines three safety classes that determine the rigor of required
documentation and processes:

| Class | Risk | Required Activities | ML Example |
|-------|------|---------------------|------------|
| **A** | No injury or damage to health | Basic planning, requirements, testing | Research-only analysis tool |
| **B** | Non-serious injury | + Architecture, detailed design, unit testing | Triaging/screening aid |
| **C** | Death or serious injury | + All activities at maximum rigor | Autonomous diagnostic system |

> **Key principle**: Safety classification determines the *depth* of required
> documentation, not whether documentation is required. Even Class A software
> needs a development plan, requirements, and system testing.

#### Clause Structure

The standard is organized into clauses 4-9, each governing a distinct aspect of
the software lifecycle:

**Clause 5 — Software Development Process** (the core):

| Sub-clause | Activity | ML Relevance |
|------------|----------|-------------|
| 5.1 | Software development planning | Training pipeline design, experiment protocols |
| 5.2 | Software requirements analysis | Performance metrics, data requirements, input/output specs |
| 5.3 | Software architectural design | Model architecture, data flow, inference pipeline |
| 5.4 | Software detailed design | Layer specifications, loss functions, augmentation strategy |
| 5.5 | Software unit implementation & verification | Model training, unit tests for data processing |
| 5.6 | Software integration & integration testing | Pipeline integration tests, end-to-end inference |
| 5.7 | Software system testing | Clinical validation, performance on hold-out sets |
| 5.8 | Software release | Model packaging, deployment manifests, release notes |

**Clause 6 — Software Maintenance**:

Defines four maintenance types, each critical for ML systems:

- **Corrective**: Bug fixes — fixing data processing errors, inference failures
- **Adaptive**: Environment changes — new OS versions, GPU drivers, framework updates
- **Perfective**: Improvements — model retraining, architecture upgrades
- **Preventive**: Proactive maintenance — dependency updates, technical debt reduction

Martina et al. (2024) demonstrate that these four types map naturally to
DevOps issue tracking workflows in JIRA (see Section 3.1).

**Clause 7 — Risk Management**:

Integrates with ISO 14971 (see Section 2.3). For ML systems, this covers:

- Training data quality risks (mislabeled data, selection bias)
- Model uncertainty and confidence calibration
- Distribution shift between training and deployment environments
- Adversarial input robustness

**Clause 8 — Configuration Management**:

> This clause is arguably the most critical for ML systems. It requires
> unique identification of all software items, configuration status accounting,
> and change control.

For ML/AI systems, Clause 8 maps directly to:

- **Model versioning**: MLflow model registry, semantic versioning
- **Data versioning**: DVC-tracked datasets with content-addressable hashes
- **Experiment tracking**: MLflow runs with full parameter/metric logging
- **Environment reproducibility**: Docker images, dependency lock files
- **SOUP management**: Third-party library tracking (see below)

**Clause 9 — Problem Resolution**:

Requires a formal process for reporting, investigating, and resolving software
problems. In MLOps terms: issue tracking, root cause analysis, regression testing.

#### SOUP — Software of Unknown Provenance

IEC 62304 defines SOUP as software that is already developed and generally
available but was not developed under the control of the medical device
manufacturer's quality management system. For ML systems, SOUP includes:

- **ML frameworks**: PyTorch, MONAI, TensorFlow
- **Pre-trained models**: MONAI Model Zoo models, foundation models used for
  transfer learning
- **Data processing libraries**: NumPy, SciPy, scikit-image, TorchIO
- **Infrastructure tools**: Docker, ONNX Runtime, BentoML

Each SOUP item requires (per Clause 8):

1. **Identification**: Name, version, manufacturer
2. **Published specifications**: What the SOUP claims to do
3. **Risk evaluation**: What happens if the SOUP fails or behaves unexpectedly
4. **Verification**: Evidence that the SOUP behaves as specified in the context
   of the medical device

> **Practical implication**: Using a pre-trained model from MONAI Model Zoo
> for transfer learning triggers SOUP requirements. The model must be identified,
> its specifications documented, risks evaluated, and behavior verified in the
> target clinical context.

#### Edition 2 (Expected September 2026)

IEC 62304 Edition 2 represents the most significant update since the standard
was published in 2006. Key expected changes:

- **Agile/DevOps compatibility**: Explicit guidance on using iterative
  development methodologies, addressing the long-standing ambiguity
- **AI/ML considerations**: New guidance on model lifecycle management,
  training data governance, and continuous learning systems
- **Cybersecurity integration**: Alignment with IEC 81001-5-1 (see Section 2.7)
- **Modernized configuration management**: Updated guidance reflecting
  modern version control and CI/CD practices
- **Software Bill of Materials (SBOM)**: Formalized SOUP documentation
  requirements aligned with modern supply chain security practices

> **Strategic note**: Teams building ML-based medical software *now* should
> design their processes to anticipate Edition 2 requirements. The "everything
> as code" approach (Stirbu et al., 2021) naturally aligns with the expected
> modernization.

### 2.2 ISO 13485 — Quality Management System

ISO 13485:2016 specifies quality management system (QMS) requirements for
organizations involved in the design, development, production, and servicing
of medical devices. While IEC 62304 governs the *software process*, ISO 13485
governs the *organizational system* within which that process operates.

#### Key Clauses for ML/AI Teams

**Clause 4.1.6 — Validation of Tools**:

> Every tool used within the scope of the QMS must be validated for its intended
> use, with documented results.

For an MLOps pipeline, this means:

| Tool | QMS Scope | Validation Approach |
|------|-----------|---------------------|
| Git | Configuration management | Verify branching, merging, history integrity |
| GitHub Actions | CI/CD automation | Verify pipeline reproducibility, artifact integrity |
| pytest | Verification testing | Verify test execution, reporting accuracy |
| MLflow | Experiment tracking | Verify metric logging, model versioning accuracy |
| DVC | Data versioning | Verify data integrity, reproducibility |
| Docker | Environment control | Verify image reproducibility, isolation |
| BentoML / MONAI Deploy | Deployment | Verify serving correctness, inference reproducibility |

Tool validation is *risk-based*: a text editor used for documentation needs less
rigorous validation than a CI/CD system that automates release decisions.

**Clause 7.3 — Design and Development**:

Defines the design control process:

1. **Design planning** (7.3.2): Development plan with phases, reviews, responsibilities
2. **Design inputs** (7.3.3): Requirements including regulatory, functional, performance
3. **Design outputs** (7.3.4): Specifications meeting input requirements
4. **Design review** (7.3.5): Systematic examination at planned stages
5. **Design verification** (7.3.6): Confirmation that outputs meet inputs
6. **Design validation** (7.3.7): Confirmation that the device meets user needs
7. **Design transfer** (7.3.8): Transition from development to production

> **Agile compatibility**: ISO 13485 does *not* mandate waterfall development.
> Clause 7.3 requires *evidence* of planning, review, verification, and validation
> — not a specific ordering. Iterative sprints with documented reviews at each
> iteration can satisfy these requirements. The AHMED project (Lähteenmäki et al.,
> 2023) provides practical guidance on agile ISO 13485 compliance.

#### Design History File (DHF)

ISO 13485 requires a Design History File containing all design control records.
For ML systems, the DHF should include:

- Software development plan (IEC 62304 Clause 5.1)
- Requirements specifications (user needs, software requirements)
- Architecture documentation
- Risk management file (ISO 14971)
- Verification and validation records (test results, clinical evidence)
- Design review minutes
- Change records (model retraining decisions, architecture changes)
- SOUP documentation

### 2.3 ISO 14971 — Risk Management

ISO 14971:2019 specifies the process for risk management throughout the lifecycle
of a medical device. The 2019 edition introduced significant changes relevant to
ML/AI systems.

#### 2019 Edition Key Changes

- **Removed ALARP** (As Low As Reasonably Practicable): Replaced with
  explicit *benefit-risk analysis* in Clause 7.4
- **Benefit-risk analysis** (Clause 7.4): When individual risk control measures
  are insufficient, the manufacturer must weigh residual risks against clinical
  benefits — particularly relevant for ML systems where zero false negatives
  may be impossible
- **Post-production monitoring** (Clause 10): New emphasis on collecting and
  reviewing production and post-production information — directly relevant to
  ML model drift monitoring
- **Cybersecurity** (Annex G): New informative annex on cybersecurity
  considerations in risk management

#### Risk Management Process for ML/AI

The ISO 14971 process, applied to ML systems:

```
Hazard Identification
    ├── Training data hazards
    │   ├── Selection bias (non-representative populations)
    │   ├── Label errors (incorrect ground truth)
    │   ├── Data leakage (training on test data)
    │   └── Privacy violations (identifiable data in training set)
    │
    ├── Model hazards
    │   ├── Overconfident predictions (poor calibration)
    │   ├── Distribution shift sensitivity
    │   ├── Adversarial vulnerability
    │   ├── Catastrophic forgetting (after retraining)
    │   └── Unexplainable decisions (black box)
    │
    ├── Deployment hazards
    │   ├── Infrastructure failures (GPU memory, latency)
    │   ├── Version mismatch (model vs. preprocessing)
    │   ├── Data format changes (DICOM variations)
    │   └── Cybersecurity threats
    │
    └── Use hazards
        ├── Automation bias (over-reliance on AI)
        ├── Misinterpretation of uncertainty estimates
        ├── Off-label use (populations not in training data)
        └── Alert fatigue (too many false positives)
```

#### AAMI/BSI TIR 34971:2023

AAMI/BSI Technical Information Report 34971:2023 provides ML-specific risk
management guidance, bridging ISO 14971 and the unique risks of ML systems:

- **Training data risks**: Systematic framework for assessing data quality,
  representativeness, and annotation reliability
- **Model uncertainty**: Guidance on characterizing and communicating model
  confidence levels
- **Dataset shift**: Methods for detecting and managing distribution shift
  between training, validation, and deployment environments
- **Adversarial inputs**: Risk assessment for intentional and unintentional
  adversarial perturbations
- **Continuous learning**: Risk framework for systems that update after deployment

> **Practical value**: TIR 34971 is not a mandatory standard but provides the
> most concrete guidance available for ML risk management. Teams should use it
> as a checklist even during research phases.

### 2.4 MDR/IVDR — EU Market Access

The EU Medical Device Regulation (MDR, 2017/745) and In Vitro Diagnostic
Regulation (IVDR, 2017/746) govern market access for medical devices in the
European Union. They replaced the earlier Medical Device Directives (MDD/IVDD)
with significantly more stringent requirements.

#### MDR Classification for Software

**Rule 11** is the primary classification rule for standalone software:

> Software intended to provide information which is used to take decisions
> with diagnosis or therapeutic purposes is classified as class IIa, except
> if such decisions have an impact that may cause:
> - death or an irreversible deterioration of a person's state of health,
>   in which case it is classified as class III; or
> - a serious deterioration of a person's state of health or a surgical
>   intervention, in which case it is classified as class IIb.

**Practical implication for ML segmentation**: A vessel segmentation tool
used for treatment planning would likely be Class IIa minimum (informing
therapeutic decisions), potentially Class IIb if the segmentation directly
guides surgical intervention.

| MDR Class | Example ML Application | Notified Body Required |
|-----------|----------------------|----------------------|
| Class I | General wellness, non-diagnostic | No (self-declaration) |
| Class IIa | Screening/triage, decision support | Yes |
| Class IIb | Diagnosis, treatment planning | Yes |
| Class III | Life-critical decisions, implant guidance | Yes (special procedure) |

#### IVDR Classification

For in vitro diagnostic software (e.g., pathology image analysis), IVDR uses
a different classification scheme (Annex VIII):

- **Class A**: Low individual risk, low public health risk
- **Class B**: Moderate individual risk OR low public health risk
- **Class C**: High individual risk OR moderate public health risk
- **Class D**: High individual risk AND high public health risk

#### Transition Timeline

The MDR/IVDR transition has been extended multiple times:

| Deadline | Requirement |
|----------|-------------|
| May 2021 | MDR entered into force (date of application) |
| May 2024 | Extended deadline for Class III and Class IIb implantable devices |
| May 2026 | Extended deadline for Class IIb (non-implantable), IIa, and Is/Im |
| May 2028 | Extended deadline for Class I devices with measuring function |

#### CE Marking Requirements

For ML-based SaMD seeking CE marking under MDR:

1. **Quality Management System** (ISO 13485) — certified by Notified Body
2. **Technical Documentation** (Annex II) — including clinical evaluation,
   risk management, software lifecycle documentation per IEC 62304
3. **EU Declaration of Conformity** — manufacturer's formal declaration
4. **Post-Market Surveillance Plan** — including ML-specific monitoring
5. **Post-Market Clinical Follow-up** (PMCF) — ongoing clinical evidence

### 2.5 FDA SaMD — US Market Access

The US Food and Drug Administration (FDA) has been actively developing
regulatory frameworks for AI/ML-based Software as a Medical Device. The
FDA's approach is increasingly sophisticated, with multiple guidance documents
addressing different aspects of the ML lifecycle.

#### Regulatory Pathways

| Pathway | Risk Level | Timeline | Cost |
|---------|-----------|----------|------|
| **510(k)** | Moderate (with predicate) | 3-12 months | $10k-50k+ |
| **De Novo** | Low-moderate (no predicate) | 6-18 months | $50k-150k+ |
| **PMA** | High risk | 12-36 months | $250k-1M+ |

The IMDRF (International Medical Device Regulators Forum) framework defines
SaMD significance levels based on the seriousness of the healthcare condition
and the significance of the information provided by the SaMD:

| | Critical | Serious | Non-serious |
|---|---------|---------|------------|
| **Treat or diagnose** | IV | III | II |
| **Drive clinical management** | III | II | I |
| **Inform clinical management** | II | I | I |

#### PCCP — Predetermined Change Control Plan (December 2024)

The FDA's final guidance on Predetermined Change Control Plans (PCCP),
published December 2024, represents a paradigm shift for ML-based SaMD.
Previously, any change to an ML model (including retraining with new data)
could require a new regulatory submission. PCCP allows manufacturers to
pre-specify categories of changes and the protocols for validating them.

A PCCP must include three elements:

1. **Description of Modifications**: Specific types of changes the manufacturer
   plans to make (e.g., retraining with expanded dataset, hyperparameter
   optimization within defined bounds, performance tuning on new hardware)

2. **Methods for Performance Assessment**: How the manufacturer will evaluate
   whether the modified device still meets performance requirements (e.g.,
   predefined test datasets, statistical equivalence tests, clinical
   performance thresholds)

3. **Verification and Validation Protocols**: Detailed protocols for testing
   each type of modification, including acceptance criteria, test procedures,
   and documentation requirements

> **Example PCCP for vessel segmentation**: "The model may be retrained with
> additional annotated 7T TOF-MRA datasets (up to 2x current training set size)
> using the same architecture (SegResNet) and loss function. After retraining,
> the model must achieve Dice >= 0.75 on the locked test set and demonstrate
> statistical non-inferiority (p < 0.05, one-sided) compared to the cleared
> version on a predefined clinical evaluation dataset."

#### GMLP — Good Machine Learning Practice (October 2021)

FDA, Health Canada, and MHRA jointly published 10 guiding principles for
Good Machine Learning Practice:

| # | Principle | MLOps Mapping |
|---|-----------|---------------|
| 1 | Multi-disciplinary expertise | Cross-functional teams (clinical, ML, regulatory) |
| 2 | Good software engineering practices | IEC 62304, CI/CD, automated testing |
| 3 | Representative clinical study participants and datasets | Data governance, bias detection |
| 4 | Independent training and test datasets | DVC-managed data splits, test set lockout |
| 5 | Reference datasets based on best available methods | Gold-standard annotations, inter-rater reliability |
| 6 | Tailored model design to intended use | Architecture selection documentation |
| 7 | Human factors considerations and human-AI teaming | UI/UX design, clinical workflow integration |
| 8 | Testing demonstrates device performance during clinical conditions | Clinical validation, real-world evidence |
| 9 | Users provided clear, essential information | Model cards, performance characteristics |
| 10 | Deployed models monitored and managed | Drift detection, post-market surveillance |

#### TPLC — Total Product Lifecycle (Draft January 2025)

The FDA's draft guidance on Total Product Lifecycle (TPLC) for AI-enabled
Decision Support Functions (January 2025) extends the lifecycle approach:

- **Pre-market**: Training data governance, algorithm design, clinical validation
- **Post-market**: Performance monitoring, real-world evidence collection, model
  updates via PCCP
- **Continuous improvement**: Feedback loops from clinical use to model refinement

> **Status**: Draft guidance, final version expected in 2025-2026. Teams should
> monitor for finalization and begin building TPLC-aligned processes now.

### 2.6 EU AI Act — AI-Specific Regulation

The EU AI Act (Regulation 2024/1689), which entered into force on August 1, 2024,
is the world's first comprehensive AI-specific legislation. Medical devices are
classified as high-risk AI systems under Annex I, Section A, meaning they must
comply with both the AI Act and MDR/IVDR.

#### High-Risk Requirements (Articles 9-15)

**Article 9 — Risk Management System**:

A continuous iterative process throughout the AI system lifecycle:

- Risk identification and analysis
- Risk estimation and evaluation
- Risk management measures (prevention, mitigation, monitoring)
- Testing to ensure appropriate risk management
- Consideration of risks to health and safety, fundamental rights

> **Overlap with ISO 14971**: The AI Act's risk management requirements
> significantly overlap with ISO 14971 but add AI-specific dimensions
> (bias, transparency, human oversight). MDCG 2025-6 clarifies that
> MDR/IVDR conformity assessment can cover these AI Act requirements.

**Article 10 — Data and Data Governance**:

Requirements for training, validation, and testing datasets:

- **Representativeness**: Data must be sufficiently representative of the
  intended population, considering geographic, demographic, and clinical diversity
- **Bias detection and mitigation**: Systematic identification and correction
  of biases in training data
- **Annotation quality**: Appropriate data labeling with quality controls
- **Data sheets**: Documentation of dataset characteristics, collection
  methodology, and known limitations
- **Gap identification**: Assessment of data gaps and their potential impact

**Article 11 — Technical Documentation**:

Before a high-risk AI system is placed on the market:

- General description and intended purpose
- Algorithm design choices and key design decisions
- Training methodology (including data preprocessing, hyperparameter selection)
- Validation and testing procedures with metrics
- Computational resources used
- Trade-off decisions (accuracy vs. speed, precision vs. recall)

**Article 12 — Record-Keeping**:

High-risk AI systems must have logging capabilities for:

- Recording events during operation (automatic logging)
- Traceability of the AI system's functioning
- Compliance monitoring and post-market surveillance

**Article 15 — Accuracy, Robustness, Cybersecurity**:

- Appropriate levels of accuracy with declared metrics
- Robustness to errors, faults, and inconsistencies
- Resilience to adversarial manipulation
- Cybersecurity measures proportionate to risk

#### Enforcement Timeline

| Date | Milestone |
|------|-----------|
| August 1, 2024 | AI Act entered into force |
| February 2, 2025 | Prohibited practices apply |
| August 2, 2025 | Provisions on notified bodies and governance apply |
| August 2, 2026 | General-purpose AI model requirements apply |
| August 2, 2027 | **High-risk AI systems (including medical devices) fully applicable** |

#### Interplay with MDR/IVDR

MDCG 2025-6 FAQ clarifies the relationship between the AI Act and MDR/IVDR:

- Medical devices with AI components undergo conformity assessment under
  *both* MDR/IVDR and the AI Act
- However, the MDR/IVDR conformity assessment (by a Notified Body) is
  deemed to cover the AI Act high-risk requirements — no separate AI Act
  conformity assessment is needed
- The Notified Body assessing MDR/IVDR conformity also verifies AI Act
  compliance
- Additional AI Act requirements not covered by MDR/IVDR (e.g., registration
  in EU database) still apply separately

### 2.7 IEC 82304 & IEC 81001-5-1 — Health Software & Cybersecurity

#### IEC 82304-1:2016 — Health Software Product Safety

IEC 82304-1 covers health software products that are *not* medical devices
but are intended for use in healthcare. This standard is relevant for:

- Research software that may later evolve into clinical tools
- Decision support tools that fall below the medical device threshold
- Wellness and population health applications

Key requirements:

- Software safety classification
- Risk management per ISO 14971 principles
- Software lifecycle per IEC 62304 principles
- Labeling and user documentation
- Post-market surveillance

> **Strategic relevance for MinIVess**: During the research phase, the project
> may not qualify as a medical device. IEC 82304-1 provides a pathway for
> maintaining safety practices without full medical device compliance,
> establishing good habits for the eventual clinical path.

#### IEC 81001-5-1 — Health Software Cybersecurity

IEC 81001-5-1 specifies requirements and guidance for managing cybersecurity
in the design and maintenance of health software. It is referenced by both
IEC 62304 Edition 2 (draft) and the EU AI Act.

Key areas:

- Security risk management process
- Secure design principles (defense in depth, least privilege)
- Secure implementation (input validation, authentication)
- Security testing (penetration testing, vulnerability scanning)
- Security maintenance (patch management, incident response)
- Security documentation

### 2.8 Upcoming Changes

The regulatory landscape for AI/ML medical devices is evolving rapidly.
Teams should monitor these upcoming developments:

| Expected Date | Development | Impact |
|---------------|-------------|--------|
| September 2026 | IEC 62304 Edition 2 | Major: agile/DevOps, AI/ML, cybersecurity |
| August 2027 | EU AI Act enforcement for medical devices | High-risk requirements fully applicable |
| 2025-2026 | FDA TPLC final guidance | Continuous monitoring requirements |
| 2026 | AAMI AI/ML standards updates | Updated ML risk management guidance |
| 2025-2026 | IEC 81001-5-1 wider adoption | Cybersecurity requirements tightening |
| 2026-2028 | MDR transition completion | All device classes under MDR |

> **Recommendation**: Design compliance processes now that anticipate these
> changes. The "everything as code" approach (Stirbu et al., 2021) provides
> the flexibility to adapt processes as standards evolve, without rebuilding
> documentation infrastructure.

---

## 3. Academic Foundations

A growing body of academic research addresses the intersection of agile/DevOps
software development and medical device regulatory compliance. This section
reviews the key papers that inform the MinIVess compliance strategy.

### 3.1 Martina et al. 2024 — DevOps-based IEC 62304

**Full title**: "DevOps-based IEC 62304 Maintenance"
**Published**: European Conference of Software Process Improvement (EuroSPI), 2024

#### Summary

Martina et al. (2024) demonstrate that IEC 62304 Clause 6 (maintenance) and
Clause 9 (problem resolution) can be fully satisfied using DevOps workflows
implemented in JIRA with CI/CD integration. The paper uses a Software Medical
Application (SWMA) as a case study.

#### Key Contributions

1. **Maintenance type mapping**: The four IEC 62304 maintenance types
   (corrective, adaptive, perfective, preventive) are mapped to JIRA issue
   types and workflows:

   | IEC 62304 Type | JIRA Issue Type | Trigger | ML Example |
   |----------------|-----------------|---------|------------|
   | Corrective | Bug | Defect report | Model producing NaN outputs |
   | Adaptive | Task (Environment) | External change | PyTorch version upgrade |
   | Perfective | Story/Enhancement | Improvement request | Model retraining with new data |
   | Preventive | Technical Debt | Proactive maintenance | Dependency vulnerability update |

2. **CI/CD integration for traceability**: Automated testing pipelines provide
   continuous verification evidence. Each JIRA ticket links to code changes
   (commits/PRs), test results (CI artifacts), and review records.

3. **Problem resolution workflow**: JIRA issue lifecycle (Open -> In Progress ->
   Review -> Resolved -> Closed) satisfies IEC 62304 Clause 9 requirements
   for problem tracking, investigation, and closure.

#### Relevance to MinIVess

The Martina et al. (2024) workflow validates the approach of using GitHub
Issues/PRs (analogous to JIRA) with GitHub Actions CI/CD for IEC 62304
compliance. The four maintenance types map directly to GitHub issue labels
and PR templates.

### 3.2 Lähteenmäki et al. 2023 — AHMED: Agile & RegOps

**Full title**: Agile and Lean Development of Medical Software (AHMED Project)
**Published**: Multi-year Finnish research project with multiple publications, 2023

#### Summary

The AHMED project is a comprehensive Finnish research initiative investigating
agile and lean development methodologies for medical device software. It produced
practical tooling recommendations, process frameworks, and empirical data on
the cost of regulatory compliance.

#### Key Findings

1. **Traceability effort**: Regulatory traceability consumes **20-25% of total
   project effort** in medical software development. This is the single largest
   compliance cost for small teams.

2. **Tool evaluation**: The project evaluated multiple ALM (Application Lifecycle
   Management) tools. Polarion ALM was found to provide **80% effort savings**
   in traceability compared to manual approaches, but at significant license cost.

3. **DVC vs. DataLad comparison**: For ML pipeline data versioning:

   | Criterion | DVC | DataLad |
   |-----------|-----|---------|
   | Git integration | Excellent (git-like commands) | Deep (built on git-annex) |
   | ML pipeline support | Native (dvc.yaml) | Requires additional tooling |
   | Learning curve | Moderate | Steep |
   | Ecosystem | Large (MLOps focused) | Smaller (neuroimaging focused) |
   | **Recommendation** | **Preferred for ML pipelines** | Better for pure data management |

4. **RegOps lifecycle concept**: Regulatory compliance as a continuous, automated
   process integrated into the development lifecycle — not a separate gate or
   phase. This is the conceptual foundation for the "RegOps" approach (see
   Section 3.5).

5. **Model Cards as ML ledger**: Model cards (Mitchell et al., 2019) serve as
   regulatory documentation artifacts, capturing model specifications,
   performance characteristics, intended use, and limitations.

6. **CI/CD pipeline**: GitLab + Jenkins pipeline demonstrated for medical
   software, with automated testing, documentation generation, and compliance
   evidence collection.

7. **Continuous documentation paradigm**: Rather than producing documentation
   in batches before regulatory submissions, documentation should be generated
   continuously as a natural byproduct of the development process.

#### Relevance to MinIVess

AHMED's findings directly validate the MinIVess approach:

- DVC (already in use) is the recommended ML data versioning tool
- MLflow model cards align with the "ML ledger" concept
- The 20-25% traceability effort figure motivates automated compliance tooling
- The RegOps concept maps to the phased compliance strategy (Section 6)

### 3.3 Granlund et al. 2022 — Calm Compliance

**Full title**: "Calm Compliance"
**Published**: IEEE Software, 2022

#### Summary

Granlund et al. (2022) articulate the thesis that regulatory compliance should
be a natural, calm byproduct of good development practices — not a separate,
stressful overhead activity. The paper introduces the "everything as code"
philosophy for compliance and the CompliancePal concept.

#### Key Thesis

> Compliance should be a calm, routine part of the development workflow.
> If you are practicing good software engineering — version control, automated
> testing, code review, documentation — then you are already generating most
> of the evidence required for regulatory compliance. The gap is in capturing
> and presenting that evidence, not in producing it.

#### "Everything as Code" Philosophy

All compliance artifacts should be stored in version-controlled, machine-readable
formats:

| Artifact | Traditional Format | "As Code" Format |
|----------|-------------------|------------------|
| Requirements | Word documents | YAML/Markdown in Git |
| Risk analysis | Excel spreadsheets | YAML/JSON in Git |
| Test plans | Word documents | pytest fixtures + YAML |
| Test results | PDF reports | CI/CD artifacts (JSON/JUnit XML) |
| Traceability matrix | Excel | Generated from Git metadata |
| Design documentation | Word/Visio | Markdown + Mermaid/PlantUML |
| Change records | Paper forms | Git commits + PR reviews |
| Audit logs | Separate database | Git history + CI/CD logs |

#### Implications for ML Systems

If ML development already follows MLOps best practices:

- **Experiments tracked in MLflow** -> Experiment records, parameter documentation
- **Data versioned in DVC** -> Data lineage, dataset documentation
- **Tests automated with pytest** -> Verification evidence
- **Code reviewed via PRs** -> Design review records
- **CI/CD pipelines in GitHub Actions** -> Automated verification, release records
- **Model cards generated** -> Performance documentation, intended use

Then **compliance evidence generation can be largely automated** through CI/CD
pipeline steps that collect, format, and archive these artifacts.

#### CompliancePal Concept

CompliancePal is a conceptual CI/CD tool that:

1. Scans Git repositories for compliance-relevant artifacts
2. Checks completeness of required documentation
3. Validates traceability links (requirements -> tests -> results)
4. Generates compliance reports for regulatory submissions
5. Alerts on missing or incomplete evidence

> **Note**: CompliancePal is a research concept, not a production tool. However,
> Ketryx (Section 4.8) implements a similar vision as a commercial product.

#### Relevance to MinIVess

The "calm compliance" paradigm is the philosophical foundation of the MinIVess
compliance strategy. The existing MLOps toolchain (MLflow, DVC, pytest, GitHub
Actions, pre-commit hooks) already generates most compliance evidence. The
remaining work is in *structuring* and *presenting* that evidence.

### 3.4 Stirbu et al. 2021-2022 — Everything as Code

Vlad Stirbu and colleagues published two foundational papers on bridging
lightweight development tools and regulatory compliance:

#### "Introducing Traceability in GitHub" (PROFES, 2021)

**Key contributions**:

- GitHub Issues and Pull Requests provide natural traceability if properly
  structured with templates and conventions
- Requirements -> code -> tests traceability can be maintained using GitHub
  native features (issue references in commits, PR reviews as design reviews)
- The lightweight nature of GitHub project management does not preclude
  regulatory compliance — it requires *discipline* in how features are used

**Practical traceability pattern**:

```
GitHub Issue #42: "Support DICOM RT-Struct format"
  ├── Type: Requirement (label)
  ├── Safety Class: B (label)
  ├── Risk Reference: RISK-007 (custom field or issue link)
  │
  ├── PR #51: "Add RT-Struct parser"
  │   ├── Commits reference Issue #42
  │   ├── Code review by qualified reviewer
  │   ├── CI/CD tests pass (automated verification)
  │   └── Links to test file: test_rtstruct_parser.py
  │
  └── Verification: Test results in CI artifacts
      ├── Unit tests: 12/12 passed
      ├── Integration tests: 3/3 passed
      └── System test: Manual verification documented in PR comment
```

#### "Towards RegOps" (Springer, 2021)

**Key contributions**:

- **RegOps** = DevOps + Regulatory compliance
- Everything-as-code approach: requirements, risks, test results, compliance
  evidence all stored as version-controlled, machine-readable files

**Proposed file formats**:

```yaml
# requirements/REQ-042.yaml
id: REQ-042
title: "DICOM RT-Struct format support"
type: software_requirement
parent: UNS-015  # User Need
safety_class: B
risk_references:
  - RISK-007
  - RISK-012
acceptance_criteria:
  - "Parser handles RT-Struct files conforming to DICOM PS3.3 C.8.8.6"
  - "Invalid files raise DicomParseError with descriptive message"
  - "Performance: parse 100MB file in < 5 seconds"
status: verified
verification_method: automated_test
test_references:
  - tests/unit/test_rtstruct_parser.py::test_valid_rtstruct
  - tests/integration/test_dicom_pipeline.py::test_rtstruct_e2e
```

```yaml
# risks/RISK-007.yaml
id: RISK-007
hazard: "Incorrect segmentation boundary from malformed RT-Struct"
severity: serious  # per ISO 14971
probability: occasional
risk_level: unacceptable  # before mitigation
risk_controls:
  - id: RC-007-1
    type: design
    description: "Input validation with schema conformance check"
    verification: tests/unit/test_rtstruct_parser.py::test_malformed_input
  - id: RC-007-2
    type: protective
    description: "Warning flag on parsing anomalies"
    verification: tests/integration/test_anomaly_warnings.py
residual_risk_level: acceptable
```

**Key insight**: The gap between GitHub's lightweight project management and
heavy regulatory requirements can be bridged with structured YAML files in the
repository and CI/CD automation that validates completeness and generates
traceability reports.

### 3.5 CompliancePal & RegOps Pipeline Concept

Building on the work of Stirbu, Granlund, and Lähteenmäki, the RegOps pipeline
concept integrates regulatory compliance into the CI/CD workflow:

```
Developer Workflow                CI/CD Pipeline                 Regulatory Output
─────────────────                ──────────────                 ─────────────────

Write code          ──────>     Build & Test    ──────>         Test Results (JSON)

Update requirements  ──────>    Validate YAML   ──────>         Requirements Matrix
(YAML files)                    completeness

Update risks         ──────>    Check risk      ──────>         Risk Mgmt Report
(YAML files)                    coverage

Create PR            ──────>    Traceability    ──────>         Traceability Matrix
                                check

Merge & Tag          ──────>    Generate DHF    ──────>         Design History File
                                artifacts

Deploy               ──────>    Compliance      ──────>         Release Record
                                report                          (IEC 62304 5.8)
```

**CompliancePal features** (conceptual, partially implemented in research):

1. **YAML schema validation**: Ensures all requirement/risk files conform
   to the defined schema
2. **Completeness checking**: Verifies that every requirement has test
   references, every risk has controls, every control has verification
3. **Traceability generation**: Automatically builds traceability matrices
   from Git metadata and YAML cross-references
4. **Gap analysis**: Identifies missing artifacts for the target safety class
5. **Report generation**: Produces formatted documents suitable for regulatory
   submission

> **Current commercial realization**: Ketryx (Section 4.8) is the closest
> commercial implementation of the CompliancePal concept, providing automated
> IEC 62304/ISO 13485 compliance checking on top of GitHub/JIRA workflows.

---

## 4. Project Management Tool Analysis

Selecting the right project management tooling is critical for ML-based medical
device development. The tool must support both the iterative nature of ML
experimentation and the traceability requirements of regulatory standards.

### 4.1 Scoring Methodology

Tools are evaluated on five criteria, weighted by importance for a small
research team pursuing a clinical deployment path:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **IEC 62304 Traceability** | 0.30 | Requirements -> design -> code -> test -> release chain |
| **Developer Experience** | 0.25 | Learning curve, daily workflow friction, modern UX |
| **Cost for Small Teams** | 0.20 | Cost for <= 5 developers, pre-revenue startup |
| **Ecosystem/Integration** | 0.15 | CI/CD, Git, test frameworks, ML tools |
| **Audit Readiness** | 0.10 | Export capabilities, immutable audit logs, evidence packages |

Each criterion is scored 1-5:

- **5**: Best in class, exceeds requirements
- **4**: Strong, meets all requirements
- **3**: Adequate, meets core requirements with workarounds
- **2**: Weak, significant gaps requiring manual effort
- **1**: Insufficient, does not meet requirements

### 4.2 JIRA + Confluence

**Overview**: Atlassian JIRA is the industry-standard project management tool,
with 57.5% market share among software teams (Atlassian, 2024). Confluence
provides integrated wiki-style documentation.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 4 | Strong with Ketryx overlay; native traceability requires configuration |
| Developer Experience | 2 | High configuration overhead, complex workflows, learning curve |
| Cost for Small Teams | 3 | $8-17/user/month; free tier limited to 10 users |
| Ecosystem/Integration | 4 | Extensive marketplace, CI/CD integrations |
| Audit Readiness | 4 | Good export, audit logs, compliance plugins |

**Weighted Score: 3.40**

**Strengths**:

- Martina et al. (2024) validate JIRA for IEC 62304 maintenance workflows
- Ketryx overlay (Section 4.8) provides UL-certified IEC 62304/ISO 13485/
  ISO 14971 automation on top of JIRA
- Extensive ecosystem of medical device compliance plugins
- Widely accepted by regulatory auditors (familiar tool)

**Weaknesses**:

- High developer friction for small teams (complex configuration)
- Requires Confluence for documentation (additional cost/complexity)
- Risk of over-engineering workflows for a research-phase project
- Cloud-only for small teams (data sovereignty concerns)

### 4.3 GitHub Projects V2

**Overview**: GitHub's built-in project management, tightly integrated with
Issues, Pull Requests, and Actions.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 3 | Achievable with discipline (Stirbu et al., 2021); semantic gap for requirement types |
| Developer Experience | 5 | Native to development workflow, minimal context switching |
| Cost for Small Teams | 5 | Free for public repos; free for private with basic features |
| Ecosystem/Integration | 4 | Native GitHub Actions, extensive marketplace |
| Audit Readiness | 3 | Git history as audit log; limited native compliance reporting |

**Weighted Score: 4.05** (rises to **4.35** with Ketryx overlay)

**Strengths**:

- Already in use for MinIVess (zero migration cost)
- Stirbu et al. (2021) demonstrate regulatory traceability is achievable
- Native integration with Git (commits, PRs, reviews = compliance evidence)
- Innolitics RDM provides open-source 510(k) documentation overlay
- Ketryx free tier provides UL-certified compliance automation

**Weaknesses**:

- No built-in requirement types or traceability links (must use labels/templates)
- Limited reporting capabilities compared to dedicated ALM tools
- Audit log export requires GitHub API scripting
- No native risk management features

**Mitigation**: The Stirbu et al. (2021) YAML-based approach + Ketryx overlay
addresses most weaknesses. GitHub Issues with structured templates provide
requirement types; Ketryx provides automated traceability and compliance reporting.

### 4.4 Linear

**Overview**: Modern, fast project management tool popular with engineering teams.
Excellent developer experience and clean UI.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 1 | No compliance features, no requirement traceability |
| Developer Experience | 5 | Best-in-class UX, keyboard-driven, fast |
| Cost for Small Teams | 3 | Free tier available; $8/user/month for standard |
| Ecosystem/Integration | 3 | GitHub integration, limited CI/CD |
| Audit Readiness | 1 | 90-day audit log only; no compliance plugins |

**Weighted Score: 2.55**

**Strengths**:

- Best developer experience of any tool evaluated
- Clean, modern interface reduces daily friction
- Good GitHub integration for code linking

**Weaknesses**:

- **90-day audit log retention**: Completely insufficient for regulatory
  compliance, which requires records for the lifetime of the device plus
  applicable retention periods (typically 10-15 years)
- No compliance plugins, extensions, or marketplace
- No requirement types or traceability features
- No path to regulatory readiness

**Verdict**: **Not recommended for SaMD path.** Linear excels for non-regulated
software teams but has no viable path to medical device compliance.

### 4.5 GitLab

**Overview**: Integrated DevOps platform with built-in CI/CD, issue tracking,
and project management.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 4 | Has IEC 62304 mapping documentation; native requirements management |
| Developer Experience | 3 | Good but complex; steeper learning curve than GitHub |
| Cost for Small Teams | 3 | Free tier; $29/user/month for Ultimate (compliance features) |
| Ecosystem/Integration | 4 | Built-in CI/CD, container registry, SAST/DAST |
| Audit Readiness | 4 | Self-hosted option, compliance frameworks, audit events |

**Weighted Score: 3.50**

**Strengths**:

- GitLab has published IEC 62304 mapping documentation showing how GitLab
  features map to standard requirements
- Self-hosted option important for data sovereignty (medical data)
- AHMED project (Lähteenmäki et al., 2023) used GitLab + Jenkins successfully
- Built-in security scanning (SAST, DAST, dependency scanning)
- Requirements management in Ultimate tier

**Weaknesses**:

- Compliance features require Ultimate tier ($29/user/month)
- Self-hosting adds operational overhead
- Smaller ecosystem than GitHub for ML tools
- Migration cost from GitHub

### 4.6 Polarion ALM

**Overview**: Siemens ALM tool purpose-built for regulated industries.
Full requirements management with bi-directional traceability.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 5 | Purpose-built; bi-directional traceability, requirement types |
| Developer Experience | 2 | Enterprise UI, slow, requires training |
| Cost for Small Teams | 1 | $3k-5k/year minimum; enterprise pricing |
| Ecosystem/Integration | 3 | Siemens ecosystem; limited modern DevOps integration |
| Audit Readiness | 5 | Best-in-class; built for regulatory audits |

**Weighted Score: 3.30**

**Strengths**:

- AHMED project found **80% traceability effort savings** compared to manual
  approaches (Lähteenmäki et al., 2023)
- Purpose-built for medical device development
- Full bi-directional traceability (requirements <-> design <-> tests <-> releases)
- Proven at scale in medical device companies (Siemens Healthineers, etc.)

**Weaknesses**:

- Enterprise pricing prohibitive for research teams ($3k-5k/year minimum)
- Poor developer experience (legacy enterprise UI)
- Limited modern CI/CD integration
- Overkill for research phase

**Verdict**: Appropriate for established medical device companies with dedicated
regulatory teams. Not recommended for research-phase projects.

### 4.7 Azure DevOps

**Overview**: Microsoft's integrated DevOps platform with project management,
CI/CD (Azure Pipelines), and artifact management.

| Criterion | Score | Notes |
|-----------|-------|-------|
| IEC 62304 Traceability | 3 | Modern Requirements extension adds IEC 62304 templates |
| Developer Experience | 3 | Good for Microsoft shops; learning curve for others |
| Cost for Small Teams | 4 | **Free for up to 5 users** (Basic plan) |
| Ecosystem/Integration | 4 | Azure Pipelines, GitHub integration, Microsoft ecosystem |
| Audit Readiness | 4 | Good audit logs, compliance frameworks |

**Weighted Score: 3.55**

**Strengths**:

- **Free for up to 5 users** — significant advantage for small teams
- Modern Requirements extension provides IEC 62304 templates
- Azure Pipelines is a capable CI/CD system
- Good integration with GitHub (Microsoft-owned)
- Enterprise support path for clinical deployment phase

**Weaknesses**:

- Microsoft ecosystem lock-in risk
- Modern Requirements extension is a paid add-on
- Less ML ecosystem support than GitHub
- Migration cost from GitHub

### 4.8 Ketryx Overlay

**Overview**: Ketryx is a UL Certified compliance overlay that works on top of
existing development tools (GitHub, JIRA). Rather than replacing the development
workflow, it adds automated compliance checking and documentation generation.

| Feature | Details |
|---------|---------|
| UL Certification | Certified for IEC 62304, ISO 13485, ISO 14971 |
| FDA Track Record | Accounted for 2% of all 2024 FDA AI/ML device approvals |
| Automation | Automates ~90% of compliance documentation |
| Pricing | Free tier for pre-market teams with < $2M revenue |
| Funding | Series B funded |
| Integration | GitHub (native), JIRA, CI/CD pipelines |

**Key capabilities**:

1. **Automated traceability**: Links requirements -> design -> code -> tests
   -> releases using Git metadata and issue references
2. **Compliance gap analysis**: Identifies missing documentation for target
   safety class and standards
3. **Evidence collection**: Automatically gathers test results, code reviews,
   and change records from CI/CD
4. **Document generation**: Produces IEC 62304/ISO 13485 compliant documents
   (SDS, SRS, risk reports)
5. **SBOM generation**: Software Bill of Materials for SOUP documentation
6. **Audit packages**: Pre-formatted evidence packages for regulatory submissions

**Free tier qualification**: Teams with less than $2M annual revenue can use
Ketryx at no cost during pre-market development. This covers the entire
research-to-submission pipeline for small teams.

### 4.9 Recommendation

#### Weighted Score Summary

| Tool | Traceability (0.30) | DevEx (0.25) | Cost (0.20) | Ecosystem (0.15) | Audit (0.10) | **Total** |
|------|---------------------|--------------|-------------|-------------------|--------------|-----------|
| GitHub + Ketryx | 4.5 | 5 | 5 | 4 | 4 | **4.35** |
| Azure DevOps | 3 | 3 | 4 | 4 | 4 | **3.55** |
| GitLab | 4 | 3 | 3 | 4 | 4 | **3.50** |
| JIRA + Ketryx | 4 | 2 | 3 | 4 | 4 | **3.40** |
| Polarion ALM | 5 | 2 | 1 | 3 | 5 | **3.30** |
| Linear | 1 | 5 | 3 | 3 | 1 | **2.55** |

#### Phased Approach

The recommended strategy is a phased tooling approach that matches compliance
investment to project maturity:

**Phase 1 — Research (current)**:
- GitHub Projects V2 (free, already in use)
- GitHub Issues with structured templates for requirement types
- GitHub Actions for CI/CD
- Cost: $0

**Phase 2 — Pre-compliance**:
- Add Ketryx free tier for automated traceability
- Add OpenRegulatory templates for IEC 62304 documents
- Structured YAML files for requirements and risks (Stirbu et al., 2021)
- Cost: $0

**Phase 3 — Clinical preparation**:
- Ketryx paid tier for full compliance automation
- Evaluate Azure DevOps or JIRA if team grows beyond 10
- Consider Polarion only if scaling to large clinical team
- Cost: varies by scale

> **PRD Decision**: `project_management_tool` = GitHub Projects V2 + Ketryx overlay.
> This decision is referenced from Section 9.1.

---

## 5. MONAI Deploy for Clinical Market Entry

MONAI (Medical Open Network for Artificial Intelligence) is an open-source
framework for healthcare AI, initiated by NVIDIA and King's College London.
MONAI Deploy is the deployment component of the MONAI ecosystem, designed
specifically for clinical integration of AI models.

### 5.1 Architecture (SDK v3.0.0)

MONAI Deploy App SDK v3.0.0 provides an operator-based Directed Acyclic Graph
(DAG) architecture for building clinical AI applications.

#### Core Concepts

```
MONAI Application Package (MAP)
├── Application (DAG of Operators)
│   ├── DICOMDataLoaderOperator
│   ├── DICOMSeriesSelectorOperator
│   ├── MonaiSegInferenceOperator
│   │   ├── Pre-transforms (TorchIO/MONAI)
│   │   ├── Model inference (ONNX/TorchScript)
│   │   └── Post-transforms (thresholding, CRF)
│   └── DICOMSegmentationWriterOperator
│
├── Manifest (app.json)
│   ├── Application metadata
│   ├── Input/output specifications
│   ├── Resource requirements (GPU, memory)
│   └── Version and provenance
│
└── OCI Container
    ├── Base image (NVIDIA runtime)
    ├── Application code
    ├── Model weights
    └── Dependencies
```

**Operators**: Reusable processing units that form the pipeline DAG. Built-in
operators handle common medical imaging tasks:

| Operator | Function |
|----------|----------|
| `DICOMDataLoaderOperator` | Loads DICOM data from study directory |
| `DICOMSeriesSelectorOperator` | Selects appropriate series by modality/protocol |
| `DICOMSeriesSelectorOperator` | Filters series based on rules |
| `MonaiSegInferenceOperator` | Runs MONAI segmentation model inference |
| `DICOMSegmentationWriterOperator` | Writes results as DICOM SEG objects |
| `PNGConverterOperator` | Generates PNG overlays for visualization |
| `PublisherOperator` | Publishes results to DICOM endpoints |

**MAP (MONAI Application Package)**: OCI-compliant container images with
immutable manifests. MAPs are the unit of deployment and versioning for
clinical AI applications.

```yaml
# Example MAP manifest (simplified)
apiVersion: v3.0.0
application:
  name: "vessel-segmentation"
  version: "1.2.0"
  description: "3D cerebral vessel segmentation from 7T TOF-MRA"
  authors:
    - name: "MinIVess Team"
input:
  formats:
    - dicom
  modalities:
    - MR
  body_parts:
    - HEAD
output:
  formats:
    - dicom-seg
    - dicom-sr
resources:
  gpu: true
  gpu_memory: "8Gi"
  cpu: "4"
  memory: "16Gi"
```

#### Runtime

MONAI Deploy applications run on the Holoscan SDK (NVIDIA), which provides:

- GPU-accelerated inference pipeline
- Low-latency streaming data processing
- GXF (Graph Execution Framework) for operator scheduling
- Hardware abstraction for deployment across GPU platforms

### 5.2 Clinical Integration

MONAI Deploy provides clinical system integration through its Informatics Gateway
and Workflow Manager components.

#### Informatics Gateway

Handles communication with clinical systems:

| Protocol | Direction | Use Case |
|----------|-----------|----------|
| DICOM SCP | Inbound | Receive studies from PACS |
| DICOM SCU | Outbound | Push results to PACS |
| DICOMweb STOW-RS | Inbound | RESTful study storage |
| DICOMweb WADO-RS | Outbound | RESTful study retrieval |
| HL7 FHIR | Bidirectional | Clinical data exchange |

#### Workflow Manager

The Workflow Manager orchestrates clinical AI workflows:

1. **Study arrival**: DICOM study received via SCP or DICOMweb
2. **Routing**: Rules engine selects appropriate MAP based on modality,
   body part, protocol
3. **Execution**: MAP container launched with study data
4. **Correlation**: Unique correlation ID tracks the study through the
   entire pipeline (critical for traceability)
5. **Result delivery**: Output DICOM SEG/SR pushed back to PACS

> **Correlation ID traceability**: Every clinical workflow execution has a
> unique correlation ID that links the incoming study to the inference
> execution, model version, processing parameters, and output results.
> This provides IEC 62304-aligned traceability for clinical operations.

### 5.3 Regulatory Facilitation

MONAI Deploy facilitates regulatory compliance but does **not** provide
compliance itself. Understanding the distinction is critical.

**What MONAI Deploy facilitates**:

| Requirement | How MONAI Deploy Helps |
|-------------|----------------------|
| IEC 62304 Clause 8 (config mgmt) | MAP provides immutable, versioned packaging |
| IEC 62304 Clause 5.8 (release) | MAP manifest documents release specifications |
| Reproducibility | OCI containers ensure identical execution environments |
| Traceability | Correlation IDs link inputs -> processing -> outputs |
| Auditability | Container immutability prevents unauthorized modifications |

**What MONAI Deploy does NOT provide**:

| Gap | Description |
|-----|-------------|
| IEC 62304 documentation | No automated SDS, SRS, or architecture document generation |
| Design History File | No DHF management or tracking |
| CAPA tracking | No Corrective and Preventive Action workflow |
| ISO 14971 integration | No risk management file support |
| SBOM generation | No Software Bill of Materials for SOUP documentation |
| Regulatory submission | No 510(k) or CE Technical File generation |

> **Important**: MONAI Deploy is a clinical deployment *platform*, not a
> regulatory compliance *tool*. It must be combined with compliance tooling
> (Ketryx, OpenRegulatory) and quality management processes to achieve
> regulatory readiness.

### 5.4 SOUP Considerations

When using MONAI Deploy and MONAI Model Zoo models, the entire stack becomes
SOUP (Software of Unknown Provenance) per IEC 62304.

#### MONAI Model Zoo SOUP Assessment

MONAI Model Zoo provides pre-trained models for various medical imaging tasks.
Using these models (even for transfer learning) triggers SOUP requirements:

| SOUP Requirement | Model Zoo Status | Gap |
|-----------------|------------------|-----|
| Identification | Good (name, version, architecture) | None |
| Published specifications | Partial (model cards, some documentation) | Incomplete performance specs |
| Risk evaluation | Not provided | Manufacturer must perform |
| Behavior verification | Not provided | Manufacturer must verify in target context |

**Required SOUP documentation for each Model Zoo model**:

```yaml
# soup/monai-segresnet-brain-vessels.yaml
id: SOUP-MONAI-001
name: "SegResNet Brain Vessel Segmentation"
version: "0.4.0"
source: "MONAI Model Zoo"
manufacturer: "MONAI Consortium"
intended_use: "3D brain vessel segmentation from TOF-MRA"
published_specifications:
  architecture: "SegResNet (Myronenko, 2018)"
  training_data: "Custom TOF-MRA dataset (details in model card)"
  reported_performance:
    dice: 0.78
    dataset: "Internal validation set"
risk_evaluation:
  failure_modes:
    - id: FM-001
      description: "Model trained on different scanner/protocol than target"
      severity: moderate
      mitigation: "Validate on MinIVess dataset before deployment"
    - id: FM-002
      description: "Model performance on rare pathologies unknown"
      severity: serious
      mitigation: "Clinician review required for all outputs"
verification_plan:
  - "Dice score >= 0.70 on MinIVess test set"
  - "No catastrophic failures (Dice < 0.30) on any individual case"
  - "Visual review of 20 representative cases by neuroradiologist"
```

### 5.5 Real-world Deployments

MONAI Deploy has been adopted by major healthcare institutions:

| Organization | Use Case | Scale |
|--------------|----------|-------|
| **Mayo Clinic** | Clinical AI integration pipeline | Production deployment |
| **Siemens Healthineers** | Digital Marketplace | 10,000+ institutions |
| **mercure DICOM** | DICOM orchestration | Open-source community |
| **King's College London** | Research to clinical translation | Academic medical center |
| **NIH/NCI** | Cancer imaging AI | Federally funded research |

**Siemens Healthineers Digital Marketplace**: Uses the MAP format as the
standard packaging for AI applications distributed to their global network
of 10,000+ healthcare institutions. This provides a proven distribution
channel for MAP-packaged applications.

**mercure DICOM orchestrator**: An open-source DICOM routing platform that
supports MONAI Deploy MAPs as processing modules. mercure handles study
routing and MAP execution, providing a lightweight alternative to the full
MONAI Deploy Informatics Gateway.

### 5.6 Hybrid Architecture

For MinIVess, a hybrid serving architecture is recommended that maintains the
current BentoML investment while adding clinical capability through MONAI Deploy.

```
                            ┌─────────────────────────────────┐
                            │         Same Model Weights       │
                            │    (MLflow Model Registry)       │
                            └───────────┬─────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                                       │
            ┌───────▼────────┐                    ┌────────▼────────┐
            │   BentoML      │                    │   MONAI Deploy  │
            │   (Research)   │                    │   (Clinical)    │
            ├────────────────┤                    ├─────────────────┤
            │ REST API       │                    │ DICOM SCP/SCU   │
            │ Gradio Demo    │                    │ DICOMweb        │
            │ ONNX Runtime   │                    │ HL7 FHIR        │
            │ Batch inference│                    │ MAP Container   │
            │ A/B testing    │                    │ Correlation ID  │
            └────────────────┘                    └─────────────────┘
                    │                                       │
            ┌───────▼────────┐                    ┌────────▼────────┐
            │  Research       │                    │  Clinical       │
            │  Consumers      │                    │  Consumers      │
            ├────────────────┤                    ├─────────────────┤
            │ Jupyter/Python  │                    │ PACS            │
            │ Web browsers    │                    │ RIS             │
            │ CI/CD pipelines │                    │ Clinical viewer │
            │ Collaborators   │                    │ Reporting       │
            └────────────────┘                    └─────────────────┘
```

**Benefits of the hybrid approach**:

1. **Preserves BentoML investment**: Current BentoML serving infrastructure
   continues to serve research use cases
2. **Same model, two frontends**: Single model training pipeline, two
   deployment targets
3. **Incremental clinical path**: MONAI Deploy can be added when clinical
   deployment is pursued, without disrupting research workflows
4. **Clear separation of concerns**: Research serving (BentoML) is not
   burdened with clinical requirements; clinical serving (MONAI Deploy)
   is purpose-built for healthcare integration

> **PRD Decisions**: `serving_architecture` = BentoML (research) + MONAI Deploy
> (clinical). `clinical_deployment_platform` = MONAI Deploy SDK v3.0.0.

### 5.7 Clara Holoscan MGX

NVIDIA Clara Holoscan MGX (Medical Grade eXtensible) is a reference hardware
platform for deploying AI inference at the edge in clinical environments.

**Key features**:

- Pre-documented IEC 62304 and IEC 60601 (medical electrical equipment)
  compliance documentation for the hardware platform
- NVIDIA GPU-accelerated inference (Jetson/dGPU)
- Designed for OEM embedding in medical devices
- Holoscan SDK runtime (same as MONAI Deploy)

**Relevance for MinIVess**: Clara Holoscan MGX is relevant for edge deployment
scenarios where inference must run locally at the clinical site (e.g., in the
scanner room or on a bedside workstation). The pre-documented IEC 62304/60601
compliance reduces the regulatory burden for hardware qualification.

**Current assessment**: Edge deployment is not a near-term priority for MinIVess.
Clara Holoscan MGX should be evaluated when edge deployment requirements arise,
likely during Phase 4 or Phase 5 of the compliance strategy.

---

## 6. Compliance Strategy for MinIVess MLOps

This section presents a five-phase compliance strategy that incrementally builds
regulatory readiness without front-loading the cost and complexity of full
medical device compliance during the research phase.

The strategy is grounded in the "calm compliance" paradigm (Granlund et al.,
2022): build compliance on top of good MLOps practices, not as a separate
overhead activity.

### 6.1 Current State

The MinIVess MLOps v2 project already implements several practices that form
a natural compliance foundation:

| Practice | Tool | Compliance Relevance |
|----------|------|---------------------|
| Experiment tracking | MLflow | IEC 62304 5.1 (planning), 8 (config mgmt) |
| Data versioning | DVC | IEC 62304 8 (config mgmt), AI Act Art. 10 |
| Automated testing | pytest (103+ tests) | IEC 62304 5.5-5.7 (verification) |
| Pre-commit hooks | ruff, mypy, pre-commit | IEC 62304 5.5 (unit verification) |
| Code review | GitHub PRs | IEC 62304 5.3-5.4 (design review) |
| CI/CD | GitHub Actions + CML | IEC 62304 5.6-5.8 (integration, release) |
| Model documentation | Model cards | FDA GMLP #9 (transparency) |
| Architecture decisions | ADRs (docs/adr/) | IEC 62304 5.3 (architecture) |
| Risk-based design | Validation onion (ADR-0003) | ISO 14971 principles |
| Observability | OpenTelemetry, Langfuse | IEC 62304 6 (maintenance), AI Act Art. 12 |

> **Key insight**: The `lightweight_audit` decision in the PRD (resolved)
> means MinIVess already operates with compliance-aware practices. The gap
> is in *formalizing* and *automating the capture* of compliance evidence,
> not in changing development practices.

### 6.2 Phase 1: Research Foundation (Current)

**Objective**: Establish MLOps best practices that naturally produce compliance
evidence. No regulatory overhead, no compliance tooling cost.

**Activities**:

- MLflow experiments with full parameter/metric logging
- DVC-tracked datasets with content-addressable hashes
- Automated test suite (unit, integration, e2e) with CI/CD execution
- Model cards for all trained models
- Architecture Decision Records for all significant design choices
- Pre-commit hooks enforcing code quality (ruff, mypy)
- Git-based change management with PR reviews

**Evidence produced** (usable for future compliance):

- Experiment records (MLflow): dates, parameters, metrics, artifacts
- Data provenance (DVC): dataset versions, transformations, splits
- Test results (pytest): verification evidence
- Code reviews (GitHub PRs): design review records
- Git history: complete change history with author/date/rationale

**Cost**: $0 additional (these are MLOps best practices)

**Duration**: Ongoing (Phases 0-6 already complete)

### 6.3 Phase 2: Pre-compliance

**Objective**: Add lightweight compliance structure without significant overhead.
Prepare for regulatory path without committing to it.

**Activities**:

1. **Add Ketryx free tier**:
   - Connect to GitHub repository
   - Enable automated traceability scanning
   - Generate initial compliance gap analysis

2. **Adopt OpenRegulatory templates**:
   - IEC 62304 Software Development Plan template
   - ISO 14971 Risk Management Plan template
   - Software Requirements Specification template
   - Customize templates for ML-specific content

3. **Implement structured YAML requirements**:
   ```yaml
   # requirements/software/REQ-SEG-001.yaml
   id: REQ-SEG-001
   title: "3D vessel segmentation accuracy"
   type: performance_requirement
   parent: UNS-001
   safety_class: B  # TBD based on intended use
   description: >
     The segmentation model shall achieve Dice >= 0.70 on the
     MinIVess hold-out test set for cerebral vessel structures.
   acceptance_criteria:
     - "Dice coefficient >= 0.70 on locked test set"
     - "95% CI lower bound >= 0.65"
     - "No catastrophic failures (Dice < 0.20) on any case"
   verification_method: automated_test
   test_references:
     - tests/e2e/test_segmentation_performance.py
   status: draft
   ```

4. **Begin SOUP inventory**:
   ```yaml
   # soup/pytorch.yaml
   id: SOUP-001
   name: "PyTorch"
   version: "2.5.0"
   manufacturer: "Meta AI / PyTorch Foundation"
   license: "BSD-3-Clause"
   intended_use: "Deep learning framework for model training and inference"
   published_specifications:
     url: "https://pytorch.org/docs/stable/"
     api_stability: "Stable API with deprecation policy"
   risk_evaluation:
     failure_impact: "Model training or inference failure"
     mitigation: "Version pinning, integration tests, ONNX export for inference"
   verification: "Integration test suite validates PyTorch behavior"
   ```

5. **Add risk placeholders**:
   ```yaml
   # risks/RISK-001.yaml
   id: RISK-001
   hazard: "Incorrect vessel segmentation leading to missed pathology"
   hazardous_situation: >
     Clinician relies on automated segmentation that fails to identify
     a vessel abnormality (aneurysm, stenosis, AVM)
   harm: "Delayed or missed diagnosis of cerebrovascular condition"
   severity: serious
   probability: occasional  # TBD with clinical input
   risk_level: unacceptable_before_mitigation
   risk_controls:
     - id: RC-001-1
       type: design
       description: "Confidence calibration with uncertainty visualization"
     - id: RC-001-2
       type: information
       description: "Intended use statement limiting to screening, not diagnosis"
     - id: RC-001-3
       type: protective
       description: "Mandatory clinician review before clinical action"
   residual_risk_level: TBD
   status: draft
   ```

**Cost**: $0 (Ketryx free tier, OpenRegulatory open-source)

**Duration**: 2-4 weeks of effort, can run in parallel with other development

### 6.4 Phase 3: RegOps Automation

**Objective**: Implement automated compliance evidence generation in the CI/CD
pipeline. Achieve "calm compliance" where regulatory artifacts are produced
as natural byproducts of development.

**Activities**:

1. **CI/CD compliance pipeline**:

   ```yaml
   # .github/workflows/compliance.yml (conceptual)
   name: Compliance Evidence Generation
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   jobs:
     traceability:
       steps:
         - name: Validate requirement YAML schemas
           run: uv run python scripts/validate_requirements.py

         - name: Check requirement-test coverage
           run: uv run python scripts/check_traceability.py

         - name: Generate traceability matrix
           run: uv run python scripts/generate_traceability.py
           # Output: artifacts/traceability-matrix.html

     risk_management:
       steps:
         - name: Validate risk YAML schemas
           run: uv run python scripts/validate_risks.py

         - name: Check risk control verification coverage
           run: uv run python scripts/check_risk_controls.py

     soup_management:
       steps:
         - name: Generate SBOM from uv.lock
           run: uv run python scripts/generate_sbom.py

         - name: Check SOUP documentation completeness
           run: uv run python scripts/check_soup_docs.py

     compliance_report:
       needs: [traceability, risk_management, soup_management]
       steps:
         - name: Generate compliance summary
           run: uv run python scripts/generate_compliance_report.py
           # Output: artifacts/compliance-report-{date}.html
   ```

2. **Automated requirements traceability**:
   - CI job validates that every requirement has test references
   - CI job validates that every test references a requirement
   - Traceability matrix generated automatically on each merge to main

3. **Risk management in YAML**:
   - Structured risk files per ISO 14971
   - CI validation of risk file completeness
   - Automated checking that all risk controls have verification references

4. **Compliance evidence as CI/CD artifacts**:
   - Test results (JUnit XML) archived per release
   - Traceability matrices generated and versioned
   - SBOM generated from dependency lock files
   - Compliance reports generated and archived

**Cost**: ~$500/year (CI/CD compute time, minor tooling)

**Duration**: 4-8 weeks of effort

### 6.5 Phase 4: Clinical Deployment

**Objective**: Achieve full regulatory readiness for clinical deployment. This
phase involves formal processes and likely requires regulatory consulting.

**Activities**:

1. **Full Verification & Validation (V&V)**:
   - Formal test protocols with acceptance criteria
   - Independent V&V (not performed by developer)
   - Clinical validation study design
   - Performance testing on representative clinical data

2. **MONAI Deploy MAP packaging**:
   - Package segmentation model as MAP
   - Validate MAP execution against reference results
   - Document MAP specifications (inputs, outputs, resource requirements)
   - Integration testing with clinical systems (DICOM SCP/SCU)

3. **Formal risk management**:
   - Complete ISO 14971 risk management file
   - Benefit-risk analysis per Clause 7.4
   - Clinical risk assessment with domain experts
   - FMEA (Failure Modes and Effects Analysis) for inference pipeline

4. **Design History File (DHF)**:
   - Compile all design control records
   - Software Development Plan (IEC 62304 5.1)
   - Software Requirements Specification (IEC 62304 5.2)
   - Software Architecture Document (IEC 62304 5.3)
   - Verification and Validation reports
   - Risk Management Report

5. **PCCP preparation**:
   - Define predetermined change boundaries (see Section 8.1)
   - Establish automated validation protocols (see Section 8.2)
   - Document impact assessment framework (see Section 8.3)

**Cost**: ~$5k-10k/year (regulatory consulting, tool upgrades, clinical validation)

**Duration**: 6-12 months

### 6.6 Phase 5: Market Access

**Objective**: Regulatory submission and market entry. This phase requires
professional regulatory consulting and significant investment.

**Activities**:

1. **Regulatory pathway selection**:
   - 510(k) if suitable predicate device exists
   - De Novo if novel but low-moderate risk
   - CE Technical File for EU market
   - Consider parallel FDA + CE submissions

2. **Regulatory submission preparation**:
   - Compile Technical Documentation (MDR Annex II)
   - Prepare 510(k) or De Novo submission package
   - Generate AI Act compliance documentation
   - Prepare PCCP (if applicable, FDA path)

3. **Notified Body / FDA engagement**:
   - Pre-submission meeting (FDA Q-Sub)
   - Notified Body selection and engagement (EU)
   - Audit preparation

4. **Post-market surveillance plan**:
   - Performance monitoring architecture
   - Drift detection and alerting
   - Complaint handling process
   - Periodic Safety Update Reports (PSURs)
   - Clinical performance follow-up plan

5. **Quality Management System**:
   - ISO 13485 certification (or subset)
   - Tool validation documentation
   - Training records
   - Document control procedures
   - Internal audit program

**Cost**: $50k-200k+ (regulatory consulting, Notified Body fees, FDA fees,
QMS implementation)

**Duration**: 12-24 months

> **Reality check**: Phase 5 represents a significant business commitment.
> It should only be pursued when there is clear clinical demand, funding,
> and a viable business model for the SaMD product. The phased approach
> ensures that the research value of the project is preserved regardless
> of whether market access is pursued.

---

## 7. Cross-Standard Traceability Matrix

One of the key challenges in medical device software development is maintaining
traceability across multiple standards simultaneously. This section provides
cross-standard traceability matrices showing how artifacts map across IEC 62304,
ISO 14971, FDA TPLC, and the EU AI Act.

### 7.1 IEC 62304: Requirements to Release

The primary traceability chain required by IEC 62304:

```
User Needs (UNS)
    │
    ▼
Software Requirements (SRS)          ← Clause 5.2
    │
    ▼
Software Architecture (SAD)          ← Clause 5.3
    │
    ▼
Detailed Design (SDD)               ← Clause 5.4
    │
    ▼
Unit Implementation & Verification   ← Clause 5.5
    │
    ▼
Integration & Integration Testing    ← Clause 5.6
    │
    ▼
System Testing                       ← Clause 5.7
    │
    ▼
Release                              ← Clause 5.8
```

**Traceability table for ML-based SaMD**:

| IEC 62304 Stage | Artifact | ML-Specific Content | Tool |
|----------------|----------|---------------------|------|
| User Needs | UNS-xxx.yaml | Clinical workflow requirements | GitHub Issues |
| Software Requirements | REQ-xxx.yaml | Performance metrics, data specs, input/output formats | GitHub Issues + YAML |
| Architecture | SAD.md | Model architecture, pipeline DAG, data flow | Markdown + Mermaid |
| Detailed Design | SDD.md | Layer specs, loss functions, augmentations, hyperparameters | Markdown + code |
| Unit Implementation | Source code | Model code, data processing, transforms | Git repository |
| Unit Verification | pytest results | Unit tests for data processing, transform correctness | pytest + CI/CD |
| Integration Testing | pytest results | Pipeline integration tests, end-to-end inference | pytest + CI/CD |
| System Testing | Validation results | Clinical performance on hold-out set, statistical analysis | pytest + notebooks |
| Release | MAP / BentoML Service | Packaged model with manifest, version, dependencies | MONAI Deploy / BentoML |

### 7.2 ISO 14971: Hazards to Verification

The risk management traceability chain per ISO 14971:

| Stage | Artifact | ML Content | Verification |
|-------|----------|------------|--------------|
| Hazard Identification | RISK-xxx.yaml | Training data risks, model failure modes, deployment risks | Risk analysis workshops |
| Risk Estimation | RISK-xxx.yaml (severity, probability) | Clinical impact assessment, failure rate estimation | Clinical expert input |
| Risk Evaluation | RISK-xxx.yaml (risk level) | Acceptable/unacceptable classification | Risk acceptance criteria |
| Risk Control | RC-xxx in RISK-xxx.yaml | Design controls (calibration, thresholds), protective measures (clinician review), information (IFU) | Implementation in code |
| Residual Risk | RISK-xxx.yaml (residual) | Post-mitigation risk level | Benefit-risk analysis |
| Verification of Controls | Test references in RC-xxx | Automated tests verifying each risk control | pytest + CI/CD |

**Example end-to-end risk traceability**:

```
RISK-003: "Model confident but wrong on out-of-distribution input"
    │
    ├── Severity: serious (missed diagnosis)
    ├── Probability: occasional (OOD inputs expected in practice)
    ├── Pre-mitigation: unacceptable
    │
    ├── RC-003-1 (Design): Uncertainty estimation via MC Dropout
    │   └── Verification: test_uncertainty_estimation.py::test_mc_dropout_variance
    │
    ├── RC-003-2 (Design): OOD detection threshold
    │   └── Verification: test_ood_detection.py::test_mahalanobis_distance
    │
    ├── RC-003-3 (Information): Warning displayed when uncertainty > threshold
    │   └── Verification: test_ui_warnings.py::test_high_uncertainty_warning
    │
    └── Residual risk: acceptable (clinician always reviews, uncertainty shown)
```

### 7.3 FDA TPLC: Data to Monitoring

The Total Product Lifecycle traceability for AI/ML-based SaMD:

| TPLC Stage | Activities | Artifacts | MLOps Tool |
|------------|-----------|-----------|------------|
| **Training Data** | Data collection, curation, annotation, splitting | Dataset documentation, annotation guidelines, split rationale | DVC, Label Studio |
| **Model Development** | Architecture selection, training, hyperparameter optimization | Experiment records, architecture rationale, training logs | MLflow, Hydra-zen |
| **Performance Validation** | Hold-out set evaluation, statistical analysis | Performance reports, confidence intervals, subgroup analysis | pytest, notebooks |
| **Clinical Validation** | Clinical study, reader study, real-world testing | Clinical evidence report, study protocol, results | Study management |
| **Post-Market Monitoring** | Drift detection, performance tracking, complaint handling | Monitoring dashboards, drift alerts, complaint records | Evidently, OpenTelemetry |
| **Model Updates** | Retraining, fine-tuning (via PCCP) | Change documentation, validation results, PCCP evidence | MLflow, DVC, CI/CD |

**PCCP traceability** (for pre-approved changes):

```
PCCP Change Category: "Retraining with expanded dataset"
    │
    ├── Change Description
    │   ├── New data: up to 2x current training set
    │   ├── Same architecture: SegResNet
    │   ├── Same loss function: Dice + Cross-Entropy
    │   └── Hyperparameters: within predefined bounds
    │
    ├── Performance Assessment
    │   ├── Predefined test set (locked, versioned in DVC)
    │   ├── Statistical non-inferiority test (p < 0.05, one-sided)
    │   ├── Subgroup analysis (scanner type, pathology type)
    │   └── Automated test suite execution
    │
    └── V&V Protocol
        ├── Unit tests: all must pass (CI/CD gate)
        ├── Integration tests: all must pass (CI/CD gate)
        ├── Performance test: Dice >= 0.75 (CI/CD gate)
        ├── Statistical test: non-inferiority confirmed (automated)
        ├── Visual review: 20 cases by qualified reviewer (manual)
        └── Documentation: automated report generation
```

### 7.4 EU AI Act: Training Data to Performance

The EU AI Act traceability chain for high-risk AI systems:

| AI Act Article | Requirement | Artifact | MLOps Mapping |
|----------------|-------------|----------|---------------|
| Art. 10 | Training data governance | Data sheet, annotation quality report, bias analysis | DVC metadata, Cleanlab, Deepchecks |
| Art. 10 | Data representativeness | Demographic analysis, scanner diversity report | Data profiling (whylogs) |
| Art. 9 | Risk management | Risk management file | YAML risk files, ISO 14971 process |
| Art. 11 | Algorithm design documentation | Architecture document, design rationale | ADRs, SAD, SDD |
| Art. 11 | Training methodology | Experiment records, hyperparameter documentation | MLflow experiments |
| Art. 11 | Validation procedures | Test protocols, acceptance criteria | pytest, CI/CD configuration |
| Art. 12 | Record-keeping (logging) | Inference logs, audit trail | OpenTelemetry, MLflow |
| Art. 13 | Transparency | Instructions for use, model card | Model cards, IFU document |
| Art. 14 | Human oversight | Clinician review workflow, override capability | MONAI Deploy workflow |
| Art. 15 | Accuracy metrics | Performance reports with confidence intervals | pytest, statistical analysis |
| Art. 15 | Robustness | Adversarial testing results, noise sensitivity | Deepchecks, custom tests |
| Art. 15 | Cybersecurity | Security assessment, vulnerability scan | IEC 81001-5-1 process |

**Cross-standard mapping table** (unified view):

| Artifact | IEC 62304 | ISO 14971 | FDA TPLC | EU AI Act |
|----------|-----------|-----------|----------|-----------|
| Data documentation | Clause 8 (config) | Input to hazard analysis | Training data | Art. 10 |
| Requirements spec | Clause 5.2 | Risk controls inform requirements | — | Art. 11 |
| Architecture doc | Clause 5.3 | — | Model development | Art. 11 |
| Risk management file | Clause 7 | Core deliverable | — | Art. 9 |
| Test results | Clause 5.5-5.7 | Verification of controls | Performance validation | Art. 15 |
| Model card | — | — | Transparency | Art. 13 |
| Release record | Clause 5.8 | — | — | — |
| Monitoring plan | Clause 6 | Clause 10 | Post-market monitoring | Art. 72 |
| Audit logs | Clause 8 | — | — | Art. 12 |
| SOUP/SBOM | Clause 8 | — | — | Art. 11 |
| Clinical evidence | — | Benefit-risk | Clinical validation | — |
| Incident reporting | Clause 9 | — | — | Art. 73 |

---

## 8. PCCP-Ready MLOps Architecture

The FDA's Predetermined Change Control Plan (PCCP) guidance (December 2024)
enables manufacturers to pre-specify types of ML model changes that can be
made without new regulatory submissions. This section describes an MLOps
architecture designed from the ground up to support PCCP workflows.

### 8.1 Predetermined Change Boundaries

A PCCP defines an "envelope" of acceptable changes. Outside this envelope,
a new submission is required. For an ML-based SaMD, the change boundaries
should be clearly defined:

#### Permitted Changes (Within PCCP Envelope)

| Change Type | Boundary | Example |
|-------------|----------|---------|
| **Retraining with new data** | Same distribution, up to Nx current set size | Adding new annotated TOF-MRA scans from same scanner type |
| **Hyperparameter tuning** | Within predefined ranges | Learning rate [1e-5, 1e-3], batch size [2, 16] |
| **Data augmentation updates** | Predefined augmentation types | Adding elastic deformation (already in augmentation library) |
| **Threshold adjustment** | Within predefined range | Confidence threshold [0.3, 0.8] |
| **Preprocessing updates** | Same pipeline, parameter tuning | Intensity normalization parameters |
| **Inference optimization** | Performance-neutral | ONNX optimization, quantization (with validation) |

#### Prohibited Changes (Require New Submission)

| Change Type | Rationale |
|-------------|-----------|
| Architecture change | Fundamentally different model behavior |
| New input modality | Different clinical validation required |
| New anatomical region | Different clinical context |
| Loss function change | May alter model behavior characteristics |
| New post-processing algorithm | Changes output interpretation |
| Training on different scanner type | Distribution shift risk |

#### Configuration Example

```yaml
# pccp/change-boundaries.yaml
pccp_version: "1.0"
device_name: "MinIVess Vessel Segmentation"
submission_reference: "K2XXXXX"  # future 510(k) number

permitted_changes:
  retraining:
    max_dataset_multiplier: 2.0
    allowed_scanners:
      - "Siemens MAGNETOM Terra 7T"
      - "Siemens MAGNETOM Plus 7T"
    allowed_sequences:
      - "TOF-MRA"
    required_annotations:
      min_annotators: 2
      agreement_threshold: 0.85

  hyperparameters:
    learning_rate:
      min: 1.0e-5
      max: 1.0e-3
    batch_size:
      min: 2
      max: 16
    epochs:
      min: 50
      max: 500
    weight_decay:
      min: 0.0
      max: 0.01

  inference:
    confidence_threshold:
      min: 0.3
      max: 0.8
    allowed_optimizations:
      - "onnx_graph_optimization"
      - "int8_quantization"  # with validation gate

  architecture:
    frozen: true  # architecture changes not permitted
    model: "SegResNet"
    backbone: "frozen"
    decoder: "frozen"
```

### 8.2 Automated Validation Protocols

Each permitted change type has an associated automated validation protocol that
must pass before the change is deployed.

#### Validation Test Suite

```yaml
# pccp/validation-protocols.yaml
protocols:
  retraining_validation:
    description: "Validation protocol for model retraining within PCCP"
    steps:
      - name: "Unit test gate"
        type: automated
        command: "uv run pytest tests/unit/ -x"
        acceptance: "all_pass"

      - name: "Integration test gate"
        type: automated
        command: "uv run pytest tests/integration/ -x"
        acceptance: "all_pass"

      - name: "Performance on locked test set"
        type: automated
        command: "uv run pytest tests/pccp/test_performance.py"
        acceptance:
          dice_mean: ">= 0.75"
          dice_95ci_lower: ">= 0.70"
          hausdorff_95: "<= 5.0mm"

      - name: "Non-inferiority test"
        type: automated
        command: "uv run pytest tests/pccp/test_noninferiority.py"
        acceptance:
          test: "paired_t_test_one_sided"
          margin: -0.03  # dice non-inferiority margin
          p_value: "< 0.05"

      - name: "Subgroup analysis"
        type: automated
        command: "uv run pytest tests/pccp/test_subgroups.py"
        acceptance:
          no_subgroup_below: 0.60  # no subgroup Dice < 0.60
          max_subgroup_drop: 0.10  # no subgroup drops > 0.10 vs baseline

      - name: "Drift detection"
        type: automated
        command: "uv run pytest tests/pccp/test_drift.py"
        acceptance:
          feature_drift: "p > 0.01 (no significant drift)"
          prediction_drift: "p > 0.01 (no significant drift)"

      - name: "Visual review"
        type: manual
        description: "Qualified reviewer examines 20 representative cases"
        reviewer_qualification: "Board-certified neuroradiologist or equivalent"
        acceptance: "No clinically significant errors identified"

      - name: "PCCP compliance report"
        type: automated
        command: "uv run python scripts/generate_pccp_report.py"
        output: "artifacts/pccp-validation-report-{version}-{date}.pdf"
```

#### Statistical Non-Inferiority Testing

The PCCP validation protocol includes a formal non-inferiority test:

```python
# tests/pccp/test_noninferiority.py (conceptual)
"""
PCCP Non-Inferiority Test

Tests that the retrained model is non-inferior to the baseline
(cleared) model on the locked test set.

Non-inferiority margin: -0.03 Dice
Significance level: alpha = 0.05 (one-sided)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def test_noninferiority_dice(
    baseline_dice_scores: np.ndarray,
    retrained_dice_scores: np.ndarray,
    margin: float = -0.03,
    alpha: float = 0.05,
) -> None:
    """Test non-inferiority of retrained model vs baseline."""
    differences = retrained_dice_scores - baseline_dice_scores
    t_stat, p_value_two_sided = stats.ttest_rel(
        retrained_dice_scores,
        baseline_dice_scores - margin,  # shifted null
    )
    # One-sided test: reject if retrained > baseline - margin
    p_value_one_sided = p_value_two_sided / 2
    if t_stat < 0:
        p_value_one_sided = 1 - p_value_one_sided

    assert p_value_one_sided < alpha, (
        f"Non-inferiority not demonstrated: "
        f"p={p_value_one_sided:.4f} >= {alpha}, "
        f"mean difference={differences.mean():.4f}"
    )
```

### 8.3 Impact Assessment Framework

When a change is proposed, the impact assessment framework classifies it and
determines the required validation level.

#### Change Classification

| Classification | Description | Validation Required | Human Review |
|---------------|-------------|---------------------|--------------|
| **Minor** | Bug fix, documentation update, non-functional change | Unit + integration tests | PR review |
| **Moderate** | Retraining within PCCP, threshold adjustment | Full PCCP validation suite | Qualified reviewer |
| **Major** | Architecture change, new modality, new indication | **New regulatory submission** | Regulatory + clinical |

#### Automated Classification Logic

```yaml
# pccp/change-classification.yaml
classification_rules:
  - name: "Documentation only"
    classification: minor
    conditions:
      - files_changed_pattern: "docs/**"
      - no_source_code_changes: true
    validation: "unit_tests_only"

  - name: "Test updates"
    classification: minor
    conditions:
      - files_changed_pattern: "tests/**"
      - no_source_code_changes: true
    validation: "unit_tests_only"

  - name: "Model weights update (retraining)"
    classification: moderate
    conditions:
      - files_changed_pattern: "models/**"
      - architecture_unchanged: true
      - within_pccp_boundaries: true
    validation: "full_pccp_suite"

  - name: "Inference threshold change"
    classification: moderate
    conditions:
      - files_changed_pattern: "configs/inference/**"
      - threshold_within_range: true
    validation: "full_pccp_suite"

  - name: "Architecture change"
    classification: major
    conditions:
      - architecture_changed: true
    validation: "new_submission_required"
    alert: "STOP — This change exceeds PCCP boundaries"

  - name: "New input modality"
    classification: major
    conditions:
      - input_specification_changed: true
    validation: "new_submission_required"
    alert: "STOP — This change exceeds PCCP boundaries"
```

#### CI/CD Integration

```yaml
# .github/workflows/pccp-gate.yml (conceptual)
name: PCCP Change Assessment
on:
  pull_request:
    branches: [release/*]

jobs:
  classify_change:
    runs-on: ubuntu-latest
    outputs:
      classification: ${{ steps.classify.outputs.level }}
    steps:
      - name: Classify change
        id: classify
        run: uv run python scripts/classify_pccp_change.py
        # Outputs: minor, moderate, or major

  minor_validation:
    needs: classify_change
    if: needs.classify_change.outputs.classification == 'minor'
    steps:
      - run: uv run pytest tests/unit/ tests/integration/ -x

  moderate_validation:
    needs: classify_change
    if: needs.classify_change.outputs.classification == 'moderate'
    steps:
      - run: uv run pytest tests/ -x
      - run: uv run pytest tests/pccp/ -x
      - run: uv run python scripts/generate_pccp_report.py
      - name: Request qualified reviewer
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.pulls.requestReviewers({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.payload.pull_request.number,
              reviewers: ['qualified-reviewer-username']
            })

  major_block:
    needs: classify_change
    if: needs.classify_change.outputs.classification == 'major'
    steps:
      - name: Block merge
        run: |
          echo "::error::This change exceeds PCCP boundaries."
          echo "::error::A new regulatory submission is required."
          exit 1
```

---

## 9. Recommendations

### 9.1 Tool Stack

Based on the analysis in this report, the recommended tool stack for MinIVess
MLOps with clinical deployment readiness:

| Layer | Tool | Phase | Cost | Rationale |
|-------|------|-------|------|-----------|
| **Source Control** | GitHub | 1 (current) | Free | Already in use, excellent ecosystem |
| **Project Management** | GitHub Projects V2 | 1 (current) | Free | Native integration, low friction |
| **Compliance Overlay** | Ketryx (free tier) | 2 | Free (<$2M rev) | UL-certified, 90% automation, GitHub native |
| **Experiment Tracking** | MLflow | 1 (current) | Free | Already in use, model registry, compliance evidence |
| **Data Versioning** | DVC | 1 (current) | Free | Already in use, AHMED-recommended |
| **CI/CD** | GitHub Actions + CML | 1 (current) | Free | Already in use, compliance pipeline capable |
| **Research Serving** | BentoML + ONNX | 1 (current) | Free | Already in use, REST/Gradio/batch |
| **Clinical Serving** | MONAI Deploy SDK v3.0.0 | 4 | Free | MAP packaging, DICOM integration |
| **Doc Templates** | OpenRegulatory | 2 | Free | Open-source IEC 62304/ISO 14971 templates |
| **Data Quality** | Cleanlab + Deepchecks | 1 (current) | Free | Label quality, model validation |
| **Drift Detection** | Evidently + whylogs | 1 (current) | Free | Post-market monitoring ready |
| **Observability** | OpenTelemetry + Langfuse | 1 (current) | Free | Inference logging, AI Act Art. 12 |

> **PRD Decision**: `project_management_tool` = GitHub Projects V2 + Ketryx.
> `regulatory_documentation_tool` = Ketryx + OpenRegulatory.
> `clinical_deployment_platform` = MONAI Deploy SDK v3.0.0.
> `serving_architecture` = BentoML (research) + MONAI Deploy (clinical).

**Cost trajectory**:

| Phase | Annual Cost | Notes |
|-------|-------------|-------|
| Phase 1 (Research) | $0 | All free/open-source tools |
| Phase 2 (Pre-compliance) | $0 | Ketryx free tier, OpenRegulatory |
| Phase 3 (RegOps) | ~$500 | CI/CD compute, minor tooling |
| Phase 4 (Clinical) | ~$5k-10k | Regulatory consulting, tool upgrades |
| Phase 5 (Market Access) | $50k-200k+ | Submission fees, Notified Body, QMS |

### 9.2 PRD Updates

Based on this research, three new PRD decisions are recommended and five
existing decisions should be enriched with clinical compliance context.

#### New Decisions

**`project_management_tool`** (L4-Infrastructure):

```yaml
id: project_management_tool
title: "Project Management Tool for Regulatory Traceability"
level: L4
status: resolved
resolution: github_projects_ketryx
confidence: 0.85
options:
  github_projects_ketryx:
    prior: 0.45
    evidence:
      - "Stirbu et al. (2021) — GitHub traceability validated"
      - "Ketryx — UL-certified, 2% of 2024 FDA AI/ML approvals"
      - "Free tier covers pre-market development"
    weighted_score: 4.35
  azure_devops:
    prior: 0.20
    evidence:
      - "Free for 5 users"
      - "Modern Requirements extension"
    weighted_score: 3.55
  jira_ketryx:
    prior: 0.20
    evidence:
      - "Martina et al. (2024) — validated for IEC 62304"
      - "Industry standard (57.5% market share)"
    weighted_score: 3.40
  gitlab:
    prior: 0.10
    evidence:
      - "AHMED project used GitLab+Jenkins"
      - "Self-hosted data sovereignty"
    weighted_score: 3.50
  polarion:
    prior: 0.05
    evidence:
      - "80% traceability savings (Lähteenmäki, 2023)"
      - "Cost prohibitive for research"
    weighted_score: 3.30
```

**`regulatory_documentation_tool`** (L4-Infrastructure):

```yaml
id: regulatory_documentation_tool
title: "Regulatory Documentation Automation Tool"
level: L4
status: resolved
resolution: ketryx_openregulatory
confidence: 0.80
options:
  ketryx_openregulatory:
    prior: 0.50
    rationale: >
      Ketryx provides automated compliance checking on GitHub.
      OpenRegulatory provides free IEC 62304/ISO 14971 templates.
      Combined, they cover automated traceability + document templates.
  innolitics_rdm:
    prior: 0.25
    rationale: >
      Open-source 510(k) documentation tool for GitHub.
      Good for FDA-focused teams. Less comprehensive than Ketryx.
  manual_yaml:
    prior: 0.25
    rationale: >
      Stirbu et al. (2021) YAML-based approach.
      Maximum flexibility, minimum cost, highest manual effort.
```

**`clinical_deployment_platform`** (L3-Technology):

```yaml
id: clinical_deployment_platform
title: "Clinical Deployment Platform for DICOM Integration"
level: L3
status: resolved
resolution: monai_deploy
confidence: 0.80
options:
  monai_deploy:
    prior: 0.60
    rationale: >
      MONAI Deploy SDK v3.0.0 with MAP packaging.
      DICOM SCP/SCU, DICOMweb, HL7 FHIR integration.
      Proven at Mayo Clinic, Siemens Healthineers.
      OCI containers for immutable deployment.
  orthanc_custom:
    prior: 0.20
    rationale: >
      Orthanc DICOM server + custom inference wrapper.
      More manual integration, less clinical orchestration.
  nvidia_triton:
    prior: 0.20
    rationale: >
      NVIDIA Triton Inference Server.
      Excellent performance, less clinical integration.
```

#### Enriched Existing Decisions

| Decision ID | Enrichment |
|-------------|------------|
| `compliance_depth` | Add IEC 62304 Edition 2 timeline, EU AI Act enforcement date, PCCP readiness requirements |
| `serving_architecture` | Add hybrid BentoML+MONAI Deploy architecture, clinical vs. research serving |
| `model_governance` | Add PCCP change boundaries, automated validation protocols, SOUP documentation |
| `documentation_standard` | Add OpenRegulatory templates, YAML-as-code approach, Ketryx automation |
| `ci_cd_platform` | Add compliance pipeline stages, PCCP gate workflow, SBOM generation |

### 9.3 Research Roadmap

The following developments should be monitored and incorporated into the
compliance strategy as they materialize:

#### Near-term (2025-2026)

| Item | Expected Date | Action |
|------|---------------|--------|
| FDA TPLC final guidance | 2025-2026 | Update post-market monitoring architecture |
| IEC 62304 Edition 2 | September 2026 | Update development process to align with new clauses |
| MONAI Deploy SDK updates | Ongoing | Track new operators, features, MAP format changes |
| Ketryx feature expansion | Ongoing | Evaluate new compliance automation features |
| AAMI TIR 34971 updates | 2025-2026 | Incorporate updated ML risk guidance |

#### Medium-term (2026-2028)

| Item | Expected Date | Action |
|------|---------------|--------|
| EU AI Act enforcement (high-risk) | August 2027 | Ensure Article 9-15 compliance |
| MDR transition completion | 2026-2028 | Prepare CE marking pathway |
| IEC 81001-5-1 wider adoption | 2026-2027 | Implement cybersecurity framework |
| Harmonized AI standards (CEN/CENELEC) | 2026-2028 | Adopt harmonized standards as they publish |

#### Long-term Research Directions

| Topic | Relevance | Key Papers to Watch |
|-------|-----------|---------------------|
| Federated learning for SaMD | Multi-site model training without data sharing | Privacy-preserving ML for regulated environments |
| Continuous learning SaMD | Models that update post-deployment | PCCP + continuous learning integration |
| Foundation models in medical imaging | Transfer learning from large pre-trained models | SOUP documentation for foundation model weights |
| Formal verification for ML | Mathematical guarantees on model behavior | Beyond statistical testing for safety-critical ML |
| Synthetic data for SaMD validation | Augmenting real clinical data with synthetic | Regulatory acceptance of synthetic validation data |

---

## References

### Standards and Regulations

- IEC 62304:2006+A1:2015. *Medical device software — Software life cycle processes*. International Electrotechnical Commission (IEC, 2015).

- IEC 62304 Edition 2 (draft, expected September 2026). *Medical device software — Software life cycle processes*. International Electrotechnical Commission.

- ISO 13485:2016. *Medical devices — Quality management systems — Requirements for regulatory purposes*. International Organization for Standardization (ISO, 2016).

- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. International Organization for Standardization (ISO, 2019).

- AAMI/BSI TIR 34971:2023. *Application of ISO 14971 to machine learning in artificial intelligence*. Association for the Advancement of Medical Instrumentation (AAMI, 2023).

- IEC 82304-1:2016. *Health software — Part 1: General requirements for product safety*. International Electrotechnical Commission (IEC, 2016).

- IEC 81001-5-1:2021. *Health software and health IT systems safety, effectiveness and security — Part 5-1: Security — Activities in the product life cycle*. International Electrotechnical Commission (IEC, 2021).

- Regulation (EU) 2017/745. *Medical Device Regulation (MDR)*. European Parliament and Council (EU, 2017).

- Regulation (EU) 2017/746. *In Vitro Diagnostic Medical Devices Regulation (IVDR)*. European Parliament and Council (EU, 2017).

- Regulation (EU) 2024/1689. *Artificial Intelligence Act*. European Parliament and Council (EU, 2024).

- MDCG 2025-6. *FAQ on the interplay between the AI Act and the MDR/IVDR*. Medical Device Coordination Group (MDCG, 2025).

### FDA Guidance Documents

- FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence/Machine Learning (AI/ML)-Enabled Device Software Functions — Final Guidance*. U.S. Food and Drug Administration (FDA, 2024).

- FDA, Health Canada, MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*. U.S. Food and Drug Administration (FDA/HC/MHRA, 2021).

- FDA. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations — Draft Guidance*. U.S. Food and Drug Administration (FDA, 2025).

- IMDRF. *Software as a Medical Device (SaMD): Key Definitions*. International Medical Device Regulators Forum (IMDRF, 2013).

- IMDRF. *Software as a Medical Device: Possible Framework for Risk Categorization and Corresponding Considerations*. International Medical Device Regulators Forum (IMDRF, 2014).

### Academic Literature

- Martina, S., et al. (2024). "DevOps-based IEC 62304 Maintenance." *European Conference of Software Process Improvement (EuroSPI)*. Springer.

- Lähteenmäki, J., et al. (2023). "AHMED: Agile and Lean Development of Medical Software." Research project publications, multiple venues. Finland.

- Granlund, T., et al. (2022). "Calm Compliance." *IEEE Software*, 39(5), pp. 56-63.

- Stirbu, V., et al. (2021). "Introducing Traceability in GitHub." *International Conference on Product-Focused Software Process Improvement (PROFES)*. Springer.

- Stirbu, V., et al. (2021). "Towards RegOps: Regulatory Compliance as Code." *Springer*.

- Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAccT)*. ACM.

### Technical Tools and Platforms

- Ketryx. *UL-Certified Compliance Automation for Medical Device Software*. Ketryx, Inc. https://ketryx.com

- Innolitics. *Regulatory Documentation Manager (RDM)*. Innolitics, LLC. https://innolitics.com/rdm/

- OpenRegulatory. *Open-source Templates for Medical Device Regulatory Compliance*. OpenRegulatory. https://openregulatory.com

- MONAI Consortium. *MONAI Deploy App SDK v3.0.0*. Project MONAI. https://monai.io/deploy.html

- Cardoso, M. J., et al. (2022). "MONAI: An open-source framework for deep learning in healthcare." *arXiv preprint arXiv:2211.02701*.

- Yang, C., et al. (2022). "BentoML: Unified Model Serving Framework." *arXiv preprint*.

- Kreuzberger, D., Kuhl, N., and Hirschl, S. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access*, 11, pp. 31866-31879.

### Additional Resources

- Atlassian. (2024). *Atlassian FY2024 Annual Report*. Market share data.

- GitLab. *IEC 62304 Compliance with GitLab*. GitLab Documentation. https://about.gitlab.com/solutions/compliance/

- NVIDIA. *Clara Holoscan MGX Developer Kit*. NVIDIA Developer. https://developer.nvidia.com/clara-holoscan-mgx

- mercure DICOM. *Open-source DICOM Orchestration Platform*. https://mercure-imaging.org

- Mayo Clinic. AI deployment infrastructure descriptions. Various press releases and publications.

- Siemens Healthineers. *Digital Marketplace*. https://www.siemens-healthineers.com/digital-health-solutions/digital-solutions-overview/clinical-decision-support

---

> **Document metadata**
>
> - **Project**: MinIVess MLOps v2
> - **Phase**: 11 — Medical Device Software Standards Research
> - **Author**: Generated with AI assistance
> - **Date**: 2026-02-23
> - **Status**: Complete
> - **PRD decisions referenced**: `compliance_depth`, `project_management_tool`, `regulatory_documentation_tool`, `clinical_deployment_platform`, `serving_architecture`, `model_governance`, `documentation_standard`, `ci_cd_platform`
