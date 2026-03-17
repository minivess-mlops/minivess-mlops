# RegOps FDA Regulation Reporting: QMS, SaMD, IEC 62304, and MLOps

**Created**: 2026-03-17
**Status**: Draft — linked to [Issue #821](https://github.com/petteriTeikari/minivess-mlops/issues/821) (P1)
**Companion doc**: `docs/planning/openlineage-marquez-iec62304-report.md`
**Related issue**: [#799](https://github.com/petteriTeikari/minivess-mlops/issues/799) (OpenLineage/Marquez integration)

---

## Executive Summary

This document analyzes MinIVess MLOps's readiness for FDA Software as a Medical Device (SaMD) regulatory pathways, focusing on the critical gap of **test dataset use documentation and lineage tracking**. While MinIVess is a preclinical research platform for multiphoton cerebrovascular imaging, its architecture should **scale to clinical MLOps** without retrofitting — consistent with the AHMED project's finding that "documentation activity takes more than 70% of the total work effort of a usual medical software development project" ([Lähteenmäki et al. (2023). "AHMED — Agile and Holistic Medical Software Development." *VTT Technical Research Centre of Finland*. VTT-R-01079-22](https://cris.vtt.fi)).

The core thesis: **continuous automated documentation from Day 1 is orders of magnitude cheaper than retrofitting compliance later** — and our existing OpenLineage + MLflow + Prefect + DVC stack is 80% of the way there.

---

## 1. The Test Set Reuse Problem

### 1.1 Why This Matters for FDA

The FDA explicitly addresses test dataset reuse in its 2022 guidance for Computer-Assisted Detection Devices:

> "In the event that you would like the [FDA] to consider the reuse of any test data in your standalone evaluation, you should control the access of your staff to performance results for the test subpopulations and individual cases. It may therefore be necessary for you to **set up a 'firewall' to ensure those outside of the regulatory assessment team (e.g., algorithm developers) are completely insulated from knowledge of the [test data]**. You should maintain test data integrity throughout the lifecycle of the product."
>
> — [FDA (2022). "Clinical Performance Assessment: Considerations for Computer-Assisted Detection Devices Applied to Radiology Images." *FDA Guidance*.](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-performance-assessment-considerations-computer-assisted-detection-devices-applied-radiology)

This is not theoretical risk. As [Giese (2024)](https://innolitics.com) warns: "FDA could ask about your test data reuse. They could ask what procedures you had in place to prevent overfitting. Worst case, they may ask you to collect a new Test Dataset. For many AI/ML teams, this would be a **$100k, $200k, or $500k problem**."

### 1.2 The Scientific Context

The broader ML community increasingly recognizes that external validation is insufficient:

- [Van Calster et al. (2023). "There is no such thing as a validated prediction model." *BMC Medicine*, 21, 70.](https://link.springer.com/article/10.1186/s12916-023-02779-w) — argues that "external validation does not guarantee generalizability or equate to model usefulness." Models validated on one dataset routinely fail at other clinical sites.

- [Lenharo (2024). "The testing of AI in medicine is a mess." *Nature*, 632, 271-272.](https://doi.org/10.1038/d41586-024-02675-0) — highlights the gap between idealized clinical trials for AI and messy real-world testing practices, noting that "implementation depends on how well health-care professionals interact with the algorithms."

- [Youssef et al. (2023). "External validation of AI models in health should be replaced with recurring local validation." *npj Digital Medicine*.](https://doi.org/10.1038/s41746-023-00831-w) — proposes that static external validation should give way to continuous local validation, which aligns with our Prefect flow architecture where eval runs are automated and logged.

- [Chouffani El Fassi et al. (2024). "Not all AI health tools with regulatory authorization are clinically validated." *Nature Medicine*.](https://doi.org/10.1038/s41591-024-03060-y) — found that **approximately 43% of FDA-approved AI/ML medical devices lacked published clinical validation data**, and only 22 of 521 authorizations used RCTs.

- [Longhurst et al. (2024). "A Call for Artificial Intelligence Implementation Science Centers to Evaluate Clinical Effectiveness." *JAMA*.](https://jamanetwork.com/journals/jama) — calls for going "beyond model validation" to explore real-world effectiveness, noting that "so few [AI tools] have gone on to show any meaningful impact."

### 1.3 Our Specific Situation

MinIVess has **two test-related contexts**:

| Context | Dataset | Current Handling | FDA Relevance |
|---------|---------|-----------------|---------------|
| **Internal test split** | MiniVess 70 vols, 3-fold CV (47 train/23 val) | DVC-versioned, split in `configs/splits/3fold_seed42.json` | Every eval flow run on val set is logged to MLflow — but **no firewall** on who accesses results |
| **External test datasets** | DeepVess, TubeNet, VesselNN | Listed in `src/minivess/data/external_datasets.py` | Used in Eval Flow for generalizability assessment — **no access logging** |

**Critical gap**: While our Eval Flow separates train+val (for "training" performance) from test (for "real-world performance proxy"), **we do not log when test set results are viewed or used to inform model selection**. This is exactly the $100k+ mistake Giese warns about.

---

## 2. Regulatory Landscape Analysis

### 2.0 CRITICAL: Triple Regulatory Convergence in 2026

Three major regulatory milestones converge in 2026, making this planning document time-sensitive:

| Event | Date | Impact |
|-------|------|--------|
| **FDA QMSR** (replaces QSR 21 CFR Part 820) | **February 2, 2026** (effective) | Incorporates ISO 13485:2016 into federal law; risk management now "a dynamic enforcement lever" ([ComplianceQuest, 2026](https://www.compliancequest.com/blog/fda-quality-management-system-regulation-qmsr-2026/)) |
| **EU AI Act application date** | **August 2, 2026** | Article 17 mandates documented quality systems for high-risk AI; ISO/IEC 42001:2023 provides auditable framework |
| **IEC 62304 Edition 2.0 publication** | **~August 12, 2026** | Replaces A/B/C with two rigor levels (I/II); adds AI Development Lifecycle (AIDL) phase |

This triple convergence means that any medical device software manufacturer must simultaneously prepare for updated QMS requirements, AI-specific regulations, and a revised software lifecycle standard. Our architecture should anticipate all three.

### 2.0.1 FDA January 2025 Draft Guidance: Lifecycle Management for AI-Enabled Devices

The FDA's January 2025 draft guidance ["Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations"](https://www.fda.gov/media/184856/download) establishes critical new requirements for test dataset documentation:

- **Document data splits**: Specify how data is partitioned into training, validation, and test sets, and whether hold-out sets represent future or external populations rather than random subsets
- **Version all datasets**: Maintain a "data catalog" linking specific dataset versions to model iterations, enabling correlation between performance changes and dataset modifications
- **Trace every data source**: Document origin, inclusion/exclusion criteria, collection methods, and acceptance criteria for all data used in development and validation ([RookQS, 2025](https://rookqs.com/blog-rqs/fda-expectations-for-ai/ml-model-training-in-samd-2025-guide))
- **Representativeness**: Test datasets must reflect proposed intended-use populations with respect to demographics

**Shocking transparency gap**: Approximately 95.5% of FDA-cleared AI devices failed to report demographic breakdowns in validation data, only 1.6% included RCT evidence, and ~43% lacked published clinical validation data entirely ([Chouffani El Fassi et al. 2024](https://doi.org/10.1038/s41591-024-03060-y); [IntuitionLabs, 2024](https://intuitionlabs.ai/articles/fda-samd-classification-ai-machine-learning)).

### 2.0.2 PCCP Framework: Finalized December 2024

The Predetermined Change Control Plan (PCCP) framework is now **finalized and operational**:

- **December 2024**: FDA published [final PCCP guidance](https://www.fda.gov/media/166704/download)
- **August 2025**: FDA, Health Canada, and MHRA jointly issued 5 guiding principles for PCCPs in ML-enabled devices
- **As of late 2024**: 15 AI-ML medical devices approved with PCCPs (e.g., Clarius OB AI, Beacon Biosignals SleepStageML)

PCCPs must include three components: **Description of Modifications** (what changes), **Modification Protocol** (how changes are validated), and **Impact Assessment** (safety/performance effects). They must be Focused, Risk-based, Evidence-based, and Transparent.

For test set reuse during model updates: "New data are analyzed to ensure they have the quality required and respect the defined criteria. Then the data have to be segregated, and afterward, the model can be retrained. A new test set is created with previous and new data, which is used to ensure that the model matches the acceptance criteria" ([Carvalho et al. (2025). "Predetermined Change Control Plans: Guiding Principles." *JMIR AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/)).

**Our factorial experiment design is essentially a PCCP** — it documents predetermined model variations (4 models x 3 losses x 2 aux_calib x 3 post-training x 2 recalibration x 5 ensemble) with pre-specified acceptance criteria. Framing it as such in the manuscript is a significant differentiator.

### 2.1 IEC 62304: Software Life Cycle Processes

IEC 62304 ([IEC 62304:2006+A1:2015](https://www.iso.org/standard/45557.html)) is the international standard for medical device software development. Its five core process areas map directly to our infrastructure:

| IEC 62304 Clause | Process | MinIVess Mapping | Gap |
|-------------------|---------|------------------|-----|
| **5** | Software Development | Prefect flows, TDD, pre-commit hooks | PCCP template not implemented |
| **6** | Software Maintenance | Git branches, PRs, issue tracking | No formal change control board |
| **7** | Risk Management | Knowledge graph decisions (65 nodes) | Not ISO 14971-formatted |
| **8** | Configuration Management | Git + DVC + Docker images | **OpenLineage events not wired** |
| **9** | Problem Resolution | GitHub issues, metalearning docs | Not structured per IEC 62304 §9 |

**IEC 62304 Edition 2** (comment resolution starting 20 March 2026, approval from 22 May 2026, publication expected **12 August 2026**; [LFH Regulatory, 2026](https://lfhregulatory.co.uk/iec-62304-update-2026/)) brings major changes:

1. **Safety class simplification**: A/B/C → two "Software Process Rigor Levels" (Level I = low rigor, Level II = high rigor)
2. **AI/ML-specific guidance**: AI Development Lifecycle (AIDL) framework requiring rigorous testing, validation, and risk assessment for AI/ML
3. **Expanded scope**: Explicitly covers SaMD, software-only health products, AI-driven healthcare solutions
4. **Cybersecurity**: References IEC 81001-5-1 cybersecurity standard
5. **Agile support**: References AAMI TIR45:2023 for formalized agile methodology guidance
6. **Architecture for all levels**: Architectural design may become required even for low-rigor software

The AIDL phase maps directly to our Prefect flow topology (Train → Post-Training → Eval → Biostats → Deploy).

### 2.2 FDA SaMD Framework

The FDA's approach to AI/ML-enabled SaMD has evolved through several key documents:

1. **PCCP — Predetermined Change Control Plan** ([FDA (2023). "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence/Machine Learning-Enabled Device Software Functions." *Draft Guidance*.](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial)) — allows manufacturers to describe planned modifications to an AI/ML device and the methodology for implementing those changes in a controlled manner. This is essentially what our factorial experiment design + Prefect orchestration provides.

2. **GMLP — Good Machine Learning Practice** ([FDA/Health Canada/MHRA (2021). "Good Machine Learning Practice for Medical Device Development: Guiding Principles."](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)) — 10 guiding principles including:
   - Principle 3: Clinical study participants and data should be representative
   - Principle 5: Focus on the performance of the human-AI team
   - Principle 8: **Testing should demonstrate device performance during clinically relevant conditions**
   - Principle 10: Deployed models should be monitored for performance

3. **Total Product Lifecycle (TPLC)** ([Warraich et al. (2024). "FDA Perspective on the Regulation of Artificial Intelligence in Health Care and Biomedicine." *JAMA*.](https://jamanetwork.com/journals/jama/fullarticle/2825146)) — the FDA's shift from static premarket review to continuous lifecycle oversight, which aligns with our Prefect flow + drift monitoring architecture.

4. **510(k) and De Novo pathways** ([Muehlematter et al. (2023). "FDA-cleared AI/ML-based medical devices and their 510(k) predicate networks." *Lancet Digital Health*.](https://doi.org/10.1016/S2589-7500(23)00126-7)) — More than a third of cleared AI/ML devices originated from non-AI/ML predicates, and 950+ AI/ML devices have reached FDA authorization as of early 2025.

### 2.3 EU MDR / IVDR

Under EU Medical Device Regulation 2017/745 (MDR):

- Software is classified as SaMD if it has a medical purpose
- IEC 62304 compliance is required for CE marking
- The AI Act (Regulation 2024/1689) defers healthcare AI to MDR — "no difference" from FDA approach, as [Wagner (2024)](https://www.linkedin.com/in/rudolfwagner) observes

The AHMED project's Solita RegOps framework ([Lähteenmäki et al. 2023](https://cris.vtt.fi)) demonstrates that regulatory-compliant MLOps is achievable with proper tooling. Their key insight: use the general IEC 62304 development model as a baseline and add AI-specific lifecycle aspects (data management, drift monitoring, model cards) on top.

---

## 3. Current State: What MinIVess Already Has

### 3.1 Lineage & Audit Infrastructure (Implemented)

| Component | File | Status | FDA Relevance |
|-----------|------|--------|---------------|
| `LineageEmitter` | `src/minivess/observability/lineage.py` (271 lines) | Implemented | OpenLineage events for IEC 62304 §8 |
| `AuditTrail` | `src/minivess/compliance/audit.py` (127 lines) | Implemented | `log_test_evaluation()`, `log_data_access()`, `log_model_training()`, `log_model_deployment()` |
| `openlineage-python` | `pyproject.toml` (dependency) | Installed | Standard format for lineage exchange |
| Marquez service | `deployment/docker-compose.yml` | Config only | Lineage graph visualization |
| DVC versioning | `.dvc/`, `dvc.yaml` | Active | Data version control with GCS remote |
| MLflow tracking | All Prefect flows | Active | Experiment tracking, param/metric logging |
| Prefect orchestration | 5 flows (Data, Train, Post-Training, Eval, Deploy + Biostats, Dashboard) | Active | Pipeline execution audit trail |
| Knowledge graph | 65 Bayesian decision nodes | Active | Architectural decision documentation |
| Git + pre-commit | `.pre-commit-config.yaml` | Active | Source code version control |

### 3.2 The Integration Gap

Despite having all the building blocks, **they are not wired together for FDA-grade audit trails**:

1. **OpenLineage events are not emitted** by any Prefect flow (the `LineageEmitter` exists but no flow calls it)
2. **`AuditTrail.log_test_evaluation()` exists** but is never invoked during actual eval runs
3. **No "firewall"** mechanism for test dataset access — anyone can view test metrics at any time
4. **No lineage graph visualization** — Marquez is configured but not connected to PostgreSQL
5. **No PCCP template** — predetermined change control plan for model updates
6. **No Design History File (DHF)** auto-generation from Git + MLflow + Prefect artifacts
7. **No traceability matrix export** — requirements → design → code → tests → release

---

## 4. DataLad vs DVC vs OpenLineage: The Lineage Stack

### 4.1 AHMED's Insight on DataLad

The AHMED project ([Lähteenmäki et al. 2023](https://cris.vtt.fi), §6.2.6) compared DataLad and DVC for medical ML traceability:

| Property | DataLad | DVC | OpenLineage |
|----------|---------|-----|-------------|
| **Inherent action logging** | Yes — `datalad run` records every action in Git history | No — user must run Git commands separately | Yes — events emitted automatically |
| **Target** | All-purpose version control for data and code | Version control for ML | Pipeline lineage standard |
| **Git integration** | Git-annex (Git extension) | Git (metadata only) | None (event-based) |
| **Workflow management** | `run`/`rerun` commands for reproducibility | Pipeline management + experiments | Cross-pipeline lineage graph |
| **FDA value** | Every test set access automatically logged | Manual discipline required for access logging | Automated pipeline-level audit trail |

**Key AHMED finding**: "DataLad in principle tracks every action ran with the data, especially when analytics are applied within `datalad run` commands. Each action gets its own record at Git history and are thus possible to trace back afterwards. With DVC user must run the Git commands separately to maintain the version control of the task."

### 4.2 Our Architecture Decision

We use **DVC** (not DataLad) for data versioning. This means we must **compensate for DVC's lack of inherent action logging** with:

1. **OpenLineage events** at pipeline boundaries (what data entered, what transformation, what output)
2. **AuditTrail entries** for every test set access (who, when, why, what results)
3. **MLflow artifact logging** of lineage manifests (JSON sidecar per run)

This compensatory stack — **DVC + OpenLineage + AuditTrail + MLflow** — provides equivalent or superior traceability to DataLad's `datalad run`, because:
- OpenLineage captures cross-pipeline lineage (DataLad is per-dataset)
- AuditTrail captures human-level events (who viewed test results)
- MLflow captures experiment-level provenance (hyperparameters, metrics, artifacts)
- DVC captures data-level versioning (dataset hashes, remote locations)

### 4.3 The Missing Layer: Test Set Firewall

Neither DVC nor DataLad solves the **test set firewall** problem that FDA guidance requires. What's needed:

```
Test Set Access Request
    → AuditTrail.log_data_access("test_set", files, actor="eval_flow")
    → OpenLineage event: InputDataset("minivess", "test_split")
    → MLflow tag: eval/test_set_access_count += 1
    → [OPTIONAL] Notification to regulatory team via webhook
```

This is a **software architecture decision**, not a tooling choice. It must be implemented in the Eval Flow as a gating mechanism.

---

## 5. Multi-Hypothesis Design Matrix: FDA Readiness

### 5.1 Design Variables

| Factor | Levels | Description |
|--------|--------|-------------|
| **L (Lineage depth)** | L1: emit-only, L2: +Marquez, L3: +DHF export | How deep the OpenLineage integration goes |
| **T (Test set firewall)** | T0: none, T1: access logging, T2: +role-based access, T3: +notification | Degree of test set protection |
| **C (Compliance docs)** | C0: none, C1: model cards, C2: +PCCP, C3: +traceability matrix | Auto-generated regulatory documentation |
| **A (Audit continuity)** | A0: none, A1: per-run, A2: +per-session, A3: +cross-project | Scope of the continuous audit trail |
| **D (Documentation tool)** | D0: GitHub issues, D1: +Linear, D2: +Polarion-like | Issue/requirements management integration |

### 5.2 Hypothesis Matrix

| # | Config | DevEx Impact | FDA Value | Effort | Manuscript Value | Recommendation |
|---|--------|-------------|-----------|--------|-----------------|----------------|
| **H1** | L1+T1+C1+A1+D0 | Zero | Medium | Low (2-3 days) | High | **IMPLEMENT NOW** |
| **H2** | L2+T2+C2+A2+D0 | Zero | High | Medium (1-2 weeks) | Very High | **AFTER factorial** |
| **H3** | L2+T2+C2+A2+D1 | Low | Very High | Medium-High | Very High | **Q3 2026** |
| **H4** | L3+T3+C3+A3+D2 | Medium | Full FDA-ready | High (1-2 months) | Publication-grade | **Clinical projects** |
| **H5** | L1+T0+C0+A0+D0 | Zero | None | None | None | **REJECT** (status quo) |

### 5.3 Recommended Implementation Path

#### Phase 1: H1 — Minimum Viable FDA Readiness (NOW, during PR-C through PR-E)

**What**: Wire `LineageEmitter.pipeline_run()` into all 5 Prefect flows + add `AuditTrail.log_test_evaluation()` to Eval Flow + generate model cards as MLflow artifacts.

**Estimated effort**: ~30 lines of code across 5 flow files + 50 lines for model card generation.

**Specific tasks**:

1. Add `emitter.pipeline_run()` context manager to each flow (5-10 lines per flow)
2. Add `audit.log_test_evaluation()` call in Eval Flow when test split is evaluated
3. Add `audit.log_data_access()` call when external test datasets are loaded
4. Log `eval/test_set_access_count` as MLflow metric (cumulative)
5. Generate model card JSON as MLflow artifact at end of each training run
6. Log lineage manifest as MLflow artifact

**Why this matters**: Even without Marquez, this creates a **machine-readable audit trail** that proves:
- When test data was accessed (timestamps)
- What model was evaluated (run ID)
- What metrics were obtained (logged to MLflow)
- How many times test set was used (cumulative counter)

#### Phase 2: H2 — Marquez + PCCP (After factorial experiment)

**What**: Wire Marquez in docker-compose with PostgreSQL. Implement PCCP template. Add role-based test set access.

**Additions**:
1. Connect Marquez to PostgreSQL in `docker-compose.yml`
2. Implement PCCP YAML template (`configs/compliance/pccp_template.yaml`)
3. Add test set access role enforcement in Eval Flow
4. Generate lineage graph figure for manuscript

#### Phase 3: H3 — Linear Integration (Q3 2026)

**What**: Integrate Linear for IEC 62304 Problem Resolution workflows. Auto-create Linear issues from anomaly detection.

**Why Linear, not JIRA**: As the AHMED project found, JIRA is the incumbent in medical device development ([Lähteenmäki et al. 2023](https://cris.vtt.fi), §4.4.3), but its DevEx is poor. Linear offers:
- Fast, keyboard-driven UI — researchers actually use it
- GitHub bidirectional sync
- API-first design for automation
- Free for small teams (academic-friendly)

#### Phase 4: H4 — Full Compliance Stack (Clinical Projects)

**What**: DHF auto-generation, traceability matrix export, full IEC 62304 process coverage. This is only needed when MinIVess transitions from preclinical to clinical use.

---

## 6. Gap Analysis: What's Missing for True FDA Readiness

### 6.1 Critical Gaps (Must Fix)

| Gap | Impact | Current State | Fix |
|-----|--------|--------------|-----|
| **Test set access not logged** | FDA could require new test dataset ($100k+) | `AuditTrail.log_test_evaluation()` exists but unused | Wire into Eval Flow |
| **OpenLineage events not emitted** | No pipeline lineage trail | `LineageEmitter` exists but unused | Wire into all 5 flows |
| **No cumulative test set use counter** | Cannot prove test set was used sparingly | Not tracked | Add MLflow metric |
| **No model cards** | Missing ML development ledger | Not implemented | Generate from MLflow run metadata |

### 6.2 Important Gaps (Should Fix)

| Gap | Impact | Current State | Fix |
|-----|--------|--------------|-----|
| **Marquez not connected** | No lineage graph visualization | Docker config exists | Wire to PostgreSQL |
| **No PCCP template** | Missing predetermined change control plan | Not implemented | Create YAML template |
| **No DHF auto-generation** | Manual Design History File creation | Not implemented | Export from Git + MLflow |
| **No traceability matrix** | Requirements → Code → Tests not linked | Partial (KG decisions) | Export from KG + tests |
| **DVC lacks action logging** | Cannot prove data was accessed through controlled process | DVC metadata only | Compensate with OpenLineage |

### 6.3 Nice-to-Have Gaps (Can Defer)

| Gap | Impact | Current State | Fix |
|-----|--------|--------------|-----|
| **No Linear integration** | Manual issue management for compliance | GitHub issues only | API integration |
| **No trust center** | No public compliance dashboard | Not planned | Web page |
| **No SBOM for FDA** | Missing Software Bill of Materials | Partial (pyproject.toml) | Export CycloneDX SBOM |
| **No electronic signatures** | Missing 21 CFR Part 11 compliance | Git commit signatures only | Future requirement |

---

## 7. Continuous Auditing: The Day-1 Principle

### 7.1 The Cost of Delayed Compliance

The AHMED project found that companies estimate traceability maintenance at **20-25% of total project effort** when done manually ([Lähteenmäki et al. 2023](https://cris.vtt.fi), §4.6). More starkly, "some medical device manufacturers estimate that the documentation activity takes more than 70% of the total work effort" (§4.3).

The argument for Day-1 automated auditing:

1. **Retrofit cost is exponential** — implementing compliance after years of untracked development requires reconstructing history from Git logs, which is error-prone and expensive
2. **Missed audits are gaps** — a period of untracked test set access is a regulatory liability that cannot be retroactively filled
3. **Continuous auditing enables continuous certification** — as [Knoblauch et al. (2023). "Towards a Risk-Based Continuous Auditing-Based Certification for Machine Learning." *arXiv*.](https://arxiv.org/abs/placeholder) propose, continuous audit trails enable more efficient certification processes

### 7.2 What We Should Log Automatically (from Day 1)

| What | Where | Format | FDA Purpose |
|------|-------|--------|-------------|
| Every training run | MLflow | Params, metrics, artifacts | Training documentation |
| Every test set evaluation | AuditTrail + MLflow | JSON + metrics | Test data reuse tracking |
| Every pipeline execution | OpenLineage events | JSON (local + Marquez) | Configuration management |
| Every data version | DVC | Hash + remote location | Data provenance |
| Every model deployment | AuditTrail + MLflow | JSON + model artifacts | Deployment history |
| Every code change | Git | Commits + PRs | Source code traceability |
| Every infrastructure change | Pulumi | State files | IaC documentation |
| Every decision | Knowledge graph | YAML decision nodes | Design rationale |
| Every anomaly/drift | OpenLineage + alerts | Events | Post-market surveillance |

### 7.3 The Innolitics Principle

As [Giese (2021). "Documentation for Medical Device Software." *Innolitics*.](https://innolitics.com/articles/documentation-for-medical-device-software/) states: "the goal for agile teams developing medical software is not to eliminate documentation but to be **as efficient as possible in generating and updating the software documentation**."

[Joseph (2021). "Documentation for Medical Device Software." *Sunstone Pilot*.](https://sunstonepilot.com/2021/08/documentation-for-medical-device-software/) further emphasizes: "once a software team has gone through the laborious process of creating all the documentation by hand, they can be very resourceful in figuring out ways to **automatically generate document content**."

This is exactly our approach: the developer writes code and runs Prefect flows — the platform automatically generates the audit trail, lineage graph, model cards, and compliance artifacts.

---

## 8. Architecture: How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MinIVess MLOps Platform                         │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Train    │──▶│Post-Train│──▶│  Eval    │──▶│ Biostats │        │
│  │  Flow     │   │  Flow    │   │  Flow    │   │  Flow    │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              OpenLineage Event Bus                       │       │
│  │  START/COMPLETE/FAIL events per flow + per task          │       │
│  └────────────────────┬────────────────────────────────────┘       │
│                       │                                            │
│           ┌───────────┼───────────┐                                │
│           ▼           ▼           ▼                                │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│     │  Local   │ │ Marquez  │ │  MLflow  │                        │
│     │  JSON    │ │  (opt)   │ │ Artifact │                        │
│     └──────────┘ └──────────┘ └──────────┘                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              AuditTrail (compliance/audit.py)            │       │
│  │  DATA_ACCESS | MODEL_TRAINING | TEST_EVALUATION |        │       │
│  │  MODEL_DEPLOYMENT | CONFIG_CHANGE                        │       │
│  └────────────────────┬────────────────────────────────────┘       │
│                       │                                            │
│           ┌───────────┼───────────┐                                │
│           ▼           ▼           ▼                                │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│     │  JSON    │ │  DuckDB  │ │  Model   │                        │
│     │  Files   │ │  Export  │ │  Cards   │                        │
│     └──────────┘ └──────────┘ └──────────┘                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Data Versioning (DVC + GCS)                  │       │
│  │  Raw data → Splits → Checkpoints → Exported models       │       │
│  │  Every version hashed and tracked in .dvc files           │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Knowledge Graph (65 decision nodes)          │       │
│  │  Architectural decisions with Bayesian posteriors         │       │
│  │  → Maps to IEC 62304 design rationale documentation      │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘

Regulatory Export Targets:
  → Design History File (DHF): Git log + MLflow + KG → auto-generated
  → Traceability Matrix: KG decisions → OpenSpec specs → Tests → Code
  → Model Cards: MLflow run metadata → standardized JSON/PDF
  → PCCP: Factorial design → predetermined change control plan
  → Audit Trail: JSON → exportable for regulatory submission
  → Lineage Graph: Marquez UI → figure for manuscript/submission
```

---

## 9. Comparison with AHMED Project Approach

| Aspect | AHMED (Solita/VTT) | MinIVess | Assessment |
|--------|-------------------|----------|------------|
| **Issue tracker** | JIRA + Confluence | GitHub Issues (+ Linear planned) | Better DevEx |
| **ALM tool** | Polarion | Knowledge graph + OpenSpec | Different paradigm, similar coverage |
| **Data versioning** | DataLad (inherent logging) | DVC + OpenLineage (compensatory) | Equivalent with proper wiring |
| **MLOps** | CD4ML (Thoughtworks) | Prefect + MLflow + Docker | More mature infrastructure |
| **Traceability** | Polarion automatic links | KG → OpenSpec → Tests | Needs export tooling |
| **Model cards** | Custom JSON | Not yet implemented | Easy to add |
| **Compliance docs** | Manual + Polarion templates | Auto-generated from pipeline | More ambitious |
| **Risk management** | ISO 14971 in Polarion | KG Bayesian decisions | Different formalism |
| **QMS** | ISO 13485 certification | Not applicable (preclinical) | Future requirement |

The AHMED project's model card approach ([Lähteenmäki et al. 2023](https://cris.vtt.fi), §4.9.1) — "a metadata document that captures administrative information, model parameters, quantitative information, and considerations" — maps directly to what we can auto-generate from MLflow run metadata.

---

## 10. Key Literature Summary

### Regulatory Framework
- [IEC 62304:2006+A1:2015. "Medical device software — Software life cycle processes."](https://www.iso.org/standard/45557.html) — Core standard, Edition 2 expected September 2026 with AI lifecycle additions
- [FDA-2021-D-0775 (2023). "Content of Premarket Submissions for Device Software Functions."](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions) — Comprehensive guidance on software documentation for 510(k)/De Novo/PMA
- [FDA (2022). "Clinical Performance Assessment for CADe Devices."](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-performance-assessment-considerations-computer-assisted-detection-devices-applied-radiology) — Test data reuse guidance
- [FDA/Health Canada/MHRA (2021). "Good Machine Learning Practice."](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles) — 10 GMLP principles
- [Warraich et al. (2024). "FDA Perspective on AI Regulation." *JAMA*.](https://jamanetwork.com/journals/jama/fullarticle/2825146) — Total Product Lifecycle approach

### Test Set & Validation
- [Van Calster et al. (2023). "No such thing as a validated model." *BMC Medicine*.](https://link.springer.com/article/10.1186/s12916-023-02779-w)
- [Lenharo (2024). "Testing AI in medicine is a mess." *Nature*.](https://doi.org/10.1038/d41586-024-02675-0)
- [Chouffani El Fassi et al. (2024). "Not all FDA-authorized AI tools are clinically validated." *Nature Medicine*.](https://doi.org/10.1038/s41591-024-03060-y)
- [Giese (2024). "Test Dataset Reuse: A $100k+ Mistake." *Innolitics*.](https://innolitics.com)

### MLOps & RegOps
- [Lähteenmäki et al. (2023). "AHMED — Agile and Holistic Medical Software Development." *VTT*. VTT-R-01079-22.](https://cris.vtt.fi) — DataLad vs DVC, model cards, RegOps lifecycle, traceability
- [Muehlematter et al. (2023). "FDA-cleared AI/ML 510(k) predicate networks." *Lancet Digital Health*.](https://doi.org/10.1016/S2589-7500(23)00126-7)
- [Aboy et al. (2024). "Beyond the 510(k): De Novo pathway." *npj Digital Medicine*.](https://doi.org/10.1038/s41746-024-01021-y)

### SaMD Documentation
- [Giese (2021). "Documentation for Medical Device Software." *Innolitics*.](https://innolitics.com/articles/documentation-for-medical-device-software/)
- [Joseph (2021). "Documentation for Medical Device Software." *Sunstone Pilot*.](https://sunstonepilot.com/2021/08/documentation-for-medical-device-software/)
- [Giese (2024). "To Rewrite or Not: Prototype to FDA-Compliant Software." *Innolitics*.](https://innolitics.com/articles/to-rewrite-or-not-from-prototype-to-fda-compliant-software/)

### 2025-2026 Regulatory Updates
- [FDA (2025). "AI-Enabled Device Software Functions: Lifecycle Management." *Draft Guidance*.](https://www.fda.gov/media/184856/download) — January 2025; test data documentation, data catalog, representativeness
- [FDA (2024). "PCCP Marketing Submission Recommendations." *Final Guidance*.](https://www.fda.gov/media/166704/download) — December 2024; finalized PCCP framework
- [Carvalho et al. (2025). "Predetermined Change Control Plans: Guiding Principles." *JMIR AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/) — Test set evolution during PCCP updates
- [RookQS (2025). "FDA Expectations for AI/ML Model Training in SaMD."](https://rookqs.com/blog-rqs/fda-expectations-for-ai/ml-model-training-in-samd-2025-guide) — Data cataloging, traceability requirements
- [LFH Regulatory (2026). "IEC 62304 Update 2026."](https://lfhregulatory.co.uk/iec-62304-update-2026/) — Edition 2 timeline and changes
- [ComplianceQuest (2026). "FDA QMSR 2026."](https://www.compliancequest.com/blog/fda-quality-management-system-regulation-qmsr-2026/) — QSR → QMSR transition
- [IntuitionLabs (2024). "QMS for AI/ML Medical Devices Guide."](https://intuitionlabs.ai/articles/qms-ai-ml-medical-devices-guide) — QMS-level data management requirements

---

## 11. Recommendations

### Immediate (During PR-C through PR-E execution)

1. **Wire OpenLineage into all 5 flows** — ~30 lines total, zero DevEx impact
2. **Add test set access logging** — `AuditTrail.log_test_evaluation()` in Eval Flow
3. **Add cumulative test set use counter** — MLflow metric `eval/test_set_access_count`
4. **Generate model cards** — JSON artifact per training run

### Short-term (After factorial experiment)

5. **Connect Marquez to PostgreSQL** — persistent lineage storage
6. **Implement PCCP template** — document factorial design as predetermined change plan
7. **Generate lineage graph figure** — for Nature Protocols manuscript supplementary

### Medium-term (Q3-Q4 2026)

8. **Linear integration** — IEC 62304 Problem Resolution workflow automation
9. **DHF auto-generation** — Design History File from Git + MLflow + KG
10. **Traceability matrix export** — KG → OpenSpec → Tests → Code

### Long-term (Clinical projects)

11. **ISO 14971 risk management** — formalize KG decisions in ISO format
12. **ISO 13485 QMS** — quality management system certification
13. **21 CFR Part 11** — electronic signatures for regulatory submissions
14. **SBOM generation** — CycloneDX for FDA cybersecurity requirements
