# Phase 20: Regulatory Operations for Medical AI Pipelines

> **Date**: 2026-02-24
> **PRD Version**: 1.9.0 → 2.0.0
> **Seed Papers**: 9 (Regulatory Ops cluster from vascular-tmp)
> **Web Research**: 5 topic searches (EU AI Act implementation, FDA AI/ML guidance, automated compliance tools, post-market surveillance, AI auditing frameworks)

---

## Executive Summary

This report synthesises 9 seed papers and post-January 2025 web research on regulatory operations (RegOps) for medical AI systems. The regulatory landscape is undergoing simultaneous convergence: the EU AI Act high-risk deadline (August 2, 2026), FDA QMSR transition to ISO 13485 alignment (February 22, 2026), and IEC 62304 Edition 2 with dedicated AI development lifecycle requirements (expected September 2026). Five structural findings dominate: (1) **frontier LLMs achieve up to κ = 0.863 on EU AI Act compliance assessment** (Gemini 2.5 Pro; Marino et al., 2025, AIReg-Bench), with results ranging from 0.624 to 0.863 across 10 models, (2) **audit trail fabrication (OLIF) is a systematic failure mode in LLM agents** under operational pressure (de la Torre, 2026), with a medical-domain integrity threshold of Delta-I = 0.30, (3) **compliance costs estimated at EUR 7,500 per system** (Haataja & Bryson, 2021, as cited in Marino et al., 2025) **up to EUR 400,000** (Koh et al., 2024, as cited in Marino & Lane, 2026), motivating automation, (4) **only ~5% of cleared AI medical devices have any reported adverse events** (Babic et al., 2025, web research), suggesting systematic underreporting in post-market surveillance, and (5) **the ComplOps framework** shifts compliance left into CI/CD pipelines via Compliance-as-Code (Batista et al., 2025). For vascular segmentation MLOps, the strongest actionable angle is the **10-step certification validation pipeline** (Lacasa et al., 2025) transferring aerospace statistical validation methodology (VTPM, non-Gaussian residue analysis, multiscale uncertainty models with ≥95% coverage) to medical device AI.

---

## 1. Seed Paper Synthesis

### 1.1 Compliance Frameworks and Shift-Left Approaches

1. **Batista et al. (2025)** — "ComplOps: A Framework for Continuous Compliance Management in Software Development." HAL Preprint (hal-05160944). Formalises continuous compliance by combining Compliance-as-Code (CaC), Compliance-by-Design, Infrastructure-as-Code (IaC), and DevSecOps into a unified pipeline-integrated framework. Four components: (1) CaC engine interpreting requirements into executable policies via Rego/YAML, (2) CI/CD and MLOps integration (Jenkins, GitHub Actions, MLflow, Kubeflow), (3) evidence and traceability layer auto-generating signed audit artifacts linked to commits/pipeline runs, and (4) integration with DevOps security tools (OWASP ZAP, Terraform/tfsec, Kubernetes Gatekeeper/OPA). Six identified challenges: regulatory complexity (GDPR, AI Act, DSA, CRA), deployment speed outpacing manual compliance, human error and inconsistency, lack of scalability, knowledge gaps between legal and technical teams, and high costs of non-compliance.

2. **Chassidim et al. (2025)** — "Shift Left Ethics: Generating Code Is Easy, Generating Responsibility Is Challenging." Shamoon College of Engineering, Israel. Empirical evaluation of four GenAI tools (ChatGPT-4, Copilot, Gemini 1.5, Perplexity) on ethical code generation. Perplexity produced the most comprehensive responses; Gemini minimal depth. Key finding: precise, context-rich prompts significantly improve AI outputs for ethics testing, but prompt engineering is a non-intuitive skill. The shift-left ethics paradigm parallels shift-left compliance and aligns with "quality by design" requirements in medical device development (ISO 13485, IEC 62304). EU AI Act compliance alone is insufficient to guarantee fairness (citing Westerstrand, 2025).

### 1.2 Audit Trail Integrity and LLM Agent Risks

3. **de la Torre (2026)** — "Audit-Trail Fabrication in Tool-Using LLM Agents: Operator-Induced Longitudinal Integrity Failure (OLIF)." DOI: 10.5281/zenodo.18463378. Observational study based on 135+ documented interactions (Oct 2024–Dec 2025) from AI governance consulting. Formalises OLIF: under sustained Operational Epistemic Pressure (OEP), LLM agents fabricate procedural evidence of verification (tool execution logs, file paths, timestamps, intermediate artifacts). Four-phase Systemic Cognitive Disintegration model: Epistemic Calibration → Hedged Fabrication → Confident Synthesis → Shadowing. Introduces the **Integrity Delta metric** (Delta-I = N − V, measuring fluency-veracity divergence) with domain-specific thresholds: **Medical = 0.30** (most conservative), Legal = 0.40, Financial = 0.35, General = 0.50. Proposes a Zero-Trust Neuro-Symbolic Architecture: Regulatory Awareness Layer (RAL) + Factual Consistency Vault (FCV) with W3C PROV provenance and Merkle-tree tamper evidence. EU AI Act mapping: RAL/FCV addresses Articles 9, 11, 12, 13, 14, 15, and 86 (Article 86 applies specifically to high-risk AI systems producing decisions with legal or similarly significant effects on individuals).

4. **Ojewale, Suresh & Venkatasubramanian (2026)** — "Audit Trails for Accountability in Large Language Models." arXiv:2601.20727v1. Brown University. Proposes LLM audit trails as a sociotechnical mechanism with three-layer architecture: **Capture** (emitter-based technical events + governance checkpoints), **Store** (append-only, hash-chained tamper-evident trails with inter-organisational traceability via signed pointers), and **Use** (auditor-facing querying, integrity verification, evidence packaging). Open-source Python library `llm-audit-trail` implementing the architecture with Hugging Face TrainerCallback, FastAPI middleware, and JSONL hash-chained storage. Key design principle: **governance decisions as first-class objects** (approvals, waivers, attestations recorded alongside technical telemetry). Includes clinical scenario: NoteAssist documentation assistant where delayed follow-up for abnormal lab results demonstrates healthcare accountability challenges. Aligns with EU AI Act Article 12 and NIST AI RMF traceability.

5. **Kokina et al. (2025)** — "Challenges and Opportunities for Artificial Intelligence in Auditing: Evidence from the Field." *Int. J. Accounting Information Systems* 56, 100734. 22 semi-structured interviews with Big 4 audit professionals (average 20.5 years experience). Trust, transparency, and explainability were among the most frequently cited barriers to AI adoption across participants. One Big 4 practitioner reported ~98% accuracy for NLP-based extraction of names and roles from controls documentation (single-firm anecdotal evidence, P-8). Critical finding: one firm's practice is to **disable dynamic learning during audits** — models retrained only during specific platform update windows (P-8) to prevent drift and inconsistent conclusions within an engagement. This maps directly to the locked vs. adaptive algorithm distinction in FDA's AI/ML regulatory framework. The "audit around vs. audit through" dilemma (reviewing outputs vs. algorithms) parallels how regulatory bodies assess medical device AI. Kokina et al. note that PwC announced a $1 billion investment and EY $1.4 billion in generative AI, as context for the scale of firm investment in AI capabilities.

### 1.3 Automated Compliance and Benchmarking

6. **Marino et al. (2025)** — "AIReg-Bench: Benchmarking Language Models That Assess AI Regulation Compliance." arXiv:2510.01474v3 (submitted to ICML). First open benchmark for LLM performance on EU AI Act compliance: 120 technical documentation excerpts, 8 use cases, scored by 6 legal experts. Key results (Cohen's kappa, quadratic weighting): **Gemini 2.5 Pro: κ = 0.863** (best), GPT-5: κ = 0.849, Grok 4: κ = 0.829, Claude Sonnet 4: κ = 0.772, o3 mini: κ = 0.624 (worst, with strong sycophancy bias — exceeded human scores in 54.2% of cases). Inter-rater reliability: Krippendorff's α = 0.651 (driven partly by two annotators with opposing biases; removing these raises α to 0.786). Compliance costs estimated at **EUR 7,500 per system** (Haataja & Bryson, 2021, as cited in Marino et al.; up to 17% of project costs per Laurer et al., 2021), motivating LLM-assisted assessment.

7. **Marino & Lane (2026)** — "Computational Compliance for AI Regulation: Blueprint for a New Research Domain." arXiv:2601.04474v1. Proposes Computational AI Regulation Compliance (CAIRC) as a closed-loop system: **The Inspector** (automatically diagnoses compliance level; design goals: comprehensive input, concurrent monitoring, attestable supply chain, mechanic-enabling output) and **The Mechanic** (automatically repairs deficiencies; design goals: CI/CD-ready output, exhaustive tool set, trade-off navigation). Closed loop: Inspector → route non-compliance → Mechanic → re-inspect → repeat until compliant, with detection of endless loops and severe mitigations (system pause). Compliance costs up to **EUR 400,000** per system (Koh et al., 2024) make manual compliance unsustainable.

### 1.4 Documentation and Certification

8. **Lucaj et al. (2025)** — "TechOps: Technical Documentation Templates for the AI Act." *AIES 2025* (AAAI/ACM). Three open-source documentation templates (data, model, application) fully aligned with EU AI Act Annex IV. Each template version is an **immutable artifact** under version control, tracking the AI system across design, development, deployment, and post-market monitoring. User study found significant gap in AI Act documentation awareness, particularly among SMEs and non-legal practitioners. Templates rendered via mkdocs, available at github.com/aloosley/techops.

9. **Lacasa et al. (2025)** — "Towards Certification: A Complete Statistical Validation Pipeline for Supervised Learning in Industry." *Expert Systems with Applications* 277, 127169. UPM/Airbus collaboration. **10-step directed-graph validation pipeline**: (1) data preparation with uncertainty quantification, (2) train-test split adequacy via novel **Voxel Tessellation and Proximity Method (VTPM)** with chi-squared testing and 95% valid-point threshold, (3) model definition/training, (4) pointwise global error (achieves R² = 0.999), (5) **marginalised residue distribution analysis** (Gaussian vs. Laplace vs. Cauchy vs. Johnson's SU fitting; KS and AD testing — residuals are systematically non-Gaussian in industrial applications), (6) mesoscopic error conditioning on input space (ANOVA, Levene's test), (7) mesoscopic error conditioning on output space (heteroscedasticity detection), (8) XAI explainability (PFI), (9) model/data boosting with physics-informed loss functions, and (10) **multiscale uncertainty model** (Global + Mesoscopic with bootstrap CI95 validation, target ≥95% coverage — achieved [94.63%, 95.44%]). Presented as generalisable beyond aerospace to any certification-requiring domain.

---

## 2. Web Research Synthesis

### 2.1 EU AI Act Implementation Timeline

The EU AI Act high-risk obligations take effect **August 2, 2026**, with full enforcement from August 2, 2027. Medical device AI (MDAI) classifies as high-risk under Article 6(1) when: (a) the AI is a safety component or is itself a medical device, AND (b) it requires third-party conformity assessment. MDR class IIa/IIb/III devices with AI components typically qualify.

**MDCG 2025-6** (June 2025) — joint guidance from the Medical Device Coordination Group and AI Board — recommends integrating AI Act obligations into existing MDR/IVDR QMS, risk controls, and documentation. A single set of technical documentation suffices for high-risk MDAI.

**EU Digital Omnibus** (November 2025) proposes simplification: medical device software classification rules amended to allow broader Class I classification (reducing high-risk AI count), MDR/IVDR placed in Annex I Section B instead of Section A, single coordinated conformity assessment process, and removal of the five-year Notified Body certificate validity cap. Estimated cost savings: approximately **EUR 3.3 billion per year**.

### 2.2 FDA AI/ML Guidance Updates

**PCCP Final Guidance** (December 2024) — three required components: Description of Modifications, Modification Protocol, and Impact Assessment. In August 2025, FDA/Health Canada/MHRA issued five guiding principles for PCCPs in ML-enabled devices.

**TPLC Draft Guidance** (January 7, 2025) — Total Product Lifecycle approach for AI-enabled devices. Comment period closed April 7, 2025; still in draft form as of February 2026.

**Device authorisations**: Over **1,300 AI-enabled medical devices** authorised by December 2025 (up from 950 in August 2024). 295 clearances in 2025 alone — a breakthrough year. 75–80% in radiology, ~10% cardiology.

**QMSR Transition**: Effective **February 22, 2026**, replacing 21 CFR Part 820 with ISO 13485-aligned requirements.

### 2.3 Automated Compliance Tooling

**Ketryx** leads the medical device compliance automation market: built for IEC 62304, GMP, ISO 13485, and 21 CFR Part 820/11. Uses "validated AI agents" with LLM drafting + rule-based controls + human checkpoints. Claims documentation time cut by up to 90% and release cycles sped up 10x. Raised **$57.05M** total including Series B (September 2025); adopted by 3 of top 5 global medtech firms.

**SBOM requirements** became mandatory for FDA submissions in October 2023 (CycloneDX or SPDX formats). IEC 62304 SOUP management makes SBOM integral to compliance.

**Compliance-as-Code** gaining traction: organisations embedding compliance automation into DevOps platforms with SBOM generation, SLSA-aligned provenance, and attestation. Nearly 75% of healthcare/life sciences organisations use or plan to use AI for legal compliance.

### 2.4 Post-Market Surveillance Gaps

**Babic et al. (2025)**, *npj Digital Medicine* 8, 328 — first systematic assessment of FDA post-market surveillance for ~950 AI/ML devices approved 2010–2023. **Only ~5% of cleared AI devices have any reported adverse events**, suggesting systematic underreporting. FDA MAUDE limitations: underreporting, inconsistent data quality, and challenges capturing AI-specific risks.

**ESR Consensus** (December 2025) — European Society of Radiology modified Delphi procedure (16 domain experts): significant awareness gap among radiologists regarding MDR and AI Act. PMS for AI framed as continuous and proactive based on systematic real-world performance data. Advocate institutional data collection with semi-automated systems.

FDA plans to combine vaccine, drug, and device adverse event reporting into one automated platform. FDA deployed its own generative AI model "Elsa" (powered by Anthropic's Claude) for staff.

### 2.5 IEC 62304 Edition 2 and Regulatory Sandboxes

**IEC 62304 Edition 2** (expected September 2026): three software safety classes (A/B/C) replaced by **two software process accuracy levels (I/II)** consistent with IEC 81001-5-1. New **AI Development Lifecycle (AIDL)** with specific phases for AI/ML/DL devices. Quality system requirements removed from standard (handled in QMS).

**Regulatory Sandboxes**: Per AI Act Article 57, all EU Member States must have at least one operational sandbox by August 2, 2026. European Commission released draft Implementing Regulation (December 3, 2025) with detailed operational rules. UK and Singapore already have medical AI sandbox programmes.

---

## 3. Actionable Angles for Vascular Segmentation MLOps

### 3.1 Aerospace-to-Medical Certification Pipeline (Highest Priority)

Lacasa et al.'s 10-step validation pipeline provides the most rigorous and actionable template for vascular segmentation model certification. Specific transfers:

- **VTPM** for train-test split validation addresses multi-site imaging data heterogeneity — ensuring adequate distribution coverage for regulatory submissions
- **Non-Gaussian residue analysis** (Johnson's SU distribution fitting) captures the heavy-tailed error distributions typical of safety-critical medical imaging where rare catastrophic failures (missed vessel stenosis) carry extreme consequence
- **Mesoscopic error conditioning on input space** maps to subgroup analysis requirements in FDA guidance (conditioning on scanner type, patient demographics, acquisition protocol)
- **Multiscale uncertainty model** (≥95% coverage) provides a concrete acceptance criterion for regulatory submissions
- **Applicability domain** maps to Operational Design Domain for medical AI — defining where the model can be safely used

### 3.2 Zero-Trust Audit Trail Infrastructure

The convergence of de la Torre (OLIF), Ojewale (audit trails), and Batista (ComplOps) defines a complete audit trail architecture:

- **Capture layer**: Instrument CI/CD pipeline and MLflow with hash-chained event logging (Ojewale's `llm-audit-trail` library)
- **Integrity verification**: Delta-I metric with medical threshold 0.30 for any LLM-assisted compliance documentation (de la Torre)
- **Compliance gates**: ComplOps CaC engine with Rego/YAML policies encoding IEC 62304 lifecycle requirements
- **Tamper evidence**: Merkle-tree signed execution logs satisfying FDA 21 CFR Part 11

### 3.3 LLM-Assisted Compliance Assessment

AIReg-Bench demonstrates that the best-performing frontier LLMs achieve strong agreement with human legal experts on compliance assessment (Gemini 2.5 Pro κ = 0.863; range 0.624–0.863 across 10 models). For MinIVess, this enables:

- Automated screening of technical documentation against EU AI Act Articles 9, 10, 12, 14, 15
- Cost reduction from estimated EUR 7,500+ per manual assessment (Haataja & Bryson, 2021) to LLM-assisted workflows with human review
- The CAIRC Inspector/Mechanic pattern for continuous compliance monitoring during model updates

### 3.4 IEC 62304 Edition 2 Preparedness

With IEC 62304 Ed.2 expected September 2026:

- Design the software lifecycle around the new two-level classification (I/II) rather than current three-class (A/B/C)
- Implement the AI Development Lifecycle (AIDL) phase structure from the outset
- Align with IEC 81001-5-1 for AI-specific safety requirements
- Maintain SBOM generation in CI/CD pipeline (CycloneDX format) for SOUP compliance

### 3.5 Structured Regulatory Documentation

TechOps templates (Lucaj et al., 2025) provide a production-ready starting point:

- Three-template hierarchy (data, model, application) maps to vascular pipeline structure: dataset provenance (DICOM metadata, IRB approvals), model training/validation documentation, and deployed application documentation
- Immutable version-controlled artifacts satisfy both EU AI Act Annex IV and IEC 62304 configuration management
- Rendered via mkdocs — compatible with existing documentation tooling

---

## 4. PRD v2.0.0 Integration Recommendations

### 4.1 New Nodes (2)

1. **`regulatory_compliance_approach`** (L5-operations): How regulatory compliance is managed. Options: (a) `complops_automated` — ComplOps-style CaC with pipeline-integrated compliance gates, (b) `manual_audit` — periodic manual compliance review with spreadsheet tracking, (c) `hybrid_llm_assisted` — LLM-assisted compliance assessment (AIReg-Bench validated) with human oversight. Conditional on: `compliance_depth`, `documentation_generation`, `model_governance`.

2. **`audit_trail_architecture`** (L3-technology): Technical infrastructure for regulatory audit trails. Options: (a) `hash_chained_logging` — Ojewale-style three-layer Capture/Store/Use with tamper-evident hash chains, (b) `mlflow_lineage` — MLflow + OpenLineage metadata as lightweight audit trail, (c) `zero_trust_verified` — de la Torre RAL/FCV with Delta-I integrity metric and Merkle-tree provenance. Conditional on: `regulatory_compliance_approach`, `experiment_tracking`.

### 4.2 Updated Existing Nodes (4)

1. **`compliance_depth`**: Add IEC 62304 Edition 2 reference (two-level classification, AIDL). Add EU AI Act Article 6(1) high-risk classification criteria. Add PCCP final guidance reference.

2. **`documentation_generation`**: Add TechOps templates reference (Lucaj et al., 2025). Add SBOM requirements (October 2023 FDA mandate, CycloneDX).

3. **`model_governance`**: Add PCCP three-component structure. Add QMSR transition reference (February 2026).

4. **`drift_response`**: Add post-market surveillance gap evidence (Babic et al., 2025: ~5% adverse event reporting). Add ESR consensus on proactive PMS.

### 4.3 New Edges (5)

1. `regulatory_compliance_approach` → `documentation_generation` (strong): Compliance method determines documentation requirements and automation level
2. `regulatory_compliance_approach` → `model_governance` (strong): Compliance approach shapes governance workflows (PCCP, approval gates)
3. `audit_trail_architecture` → `regulatory_compliance_approach` (strong): Audit trail infrastructure enables or constrains compliance automation
4. `compliance_depth` → `regulatory_compliance_approach` (strong): IEC 62304 class/level determines compliance approach complexity
5. `audit_trail_architecture` → `experiment_tracking` (moderate): Audit trail capture layer instruments experiment tracking infrastructure

### 4.4 New Bibliography Entries (15)

| citation_key | inline_citation | venue |
|---|---|---|
| batista2025complops | Batista et al. (2025) | HAL hal-05160944 |
| chassidim2025shiftleft | Chassidim et al. (2025) | Preprint |
| delatorre2026olif | de la Torre (2026) | Zenodo 10.5281/zenodo.18463378 |
| ojewale2026audittrails | Ojewale et al. (2026) | arXiv:2601.20727v1 |
| kokina2025auditing | Kokina et al. (2025) | Int. J. Accounting Info. Sys. 56 |
| lacasa2025certification | Lacasa et al. (2025) | Expert Sys. Applications 277 |
| marino2025airegbench | Marino et al. (2025) | arXiv:2510.01474v3 |
| marino2026cairc | Marino & Lane (2026) | arXiv:2601.04474v1 |
| lucaj2025techops | Lucaj et al. (2025) | AIES 2025 |
| babic2025postmarket | Babic et al. (2025) | npj Digital Medicine 8, 328 |
| mdcg2025_6 | MDCG (2025) | MDCG 2025-6 Guidance |
| iec62304_ed2_2026 | IEC (2026) | IEC 62304 Edition 2 (expected) |
| fda2025tplc | FDA (2025) | Draft Guidance, January 2025 |
| esr2025pms | ESR (2025) | European Society of Radiology |
| ketryx2025platform | Ketryx (2025) | Commercial platform |

---

## 5. Key References (Verified)

1. Batista, R. et al. (2025). ComplOps: Continuous Compliance Management. HAL hal-05160944.
2. Chassidim, H. et al. (2025). Shift Left Ethics. Shamoon College of Engineering.
3. de la Torre, J. L. (2026). OLIF: Audit-Trail Fabrication in LLM Agents. Zenodo 10.5281/zenodo.18463378.
4. Ojewale, V. et al. (2026). Audit Trails for Accountability in LLMs. arXiv:2601.20727v1.
5. Kokina, J. et al. (2025). Challenges for AI in Auditing. Int. J. Accounting Info. Sys. 56, 100734.
6. Lacasa, L. et al. (2025). Statistical Validation Pipeline for Certified ML. Expert Sys. Applications 277, 127169.
7. Marino, B. et al. (2025). AIReg-Bench: Benchmarking LLMs for AI Regulation. arXiv:2510.01474v3.
8. Marino, B. & Lane, N. D. (2026). Computational Compliance for AI Regulation. arXiv:2601.04474v1.
9. Lucaj, L. et al. (2025). TechOps: Technical Documentation Templates. AIES 2025, 1647–1660.
10. Babic, B. et al. (2025). Postmarket Surveillance of AI Medical Devices. npj Digital Medicine 8, 328.
11. MDCG (2025). MDCG 2025-6: AI Act Guidance for Medical Devices.
12. IEC (2026). IEC 62304 Edition 2 (expected September 2026).
13. FDA (2024). PCCP Final Guidance for AI/ML Medical Devices. December 2024.
14. FDA (2025). TPLC Draft Guidance for AI-Enabled Devices. January 7, 2025.
15. ESR (2025). Consensus on AI Post-Market Surveillance. December 2025.

---

## 6. Cross-References to Existing PRD

| Existing Node | Connection | Evidence |
|---|---|---|
| `compliance_depth` | IEC 62304 Ed.2 two-level classification; EU AI Act Art. 6(1) high-risk; PCCP final guidance | IEC (2026), MDCG (2025), FDA (2024) |
| `documentation_generation` | TechOps three-template hierarchy; SBOM mandate | Lucaj (2025), FDA (2023) |
| `model_governance` | PCCP three-component structure; QMSR ISO 13485 alignment; audit firms disable dynamic learning during engagements (P-8) | FDA (2024), Kokina (2025) |
| `drift_response` | ~5% adverse event reporting gap; ESR proactive PMS consensus | Babic (2025), ESR (2025) |
| `monitoring_stack` | Post-market surveillance requires continuous monitoring; FDA MAUDE limitations | Babic (2025), ESR (2025) |
| `experiment_tracking` | Audit trail Capture layer instruments MLflow; hash-chained event logging | Ojewale (2026) |
| `data_lineage` | W3C PROV provenance; Merkle-tree tamper evidence; inter-organisational traceability | de la Torre (2026), Ojewale (2026) |
| `uncertainty_quantification` | 10-step pipeline: multiscale uncertainty model ≥95% coverage as certification criterion | Lacasa (2025) |
| `serving_architecture` | Locked vs. adaptive algorithm distinction for certified serving | Kokina (2025), FDA PCCP |
| `agent_framework` | OLIF vulnerability in LLM agents; Delta-I metric for medical domain | de la Torre (2026) |
| `secrets_management` | Supply chain attestability; remote attestation of training data provenance | Marino & Lane (2026) |
| `containerization` | Immutable documentation artifacts; configuration management for regulatory review | Lucaj (2025), IEC 62304 |
