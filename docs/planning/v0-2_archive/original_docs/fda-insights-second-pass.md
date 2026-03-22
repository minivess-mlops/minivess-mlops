# FDA Readiness Second Pass: Innolitics Trends, SBOM, SecOps, QMSR, PCCP & MLOps Maturity

**Created**: 2026-03-17
**Updated**: 2026-03-18 (enriched with post-June 2025 academic/industry sources)
**Status**: Active — extends [first-pass report](regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md)
**Seed document**: `docs/planning/fda-insights-innolitics-trends-seed-from-linkedin.md`
**Issue**: [#821](https://github.com/petteriTeikari/minivess-mlops/issues/821) (P1)

---

## Original User Prompt (verbatim)

> Could we extend this FDA readiness planning from these Linkedin posts from Innolitics: fda-insights-innolitics-trends-seed-from-linkedin.md. The posts obviously themselves are not very in-depth but read the .md doc line-by-line to get an idea what are recent hot topics in FDA regulatory stack. Can we easily describe our SBOM? What should we do to improve our cybersecurity / SecOps / MLSecOps readiness, is our repo ready for the new QMSR framework? Is our stack aligned optimally with PCCP? How about the MLOps maturity level 4 and continuous automatic re-training? Drift detection and retraining with postmarket surveillance study? How does our system keep collecting information (if allowed) from all the sites that are using this? As in the data sharing cannot be enabled by default, but different institutions (and team of one deparment) could opt-in to pool their "neurovex installations" both for "ML audit" and also any UX issues for the data annotation and dashboard (e.g. monitored with Posthog, Sentry, Intercom, and what have you for example for the malleable agentic generative UI with AG-UI and A2UI?). How is the SaMD stack converging with the "regular software" and how about other non-medical software in other regulated domains (digital product passports in EU, UAD 3.6 in USA for real estate, etc.). This is quite comprehensive task so create an improved second pass research report with our iterated-llm-council Skill and iterate as an academic reviewer for this open-ended multi-hypothesis research report on robustifying our repo to be used in regular FDA contexts and contextualize these into preclinical neuroscience biomedical tasks! As in we sure benefit as a preclinical mlops platform for audit trails on what have been done for example with OpenLineage, and we are interested in possible overuse of test dataset and planning how to be compliance ready from day one! Is this clear? Ask any questions and use the innolitics trends as a seed document and now focus on blog posts, industry reports, academic articles, preprints, etc published after June 2025 to keep this fresh and what libraries should we think of adding, and how could we use the tool stack better?

---

## Executive Summary

This second-pass report extends the first-pass FDA readiness analysis by integrating 20+ LinkedIn posts from Innolitics (J. David Giese, 2025-2026) with deep-dive research into seven regulatory/engineering themes that the Innolitics posts flag as hot topics. The analysis is contextualized for **preclinical neuroscience biomedical imaging** (2-photon microscopy of rodent cerebrovasculature) while maintaining architectural readiness for clinical SaMD pathways.

**Key takeaways from the Innolitics trend scan**:
1. **SBOM is no longer optional** — FDA requires it in premarket submissions; automated CI/CD generation is mandatory
2. **Cybersecurity is now a QMS requirement** under QMSR (Feb 2, 2026) — not just product design
3. **CI/CD is explicitly recognized as production controls** under the updated cybersecurity guidance
4. **PCCP adoption is accelerating** but unevenly — neurology panels are less AI-fluent than radiology
5. **The Senate "FDA of the Future" report** signals PCCPs have "tremendous potential" and post-market surveillance should be "core architecture, not afterthought"
6. **Modular architecture** creates compounding ROI for multi-indication platforms (N x months saved)
7. **Agentic AI tools can be used for FDA-regulated development** — the same principles apply
8. **Harrison.ai's 510(k) partial exemption petition** could make PCCP obsolete for certain product codes

---

## 1. SBOM: Can We Easily Describe Ours?

### 1.1 What FDA Expects

Per Innolitics and the Feb 2026 cybersecurity guidance update:

> "FDA requires SBOMs in premarket submissions — but most teams treat them as a checkbox. A well-structured SBOM does more than list your dependencies. It signals to FDA that you understand your software supply chain and have a plan for monitoring it post-market." — [Giese (2026), Innolitics](https://innolitics.com)

Six practices for strong SBOM submissions:
1. **Automate generation in CI/CD pipeline** — manual SBOMs go stale instantly
2. **Include software metadata** — company name, contact, git hash, timestamp
3. **Cover every NTIA baseline field** + FDA additions: level of support, end-of-support date
4. **Continuously monitor vulnerabilities** in the field, not just at submission time
5. **Provide both human-readable and machine-readable** versions — JSON alone doesn't cut it
6. **Periodically review vulnerabilities for patient safety impact** — not just logging

**August 2024 change**: FDA pushed back on email-only SBOM distribution. SBOMs must be readily available to end users at all times as part of device labeling.

### 1.1.1 Post-June 2025 Regulatory Developments

SBOMs are now **mandatory** for medical devices under **Section 524B of the FD&C Act**. The Feb 2026 cybersecurity guidance reissue confirmed SBOM requirements are woven into QMSR-aligned processes ([RAPS, 2026](https://www.raps.org/news-and-articles/news-articles/2026/2/fda-reissues-cybersecurity-guidance-to-align-with); [DLA Piper, 2026](https://www.dlapiper.com/en-us/insights/publications/2026/02/fda-issues-revised-cybersecurity-premarket-submission-guidance)).

- **CycloneDX** is security-focused with native vulnerability tracking (VEX); **SPDX** is compliance-focused with deep license documentation. FDA accepts both formats plus SWID tags ([Sbomify, 2026](https://sbomify.com/2026/01/15/sbom-formats-cyclonedx-vs-spdx/); [Blue Goat Cyber, 2025](https://bluegoatcyber.com/blog/navigating-the-fdas-sbom-requirements-for-medical-device-manufacturing-spdx-cdx-and-more/))
- **CycloneDX v7.2.2** now natively supports `uv` virtual environments, producing near Level-2 OWASP SCVS-compliant SBOMs ([Sbomify Python guide, 2026](https://sbomify.com/guides/python/))
- NTIA Minimum Elements remain the baseline: supplier name, component name, version, unique identifier, relationship, plus FDA additions (level of support, end-of-support date)

### 1.2 Our Current State

| Aspect | Status | Gap |
|--------|--------|-----|
| **pyproject.toml** | Lists all dependencies | Not in SBOM format |
| **uv.lock** | Pinned versions with hashes | Machine-readable but not CycloneDX/SPDX |
| **Docker images** | Multi-stage builds with pinned base images | No SBOM embedded in image labels |
| **CI/CD SBOM generation** | Not implemented | Need `cyclonedx-bom` or `syft` in pre-commit |
| **Vulnerability monitoring** | Not implemented | Need `grype` or `osv-scanner` |
| **Human-readable SBOM** | Not generated | Need PDF/Excel export alongside JSON |

### 1.3 Recommended Actions

**Immediate** (can add to pre-commit or Makefile):

```bash
# Generate CycloneDX SBOM from uv.lock
uv run cyclonedx-py environment -o sbom.json --format json
uv run cyclonedx-py environment -o sbom.xml --format xml

# Scan for known vulnerabilities
grype sbom:sbom.json --output table
```

**Libraries to add**:
- `cyclonedx-bom` (or `cyclonedx-python-lib`) — CycloneDX SBOM generation from Python environment
- `syft` (Anchore) — container-level SBOM for Docker images
- `grype` (Anchore) — vulnerability scanner that reads CycloneDX/SPDX SBOMs
- `osv-scanner` (Google) — alternative vuln scanner using the OSV database

**Integration point**: Generate SBOM as a Docker build artifact, embed in image labels, and log as MLflow artifact per deployment.

### 1.4 SBOM and the Knowledge Graph

The KG already has `sbom_generation` as a decision node with status `not_started` and candidates `[syft_uv_lock, cyclonedx]`. **Recommendation**: resolve to `cyclonedx_plus_grype` — CycloneDX for SBOM generation (Python-native, integrates with `uv.lock`), Grype for vulnerability scanning.

---

## 2. Cybersecurity / SecOps / MLSecOps Readiness

### 2.1 The QMSR-Cybersecurity Convergence

The Feb 2026 cybersecurity guidance update is the most significant development from the Innolitics posts:

> "Your SBOM, threat modeling, security architecture, and vulnerability management are now QMS compliance documentation." — [Giese (2026), Innolitics](https://innolitics.com)

Key changes:
- All CFR 820 references replaced with **QMSR citations**
- Full alignment with **ISO 13485:2016** requirements incorporated by reference
- **Tool validation** requirements now referenced under QMSR 4.1.6
- **CI/CD explicitly recognized as production controls** under QMSR

This means our Docker + Prefect + pre-commit pipeline is already a "production control" in FDA terms. The gap: we don't **document it as such**.

### 2.2 The 14 Common Cybersecurity Deficiencies

From Innolitics' analysis of real AINN (Additional Information) letters:

1. Incomplete threat modeling (misses insider threats, supply chain risks)
2. Missing/vague authentication and authorization controls
3. No SBOM, or SBOM that omits OTS and open-source components
4. Weak encryption documentation without algorithm justification
5. No clear timelines for patch management and security updates
6. ...and 9 more detailed in their article

### 2.3 Our SecOps Stack Assessment

| Component | Status | FDA Relevance |
|-----------|--------|---------------|
| **Docker multi-stage builds** | Implemented | Isolates build dependencies from runtime |
| **Pre-commit hooks** | Active (ruff, mypy, secrets detection) | Automated code quality gates = production controls |
| **Git signed commits** | Not enforced | Future requirement for 21 CFR Part 11 |
| **Dependency pinning** | `uv.lock` with hashes | Supply chain integrity |
| **Container scanning** | Not implemented | Need `trivy` or `grype` on Docker images |
| **Secret detection** | `detect-secrets` in pre-commit | Prevents credential leaks |
| **Network isolation** | Docker Compose networks | Documented but not threat-modeled |
| **SBOM generation** | Not implemented | Critical gap (see Section 1) |
| **Threat model** | Not documented | Needed for any premarket submission |
| **Penetration testing** | Not applicable (preclinical) | Future requirement |

### 2.4 MLSecOps: AI-Specific Security

Three frameworks now anchor medical AI security assessment (post-June 2025):

- **MITRE ATLAS** (Adversarial Threat Landscape for AI Systems) — threat taxonomy mapping AI-specific attacks to defensive controls. Updated continuously; provides the vocabulary for FDA cybersecurity threat modeling ([Practical DevSecOps, 2026](https://www.practical-devsecops.com/mitre-atlas-framework-guide-securing-ai-systems/))
- **OWASP Top 10 for LLMs/GenAI (2025 edition)** — though primarily for LLM applications, the taxonomy of prompt injection, training data poisoning, and model denial of service applies to our BentoML serving endpoint
- **NIST AI-RMF** — risk management framework that provides the process scaffolding for medical AI security assessment

A recent medRxiv preprint introduces a taxonomy of **8 adversarial attack categories with 24 sub-strategies** for medical AI red-teaming ([medRxiv, 2026](https://www.medrxiv.org/content/10.64898/2026.02.26.26347212v1)). Research confirms that multimodal models show enhanced adversarial resilience vs single-modality counterparts ([Frontiers in Medicine, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328349/); [Springer AI Review, 2024](https://link.springer.com/article/10.1007/s10462-024-11005-9)).

Beyond traditional cybersecurity, ML models have unique attack surfaces:

| Threat | MinIVess Relevance | Mitigation |
|--------|-------------------|------------|
| **Data poisoning** | MiniVess dataset could be tampered | DVC hash verification + audit trail |
| **Model extraction** | BentoML endpoint could be probed | API rate limiting, authentication |
| **Adversarial inputs** | Malformed volumes at inference time | Input validation in serving layer |
| **Model inversion** | Privacy leakage from model weights | Not applicable (preclinical rodent data) |
| **Supply chain attacks** | Compromised PyTorch/MONAI dependencies | SBOM + vulnerability monitoring |

**Recommended library additions**:
- `safety` or `pip-audit` — Python package vulnerability scanning
- `trivy` — comprehensive container vulnerability scanner
- `ciso-assistant-community` — open-source GRC framework (from SOC2 research)

### 2.5 SOC2/ISO 27001 Convergence with SaMD

From the SOC2 strategy research: the principles are **converging** with medical device QMS:

| SOC2 Principle | FDA/QMSR Equivalent | Our Coverage |
|----------------|---------------------|--------------|
| Security (CC1-CC9) | QMSR cybersecurity, IEC 81001-5-1 | Partial (secrets, Docker isolation) |
| Availability (A1) | Postmarket surveillance, uptime | Not applicable (preclinical) |
| Processing Integrity (PI1) | Data integrity, IEC 62304 §8 | DVC hashing, OpenLineage |
| Confidentiality (C1) | HIPAA/GDPR (clinical) | N/A for preclinical rodent data |
| Privacy (P1-P8) | GDPR (EU), secondary use laws | N/A for preclinical |

**Key insight from Pulumi CrossGuard**: Infrastructure-as-Code policy enforcement (Pulumi CrossGuard) can automatically validate compliance policies on every `pulumi up`. Since we already use Pulumi for GCP IaC, adding compliance policies is trivial.

---

## 3. QMSR Readiness Assessment

### 3.1 What Changed (February 2, 2026)

The QMSR replaced 21 CFR Part 820 by incorporating ISO 13485:2016 into federal law ([ComplianceQuest, 2026](https://www.compliancequest.com/blog/fda-quality-management-system-regulation-qmsr-2026/); [WQS, 2026](https://us.wqs.de/fda-qmsr-2026/)). The Feb 2026 cybersecurity guidance reissue ([RAPS, 2026](https://www.raps.org/news-and-articles/news-articles/2026/2/fda-reissues-cybersecurity-guidance-to-align-with)) confirmed:

- All CFR 820 references replaced with QMSR citations
- Tool validation now under **QMSR 4.1.6** (our pre-commit hooks, Docker build system, Prefect orchestrator)
- **CI/CD explicitly recognized as production controls** — software deployments are production events requiring build/commit approvals, environment snapshots, acceptance evidence, rollback plans, and SBOM documentation ([RookQS, 2026](https://rookqs.com/blog-rqs/2026-fda-guidance-critical-impacts-of-qmsr-alignment))
- SBOMs required for all cyber devices under Section 524B(b)(3)
- Cybersecurity controls integrated into design controls, validated through testing, maintained through CAPA

From the Innolitics analysis:

> "FDA is auditing your project management discipline, even if they don't call it that. Inspectors use the manufacturer's own risk management documentation to focus the inspection — and review it throughout, not just at one checkpoint." — [Giese (2026), Innolitics](https://innolitics.com)

### 3.2 Our Stack vs QMSR Requirements

| QMSR Requirement | ISO 13485 Clause | Our Implementation | Gap |
|------------------|-----------------|-------------------|-----|
| **Quality Manual** | 4.2.2 | CLAUDE.md + KG navigator | Not ISO-formatted |
| **Document Control** | 4.2.4 | Git + pre-commit + PR reviews | Adequate |
| **Design and Development Planning** | 7.3.2 | XML execution plans + KG | Adequate |
| **Design Input** | 7.3.3 | OpenSpec specs (GIVEN/WHEN/THEN) | Adequate |
| **Design Output** | 7.3.4 | Prefect flow artifacts + MLflow | Adequate |
| **Design Verification** | 7.3.5 | pytest (3-tier: staging/prod/GPU) | Adequate |
| **Design Validation** | 7.3.6 | Eval flow + biostatistics flow | Adequate |
| **Design Transfer** | 7.3.8 | Docker images, BentoML export, ONNX | Adequate |
| **Design History File** | 7.3.10 | Git log + MLflow, but no auto-export | **Gap** |
| **Risk Management** | 7.1 (ISO 14971) | KG Bayesian decisions, not ISO-formatted | **Gap** |
| **Software Lifecycle** | IEC 62304 | Prefect flows map to AIDL phases | **Partial** |
| **Purchasing/Supplier Control** | 7.4 | pyproject.toml, Docker base images | Not documented |
| **Production Controls** | 7.5 | CI/CD, Docker, pre-commit | **Not documented as QMSR controls** |
| **CAPA** | 8.5.2-8.5.3 | GitHub issues + metalearning | Not structured per QMSR |
| **Management Review** | 5.6 | Not applicable (preclinical) | N/A |

**Key gap**: Our CI/CD pipeline IS a production control system under QMSR — pre-commit hooks enforce code quality, Docker builds produce reproducible artifacts, MLflow logs every parameter. But we don't **document it as such**. Adding a `docs/compliance/production-controls.md` that maps our pipeline to QMSR 7.5 requirements would be a significant step.

---

## 4. PCCP Alignment: Is Our Factorial Design a PCCP?

### 4.1 Yes — Our Factorial Experiment IS a PCCP Template

From the Innolitics analysis, PCCPs work best when:
1. Well-understood, repeatable changes planned multiple times
2. Each change would otherwise require a separate 510(k)
3. Precise acceptance criteria defined upfront
4. Documentation educates reviewers on ML fundamentals
5. Changes stay within original intended use

**Our factorial design (4 models x 3 losses x 2 aux_calib x 3 post-training x 2 recalibration x 5 ensemble)** satisfies ALL five criteria:

| PCCP Criterion | Our Implementation |
|----------------|-------------------|
| Repeatable changes | Model family swap, loss function change, post-training strategy |
| Each requires separate submission | Yes — changing the CNN architecture is a material change |
| Precise acceptance criteria | Pre-specified: clDice > threshold, MASD < threshold, Brier < threshold |
| Educates reviewers | Biostatistics flow generates two-way ANOVA, effect sizes, interaction plots |
| Within intended use | All variants segment cerebrovascular structures |

### 4.2 The K252366 Blueprint (a2z-Unified-Triage)

From the Innolitics 510(k) Analyzer figures, K252366 shows an exemplary PCCP for a radiological AI device:

- **Covers adaptive algorithm updates** across all 7 abdominopelvic findings
- **Authorized modifications**: training data expansion, annotation refinement, ensemble optimization
- **Permits changes to neural network architecture** components and data augmentation parameters
- **Allows expansion of validated slice thickness ranges**
- **Validation requirements**: Sensitivity/Specificity >80% for all findings; AUC >0.95 for QFM findings
- **Updates verified via sequestered data testing** before release without new 510(k)

**This is exactly our architecture**: factorial model variants validated against sequestered test data with pre-specified acceptance criteria, deployed via BentoML without requiring re-submission for each variant.

### 4.2.1 PCCP Regulatory Timeline (Post-June 2025)

- **December 2024**: FDA finalized PCCP guidance, expanding scope from ML-only to **all AI-enabled devices** ([McDermott+, 2024](https://www.mcdermottplus.com/insights/fda-issues-final-guidance-on-predetermined-change-control-plans-for-ai-enabled-devices/))
- **August 2025**: FDA, Health Canada, and MHRA jointly issued **five guiding principles**: PCCPs must be Focused, Risk-based, Evidence-based, Transparent, and subject to Lifecycle oversight ([Ballard Spahr, 2025](https://www.ballardspahr.com/insights/alerts-and-articles/2025/08/fda-issues-guidance-on-ai-for-medical-devices))
- **October 2025**: Harrison.ai (via Rubrum Advising) submitted a **34-page petition** to FDA requesting **partial 510(k) exemption** for radiology AI devices (CADx, CADt, CADe/x product codes POK, MYN, QAS, QFM, QDQ). If granted, manufacturers with one existing clearance could launch similar devices without new 510(k)s, provided they maintain postmarket monitoring ([STAT News, 2026](https://www.statnews.com/2026/02/23/harrisonai-fda-petition-exempt-ai-devices-premarket-review/); [Federal Register, 2025](https://www.federalregister.gov/documents/2025/12/29/2025-23901/medical-devices-exemption-from-premarket-notification-radiology-computer-aided-detection-andor)). Comment period closed Feb 27, 2026 — if FDA doesn't deny by mid-April, the exemption takes effect
- **Implication**: If this exemption passes for radiology, it could set precedent for other imaging modalities. Building robust postmarket monitoring NOW positions MinIVess for either pathway

### 4.3 Neurology-Specific PCCP Considerations

From the Innolitics neurology post: "Unlike radiology (where FDA reviewers see high volumes of AI/ML submissions), neurology panels see fewer AI/ML devices." Only 2 of 6 recent AI/ML neurology devices included a PCCP.

**Implication for NEUROVEX**: If we pursue clinical translation for cerebrovascular segmentation, we should expect more conservative reviewers. The submission must "educate, not just demonstrate" — our biostatistics flow's ANOVA tables, interaction plots, and specification curves serve exactly this purpose.

---

## 5. MLOps Maturity Level 4: Continuous Retraining

### 5.1 The Maturity Model

| Level | Description | Our Status |
|-------|-------------|------------|
| **0** | No MLOps | Past this |
| **1** | Manual ML pipeline | Past this |
| **2** | ML pipeline automation | **Current** — Prefect orchestrates 5 flows |
| **3** | CI/CD for ML | **Approaching** — Docker + pre-commit, but no automated model registry gates |
| **4** | Continuous training & monitoring | **Not yet** — drift detection exists but no automated retraining trigger |
| **5** | Full ML automation with feedback loops | **Target** — requires postmarket surveillance |

### 5.1.1 The MedMLOps Framework (Post-June 2025)

The **MedMLOps framework** ([de Almeida et al. (2025). "Medical machine learning operations: a framework to facilitate clinical AI development and deployment in radiology." *European Radiology*.](https://link.springer.com/article/10.1007/s00330-025-11654-6)) defines four pillars specific to medical imaging:

1. **Availability** — reproducible model training and serving infrastructure
2. **Continuous monitoring/validation/(re)training** — drift detection, performance degradation alerts
3. **Patient privacy/data protection** — federated learning, anonymization, access controls
4. **Ease of use** — DevEx for researchers, minimal friction for clinical adoption

A JMIR scoping review ([2025](https://www.jmir.org/2025/1/e66559)) maps MLOps maturity along two dimensions: **operational excellence** and **regulatory compliance** — both must advance together for medical AI platforms.

An empirical MLOps guide ([ScienceDirect, 2025](https://www.sciencedirect.com/science/article/pii/S0950584925000643)) and a 2026 maturity model ([Flexiana, 2026](https://flexiana.com/machine-learning-architecture/mlops-maturity-model-2026-4-stages-to-resilient-risk-free-machine-learning)) both emphasize that Level 4 requires **governance and compliance embedded** in automated pipelines, not bolted on afterward.

### 5.2 What's Needed for Level 4

| Component | Status | Gap |
|-----------|--------|-----|
| **Drift detection** | Evidently DataDriftPreset + kernel MMD implemented | Not connected to retraining trigger |
| **Automated retraining** | `retraining_trigger` decision: `not_started` | Need drift → retrain → eval → gate pipeline |
| **Model registry gates** | MLflow model registry exists | No automated promotion/rejection based on metrics |
| **A/B testing** | Not implemented | Needed for champion/challenger pattern in deployment |
| **Monitoring dashboard** | Grafana + Prometheus configured | Not connected to drift alerts |

### 5.3 The "Locked" vs "Adaptive" Dilemma

From the AHMED project and PCCP framework: FDA traditionally promotes "locked" algorithms that are trained during development and cannot change in the operational environment. The PCCP framework is the bridge — it pre-approves specific types of changes with pre-specified validation protocols.

**Our approach**: Train the factorial models (locked), deploy the champion (locked), but have a PCCP-compatible retraining pipeline ready:

```
Drift Alert → New Data Ingestion → Retrain → Evaluate on Sequestered Test Set
    → If passes acceptance criteria → Log to AuditTrail → Deploy new champion
    → If fails → Alert, do NOT deploy, create GitHub issue
```

This maps directly to [Carvalho et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/)'s PCCP framework.

---

## 6. Postmarket Surveillance and Multi-Site Data Collection

### 6.1 522 Postmarket Surveillance Studies & FDA Real-World Monitoring

From Innolitics: FDA guidance now recognizes 522 studies as appropriate for Real-World Data (RWD) and Real-World Evidence (RWE). "They no longer need to be fully prospective, site-intensive, or trial-like."

**Critical Sept 2025 development**: On **September 30, 2025**, the FDA issued a **Request for Public Comment** on practical approaches to measuring real-world performance of AI-enabled medical devices, focusing on systematic performance monitoring across the total product lifecycle ([FDA, 2025](https://www.fda.gov/medical-devices/medical-device-regulatory-science-research-programs-conducted-osel/methods-and-tools-effective-postmarket-monitoring-artificial-intelligence-ai-enabled-medical-devices); [Covington, 2025](https://www.covingtondigitalhealth.com/2025/10/fda-requests-public-comment-on-real-world-evaluation-of-ai-enabled-medical-devices/); [Hogan Lovells, 2025](https://www.hoganlovells.com/en/publications/fda-seeks-public-comment-on-monitoring-strategies)).

FDA explicitly names **data drift, concept drift, and model drift** as threats to device safety and reliability. Teams must demonstrate:
- How they **detect** performance drift
- How they handle **cybersecurity incidents** related to drift
- **Escalation paths** and decision-making authority for emerging risks
- Intake processes for safety signals from the field

**Our existing drift simulation flow (`src/minivess/pipeline/drift_detection.py`) and Evidently DataDriftPreset integration are DIRECT regulatory assets** that demonstrate postmarket monitoring capability.

The Senate "FDA of the Future" report (Feb 17, 2026; [Buchanan Ingersoll & Rooney, 2026](https://www.bipc.com/senate-help-committee-releases-fda-modernization-report-); [InsideHealthPolicy, 2026](https://insidehealthpolicy.com/daily-news/cassidy-report-calls-fda-ai-strategy-outdated-urges-bias-monitoring)) signals:
> "Post-market surveillance is shifting from stick to carrot. Companies that build real-world evidence infrastructure into their product — 'as core architecture, not afterthought' — should get reduced pre-market burden."

The report calls FDA's AI/ML Action Plan **"already too outdated"** and recommends: broadly applicable FDA guidance for consistent AI treatment across review divisions, international harmonization, expanded internal AI expertise, and a fresh risk-adjusted regulatory approach through interagency collaboration.

### 6.2 The NEUROVEX Multi-Site Opt-In Architecture

For a preclinical neuroscience platform deployed across multiple research labs, data sharing must be **opt-in only**. The architecture:

```
┌─────────────────────────────────────────────────────┐
│              NEUROVEX Installation (Site A)           │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Train    │  │ Eval     │  │ Dashboard/Annot  │   │
│  │ Flow     │  │ Flow     │  │ (Gradio + Sentry)│   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │              │                 │              │
│       ▼              ▼                 ▼              │
│  ┌─────────────────────────────────────────────┐     │
│  │         Local MLflow + PostgreSQL            │     │
│  │         Local OpenLineage events             │     │
│  │         Local AuditTrail JSON                │     │
│  └─────────────────┬───────────────────────────┘     │
│                    │                                  │
│            [OPT-IN GATE]                              │
│                    │                                  │
│  ┌─────────────────▼───────────────────────────┐     │
│  │         Sync Agent (if opted-in)             │     │
│  │  - Anonymized metrics → Central MLflow       │     │
│  │  - Drift reports → Central Grafana           │     │
│  │  - UX telemetry → PostHog (anonymized)       │     │
│  │  - Error reports → Sentry (no PII)           │     │
│  │  - Lineage events → Central Marquez          │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘

                          │ (opt-in only)
                          ▼

┌─────────────────────────────────────────────────────┐
│              NEUROVEX Central (GCP)                   │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Central  │  │ Drift    │  │ UX Analytics     │   │
│  │ MLflow   │  │ Monitor  │  │ (PostHog)        │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Federated│  │ Error    │  │ Cross-site       │   │
│  │ Learning │  │ Tracking │  │ Evaluation       │   │
│  │ (Flower) │  │ (Sentry) │  │ Comparison       │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 6.3 Monitoring Tool Stack

| Tool | Purpose | SaMD Relevance | Preclinical Use | Compliance |
|------|---------|---------------|-----------------|------------|
| **PostHog** | Product analytics, session replay | UX issue tracking for annotation tool | Opt-in telemetry from labs | **SOC 2 Type II certified** (May 2025), **HIPAA-ready** with BAA, **GDPR-compliant** with EU-U.S. DPF, supports **self-hosted Docker deployment** ([PostHog HIPAA docs, 2025](https://posthog.com/docs/privacy/hipaa-compliance)) |
| **Sentry** | Error tracking, performance monitoring | Crash reporting, latency anomalies | Stack traces from deployed flows | SOC 2 certified, BAA available |
| **Grafana + Prometheus** | Infrastructure monitoring | Drift dashboards, model performance | Already configured | Self-hosted, no data leaves infra |
| **OpenLineage/Marquez** | Data lineage | IEC 62304 traceability | Already implemented (needs wiring) | LF AI & Data Foundation standard; IBM adopted for explainable AI ([IBM, 2025](https://www.ibm.com/new/announcements/openlineage-for-a-unified-lineage-view)); Debezium added native integration ([Debezium, 2025](https://debezium.io/blog/2025/06/13/openlineage-integration/)) |
| **Evidently** | ML drift detection | Postmarket surveillance input | Already implemented | Directly maps to FDA Sept 2025 RfC on drift monitoring |
| **Intercom/Crisp** | User communication | Support channel for annotation users | Optional for research labs | — |

**PostHog self-hosted note**: PostHog's Docker Compose deployment aligns with our Docker-per-flow architecture. Product analytics (researcher usage patterns, error rates, flow completion times) could be volume-mounted alongside MLflow artifacts — a unified observability layer satisfying both UX monitoring and regulatory audit.

### 6.3.1 Federated Learning: NVIDIA FLARE for Multi-Site Training

**NVIDIA FLARE v2.7.0** remains the dominant open-source FL framework for medical imaging, with real-world deployments at Mass General (brain aneurysm detection) and NCI (pancreatic cancer screening via Rhino Health) ([NVIDIA GTC 2025, Session S73112](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73112/); [PubMed, 2025](https://pubmed.ncbi.nlm.nih.gov/39895208/)).

Key insight: only **5.2% of FL publications have resulted in real clinical deployment** — a significant gap. Blockchain-based FL (BCFL) is emerging for tamper-resistant audit trails ([JMIR, 2026](https://www.jmir.org/2026/1/e79052)). A 2025 Frontiers paper proposes FL as a mechanism for **regulatory agencies themselves** to collaborate on model training without sharing data ([Frontiers in Drug Safety, 2025](https://www.frontiersin.org/journals/drug-safety-and-regulation/articles/10.3389/fdsfr.2025.1579922/full)).

**For MinIVess**: NVIDIA FLARE integration via the existing `ModelAdapter` ABC would enable multi-site training for multiphoton imaging data while keeping raw vascular data at each institution — particularly relevant for EBRAINS data sharing scenarios. The Flower framework (already in our library recommendations) is the lighter-weight alternative.

### 6.4 AG-UI / A2UI: Agentic Generative UI

For the annotation dashboard, agentic generative UI (AG-UI) could provide:
- **Adaptive annotation interfaces** that adjust based on user expertise and annotation quality
- **LLM-assisted quality checks** that flag potential annotation errors in real-time
- **Malleable UI** that researcher can customize without code changes

**Regulatory consideration**: If the annotation UI provides clinical decision support, it may fall under CDS regulation (see Innolitics CDS guidance analysis). For preclinical use, this is not regulated — but the architecture should support both modes.

---

## 7. Regulatory Convergence: SaMD, Regular Software, and Other Regulated Domains

### 7.1 SaMD and Regular Software Are Converging

From the Innolitics posts on agentic AI in regulated development:

> "The same principles that make AI work for general software make it work for regulated software: explicit requirements, traceable design decisions, automated verification, rigorous review, continuous cleanup. The mechanics don't change because the stakes are higher. What does change is the cost of skipping them."

**The convergence pattern**: SOC2 → ISO 27001 → ISO/IEC 42001 (AI) → IEC 62304 (medical) → FDA QMSR. Each layer adds specificity but the fundamentals are identical: version control, CI/CD, testing, audit trails, vulnerability management.

### 7.2 EU Digital Product Passport (DPP) and AI Omnibus

The EU Digital Product Passport Regulation (ESPR) is rolling out with a **central registry by July 2026** and first mandatory DPPs for batteries by February 2027 ([Fiegenbaum Solutions, 2026](https://www.fiegenbaum.solutions/en/blog/digital-product-passport-from-european-regulation-to-global-standard)). While not directly applicable to software, the DPP infrastructure is converging with SaMD requirements:

- **DPP data carrier** = our SBOM + OpenLineage lineage manifest
- **DPP lifecycle tracking** = our MLflow + AuditTrail
- **DPP sustainability reporting** = our cost/carbon tracking (TRIPOD cost appendix)

**EU AI Omnibus Proposal (February 2026)**: Clarifies that AI Act requirements for high-risk AI systems that are also regulated medical products should be applied **within existing conformity assessment procedures** rather than through separate certification ([Arnold & Porter, 2026](https://www.arnoldporter.com/en/perspectives/advisories/2026/02/eu-digital-omnibus-what-the-proposed-reforms-mean-for-pharma-and-medtech); [Petrie-Flom Center, Harvard Law, 2026](https://petrieflom.law.harvard.edu/2026/03/05/simplification-or-back-to-square-one-the-future-of-eu-medical-ai-regulation/)).

**December 2025 proposed MDR/IVDR revision**: Sharpens software classification rules, extends "well-established technology" to digital products, and integrates cybersecurity into General Safety and Performance Requirements. A Nature npj paper maps standards gaps for data-driven devices under EU MDR ([Nature, 2026](https://www.nature.com/articles/s44401-026-00075-2)).

**Implication for MinIVess**: A unified metadata/provenance layer (OpenLineage + MLflow + SBOM) satisfies multiple jurisdictions simultaneously — FDA Section 524B, EU MDR/IVDR, EU AI Act, and DPP transparency mandates.

### 7.3 USA UAD 3.6 (Real Estate)

The Uniform Appraisal Dataset (UAD) 3.6 for real estate requires **standardized data formats and audit trails** for automated valuation models (AVMs). The pattern is identical: regulated AI models need traceable data lineage, version-controlled models, and documented validation procedures.

### 7.4 The Meta-Pattern

All regulated domains are converging on the same stack:

```
Data Versioning (DVC/DataLad) + Pipeline Orchestration (Prefect/Airflow) +
Experiment Tracking (MLflow/W&B) + Lineage (OpenLineage) +
SBOM (CycloneDX) + CI/CD (Docker + pre-commit) +
Audit Trail (structured JSON) + Vulnerability Monitoring (Grype/Trivy)
```

We have 80% of this stack. The missing 20% is: SBOM generation, vulnerability monitoring, threat modeling documentation, and wiring the audit trail into flows.

---

## 8. Technical Debt as Regulatory Risk

### 8.1 The Innolitics Warning

From Giese on technical debt in AI-augmented development:

> "Technical debt used to be a low-grade, chronic pain. Now it's an acute tax on every feature you build. The types of debt that are especially toxic: Undocumented/Tribal-Knowledge Code, Hidden Side Effects, Deep Nesting, Inconsistent Patterns, Missing tests."

The figure from Innolitics shows three layers of risk: **Technical Debt** → **Regulatory Debt** → **Quality Debt** → cascading to 483 Observations → Warning Letters → Recalls → Patient Harm. "AI amplifies all of this."

### 8.2 Our Technical Debt Posture

| Debt Category | Our Mitigation | Residual Risk |
|---------------|---------------|---------------|
| **Undocumented code** | CLAUDE.md, KG, docstrings | Some modules lack full docstrings |
| **Missing tests** | TDD mandatory, 3-tier testing | Coverage gaps in compliance/ module wiring |
| **Inconsistent patterns** | Ruff + mypy + pre-commit | Slash-prefix migration was a cleanup |
| **Hidden side effects** | Type annotations, `from __future__ import annotations` | Some dynamic dispatch in Hydra configs |
| **Tribal knowledge** | Metalearning docs, CLAUDE.md rules | Session continuation != authorization rule |

---

## 9. Library Recommendations: What to Add

### 9.1 Priority 1: SBOM & Vulnerability

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| `cyclonedx-bom` | CycloneDX SBOM generation from Python env | Pre-commit hook or Makefile target |
| `grype` (binary) | Vulnerability scanning of SBOMs | CI/CD gate, `make sbom-scan` |
| `syft` (binary) | Container-level SBOM generation | Docker build pipeline |
| `trivy` (binary) | Comprehensive container scanner | Docker image scanning |

### 9.2 Priority 2: Compliance Automation

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| `model-card-toolkit` (Google) | Standardized model card generation | Post-training flow artifact |
| `openlineage-prefect` | Native Prefect → OpenLineage integration | All 5 Prefect flows |
| `ciso-assistant-community` | Open-source GRC framework | Compliance dashboard |

### 9.3 Priority 3: Monitoring & Telemetry

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| `posthog-python` | Product analytics (opt-in) | Dashboard/annotation tool |
| `sentry-sdk` | Error tracking | All Python entry points |
| `flower` | Federated learning framework | Multi-site model aggregation |

### 9.4 Priority 4: Security

| Library | Purpose | Integration Point |
|---------|---------|-------------------|
| `pip-audit` | Dependency vulnerability audit | Pre-commit or CI |
| `bandit` | Python security linting | Pre-commit hook |
| `safety` | PyPI vulnerability database check | CI/CD gate |

---

## 10. Action Items by Timeline

### Immediate (During PR-C through PR-E)

1. Wire `LineageEmitter.pipeline_run()` into all 5 Prefect flows (~30 lines)
2. Wire `AuditTrail.log_test_evaluation()` into Eval Flow
3. Add cumulative `eval/test_set_access_count` MLflow metric
4. Generate model cards as MLflow artifacts

### Short-term (After factorial experiment, Q2 2026)

5. Add CycloneDX SBOM generation to Makefile (`make sbom`)
6. Add Grype vulnerability scanning (`make sbom-scan`)
7. Connect Marquez to PostgreSQL for persistent lineage
8. Document CI/CD pipeline as QMSR production controls
9. Implement PCCP YAML template based on factorial design

### Medium-term (Q3-Q4 2026)

10. Add Sentry error tracking to all Python entry points
11. Add PostHog analytics to dashboard (opt-in)
12. Implement multi-site sync agent for opt-in data pooling
13. Add Trivy container scanning to Docker build pipeline
14. Create threat model document
15. Resolve `sbom_generation` KG node → `cyclonedx_plus_grype`

### Long-term (Clinical transition)

16. ISO 14971 risk management formalization
17. ISO 13485 QMS certification
18. 21 CFR Part 11 electronic signatures
19. Federated learning (Flower) for multi-site training
20. Full PCCP submission template

---

## 11. Key References

### Innolitics (Giese, 2025-2026)

- [Giese (2026). "6 SBOM Best Practices for Medical Device Manufacturers." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "FDA Premarket Cybersecurity Guidance Updated for QMSR." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "14 Common FDA Cybersecurity Deficiencies." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "The Senate Just Signaled the Future of FDA for AI/ML Devices." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "PCCPs in Low-AI-Volume Panels (Neurology)." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "Agentic AI Tools in Medical Device Development." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "Medical Device Software: Technical Debt = Regulatory Risk." *Innolitics*.](https://innolitics.com)
- [Giese (2025). "CorticoMetrics: From FreeSurfer to FDA-Cleared SaMD." *Innolitics*.](https://innolitics.com)
- [Giese (2025). "PAS vs 522 Postmarket Surveillance Studies." *Innolitics*.](https://innolitics.com)
- [Giese (2025). "TEMPO Pilot Program." *Innolitics*.](https://innolitics.com)
- [Giese (2025). "510(k) Analyzer: K252366 (a2z-Unified-Triage) with PCCP." *Innolitics*.](https://innolitics.com)
- [Giese (2025). "Harrison.ai 510(k) Partial Exemption Petition." *Innolitics*.](https://innolitics.com)
- [Giese (2026). "FDA CDS Guidance 2026 Analysis." *Innolitics*.](https://innolitics.com)

### Regulatory & Standards (Post-June 2025 Sources)

- [FDA (2026). "Premarket Cybersecurity Guidance" (Feb 2026 QMSR update).](https://www.fda.gov) — [RAPS analysis](https://www.raps.org/news-and-articles/news-articles/2026/2/fda-reissues-cybersecurity-guidance-to-align-with); [DLA Piper analysis](https://www.dlapiper.com/en-us/insights/publications/2026/02/fda-issues-revised-cybersecurity-premarket-submission-guidance); [Hattrick IT analysis](https://www.hattrick-it.com/blog/cybersecurityguidanceupdate/); [BIOT-MED analysis](https://www.biot-med.com/resources/fda-cybersecurity-requirements-connected-medical-devices-2026)
- [FDA (2025). "Methods and Tools for Effective Postmarket Monitoring of AI-Enabled Medical Devices." *Request for Public Comment* (Sept 30, 2025).](https://www.fda.gov/medical-devices/medical-device-regulatory-science-research-programs-conducted-osel/methods-and-tools-effective-postmarket-monitoring-artificial-intelligence-ai-enabled-medical-devices) — [Covington](https://www.covingtondigitalhealth.com/2025/10/fda-requests-public-comment-on-real-world-evaluation-of-ai-enabled-medical-devices/); [Hogan Lovells](https://www.hoganlovells.com/en/publications/fda-seeks-public-comment-on-monitoring-strategies)
- [Senate HELP Committee (2026). "Patients and Families First: Building the FDA of the Future."](https://www.help.senate.gov) — [Buchanan Ingersoll](https://www.bipc.com/senate-help-committee-releases-fda-modernization-report-); [InsideHealthPolicy](https://insidehealthpolicy.com/daily-news/cassidy-report-calls-fda-ai-strategy-outdated-urges-bias-monitoring)
- [Carvalho et al. (2025). "Predetermined Change Control Plans: Guiding Principles." *JMIR AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/)
- [FDA PCCP Final Guidance (Dec 2024).](https://www.fda.gov/media/166704/download) — [McDermott+](https://www.mcdermottplus.com/insights/fda-issues-final-guidance-on-predetermined-change-control-plans-for-ai-enabled-devices/); [Ballard Spahr](https://www.ballardspahr.com/insights/alerts-and-articles/2025/08/fda-issues-guidance-on-ai-for-medical-devices); [Intertek](https://www.intertek.com/blog/2025/03-25-fdas-pccp-framework-and-ai-enabled-medical-devices/)
- [Harrison.ai / Rubrum Advising (2025). "510(k) Partial Exemption Petition."](https://www.federalregister.gov/documents/2025/12/29/2025-23901/medical-devices-exemption-from-premarket-notification-radiology-computer-aided-detection-andor) — [STAT News](https://www.statnews.com/2026/02/23/harrisonai-fda-petition-exempt-ai-devices-premarket-review/); [AuntMinnie](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15775081/petition-to-us-fda-proposes-alternative-pathway-for-radiology-ai)
- [ComplianceQuest (2026). "FDA QMSR 2026."](https://www.compliancequest.com/blog/fda-quality-management-system-regulation-qmsr-2026/); [WQS (2026)](https://us.wqs.de/fda-qmsr-2026/); [RookQS (2026)](https://rookqs.com/blog-rqs/2026-fda-guidance-critical-impacts-of-qmsr-alignment)
- [IEC 62304 Edition 2 (expected Aug 2026).](https://www.iso.org) — [LFH Regulatory](https://lfhregulatory.co.uk/iec-62304-update-2026/); [8fold Governance](https://8foldgovernance.com/iec-62304-edition-2-big-changes-in-samd-requirements/); [NSF Prosystem](https://www.nsf-prosystem.org/en/news/detail/iec-62304-2-ausgabe-optimierte-sicherheitsklassen-erweiterter-anwendungsbereich-fuer-gesundheitssoftware-und-ki/); [IEC SC62A Change Rationales](https://assets.iec.ch/public/sc62a/N0166_IEC62304_ED2_ChangeRationales_CoverLetter.pdf)

### SBOM & Supply Chain
- [Sbomify (2026). "FDA Medical Device SBOM Requirements."](https://sbomify.com/2026/01/09/fda-medical-device-sbom-requirements/)
- [Sbomify (2026). "SBOM Formats Compared: CycloneDX vs SPDX."](https://sbomify.com/2026/01/15/sbom-formats-cyclonedx-vs-spdx/)
- [Sbomify (2026). "SBOM Generation Guide for Python — UV, Poetry, Pipenv."](https://sbomify.com/guides/python/)
- [Blue Goat Cyber (2025). "Navigating FDA SBOM Requirements: SPDX, CDX."](https://bluegoatcyber.com/blog/navigating-the-fdas-sbom-requirements-for-medical-device-manufacturing-spdx-cdx-and-more/)

### MLSecOps & Adversarial Robustness
- [Frontiers in Medicine (2025). "Assessing adversarial robustness of multimodal medical AI."](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328349/)
- [arXiv (2025). "A Practical Framework for Evaluating Medical AI Security."](https://arxiv.org/pdf/2512.08185)
- [medRxiv (2026). "Red-Teaming Medical AI: Systematic Adversarial Evaluation."](https://www.medrxiv.org/content/10.64898/2026.02.26.26347212v1)
- [Practical DevSecOps (2026). "MITRE ATLAS Framework 2026."](https://www.practical-devsecops.com/mitre-atlas-framework-guide-securing-ai-systems/)
- [Springer AI Review (2024). "Robustness in deep learning models for medical diagnostics."](https://link.springer.com/article/10.1007/s10462-024-11005-9)

### MLOps Maturity & MedMLOps
- [de Almeida et al. (2025). "Medical machine learning operations: a framework for radiology." *European Radiology*.](https://link.springer.com/article/10.1007/s00330-025-11654-6)
- [JMIR (2025). "Maturity Framework for Operationalizing ML in Health Care."](https://www.jmir.org/2025/1/e66559)
- [ScienceDirect (2025). "An empirical guide to MLOps adoption."](https://www.sciencedirect.com/science/article/pii/S0950584925000643)
- [Flexiana (2026). "MLOps Maturity Model 2026."](https://flexiana.com/machine-learning-architecture/mlops-maturity-model-2026-4-stages-to-resilient-risk-free-machine-learning)
- [IntuitionLabs (2025). "AI Post-Market Surveillance: Locked vs. Continuous Learning."](https://intuitionlabs.ai/articles/post-market-surveillance-ai-locked-continuous-learning)

### EU Regulatory Convergence
- [Fiegenbaum Solutions (2026). "Digital Product Passport 2026."](https://www.fiegenbaum.solutions/en/blog/digital-product-passport-from-european-regulation-to-global-standard)
- [Arnold & Porter (2026). "EU Digital Omnibus: Pharma and MedTech Reforms."](https://www.arnoldporter.com/en/perspectives/advisories/2026/02/eu-digital-omnibus-what-the-proposed-reforms-mean-for-pharma-and-medtech)
- [Petrie-Flom Center, Harvard Law (2026). "The Future of EU Medical AI Regulation."](https://petrieflom.law.harvard.edu/2026/03/05/simplification-or-back-to-square-one-the-future-of-eu-medical-ai-regulation/)
- [Nature npj Health Systems (2026). "Data-driven medical devices and the EU MDR."](https://www.nature.com/articles/s44401-026-00075-2)
- [Osborne Clarke (2025). "Revised EU medtech regulations: software and cybersecurity."](https://www.osborneclarke.com/insights/revised-eu-medtech-regulations-proposal-sharpens-software-and-cybersecurity-rules-digital)

### Federated Learning & Multi-Site
- [NVIDIA GTC 2025, Session S73112. "Federated Learning in Medical Imaging."](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73112/)
- [PubMed (2025). "Real-world applications of federated learning with NVIDIA FLARE."](https://pubmed.ncbi.nlm.nih.gov/39895208/)
- [JMIR (2026). "Securing Federated Learning With Blockchain in Medical Field."](https://www.jmir.org/2026/1/e79052)
- [Frontiers in Drug Safety (2025). "Federated learning: privacy-preserving regulatory cooperation."](https://www.frontiersin.org/journals/drug-safety-and-regulation/articles/10.3389/fdsfr.2025.1579922/full)

### Product Analytics in Regulated Software
- [PostHog (2025). "PostHog & HIPAA compliance."](https://posthog.com/docs/privacy/hipaa-compliance)
- [PostHog (2025). "The 7 best HIPAA-compliant analytics tools."](https://posthog.com/blog/best-hipaa-compliant-analytics-tools)

### Agentic UI
- [CopilotKit (2025). "AG-UI and A2UI: Understanding the Differences."](https://www.copilotkit.ai/ag-ui-and-a2ui)
- [Google Developers (2025). "Introducing A2UI: agent-driven interfaces."](https://developers.googleblog.com/introducing-a2ui-an-open-project-for-agent-driven-interfaces/)
- [ScienceDirect (2025). "Next-generation agentic AI for transforming healthcare."](https://www.sciencedirect.com/science/article/pii/S2949953425000141)

### First-Pass Report
- [`docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md`](regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md)
- [`docs/planning/openlineage-marquez-iec62304-report.md`](openlineage-marquez-iec62304-report.md)
