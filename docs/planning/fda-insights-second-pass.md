# FDA Readiness Second Pass: Innolitics Trends, SBOM, SecOps, QMSR, PCCP & MLOps Maturity

**Created**: 2026-03-17
**Status**: Draft — extends [first-pass report](regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md)
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

The QMSR replaced 21 CFR Part 820 by incorporating ISO 13485:2016 into federal law. From the Innolitics analysis:

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

### 6.1 522 Postmarket Surveillance Studies

From Innolitics: FDA guidance now recognizes 522 studies as appropriate for Real-World Data (RWD) and Real-World Evidence (RWE). "They no longer need to be fully prospective, site-intensive, or trial-like."

The Senate "FDA of the Future" report (Feb 17, 2026) signals:
> "Post-market surveillance is shifting from stick to carrot. Companies that build real-world evidence infrastructure into their product — 'as core architecture, not afterthought' — should get reduced pre-market burden."

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

| Tool | Purpose | SaMD Relevance | Preclinical Use |
|------|---------|---------------|-----------------|
| **PostHog** | Product analytics, session replay | UX issue tracking for annotation tool | Opt-in telemetry from labs |
| **Sentry** | Error tracking, performance monitoring | Crash reporting, latency anomalies | Stack traces from deployed flows |
| **Grafana + Prometheus** | Infrastructure monitoring | Drift dashboards, model performance | Already configured |
| **OpenLineage/Marquez** | Data lineage | IEC 62304 traceability | Already implemented (needs wiring) |
| **Evidently** | ML drift detection | Postmarket surveillance input | Already implemented |
| **Intercom/Crisp** | User communication | Support channel for annotation users | Optional for research labs |

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

### 7.2 EU Digital Product Passport (DPP)

The EU Digital Product Passport Regulation (2024/1781) requires **machine-readable lifecycle documentation** for products sold in the EU. While not directly applicable to software, the DPP infrastructure (data carriers, digital twins, lifecycle tracking) is converging with SaMD requirements:

- **DPP data carrier** = our SBOM + OpenLineage lineage manifest
- **DPP lifecycle tracking** = our MLflow + AuditTrail
- **DPP sustainability reporting** = our cost/carbon tracking (TRIPOD cost appendix)

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

### Regulatory
- [FDA (2026). "Premarket Cybersecurity Guidance" (Feb 2026 update).](https://www.fda.gov)
- [Senate HELP Committee (2026). "Patients and Families First: Building the FDA of the Future."](https://www.help.senate.gov)
- [Carvalho et al. (2025). "Predetermined Change Control Plans: Guiding Principles." *JMIR AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/)
- [IEC 62304 Edition 2 (expected Aug 2026). AI Development Lifecycle additions.](https://www.iso.org)

### First-Pass Report
- [`docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md`](regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md)
- [`docs/planning/openlineage-marquez-iec62304-report.md`](openlineage-marquez-iec62304-report.md)
