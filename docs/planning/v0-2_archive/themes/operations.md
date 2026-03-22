---
title: "Theme: Operations — FDA, IEC 62304, EU AI Act, Compliance"
theme_id: operations
doc_count: 19
archive_path: docs/planning/v0-2_archive/original_docs/
kg_domain: knowledge-graph/domains/operations.yaml
created: "2026-03-22"
status: archived
---

# Theme: Operations

Regulatory compliance (FDA, IEC 62304, EU AI Act), quality management, post-market
surveillance, OpenLineage audit trails, federated learning, and documentation
card frameworks. This theme covers the "compliance-native MLOps" architecture that
distinguishes MinIVess from standard ML pipelines.

---

## Key Scientific Insights

### 1. FDA QMSR (Feb 2026) Replaces QSR with ISO 13485 Alignment

The `regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md` (comprehensive,
30+ citations) documents that the FDA Quality Management System Regulation (QMSR),
effective February 2026, aligns the US device quality system with ISO 13485. This
simplifies dual compliance for academic projects targeting both FDA and EU MDR markets.
MinIVess's Prefect-orchestrated pipeline maps naturally to QMSR's process-based
quality system requirements.

### 2. PCCP (Predetermined Change Control Plan) Enables Locked-to-Adaptive Models

The FDA's PCCP framework (K252366 as blueprint) allows SaMD manufacturers to pre-specify
modification protocols. MinIVess's factorial design IS a PCCP template: the experimental
matrix pre-specifies which model variants, loss functions, and post-training methods will
be evaluated, with statistical acceptance criteria defined before execution. This maps
the academic experiment directly to regulatory language.

### 3. IEC 62304 Maps to Prefect Flows + OpenLineage

IEC 62304 (Medical device software lifecycle) Clause 8 requires traceability of all
software changes and their verification. The `openlineage-marquez-iec62304-report.md`
demonstrates that OpenLineage's InputDataset/OutputDataset events, when wired into
Prefect flows, produce a compliant audit trail automatically. Each Prefect task execution
becomes a traceable software activity; each MLflow artifact becomes a controlled document.

### 4. EU AI Act Classification: High-Risk (Article 6)

Medical AI devices are automatically classified as high-risk under EU AI Act Article 6
(products covered by EU harmonized legislation including MDR). The `eu-ai-act-plan.md`
maps MinIVess's existing infrastructure to the seven requirements for high-risk AI:
risk management (FMEA), data governance (DVC + Pandera), technical documentation
(auto-generated DHF), record-keeping (MLflow + OpenLineage), transparency (Model Cards),
human oversight (Prefect UI approval gates), and robustness (conformal prediction sets).

### 5. ComplOps = Compliance-as-Code Is Achievable Today

The `complops-plan.md` envisions automated compliance validation: 510(k) summary
templates, EU MDR technical file templates, and automated gap analysis across all
regulatory frameworks. The implementation exists in `src/minivess/compliance/` with
modules for audit trails, EU AI Act, IEC 62304, regulatory document generation, and
fairness evaluation. However, these modules are config-only stubs -- they define the
data structures and enums but are not wired into the Prefect flows.

### 6. Five-Hypothesis FDA Readiness Design Matrix

The comprehensive RegOps report structures FDA readiness as 5 hypotheses x 5 factors:

| Hypothesis | Key Factor |
|-----------|-----------|
| H1: Continuous audit reduces submission time | OpenLineage wiring completeness |
| H2: PCCP with factorial matches K252366 | Pre-specified acceptance criteria |
| H3: SBOM generation meets QMSR Clause 7 | CycloneDX automation |
| H4: Test set firewall prevents leakage | AuditTrail.log_test_evaluation() |
| H5: DHF auto-generation is reviewer-acceptable | Template quality + completeness |

Phased implementation: H1 now, H2 after factorial, H3 Q3 2026, H4 clinical projects.

### 7. OpenLineage + Marquez Provides Lineage Graph Visualization

The `openlineage-plan.md` and its companion report document that Marquez (OpenLineage's
reference implementation) provides a DAG visualization of data flow through the pipeline.
This directly addresses IEC 62304 Clause 8 traceability requirements and enables auditors
to verify that every training input can be traced to its data source and every model
artifact can be traced to its training configuration.

### 8. PPRM (Post-Publication Reproducibility Monitoring) Is Novel

The `pprm-plan.md` proposes automated post-publication monitoring: after the Nature
Protocols paper is published, CI jobs periodically verify that the published Docker
images, DVC data hashes, and MLflow artifacts still reproduce the reported results.
This goes beyond standard archival (Zenodo) to provide active verification that
computational results remain valid as dependencies evolve.

### 9. Federated Learning Deferred but Architecture-Ready

The `federated-learning-plan.md` scopes MONAI FL integration for future multi-site
studies. The current single-site architecture (MiniVess + DeepVess) does not require
federated learning, but the config-driven design (Hydra groups for data sources,
SkyPilot for compute) means adding a federated aggregation flow is a YAML config
change, not a code change. Status: not_started in KG.

---

## Architectural Decisions Made

| Decision | Winner | Evidence Doc | KG Node |
|----------|--------|-------------|---------|
| Audit trail format | Structured JSON + OTEL | regops-pipeline-plan.md | `operations.audit_trail` |
| FDA readiness strategy | Preclinical FDA-ready, phased | regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md | `operations.fda_readiness` |
| Cost optimization | SkyPilot spot + FinOps | (cloud theme) | `operations.cost_optimization` |
| Dashboard framework | Grafana + Prometheus | (infrastructure theme) | `operations.dashboard_framework` |
| Documentation cards | Model > Data > Environment | ai-cards-comprehensive-reference.md | `operations.documentation_cards` |
| Test set firewall | Audit log with counter | regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md | `operations.test_set_firewall` |
| SBOM generation | CycloneDX (not started) | complops-plan.md | `operations.sbom_generation` |
| Federated learning | Deferred (MONAI FL candidate) | federated-learning-plan.md | `operations.federated_learning` |
| Drift monitoring | Evidently DataDriftPreset | (data theme) | `operations.drift_monitoring` |
| Retraining trigger | Not started | N/A | `operations.retraining_trigger` |

---

## Implementation Status

| Document | Type | Status | Key Impl Files |
|----------|------|--------|----------------|
| complops-plan.md | plan | Implemented (structure) | `compliance/complops.py` (228 lines) |
| eu-ai-act-plan.md | plan | Implemented (structure) | `compliance/eu_ai_act.py` (155 lines) |
| fda-insights-innolitics-trends-seed-from-linkedin.md | reference | Reference only | LinkedIn-sourced FDA trends |
| fda-insights-second-pass-executable.xml | execution_plan | Partial | XML execution plan for FDA insights |
| fda-insights-second-pass-prompt-to-xml.md | prompt | Consumed | Prompt for FDA XML plan generation |
| fda-insights-second-pass.md | reference | Reference only | FDA landscape analysis |
| federated-learning-plan.md | plan | Not started | Architecture-ready via Hydra config groups |
| ge-validation-gates-plan.md | plan | Implemented | `validation/ge_runner.py` (162 lines), `validation/gates.py` (108 lines) |
| iec62304-framework-plan.md | plan | Implemented (structure) | `compliance/iec62304.py` (226 lines) |
| lit-report-regulatory-postmarket.md | literature_report | Complete | 35 papers on QMSR, PCCP, IEC 62304 |
| lit-report-regulatory-postmarket.xml | execution_plan | Consumed | Invocation plan for regulatory lit report |
| medical-mlops-standards.md | reference | Reference only | Medical MLOps standards landscape |
| openlineage-marquez-iec62304-report.md | reference | Planning complete | OpenLineage-to-IEC62304 mapping |
| openlineage-plan.md | plan | Not started | OpenLineage event wiring not implemented |
| pprm-plan.md | plan | Not started | Post-Publication Reproducibility Monitoring |
| regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md | reference | Planning complete | 30+ citations, 5 hypotheses x 5 factors |
| regops-pipeline-plan.md | plan | Implemented (structure) | `compliance/regops.py` (163 lines) |
| regulatory-docs-plan.md | plan | Implemented | `compliance/regulatory_docs.py` (229 lines) |
| regulatory-ops-report.md | reference | Reference only | RegOps landscape and recommendations |

---

## Cross-References

- **KG Domain**: `knowledge-graph/domains/operations.yaml` -- 13 decision nodes, phased FDA roadmap
- **Manuscript Theme**: Discussion section on regulatory-native MLOps (future work)
- **Data Theme**: Data governance (DVC, Pandera) satisfies EU AI Act data quality requirements
- **Infrastructure Theme**: Docker isolation satisfies software lifecycle traceability (IEC 62304)
- **Observability Theme**: MLflow + Prometheus satisfy record-keeping requirements
- **Key Source Files**:
  - `src/minivess/compliance/` (10 files, 1925 lines total):
    - `audit.py` (126 lines) -- structured JSON audit trail
    - `complops.py` (228 lines) -- compliance-as-code framework
    - `eu_ai_act.py` (155 lines) -- EU AI Act risk classification
    - `fairness.py` (227 lines) -- CyclOps-inspired subgroup fairness
    - `iec62304.py` (226 lines) -- IEC 62304 lifecycle mapping
    - `model_card.py` (177 lines) -- Model Card generation
    - `regops.py` (163 lines) -- RegOps pipeline
    - `regulatory_docs.py` (229 lines) -- DHF, SRS, risk analysis generation
    - `reporting_templates.py` (314 lines) -- automated reporting templates

**Implementation gap note**: All compliance modules define correct data structures and
enums but are primarily config-only stubs. The critical gap is wiring them into
Prefect flows -- the modules exist but are never called from orchestration code.
OpenLineage integration (the foundation for IEC 62304 traceability) is not started.

---

## Constituent Documents

1. `complops-plan.md` -- ComplOps: automated 510(k) summaries, EU MDR tech files, gap analysis (Issue #49)
2. `eu-ai-act-plan.md` -- EU AI Act compliance checklist: high-risk classification, 7 requirements (Issue #20)
3. `fda-insights-innolitics-trends-seed-from-linkedin.md` -- LinkedIn-sourced FDA SaMD trends and insights
4. `fda-insights-second-pass-executable.xml` -- XML execution plan for second-pass FDA analysis
5. `fda-insights-second-pass-prompt-to-xml.md` -- Prompt for generating FDA executable plan
6. `fda-insights-second-pass.md` -- Second-pass FDA landscape analysis
7. `federated-learning-plan.md` -- MONAI FL / Flower federated learning plan (deferred)
8. `ge-validation-gates-plan.md` -- Great Expectations validation gates for pipeline quality
9. `iec62304-framework-plan.md` -- IEC 62304 medical device software lifecycle mapping
10. `lit-report-regulatory-postmarket.md` -- Literature report: QMSR, PCCP, IEC 62304, post-market surveillance (35 papers)
11. `lit-report-regulatory-postmarket.xml` -- XML invocation plan for regulatory literature report
12. `medical-mlops-standards.md` -- Medical MLOps standards landscape overview
13. `openlineage-marquez-iec62304-report.md` -- OpenLineage + Marquez as IEC 62304 Clause 8 audit trail
14. `openlineage-plan.md` -- OpenLineage event wiring plan for all 5 Prefect flows
15. `pprm-plan.md` -- Post-Publication Reproducibility Monitoring (novel concept)
16. `regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md` -- Comprehensive FDA readiness: 5 hypotheses x 5 factors, 30+ citations
17. `regops-pipeline-plan.md` -- RegOps CI/CD pipeline for automated regulatory doc generation
18. `regulatory-docs-plan.md` -- Plan for DHF, SRS, risk analysis, validation protocol generation
19. `regulatory-ops-report.md` -- RegOps landscape analysis and recommendations
