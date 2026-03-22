---
title: "Theme: Manuscript — Literature Reports, TRIPOD, Preregistration"
theme_id: manuscript
doc_count: 26
archive_path: docs/planning/v0-2_archive/original_docs/
kg_domain: knowledge-graph/domains/manuscript.yaml
created: "2026-03-22"
status: archived
---

# Theme: Manuscript

Literature research reports, publication strategy, TRIPOD compliance, preregistration
mapping, AI documentation cards, and the repo-to-manuscript pipeline. This theme
covers the scholarly infrastructure supporting the Nature Protocols paper (working
name: NEUROVEX).

---

## Key Scientific Insights

### 1. Six Literature Domains Required for Nature Protocols

The batch literature research plan identified six interleaved research themes needed for
a comprehensive Nature Protocols manuscript:

- **R1**: Post-Training Methods (SWA, SWAG, ensembles, calibration)
- **R2**: Biostatistical Evaluation (factorial design, ANOVA, specification curve)
- **R3**: Biomedical Segmentation (tubular structures, foundation models, MONAI)
- **R4**: Multiphoton Neuroscience (smart acquisition, closed-loop feedback)
- **R5**: Regulatory/Post-Market (FDA, IEC 62304, OpenLineage audit)
- **R6**: Platform Engineering / PaaS (MLOps scaffolds, reproducibility)

Each report targets 35-50 papers at MINOR_REVISION quality level. The XML plans
(`lit-report-*.xml`) provide structured invocation prompts for the literature report
creation skill.

### 2. TRIPOD+AI Is the Primary Reporting Guideline (27 Items)

Three TRIPOD variants apply to this project:

- **TRIPOD+AI** (Collins et al. 2024, BMJ) -- 27 items, primary compliance target
- **TRIPOD-LLM** (Gallifant et al. 2025, Nat Med) -- 19 items, for Pydantic AI agents
- **TRIPOD-Code** (Pollard et al. 2026, Diagn Progn Res) -- protocol stage, proactive

As of 2026-03-22: 15 of 27 TRIPOD+AI items fully/partially addressed. Four priority
gaps remain: sample size justification (9a), participant flow diagram (6, 20a),
fairness statement (3c, 12f, 26), and cost transparency (TRIPOD-LLM 12).

### 3. This Is a Platform Paper, Not a SOTA Segmentation Paper

Recurring metalearning: the contribution is the PLATFORM's capabilities (MLOps scaffold
extending MONAI), not the segmentation numbers. External test evaluation on DeepVess
demonstrates the platform's ability to handle arbitrary test datasets with subgroup
analysis -- the actual numbers are standard practice. This framing governs all manuscript
sections: Methods describes the platform architecture, Results demonstrates platform
capabilities, Discussion frames regulatory-native MLOps as the contribution.

### 4. AI Cards Landscape Is Vast and Growing

The comprehensive reference catalogs 55+ documentation artifact types across 11
categories: data-focused (Datasheets, Data Cards, CrowdWorkSheets), model-focused
(Model Cards, FactSheets, Reward Reports), system-focused (System Cards, AI Service
Cards), risk-focused (Ethics Cards, Consequence Scanning), explainability-focused
(Explanation Cards), sustainability-focused (Carbon Cards, Energy Cards), and
regulatory frameworks (EU AI Act, FDA guidance). For MinIVess, the priority stack is:
Model Cards (existing) > Data Cards (gap) > Environment Cards (gap).

### 5. Computational Reproducibility Is a Solved Problem -- If You Use Docker

Literature report R6 (computational reproducibility) confirmed that Docker + DVC + MLflow
is the canonical reproducibility stack. The key gap in most academic repos is not
technology but *discipline*: researchers bypass Docker for speed, use bare-metal
execution, and lose reproducibility. MinIVess enforces Docker as the execution model
via the STOP protocol (Rules 17-19), making it one of the few academic repos where
reproducibility is a hard constraint, not an aspiration.

### 6. Federated Learning Is Research-Grade, Not Production-Ready

Literature report R4 (federated/multisite) found that MONAI FL and Flower provide
frameworks, but real-world federated learning for medical imaging remains limited by
data governance barriers, not technical ones. MinIVess's dataset strategy (MiniVess
primary + DeepVess external test) is single-site; federated learning is deferred to
future work (Issue #574 scope).

### 7. Platform Engineering / PaaS Is an Emerging ML Category

Literature report R6 (platform engineering) identified that MLOps-as-a-platform for
research labs is underserved. Existing platforms (Vertex AI, SageMaker, Azure ML) are
cloud-locked. MinIVess's intercloud portability via SkyPilot + Docker + Hydra config
groups fills this gap. The Nature Protocols audience (biomedical researchers) needs
this scaffolding more than any other community.

### 8. KG-to-Manuscript Pipeline Enables Automated Section Generation

The `repo-to-manuscript.md` plan defines a pipeline: KG decision nodes produce
`claims.yaml` (12 claims linked to evidence), `methods.yaml` (13 stubs M0-M12),
`results.yaml` (6 stubs R0-R5), and `limitations.yaml` (6 limitations). The
`kg-sync` skill exports snapshots to `sci-llm-writer` for LaTeX generation. Currently
scaffold-only; R3b blocked on GPU benchmark runs.

---

## Architectural Decisions Made

| Decision | Winner | Evidence Doc | KG Node |
|----------|--------|-------------|---------|
| Publication target | Nature Protocols | repo-to-manuscript.md | `manuscript.publication_target` |
| Primary reporting guideline | TRIPOD+AI (Collins 2024) | tripod-compliance-matrix.md | `manuscript.tripod_compliance` |
| Documentation cards priority | Model > Data > Environment | ai-cards-comprehensive-reference.md | `operations.documentation_cards` |
| Manuscript architecture | KG claims -> methods/results stubs -> LaTeX | kg-manuscript-execution-plan.yaml | N/A |
| Literature report format | XML invocation plans + MD output | batch-literature-research-report.xml | N/A |
| Citation format | Author-year with hyperlinks | CLAUDE.md Rule Citations | N/A |
| Preregistration scope | TRIPOD+AI 27 items + TRIPOD-LLM 19 items | preregistration-tripod-mapping.md | N/A |

---

## Implementation Status

| Document | Type | Status | Key Impl Files |
|----------|------|--------|----------------|
| ai-cards-comprehensive-reference.md | reference | Reference only | 55+ card types cataloged, no code needed |
| batch-literature-research-prompt.md | prompt | Consumed | Input to batch-literature-research-report.xml |
| batch-literature-research-report.xml | execution_plan | Partial | 6 lit reports planned, 5 generated |
| cover-letter-to-sci-llm-writer-for-knowledge-graph.md | reference | Reference only | Describes KG export strategy for sci-llm-writer |
| fuel-for-manuscript.md | reference | Reference only | Raw notes on manuscript material |
| kg-manuscript-execution-plan.yaml | execution_plan | Partial | `knowledge-graph/manuscript/` bootstrapped |
| lit-report-ai-cards.md | literature_report | Complete | Survey of 55+ documentation artifact types |
| lit-report-biomedical-segmentation.md | literature_report | Complete | 50 papers on tubular segmentation + foundation models |
| lit-report-biomedical-segmentation.xml | execution_plan | Consumed | Invocation plan for R3 report |
| lit-report-computational-reproducibility.md | literature_report | Complete | Docker + DVC + MLflow reproducibility stack |
| lit-report-computational-reproducibility.xml | execution_plan | Consumed | Invocation plan for R6 report |
| lit-report-federated-multisite.md | literature_report | Complete | MONAI FL, Flower, data governance barriers |
| lit-report-federated-multisite.xml | execution_plan | Consumed | Invocation plan for R4 (multisite) report |
| lit-report-medical-mlops.md | literature_report | Complete | MedMLOps pillars, MLOps maturity models |
| lit-report-medical-mlops.xml | execution_plan | Consumed | Invocation plan for R5 report |
| lit-report-multiphoton-neuroscience.md | literature_report | Complete | Smart acquisition, closed-loop, OME-Zarr |
| lit-report-multiphoton-neuroscience.xml | execution_plan | Consumed | Invocation plan for R4 (neuroscience) report |
| lit-report-platform-engineering-paas-prompt.md | prompt | Consumed | Input to R6 platform engineering report |
| lit-report-platform-engineering-paas.md | literature_report | Complete | MLOps-as-PaaS, intercloud portability |
| lit-report-platform-engineering-paas.xml | execution_plan | Consumed | Invocation plan for R6 report |
| preregistration-tripod-mapping.md | reference | Active | Maps factorial design to TRIPOD+AI items |
| repo-to-manuscript-prompt.md | prompt | Reference only | Verbatim user prompts for manuscript planning |
| repo-to-manuscript.md | plan | Active | Primary plan: paper framing, IMRAD structure |
| reporting-templates-plan.md | plan | Implemented | `compliance/reporting_templates.py` (314 lines) |
| research-reports-general-plan-for-manuscript-writing.md | plan | Implemented | 6-theme research plan for manuscript writing |
| tripod-compliance-matrix.md | reference | Active | 27+19 items mapped; 15/27 addressed |

---

## Cross-References

- **KG Domain**: `knowledge-graph/domains/manuscript.yaml` -- publication target, TRIPOD compliance, 5 section stubs
- **Training Theme**: Results section R3 depends on loss function experiment outcomes
- **Evaluation Theme**: Biostatistics flow results feed into Results section R2
- **Operations Theme**: Regulatory compliance (FDA, IEC 62304) feeds Discussion section
- **Observability Theme**: MLflow artifacts are the data source for all results tables
- **Key Source Files**:
  - `knowledge-graph/manuscript/claims.yaml` -- 12 scientific claims linked to evidence
  - `knowledge-graph/manuscript/methods.yaml` -- 13 Methods stubs (M0-M12)
  - `knowledge-graph/manuscript/results.yaml` -- 6 Results stubs (R0-R5, R3b blocked)
  - `knowledge-graph/manuscript/limitations.yaml` -- 6 known limitations
  - `docs/planning/tripod-compliance-matrix.md` -- TRIPOD+AI/LLM/Code compliance matrix
  - `src/minivess/compliance/reporting_templates.py` (314 lines) -- reporting template generation
  - `docs/manuscript/latent-methods-results/` -- LaTeX scaffold (partial)

---

## Constituent Documents

1. `ai-cards-comprehensive-reference.md` -- 55+ AI documentation artifact types across 11 categories
2. `batch-literature-research-prompt.md` -- User prompt for batch literature research session
3. `batch-literature-research-report.xml` -- XML master plan for 6 themed literature reports
4. `cover-letter-to-sci-llm-writer-for-knowledge-graph.md` -- KG export strategy for external manuscript tool
5. `fuel-for-manuscript.md` -- Raw notes and material for manuscript sections
6. `kg-manuscript-execution-plan.yaml` -- YAML execution plan for KG-to-manuscript pipeline
7. `lit-report-ai-cards.md` -- Literature report: AI Cards and structured documentation artifacts (55+ refs)
8. `lit-report-biomedical-segmentation.md` -- Literature report R3: tubular segmentation, foundation models, biostatistics (50 papers)
9. `lit-report-biomedical-segmentation.xml` -- XML invocation plan for R3 literature report
10. `lit-report-computational-reproducibility.md` -- Literature report R6: Docker, DVC, MLflow reproducibility
11. `lit-report-computational-reproducibility.xml` -- XML invocation plan for R6 report
12. `lit-report-federated-multisite.md` -- Literature report R4: federated learning, multisite data governance
13. `lit-report-federated-multisite.xml` -- XML invocation plan for R4 (multisite) report
14. `lit-report-medical-mlops.md` -- Literature report R5: MedMLOps pillars, regulatory mapping
15. `lit-report-medical-mlops.xml` -- XML invocation plan for R5 report
16. `lit-report-multiphoton-neuroscience.md` -- Literature report R4: multiphoton microscopy, smart acquisition
17. `lit-report-multiphoton-neuroscience.xml` -- XML invocation plan for R4 (neuroscience) report
18. `lit-report-platform-engineering-paas-prompt.md` -- User prompt for platform engineering report
19. `lit-report-platform-engineering-paas.md` -- Literature report R6: MLOps-as-PaaS, intercloud portability
20. `lit-report-platform-engineering-paas.xml` -- XML invocation plan for R6 (PaaS) report
21. `preregistration-tripod-mapping.md` -- Maps factorial design to TRIPOD+AI 27 items
22. `repo-to-manuscript-prompt.md` -- Verbatim user prompts for manuscript planning sessions
23. `repo-to-manuscript.md` -- Primary manuscript plan: Nature Protocols framing, IMRAD structure, KG architecture
24. `reporting-templates-plan.md` -- Implementation plan for automated reporting templates
25. `research-reports-general-plan-for-manuscript-writing.md` -- 6-theme research plan for comprehensive manuscript coverage
26. `tripod-compliance-matrix.md` -- TRIPOD+AI (27 items) + TRIPOD-LLM (19 items) + TRIPOD-Code compliance tracking
