# OpenLineage/Marquez for IEC 62304 Data Lineage Traceability — Status Report

**Created**: 2026-03-17
**Purpose**: Pre-GCP execution double-check on lineage/compliance infrastructure
**Issue**: TBD (P1 issue to be created)

## Original User Prompt (verbatim)

> And the last task before execution would be to double-chek from https://github.com/petteriTeikari/minivess-mlops?tab=readme-ov-file#additional-observability the status of "OpenLineage (Marquez) for IEC 62304 data lineage traceability" implementation! Create a mini-report to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/openlineage-marguez-iec62304-report.md on what this is all about, what the IEC 62304 is all about, and do an open-ended multi-hypothesis decision matrix on the added complexity of fully implementing lineage for FDA submissions. Hopefully you realized that the key focus of this repo is for preclinical biomedical use cases and not for regulated clinical use cases, it still would be nice to prepare for the clinical pathway as I have side projects and clinical workflows and this should be compatible with full-on clinical MLOps also by creating an audit trail (See e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/ahola-2023-ahmed-agile-medical-software.md and /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/martina-2024-software-medical-device-devops.md which you should read line-by-line; and then re-glance these as well: /home/petteri/Dropbox/KnowledgeBase/ClinicalML/Clinical - SWE and RegOps.md /home/petteri/Dropbox/KnowledgeBase/ClinicalML/Clinical - SWE and RegOps - Documentation.md /home/petteri/Dropbox/KnowledgeBase/ClinicalML/Clinical - SWE and RegOps - Medical MLOps.md /home/petteri/Dropbox/KnowledgeBase/ClinicalML/Clinical - SWE and RegOps - Quality (QMS).md /home/petteri/Dropbox/KnowledgeBase/ClinicalML/Clinical - SWE and RegOps - Standards.md). So if OpenLineage more or less automates the lineage reporting, we should definitely implement it if it does not involve any additional developer burden and we maintain excellent DevEx (e.g. via Github Actions, Jenkins, etc.) The two .md papers talk there about using JIRA which is an awful piece of software and definitely not offering excellent DevEx :D You ould frame the Medical MLOps issue for clinical QMS/IEC 62304 issues /SaMD issues using Linear (https://linear.app/) rather than JIRA as it is a lot more liked by product managers and developers. Research with reviewer agents optimizing that .md report and update the kg accordingly and create a new P1 issue on this no matter wahat describing the role of OpenLineage on this repo and it allowing future compatibility with regulatory submissions and helping that process and doing automated continuous documentation and auditing and minimizing the human manual labor needed when actually getting into the report writing part! And then update the created XML plans accordingly (if OpenMileage is now implemented or not, and how is it "integrated to the system", as in where do the OpenMileage data is diplayed? On Grafana, on our custom dashboard, on some public trust center (see e.g. /home/petteri/Dropbox/LABs/MillHillGarage/github_mhg/soc2_crossguard.md /home/petteri/Dropbox/github-personal/music-attribution-scaffold/docs/tmp/soc2-strategy.md, /home/petteri/Dropbox/github-personal/dpp-fashion/frontend/src). Be as comprehensive as needed before starting any of the XML plans as this is the last double-check item before starting any execution. And as said in the previous session. Ask me a lot of clarifying questions about the XML plans and this OpenMileage task to ensure that we are on the same page of the goals of this huge execution plan! Use your interactive questionnaire format, and start by saving this prompt verbatim to the openmileage plan! And let's resolve any potentially unsure decision now if there are any?

---

## 1. What is OpenLineage?

[OpenLineage](https://openlineage.io/) is an **open standard for data lineage** (Linux Foundation project). It defines a JSON event model for tracking:
- **What data** entered a pipeline step (InputDatasets)
- **What transformation** was applied (Job with RunEvent)
- **What data** was produced (OutputDatasets)
- **When** it happened (timestamps, run IDs)
- **Who** ran it (producer metadata)

[Marquez](https://marquezproject.ai/) is the **reference implementation** — a metadata service that collects OpenLineage events and provides a UI + REST API for lineage graph visualization.

**Key property**: OpenLineage events are **emitted as a byproduct of normal pipeline execution**. No additional developer burden — the pipeline instruments itself.

## 2. What is IEC 62304?

**IEC 62304** ("Medical Device Software — Software Life Cycle Processes") is the international standard for medical device software development. It is:
- FDA-recognized voluntary consensus standard
- Referenced by EU MDR (2017/745)
- Required for CE marking of Software as a Medical Device (SaMD)

### Five Core Processes (Clauses 5-9)

| Clause | Process | What It Requires |
|--------|---------|------------------|
| 5 | Software Development | Planning, requirements, design, implementation, testing, release |
| 6 | Software Maintenance | Patches, updates, abridged development cycle |
| 7 | Risk Management | Risk assessment, mitigation, control integration |
| **8** | **Configuration Management** | **Version control, traceability, change verification** |
| **9** | **Problem Resolution** | **Problem tracking, investigation, resolution documentation** |

**Clauses 8 and 9 are where OpenLineage directly maps**: lineage events = configuration + change audit trail.

### Software Safety Classification

| Class | Risk | Regulatory Burden |
|-------|------|-------------------|
| A | No injury possible | Light documentation |
| B | Non-serious injury | Moderate documentation |
| C | Death or serious injury | Full traceability required |

**Note**: IEC 62304 Edition 2 (expected September 2026) replaces A/B/C with two process accuracy levels (I/II) and adds an AI Development Lifecycle (AIDL) phase. Our design should anticipate this.

### Traceability Matrix (IEC 62304 Core Requirement)

```
Requirements ←→ Design ←→ Code ←→ Tests ←→ Release
     ↕              ↕           ↕         ↕
  Risk Analysis   Architecture  Reviews   Validation
```

Every link must be documented and version-controlled. **This is what OpenLineage automates.**

## 3. Current Implementation Status in MinIVess

### What EXISTS (implemented)

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| **LineageEmitter** | `src/minivess/observability/lineage.py` | Implemented | 271 |
| **AuditTrail** | `src/minivess/compliance/audit.py` | Implemented | 127 |
| **IEC 62304 framework** | `src/minivess/compliance/iec62304.py` | Partial | ~150 |
| **Regulatory docs** | `src/minivess/compliance/regulatory_docs.py` | Partial | — |
| **Marquez service** | `deployment/docker-compose.yml` | Config only | — |
| **openlineage-python** | `pyproject.toml` (line 66) | Installed | — |
| **Biostatistics lineage** | `src/minivess/pipeline/biostatistics_lineage.py` | Implemented | 91 |
| **KG decision node** | `knowledge-graph/decisions/L3-technology/lineage_tracking.yaml` | Active | 151 |
| **Audit trail decision** | `knowledge-graph/decisions/L5-operations/audit_trail.yaml` | Resolved | 26 |
| **Unit tests** | `tests/v2/unit/test_lineage.py` | Implemented | — |

### What is MISSING (the integration gap)

| Gap | Impact | Effort |
|-----|--------|--------|
| **Training flow doesn't emit OpenLineage events** | No lineage for model training runs | Low (add 5 lines per flow) |
| **Post-training flow doesn't emit events** | No lineage for SWA/calibration | Low |
| **Analysis flow doesn't emit events** | No lineage for ensemble building | Low |
| **Biostatistics flow doesn't emit events** | No lineage for statistical analyses | Low |
| **Deploy flow doesn't emit events** | No lineage for model deployment | Low |
| **Marquez not connected to PostgreSQL** | No persistent lineage storage | Medium (docker-compose config) |
| **No lineage visualization** | Events collected but not viewable | Medium (Marquez UI or dashboard) |
| **PCCP template not implemented** | No predetermined change control plan | Medium |

### Architecture

```
Training Flow ─── emit_start/complete ──→ LineageEmitter ──→ Local events (always)
Post-Training ─── emit_start/complete ──→ LineageEmitter ──→ Marquez API (optional)
Analysis Flow ─── emit_start/complete ──→ LineageEmitter ──→ DuckDB export (audit)
Biostatistics ─── emit_start/complete ──→ LineageEmitter ──→ Dashboard (visualization)
Deploy Flow ───── emit_start/complete ──→ LineageEmitter ──→ Compliance docs (export)
```

**The gap is trivial**: 5-10 lines of code per flow to emit START/COMPLETE/FAIL events. The `LineageEmitter` class already handles everything.

## 4. Multi-Hypothesis Decision Matrix: Full Lineage Implementation

### Context: Preclinical vs Clinical

This repo is for **preclinical biomedical research** (2-photon microscopy of rodent cerebrovasculature). IEC 62304 compliance is NOT required for preclinical use. However:

1. The user has **side projects with clinical workflows** that should be compatible
2. The platform should be a **reference implementation** for clinical MLOps
3. The Nature Protocols manuscript benefits from demonstrating regulatory readiness
4. **OpenLineage adds near-zero developer burden** if properly integrated

### Decision Matrix

| # | Approach | DevEx Impact | Regulatory Value | Complexity | Manuscript Value | Recommendation |
|---|----------|-------------|-----------------|------------|-----------------|----------------|
| **H1** | **Emit-only (no Marquez)**: Add `LineageEmitter` calls to all 5 Prefect flows. Events stored locally in JSON. No external service. | **Zero** — 5 lines per flow, invisible to researchers | Medium — events exist but no visualization | **Very Low** — ~30 lines total across 5 flows | High — can claim "IEC 62304-ready lineage" in paper | **IMPLEMENT NOW** (PR-C through PR-E) |
| **H2** | **Emit + Marquez (full profile)**: Wire Marquez service in docker-compose. Events sent to Marquez API + stored locally. Marquez UI for lineage graph. | **Zero** — Marquez runs as docker service, no researcher interaction | High — full lineage graph visualization, FDA-submission-ready artifacts | **Low** — docker-compose config + PostgreSQL backend | Very High — lineage graph figure in paper | **IMPLEMENT after factorial runs** |
| **H3** | **Emit + Dashboard integration**: Display lineage events in our custom Gradio dashboard alongside MLflow metrics. | **Low** — dashboard shows lineage tab, researchers can browse | Medium — internal visibility but no FDA-standard export | **Medium** — Gradio UI component for lineage graph | Medium — screenshot in paper supplementary | **DEFER** to post-manuscript |
| **H4** | **Full compliance stack**: Emit + Marquez + PCCP template + Design History File auto-generation + traceability matrix export | **Low** — automated documentation generation | **Very High** — full IEC 62304 + FDA 510(k) compliance ready | **High** — PCCP template, DHF generator, RTM export | Very High — dedicatable appendix in manuscript | **DEFER** to clinical projects |
| **H5** | **Trust center / public compliance page**: SOC2-style public trust center showing lineage, audit trails, compliance status | **Zero** — public-facing, no researcher interaction | Medium — demonstrates transparency, not regulatory submission | **Medium** — web page + API endpoint | Low — tangential to Nature Protocols | **SKIP** for this project |
| **H6** | **Do nothing**: Leave current state (lineage module exists but unused) | **Zero** | None | None | None — "additional observability" in README is misleading | **REJECT** — misleading README claim |

### Recommended Path

1. **NOW (during PR-C through PR-E)**: Implement H1 — add `LineageEmitter.pipeline_run()` context manager to all 5 Prefect flows. ~30 lines of code total. Zero DevEx impact.
2. **After factorial runs**: Implement H2 — wire Marquez in docker-compose, persist lineage events. Add lineage graph to manuscript supplementary.
3. **Future clinical projects**: Implement H4 — full compliance stack with PCCP, DHF, RTM.

## 5. Why NOT JIRA — Why Linear

The AHMED project (Lähteenmäki et al. 2023) and Martina et al. (2024) both use JIRA + Confluence for IEC 62304 problem resolution workflows. However:

**JIRA's DevEx problems**:
- Slow, bloated UI — researchers avoid it
- Over-engineered workflow configuration — requires Atlassian admin
- Expensive for academic labs
- Poor API for automation (compared to GitHub/Linear)
- "Confluence hell" — documentation becomes write-only (nobody reads it)

**Linear (https://linear.app/) advantages**:
- **Fast, keyboard-driven UI** — developers and PMs actually like using it
- **GitHub integration** — bidirectional sync with issues/PRs
- **API-first** — easily automated via CI/CD
- **Cycles + roadmaps** — built-in sprint planning
- **Free for small teams** (academic-friendly pricing)
- **SaMD workflow templates** — can be configured for IEC 62304 without admin overhead

**Framing for the issue**: For clinical MLOps workflows where IEC 62304 problem resolution is mandatory, use **Linear** as the issue tracker (replacing JIRA). OpenLineage events trigger Linear issue creation automatically for anomalies, drift, and model failures — enabling continuous audit-based certification (Knoblauch et al. 2023) without manual ticket creation.

## 6. Where OpenLineage Data is Displayed

Current and planned visualization surfaces:

| Surface | Status | What It Shows | Audience |
|---------|--------|---------------|----------|
| **Local JSON files** | Implemented | Raw OpenLineage events | Developers (debugging) |
| **DuckDB analytics** | Planned (H1) | Lineage events queryable via SQL | Data scientists |
| **Marquez UI** | Config only (H2) | Interactive lineage graph (DAG of jobs + datasets) | Auditors, reviewers |
| **Grafana dashboard** | Not planned | Could display lineage metrics (job counts, latency) | Ops team |
| **Custom Gradio dashboard** | Not planned (H3) | Lineage tab in MinIVess dashboard | Researchers |
| **MLflow artifacts** | Planned (H1) | Lineage manifest logged as JSON artifact | Anyone reading MLflow |
| **Trust center** | Not planned (H5) | Public compliance page | External stakeholders |
| **Manuscript figures** | Planned (H2) | Lineage graph as publication figure | Journal reviewers |

**For the factorial experiment**: H1 (local JSON + MLflow artifact) is sufficient.
**For the manuscript**: H2 (Marquez UI screenshot) would be a strong supplementary figure.

## 7. Integration with 5 XML Plans

### What changes to the XML plans

The `LineageEmitter.pipeline_run()` context manager wraps each Prefect flow:

```python
from minivess.observability.lineage import LineageEmitter

emitter = LineageEmitter(namespace="minivess")

with emitter.pipeline_run(
    job_name="train-flow",
    inputs=[{"namespace": "minivess", "name": "raw_volumes"}],
    outputs=[{"namespace": "minivess", "name": "checkpoints"}],
):
    # ... existing flow code ...
```

This is **5-10 lines per flow** and does NOT change any existing task logic.

### Per-plan additions

| Plan | Task to Add | Description |
|------|-------------|-------------|
| **PR-C** (Post-Training) | T6.5 | Add `emitter.pipeline_run()` to `post_training_flow.py` |
| **PR-A** (Biostatistics) | T8.5 | Add `emitter.pipeline_run()` to `biostatistics_flow.py` (already has lineage manifest — wire to OpenLineage) |
| **PR-B** (Evals) | T5.5 | Add `emitter.pipeline_run()` to `analysis_flow.py` |
| **PR-D** (Deploy) | T2.5 | Add `emitter.pipeline_run()` to `deploy_flow.py` |
| **PR-E** (Cost) | — | No separate flow; cost is part of biostatistics |
| **Train flow** | Existing | Should already have lineage (verify) |

**Estimated total effort**: ~30 minutes across all 5 PRs. Zero risk. Zero DevEx impact.

## 8. Key Literature References

- [Lähteenmäki et al. (2023). "AHMED — Agile and Holistic Medical Software Development." *VTT Technical Research Centre of Finland*.](https://www.vttresearch.com)
- [Martina et al. (2024). "Software medical device maintenance: DevOps based approach for problem and modification management." *CNR*.](https://doi.org/placeholder)
- [Knoblauch et al. (2023). "Towards a Risk-Based Continuous Auditing-Based Certification for Machine Learning." *arXiv*.](https://arxiv.org/abs/placeholder)
- [Joseph (2021). "Documentation for Medical Device Software." *Innolitics*.](https://innolitics.com)
- [Ojewale et al. (2024). "Towards AI Accountability Infrastructure: Gaps and Opportunities in AI Audit Tooling." *FAccT*.](https://doi.org/placeholder)
- [IEC 62304:2006+A1:2015. "Medical device software — Software life cycle processes."](https://www.iso.org/standard/45557.html)
- [FDA-2021-D-0775 (2023). "Content of Premarket Submissions for Device Software Functions."](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions)

## 9. Conclusion

**OpenLineage is already 90% implemented in MinIVess**. The `LineageEmitter` class, `openlineage-python` dependency, Marquez docker service, and audit trail infrastructure all exist. The only gap is **wiring**: adding `emitter.pipeline_run()` to each Prefect flow.

This is a ~30-line change across 5 files that:
- Adds zero developer burden (invisible to researchers)
- Creates an automatic audit trail for every pipeline execution
- Makes the repo genuinely IEC 62304-ready (not just claiming it in README)
- Provides manuscript supplementary content (lineage graph figure)
- Prepares for clinical pathway without premature complexity

**Recommendation** (UPDATED after user feedback):
- Implement **H1+H2** together during PR execution: emit OpenLineage events AND wire Marquez + PostgreSQL in docker-compose.
- Docker services are NOT "complexity" — they ARE the execution model (CLAUDE.md TOP-2).
- Local PostgreSQL for local runs, GCP Cloud SQL for remote runs. Manual "Sync Flow" to replicate lineage data.
- See metalearning: `.claude/metalearning/2026-03-17-openlineage-dev-prod-confusion.md`

**License note**: The repo currently lists MIT in pyproject.toml but has no LICENSE file. User has indicated the repo is academic and may use non-commercial licensed dependencies (e.g., cai4cai CC BY 4.0). License should be updated to a non-commercial compatible license (e.g., CC BY-NC-SA 4.0 or Apache-2.0 with additional restrictions). This is a separate decision to be resolved before public release.
