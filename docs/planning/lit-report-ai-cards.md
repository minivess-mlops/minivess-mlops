# AI Cards: Comprehensive Taxonomy and Application to Biomedical MLOps

**Date**: 2026-03-19
**Target publication**: Nature Protocols
**Platform**: MinIVess MLOps v2
**Reference catalog**: `docs/planning/ai-cards-comprehensive-reference.md` (55+ references)

---

## 1. Executive Summary

Structured documentation artifacts -- collectively termed "cards" -- have become the
primary mechanism for communicating the properties, limitations, and risks of ML systems
to diverse stakeholders. Since [Mitchell et al. (2019). "Model Cards for Model Reporting." *Proc. FAT* 2019*.](https://dl.acm.org/doi/10.1145/3287560.3287596) introduced Model Cards,
the field has expanded to over 30 distinct card types spanning data provenance, system
architecture, regulatory compliance, sustainability, and human factors.

A 2024 analysis of 32,111 model cards on Hugging Face found that most omit safety, bias,
and environmental impact information ([Liang et al. (2024). "Systematic analysis of 32,111 AI model cards." *Nature Machine Intelligence*, 6, 744-753.](https://www.nature.com/articles/s42256-024-00857-z)), revealing a documentation
completeness problem that is especially acute in regulated domains like medical imaging.
For biomedical segmentation platforms targeting FDA SaMD submission and Nature Protocols
publication, a principled card stack must satisfy three distinct audiences: (1) peer
reviewers demanding scientific reproducibility, (2) regulators requiring traceability
under IEC 62304 and the EU AI Act, and (3) clinicians needing interpretable performance
summaries.

This report catalogs the full AI Cards taxonomy, maps it to the MinIVess MLOps platform's
existing compliance infrastructure, identifies gaps, and proposes a prioritized
implementation plan that distinguishes between the minimum viable card stack for the
Nature Protocols paper and the full card stack for regulatory submission.

The MinIVess platform already implements 9 of the 18 card types identified as relevant:
Model Cards, Audit Trail, Regulatory Document Generator, CONSORT-AI, MI-CLEAR-LLM,
IEC 62304 framework, EU AI Act checklist, Fairness reporting, and SBOM/CycloneDX
generation. This report identifies 9 missing card types and proposes a phased
implementation prioritized by scientific and regulatory impact.

---

## 2. Card Taxonomy (General -- Any ML Project)

### 2.1 Foundation Cards (Model Cards, Datasheets for Datasets)

Foundation cards document the two irreducible inputs to any ML system: the model and
the data. They are the most mature card types with the broadest adoption.

**Model Cards** ([Mitchell et al. (2019). "Model Cards for Model Reporting." *Proc. FAT* 2019*.](https://dl.acm.org/doi/10.1145/3287560.3287596)) are short documents accompanying
trained ML models that provide benchmarked evaluation across different conditions,
intended use, limitations, and ethical considerations. Model Cards are the foundational
"card" that inspired all subsequent card types. The format has been adopted as a de facto
standard by Hugging Face ([Ozoani et al. (2022). "Model Card Guidebook." Hugging Face.](https://huggingface.co/docs/hub/en/model-card-guidebook)), Google's Model Card Toolkit
([TensorFlow Team (2020). "Model Card Toolkit."](https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide)), and is embedded in the
CycloneDX ML-BOM specification ([OWASP CycloneDX (2023). "Machine Learning Bill of Materials." ECMA-424.](https://cyclonedx.org/capabilities/mlbom/)).

**Datasheets for Datasets** ([Gebru et al. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92.](https://dl.acm.org/doi/10.1145/3458723)) adapt the electronics industry's
datasheet concept to ML datasets, documenting motivation, creation process, composition,
intended uses, distribution, and maintenance. Adopted internally by Microsoft, Google,
and IBM.

**Data Cards** ([Pushkarna et al. (2022). "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI." *Proc. ACM FAccT 2022*.](https://dl.acm.org/doi/10.1145/3531146.3533231)) extend datasheets with structured summaries optimized for
stakeholder communication across the dataset lifecycle. The accompanying Data Cards
Playbook provides step-by-step guidance for creating cards through participatory design.

**Dataset Nutrition Labels** ([Holland et al. (2018). "The Dataset Nutrition Label." arXiv:1805.03677](https://datanutrition.org/); [Chmielinski et al. (2022). "The Dataset Nutrition Label (2nd Gen)." arXiv:2201.03954.](https://datanutrition.org/)) use a nutrition-facts-for-food
metaphor, distilling dataset properties into a concise visual format. The 2nd generation
adds context-specific Use Cases and Alerts. The metaphor is particularly effective for
communicating dataset quality to clinicians.

For biomedical imaging, the most critical foundation card gap is **Data Cards**. While
Model Cards already exist in MinIVess (`src/minivess/compliance/model_card.py`), there
is no equivalent structured documentation for the MiniVess dataset itself -- its imaging
protocol, annotation inter-rater agreement, demographic coverage, or known biases.

### 2.2 Operational Cards (System Cards, Deployment Cards, SBOM)

Operational cards document how multiple components work together as a deployed system,
extending beyond individual model or dataset documentation.

**System Cards** ([Alsallakh et al. (2022). "System-Level Transparency of Machine Learning." Meta Technical Report.](https://ai.meta.com/research/publications/system-level-transparency-of-machine-learning/)) document how multiple ML models,
AI tools, and non-AI technologies interact as a system. Meta published System Cards for
Instagram Feed Ranking, demonstrating the format's utility for multi-component pipelines.
For MinIVess, a System Card would document the data-train-evaluate-deploy flow topology,
inter-flow contracts, and the interaction between Prefect orchestration, MLflow tracking,
and BentoML serving.

**IBM AI FactSheets** ([Arnold et al. (2019). "FactSheets: Increasing Trust in AI Services through Supplier's Declarations of Conformity." *IBM Journal of Research and Development*, 63(4/5).](https://ieeexplore.ieee.org/document/8843893)) are supplier's declarations of
conformity for AI services, documenting purpose, performance, safety, security, and
provenance. More comprehensive than Model Cards because they cover entire AI services
(potentially multiple models and APIs). The FactSheets 360 toolkit provides automated
generation.

**CycloneDX ML-BOM** ([OWASP CycloneDX (2023). "Machine Learning Bill of Materials (AI/ML-BOM)." ECMA-424.](https://cyclonedx.org/capabilities/mlbom/)) extends the software bill
of materials (SBOM) standard to ML systems, documenting datasets, models, configurations,
dependencies, and provenance in machine-readable JSON/XML. Part of the ECMA-424 standard.
MinIVess already generates CycloneDX SBOMs (`tests/v2/unit/test_sbom_generation.py`).

### 2.3 Governance Cards (Audit Cards, AI Cards for EU AI Act, Fairness Cards)

Governance cards address accountability, regulatory compliance, and bias mitigation.

**AI Cards** ([Golpayegani et al. (2024). "AI Cards: Towards an Applied Framework for Machine-Readable AI and Risk Documentation Inspired by the EU AI Act." *Proc. APF 2024*, Springer LNCS.](https://arxiv.org/abs/2406.18211)) provide a holistic framework for
machine-readable AI documentation aligned with EU AI Act Annex IV requirements. They
bridge the gap between human-readable transparency artifacts and machine-readable
compliance verification, enabling automated compliance checking.

**TechOps** ([Lucaj et al. (2025). "TechOps: Technical Documentation Templates for the AI Act." *Proc. AAAI/ACM AIES 2025*, 8(2), 1647-1660.](https://ojs.aaai.org/index.php/AIES/article/view/36663)) provides open-source templates
specifically designed for EU AI Act compliance, validated on real-world image segmentation
scenarios. The templates cover data, models, and applications across the full AI lifecycle.

**Audit Cards** ([Staufer et al. (2025). "Audit Cards: Contextualizing AI Evaluations." arXiv:2504.13839.](https://arxiv.org/abs/2504.13839)) are a structured format for contextualizing
AI evaluation reports, documenting auditor identity, evaluation scope, methodology,
resource access, process integrity, and review mechanisms. Their analysis found that most
existing evaluation reports omit crucial context about who performed the evaluation and
under what constraints.

**Fairness reporting** in MinIVess (`src/minivess/compliance/fairness.py`) implements
subgroup disparity analysis following [Krishnan et al. (2022). "CyclOps: Toolkit for Healthcare ML Auditing and Monitoring." NeurIPS 2022 Workshop.](https://cyclops.readthedocs.io/) patterns, with max-min disparity
computation across demographic subgroups and automated pass/fail thresholds.

### 2.4 Sustainability Cards (Carbon Cards, ESG Cards)

Sustainability documentation has become a publication requirement in several venues and
is increasingly mandated by funding agencies.

**Carbon emissions tracking** follows the framework of [Lacoste et al. (2019). "Quantifying the Carbon Emissions of Machine Learning." arXiv:1910.09700.](https://arxiv.org/abs/1910.09700), which identifies
GPU type, training duration, energy grid carbon intensity, and server location as the
critical factors. The seminal work by [Strubell et al. (2019). "Energy and Policy Considerations for Deep Learning in NLP." *Proc. ACL 2019*.](https://aclanthology.org/P19-1355/) quantified the
environmental cost of training, finding that a single transformer NAS run produces CO2
equivalent to five cars' lifetimes.

**CodeCarbon** ([CodeCarbon contributors (2020-present). "CodeCarbon: Track emissions from Compute."](https://codecarbon.io/)) is the standard Python library for tracking
carbon emissions during training, measuring GPU + CPU + RAM power consumption and
applying regional carbon intensity factors. It generates LaTeX snippets suitable for
research papers.

**AI ESG Protocol** ([Saetra (2023). "The AI ESG Protocol." *Sustainable Development*, 31(2), 1027-1037.](https://onlinelibrary.wiley.com/doi/10.1002/sd.2438)) provides a flexible
framework for evaluating and disclosing the environmental, social, and governance impacts
of AI systems across micro (individual model) to macro (societal) impact scopes.

MinIVess currently has `src/minivess/observability/cost_logging.py` for cloud instance
cost tracking (GPU hours, spot vs. on-demand pricing), but lacks carbon emission
estimation and ESG impact documentation.

### 2.5 Human-Centric Cards (Team Cards, Use Case Cards, Value Cards)

Human-centric cards document the people and purposes behind AI systems.

**Team Cards** ([Modise et al. (2025). "Introducing the Team Card." *PLOS Digital Health*, 4(3), e0000495.](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000495)) document researcher positionality -- worldviews, training,
backgrounds, and experiences -- that shape decisions during clinical AI development. This
is the only card type that addresses the team dimension, promoting reflexivity to
identify blind spots. For a biomedical imaging platform used across heterogeneous
research labs, Team Cards are essential for disclosing whose perspectives shaped the
annotation protocols, loss function choices, and evaluation metrics.

**Use Case Cards** ([Hupont et al. (2024). "Use case cards: a use case reporting framework inspired by the European AI Act." *Ethics and Information Technology*, 26(2).](https://link.springer.com/article/10.1007/s10676-024-09757-7)) use UML-based
templates to document AI system use cases and implicitly assess risk levels aligned with
the EU AI Act. For MinIVess, Use Case Cards would document the distinct clinical
applications of vascular segmentation: surgical planning, drug delivery assessment,
disease monitoring, and research quantification.

**AI Usage Cards** ([Wahle et al. (2023). "AI Usage Cards: Responsibly Reporting AI-generated Content." *Proc. ACM/IEEE JCDL 2023*.](https://dl.acm.org/doi/10.1109/JCDL57899.2023.00060)) address a three-dimensional model
(transparency, integrity, accountability) for reporting AI use in scientific research.
Given MinIVess's use of Claude Code for development automation, AI Usage Cards are
directly applicable for disclosing AI-assisted development in the Nature Protocols paper.

**Value Cards** ([Shen et al. (2021). "Value Cards: An Educational Toolkit." *Proc. ACM FAccT 2021*.](https://dl.acm.org/doi/abs/10.1145/3442188.3445971)) are primarily an educational
tool for deliberating ethical tradeoffs in ML systems. Lower priority for MinIVess but
the persona-based deliberation approach could inform clinical stakeholder engagement.

### 2.6 Clinical/Regulatory Cards (TRIPOD+AI, CONSORT-AI, FDA PCCP)

Clinical and regulatory cards are mandatory for any path toward clinical deployment or
publication in medical journals.

**TRIPOD+AI** ([Collins et al. (2024). "TRIPOD+AI statement." *BMJ*, 385.](https://pubmed.ncbi.nlm.nih.gov/38626948/)) is a 27-item checklist harmonizing reporting for
prediction model studies regardless of methodology (regression or ML). It is the
mandatory reporting standard for clinical prediction and segmentation models published
in medical journals. TRIPOD+AI supersedes the 2015 TRIPOD statement.

**CONSORT-AI** is the minimum reporting standard for AI interventions in clinical trials.
MinIVess implements this as `src/minivess/compliance/reporting_templates.py::ConsortAIChecklist`,
generating markdown reports with sections for AI intervention, training data, input/output
descriptions, preprocessing, evaluation protocol, performance metrics, confidence
intervals, failure analysis, and limitations.

**FDA PCCPs** ([U.S. FDA (2024). "Marketing Submission Recommendations for a Predetermined Change Control Plan." Final Guidance.](https://www.fda.gov/media/166704/download)) allow manufacturers to
pre-specify anticipated modifications at initial market authorization. The five guiding
principles (focused, risk-based, evidence-based, transparent, lifecycle-oriented) were
developed jointly with Health Canada and UK MHRA. MinIVess has a stub PCCP template
(`src/minivess/compliance/iec62304.py::PCCPTemplate`) that generates the document
structure but does not yet populate it from MLflow metadata.

**IMDRF SaMD Characterization** ([IMDRF SaMD Working Group (2025). "Characterization Considerations for Medical Device Software." IMDRF/SaMD WG/N81 FINAL:2025.](https://www.imdrf.org/)) defines the
international regulatory classification pathway for software as a medical device,
adopted by FDA, EU, and other regulators.

**MI-CLEAR-LLM** is implemented in `src/minivess/compliance/reporting_templates.py::MiClearLLMChecklist` for documenting LLM components used in the
pipeline (e.g., AI-assisted literature review, code generation).

### 2.7 Pipeline Cards (Environment Cards, Lineage Cards, Method Cards)

Pipeline cards document the computational environment, data lineage, and methodological
choices that determine reproducibility.

**Method Cards** ([Adkins et al. (2022). "Method Cards for Prescriptive Machine-Learning Transparency." *Proc. CAIN 2022*.](https://dl.acm.org/doi/10.1145/3522664.3528600)) are the prescriptive
complement to descriptive Model Cards. While Model Cards describe the cooked meal,
Method Cards provide the recipe. They guide ML engineers through model development with
actionable guidance on methodological and algorithmic choices. For a Nature Protocols
paper, Method Cards are arguably the most important card type: the entire purpose of a
protocol is to be a prescriptive, reproducible recipe.

**OpenLineage lineage tracking** provides data lineage documentation through the
OpenLineage standard. MinIVess has a lineage emitter (`src/minivess/observability/lineage.py::LineageEmitter`) that emits START/COMPLETE/FAIL
events at pipeline stage boundaries. The emitter supports both local event collection and
remote Marquez API integration. However, the emitter is not yet wired to all five Prefect
flows -- it operates in all flows but the dashboard flow, with lineage events emitted
for data ingestion, training, post-training analysis, and deployment stages.

**Reward Reports** ([Gilbert et al. (2022). "Reward Reports for Reinforcement Learning." *AAAI/ACM AIES 2023*.](https://dl.acm.org/doi/10.1145/3600211.3604698)) introduce the concept of
"living documents" that track updates to design choices and optimization targets over
time. While designed for RL, the living document paradigm is relevant to tracking loss
function evolution and metric selection rationale across HPO campaigns in MinIVess.

**fAIlureNotes** ([Moore et al. (2023). "fAIlureNotes: Supporting Designers in Understanding the Limits of AI Models for Computer Vision Tasks." *Proc. CHI 2023*.](https://dl.acm.org/doi/10.1145/3544548.3581242)) provide a taxonomy of failure modes
for CV tasks across three levels: input-level, model-level, and response-level. For
medical imaging segmentation, documenting failure modes (missed thin vessels, false
positive noise regions, boundary imprecision) is critical because false negatives have
direct clinical consequences.

**FeedbackLogs** ([Barker et al. (2023). "FeedbackLogs: Recording and Incorporating Stakeholder Feedback into Machine Learning Pipelines." arXiv:2307.15475.](https://arxiv.org/abs/2307.15475)) are addenda to
ML pipeline documentation that track stakeholder feedback collection, the feedback
itself, and how feedback updates the pipeline. They serve as evidence for algorithmic
auditing and iterative improvement.

---

## 3. Application to MinIVess MLOps Platform

### 3.1 Current Implementation Status

The MinIVess compliance infrastructure (`src/minivess/compliance/`) implements a
substantial card and documentation stack. The table below maps each card type to its
implementation status.

| Card Type | Status | Implementation | File Path |
|-----------|--------|----------------|-----------|
| Model Card | **Implemented** | `ModelCard` dataclass with `to_markdown()` | `src/minivess/compliance/model_card.py` |
| Audit Trail (IEC 62304) | **Implemented** | `AuditTrail` with event logging, SHA-256 hashing, save/load | `src/minivess/compliance/audit.py` |
| Regulatory Doc Generator | **Implemented** | Design History File, Risk Analysis, SRS, Validation Summary | `src/minivess/compliance/regulatory_docs.py` |
| CONSORT-AI Checklist | **Implemented** | 11-section markdown report generator | `src/minivess/compliance/reporting_templates.py` |
| MI-CLEAR-LLM Checklist | **Implemented** | 7-section LLM usage report | `src/minivess/compliance/reporting_templates.py` |
| IEC 62304 Framework | **Implemented** | Safety classification, lifecycle stages, traceability matrix, PCCP stub | `src/minivess/compliance/iec62304.py` |
| EU AI Act Checklist | **Implemented** | 4-tier risk classification, Articles 9-15 + 43 gap analysis | `src/minivess/compliance/eu_ai_act.py` |
| Fairness Reporting | **Implemented** | Subgroup disparity analysis with pass/fail thresholds | `src/minivess/compliance/fairness.py` |
| SBOM/CycloneDX | **Implemented** | ML-BOM generation (tests exist) | `tests/v2/unit/test_sbom_generation.py` |
| RegOps Pipeline | **Implemented** | CI/CD-triggered regulatory artifact generation | `src/minivess/compliance/regops.py` |
| ComplOps (510(k)/EU MDR) | **Implemented** | FDA 510(k) summary, EU MDR technical file, compliance gap assessment | `src/minivess/compliance/complops.py` |
| OpenLineage Lineage | **Partial** | Emitter exists, wired to 4/5 flows | `src/minivess/observability/lineage.py` |
| Cost/Cloud Tracking | **Partial** | SkyPilot env var collection, spot savings computation | `src/minivess/observability/cost_logging.py` |
| PCCP Template | **Stub** | Document structure only, not populated from MLflow | `src/minivess/compliance/iec62304.py::PCCPTemplate` |
| Data Card / Datasheet | **Missing** | No structured dataset documentation | -- |
| System Card | **Missing** | No system-level architecture documentation | -- |
| Carbon/ESG Card | **Missing** | No emission tracking (cost logging exists but not carbon) | -- |
| Team Card | **Missing** | No researcher positionality documentation | -- |
| Audit Card | **Missing** | No evaluation contextualization | -- |
| Method Card | **Missing** | No prescriptive methodology documentation | -- |
| Use Case Card | **Missing** | No UML-based use case documentation | -- |
| Saliency Card | **Missing** | No explainability method documentation | -- |
| fAIlureNotes | **Missing** | No structured failure mode documentation | -- |

**Summary**: 11 implemented, 2 partial, 1 stub, 9 missing.

### 3.2 Scientific Reproducibility Audit Trail (Non-Regulated)

For the Nature Protocols paper, the reproducibility audit trail requires a different
emphasis than regulatory compliance. The key question is: "Can another lab reproduce
these results?" This requires:

1. **Method Cards** -- the protocol itself is a Method Card. MONAI transform pipelines,
   loss function selection rationale (why `cbdice_cldice`?), augmentation strategies,
   and hyperparameter search spaces must be documented prescriptively. This is the highest
   priority missing card for Nature Protocols.

2. **Data Cards** -- the MiniVess dataset needs structured documentation: imaging protocol
   (multiphoton microscopy parameters), resolution, voxel spacing, annotation protocol
   (number of annotators, inter-rater agreement), demographic coverage, known biases,
   and split strategy (3-fold, seed=42, 47 train / 23 val per
   `configs/splits/3fold_seed42.json`).

3. **Model Cards** -- already implemented in `src/minivess/compliance/model_card.py`.
   Current implementation covers model details, intended use, training/evaluation data,
   metrics, limitations, and ethical considerations. Needs extension for multi-model
   comparison (DynUNet vs. SAM3 vs. ensemble).

4. **Carbon Cards** -- increasingly expected by Nature journals. CodeCarbon integration
   would provide GPU power consumption, regional carbon intensity, and total CO2
   emissions per training run.

5. **AI Usage Cards** -- disclose Claude Code's role in development automation, following
   [Wahle et al. (2023)](https://dl.acm.org/doi/10.1109/JCDL57899.2023.00060).

6. **OpenLineage Lineage** -- already partially implemented. Full wiring to all flows
   provides the data provenance chain from raw NIfTI volumes through preprocessing,
   training, evaluation, and artifact generation.

### 3.3 Regulated SaMD Audit Trail (FDA/EU MDR)

For FDA SaMD submission and EU MDR compliance, the audit trail requirements are
substantially more stringent and legally binding.

**EU AI Act (Regulation 2024/1689)** classifies medical device AI as high-risk under
Annex I, Section A ([EU Parliament (2024). "Regulation (EU) 2024/1689."](https://artificialintelligenceact.eu/article/11/)). Article 11
requires comprehensive technical documentation per Annex IV, covering:

- System description and intended purpose (MinIVess: **implemented** via `eu_ai_act.py`)
- Development process and data governance (MinIVess: **partial** via `audit.py`)
- Risk management system (MinIVess: **implemented** via `regulatory_docs.py`)
- Technical documentation (MinIVess: **implemented** via `regops.py`)
- Automatic logging / record-keeping (MinIVess: **implemented** via `AuditTrail`)
- Transparency provisions (MinIVess: **gap** -- no System Card)
- Human oversight mechanisms (MinIVess: **gap** -- not documented)
- Robustness and cybersecurity (MinIVess: **partial** -- drift detection exists)

**ALTAI self-assessment** ([EU HLEG (2020). "Assessment List for Trustworthy Artificial Intelligence."](https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment)) covers 7 requirements with
a technology-neutral checklist. MinIVess should generate an ALTAI report from existing
compliance artifacts.

**FDA PCCPs** are essential for the AI/ML SaMD pathway because they allow pre-specifying
model updates (retraining on new data, architecture changes) at initial authorization.
MinIVess's `PCCPTemplate` stub needs to be populated from MLflow experiment metadata:
which hyperparameters may change, what performance bounds must be maintained, and what
verification protocols apply.

**IEC 62304 lifecycle** is fully modeled in `src/minivess/compliance/iec62304.py` with
safety classification (Class A/B/C), lifecycle stages (development through maintenance),
and traceability matrix (requirements to implementation to tests).

**NIST AI RMF 1.0** ([NIST (2023). "AI Risk Management Framework 1.0."](https://www.nist.gov/itl/ai-risk-management-framework)) provides a
complementary US risk management vocabulary (Govern, Map, Measure, Manage). MinIVess's
compliance infrastructure maps naturally to the Measure and Manage functions but lacks
explicit Govern and Map documentation.

### 3.4 Gap Analysis: What's Missing

The following table summarizes the gaps between current implementation and the two target
use cases (Nature Protocols and FDA SaMD).

| Card Type | Nature Protocols Need | FDA SaMD Need | Current Gap | Effort (days) |
|-----------|----------------------|---------------|-------------|---------------|
| Data Card | **Critical** | **Critical** | Full implementation needed | 3-4 |
| Method Card | **Critical** | Medium | Full implementation needed | 2-3 |
| Carbon Card | **High** | Low | CodeCarbon integration + report generator | 1-2 |
| System Card | Medium | **Critical** | Full implementation needed | 2-3 |
| Team Card | **High** | Medium | Template + manual authoring workflow | 1 |
| AI Usage Card | **High** | Low | Template for Claude Code disclosure | 0.5 |
| Use Case Card | Low | **High** | UML template + clinical scenario docs | 1-2 |
| Audit Card | Low | **High** | Evaluation context documentation | 1 |
| Saliency Card | Medium | Medium | Captum method documentation framework | 1-2 |
| fAIlureNotes | Medium | **High** | Failure taxonomy + automated collection | 2-3 |
| PCCP (populated) | Low | **Critical** | Connect stub to MLflow metadata | 1-2 |
| FeedbackLogs | Low | Medium | Clinician feedback tracking | 1 |
| ALTAI Report | Low | **High** | Generate from existing compliance data | 1 |

**Total estimated effort for Nature Protocols minimum viable stack**: 8-11 days.
**Total estimated effort for full FDA SaMD stack**: 18-25 days.

---

## 4. Card Generation Architecture

### 4.1 Automated Card Generation (from MLflow + Prefect Metadata)

The MinIVess platform's existing observability infrastructure provides rich metadata
that can be automatically extracted into card fields.

**MLflow as the card data source.** Every training run logs 17+ parameters with
`train/` prefix, architecture parameters with `arch/` prefix, system information with
`sys/` prefix, and metrics with `val/` prefix (see `src/minivess/observability/CLAUDE.md`
for the full prefix taxonomy). This metadata directly populates:

- **Model Card** fields: model type, hyperparameters, performance metrics, training
  hardware, software versions
- **Data Card** fields: number of volumes (`data/n_volumes`), augmentation pipeline
  (`data/augmentation_pipeline`), split configuration
- **Carbon Card** fields: GPU type (`sys/gpu_model`), training duration
  (`prof/first_epoch_seconds`), cloud provider and region (`cost/provider`, `cost/region`)
- **Cost fields**: spot vs. on-demand pricing, total GPU hours, savings percentage

**Prefect as the orchestration metadata source.** Flow runs provide:

- **Lineage Card** fields: flow topology, task dependencies, data flow between stages
- **System Card** fields: which flows ran, in what order, with what Docker images
- **Audit Card** fields: run timestamps, success/failure status, actor identity

**OpenLineage events** (`src/minivess/observability/lineage.py`) provide:

- **Lineage Card** fields: input/output datasets per pipeline stage, run provenance
  chain, parent-child run relationships

The following card fields can be **fully automated** (no human input needed):

| Field | Source | Card Type |
|-------|--------|-----------|
| Model architecture | `arch/*` MLflow params | Model Card |
| Training metrics | `val/*`, `train/*` MLflow metrics | Model Card |
| Hardware/software | `sys/*` MLflow params | Model Card, Carbon Card |
| Dataset statistics | `data/*` MLflow params | Data Card |
| Cost/GPU hours | `cost/*` MLflow params | Carbon Card |
| Data lineage | OpenLineage events | Lineage Card |
| Pipeline topology | Prefect flow metadata | System Card |
| CI/CD provenance | `CIContext` from env vars | Audit Trail |
| Dependencies (SBOM) | `uv lock` + CycloneDX | SBOM |
| Traceability | pytest markers + IEC 62304 matrix | Regulatory Docs |

### 4.2 Manual Card Templates (for Human-Authored Sections)

Certain card fields require domain expertise and cannot be auto-generated:

| Field | Required Expertise | Card Type |
|-------|-------------------|-----------|
| Intended clinical use | Clinical collaborators | Model Card, Use Case Card |
| Known biases / limitations | Domain scientists | Model Card, Data Card |
| Annotation protocol | Annotators / pathologists | Data Card, CrowdWorkSheets |
| Ethical considerations | Ethics review board | Model Card, Value Card |
| Team positionality | Team members (self-authored) | Team Card |
| Failure mode clinical impact | Clinicians | fAIlureNotes |
| Regulatory intended purpose | Regulatory affairs | EU AI Act Checklist |
| PCCP permitted changes | Regulatory + engineering | PCCP |

For these fields, the architecture should provide:

1. **YAML templates** in `configs/cards/` with structured fields and placeholder text
2. **Validation schemas** that flag empty required fields at card generation time
3. **Merge logic** that combines auto-generated fields with manually authored YAML
4. **Version tracking** via git (cards are committed alongside code changes)

### 4.3 Card Storage and Versioning

Cards should be stored as **MLflow artifacts** attached to the run that generated them,
with a parallel copy committed to git for version control.

**MLflow artifact storage:**
```
mlruns/<experiment_id>/<run_id>/artifacts/
    cards/
        model_card.md
        data_card.md
        carbon_card.md
        lineage_card.json    (OpenLineage events)
        sbom.json            (CycloneDX ML-BOM)
```

**Git-committed templates:**
```
compliance/cards/
    templates/
        data_card_template.yaml
        team_card_template.yaml
        method_card_template.yaml
    generated/               (gitignored -- generated at build time)
```

This dual-storage approach ensures that:
- Auto-generated cards are tied to specific MLflow runs (reproducibility)
- Manual templates evolve with the codebase (version control)
- Generated cards are not committed to git (avoiding stale artifacts)

---

## 5. Recommended Card Stack for MinIVess

### 5.1 Minimum Viable Card Stack (for Nature Protocols Paper)

The Nature Protocols paper requires reproducibility, transparency, and methodological
rigor. The minimum viable card stack prioritizes cards that directly serve these goals.

| Priority | Card Type | Rationale | Auto-gen | Implementation Approach |
|----------|-----------|-----------|----------|------------------------|
| **P0** | Method Card | Nature Protocols IS a method card. Protocol = recipe. | Partial | YAML template + auto-populated from Hydra config |
| **P0** | Model Card | Required for each model in the comparison (DynUNet, SAM3) | Yes | Extend existing `ModelCard` with multi-model comparison |
| **P0** | Data Card | MiniVess dataset documentation for reproducibility | Partial | YAML template + auto-populated from `data/*` MLflow params |
| **P1** | Carbon Card | Increasingly expected by Nature journals | Yes | CodeCarbon integration into train flow |
| **P1** | AI Usage Card | Disclose Claude Code usage in development | Manual | YAML template following Wahle et al. (2023) |
| **P1** | Team Card | Researcher positionality for clinical AI | Manual | YAML template following Modise et al. (2025) |
| **P2** | Lineage Card | Data provenance for reproducibility | Yes | Export OpenLineage events as card |
| **P2** | CONSORT-AI | Already implemented, needs population from real runs | Yes | Populate existing `ConsortAIChecklist` from MLflow |
| **P2** | TRIPOD+AI | 27-item checklist for clinical prediction models | Manual | Checklist template with auto-fill for applicable items |
| **P2** | Saliency Card | Document Captum-based explanations | Partial | Template per Boggust et al. (2023) |

**Estimated total effort: 8-11 days.**

The P0 items (Method Card, Model Card extension, Data Card) constitute the core
deliverable. They provide the three things every Nature Protocols reviewer will look for:
(1) how to reproduce the pipeline (Method Card), (2) what models were used and how they
performed (Model Card), and (3) what data was used and how it was curated (Data Card).

### 5.2 Full Card Stack (for FDA SaMD Submission)

FDA SaMD submission requires the Nature Protocols stack plus regulatory-specific
documentation. The incremental additions are:

| Priority | Card Type | Regulatory Requirement | Auto-gen | Implementation Approach |
|----------|-----------|----------------------|----------|------------------------|
| **P0** | System Card | EU AI Act Art. 11 transparency | Partial | Document multi-flow architecture, inter-flow contracts |
| **P0** | PCCP (populated) | FDA December 2024 guidance | Partial | Connect `PCCPTemplate` to MLflow run history |
| **P0** | Use Case Card | EU AI Act risk classification | Manual | UML template per Hupont et al. (2024) |
| **P0** | EU AI Act Card | Annex IV machine-readable documentation | Yes | Extend existing `EUAIActChecklist` per Golpayegani et al. (2024) |
| **P1** | fAIlureNotes | FDA clinical evaluation | Partial | Failure taxonomy + automated collection from eval errors |
| **P1** | Audit Card | Third-party evaluation context | Manual | Template per Staufer et al. (2025) |
| **P1** | ALTAI Report | EU Trustworthy AI self-assessment | Yes | Generate from existing compliance data |
| **P1** | FeedbackLogs | FDA post-market surveillance | Partial | Template per Barker et al. (2023) |
| **P2** | TechOps Templates | EU AI Act compliance | Yes | Adopt open-source templates from Lucaj et al. (2025) |
| **P2** | 510(k) Summary | FDA predicate comparison | Manual | Already implemented in `complops.py` |

**Estimated incremental effort beyond Nature Protocols stack: 10-14 days.**
**Total estimated effort for full FDA stack: 18-25 days.**

### 5.3 Implementation Priority Matrix

The following matrix cross-references card types against two dimensions: (1) scientific
value (reproducibility, transparency) and (2) regulatory value (FDA, EU AI Act, IEC 62304).

```
                        Regulatory Value
                    Low         Medium        High
                +----------+-----------+-----------+
Scientific  H   | AI Usage | Carbon    | Data Card |
Value       i   | Card     | Card      | Method    |
            g   |          | Saliency  | Card      |
            h   |          | Card      |           |
                +----------+-----------+-----------+
            M   | Value    | Team Card | System    |
            e   | Card     | Feedback  | Card      |
            d   |          | Logs      | PCCP      |
                +----------+-----------+-----------+
            L   | Consumer | ABOUT ML  | Audit     |
            o   | Labels   |           | Card      |
            w   |          |           | Use Case  |
                +----------+-----------+-----------+
```

**Quadrant priorities:**
- **High Scientific + High Regulatory** (top-right): implement first (Data Card, Method Card)
- **High Scientific + Low Regulatory** (top-left): implement for Nature Protocols (AI Usage Card)
- **Low Scientific + High Regulatory** (bottom-right): implement for FDA (Audit Card, Use Case Card)
- **Low Scientific + Low Regulatory** (bottom-left): defer or skip (Consumer Labels, Value Cards)

---

## 6. Bibliography

### Foundational Card Frameworks

1. [Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). "Model Cards for Model Reporting." *Proc. FAT* 2019*, ACM.](https://dl.acm.org/doi/10.1145/3287560.3287596)
2. [Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92.](https://dl.acm.org/doi/10.1145/3458723)
3. [Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI." *Proc. ACM FAccT 2022*.](https://dl.acm.org/doi/10.1145/3531146.3533231)
4. [Adkins, D., Alsallakh, B., Cheema, A., et al. (2022). "Method Cards for Prescriptive Machine-Learning Transparency." *Proc. CAIN 2022*.](https://dl.acm.org/doi/10.1145/3522664.3528600)

### Data Documentation

5. [Holland, S., Hosny, A., Newman, S., et al. (2018). "The Dataset Nutrition Label." arXiv:1805.03677.](https://datanutrition.org/)
6. [Chmielinski, K.S., Newman, S., Taylor, M., et al. (2022). "The Dataset Nutrition Label (2nd Gen)." arXiv:2201.03954.](https://datanutrition.org/)
7. [Bender, E.M. & Friedman, B. (2018). "Data Statements for NLP." *TACL*, 6, 587-604.](https://aclanthology.org/Q18-1041/)
8. [McMillan-Major, A., Auli, M., Barrault, L., et al. (2021). "Reusable Templates for NLP Documentation." *Proc. GEM Workshop at ACL 2021*.](https://aclanthology.org/2021.gem-1.11/)
9. [Hutchinson, B., Smart, A., Hecht, B., et al. (2021). "Towards Accountability for ML Datasets." *Proc. ACM FAccT 2021*.](https://dl.acm.org/doi/pdf/10.1145/3442188.3445918)
10. [Diaz, M., Kivlichan, I.D., Rosen, R., et al. (2022). "CrowdWorkSheets." *Proc. ACM FAccT 2022*.](https://dl.acm.org/doi/abs/10.1145/3531146.3534647)
11. [MLCommons (2024). "Croissant: A Metadata Format for ML-Ready Datasets." *Proc. DEEM Workshop at SIGMOD 2024*.](https://docs.mlcommons.org/croissant/)

### System and Service Documentation

12. [Alsallakh, B., Cheema, A., Procope, C., et al. (2022). "System-Level Transparency of Machine Learning." Meta Technical Report.](https://ai.meta.com/research/publications/system-level-transparency-of-machine-learning/)
13. [Arnold, M., Bellamy, R.K.E., Hind, M., et al. (2019). "FactSheets: Increasing Trust in AI Services." *IBM JRD*, 63(4/5).](https://ieeexplore.ieee.org/document/8843893)
14. [Gilbert, T.K., Lambert, N., Dean, S., et al. (2022). "Reward Reports for Reinforcement Learning." *AAAI/ACM AIES 2023*.](https://dl.acm.org/doi/10.1145/3600211.3604698)
15. [Goel, K., Rajani, N.F., Vig, J., et al. (2021). "Robustness Gym." *Proc. NAACL-HLT 2021 Demonstrations*.](https://aclanthology.org/2021.naacl-demos.6/)
16. [Raji, I.D. & Yang, J. (2019). "ABOUT ML." *NeurIPS 2019 Workshop on Human-Centric ML*.](https://partnershiponai.org/workstream/about-ml/)

### Use Case, Risk, and Explainability

17. [Hupont, I., Fernandez-Llorca, D., Baldassarri, S., & Gomez, E. (2024). "Use case cards." *Ethics and Information Technology*, 26(2).](https://link.springer.com/article/10.1007/s10676-024-09757-7)
18. [Golpayegani, D., Hupont, I., Panigutti, C., et al. (2024). "AI Cards: Machine-Readable AI Documentation." *Proc. APF 2024*, Springer LNCS.](https://arxiv.org/abs/2406.18211)
19. [Wahle, J.P., Ruas, T., Mohammad, S.M., et al. (2023). "AI Usage Cards." *Proc. ACM/IEEE JCDL 2023*.](https://dl.acm.org/doi/10.1109/JCDL57899.2023.00060)
20. [Moore, S., Liao, Q.V., & Subramonyam, H. (2023). "fAIlureNotes." *Proc. CHI 2023*, ACM.](https://dl.acm.org/doi/10.1145/3544548.3581242)
21. [Barker, M., Kallina, E., Ashok, D., et al. (2023). "FeedbackLogs." arXiv:2307.15475.](https://arxiv.org/abs/2307.15475)
22. [Boggust, A., Suresh, H., Strobelt, H., et al. (2023). "Saliency Cards." *Proc. ACM FAccT 2023*.](https://vis.csail.mit.edu/pubs/saliency-cards/)
23. [Staufer, L., Yang, M., Reuel, A., & Casper, S. (2025). "Audit Cards." arXiv:2504.13839.](https://arxiv.org/abs/2504.13839)
24. [Shen, H., Deng, W.H., Chattopadhyay, S., et al. (2021). "Value Cards." *Proc. ACM FAccT 2021*.](https://dl.acm.org/doi/abs/10.1145/3442188.3445971)
25. [Seifert, C., Scherzinger, S., & Wiese, L. (2019). "Consumer Labels for ML Models." *Proc. IEEE CogMI 2019*.](https://ieeexplore.ieee.org/document/8998974)

### Sustainability and ESG

26. [Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). "Quantifying the Carbon Emissions of Machine Learning." arXiv:1910.09700.](https://arxiv.org/abs/1910.09700)
27. [Strubell, E., Ganesh, A., & McCallum, A. (2019). "Energy and Policy Considerations for Deep Learning in NLP." *Proc. ACL 2019*.](https://aclanthology.org/P19-1355/)
28. [CodeCarbon contributors (2020-present). "CodeCarbon: Track emissions from Compute."](https://codecarbon.io/)
29. [Saetra, H.S. (2023). "The AI ESG Protocol." *Sustainable Development*, 31(2), 1027-1037.](https://onlinelibrary.wiley.com/doi/10.1002/sd.2438)

### Regulatory and Compliance

30. [European Parliament and Council (2024). "Regulation (EU) 2024/1689 (Artificial Intelligence Act)." Article 11.](https://artificialintelligenceact.eu/article/11/)
31. [European Parliament and Council (2024). "Regulation (EU) 2024/1689 (Artificial Intelligence Act)." Annex IV.](https://artificialintelligenceact.eu/annex/4/)
32. [EU HLEG on AI (2020). "Assessment List for Trustworthy Artificial Intelligence (ALTAI)."](https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment)
33. [NIST (2023). "AI Risk Management Framework (AI RMF 1.0)." NIST AI 100-1.](https://www.nist.gov/itl/ai-risk-management-framework)
34. [Lucaj, L., Loosley, A., Jonsson, A., et al. (2025). "TechOps: Templates for the AI Act." *Proc. AAAI/ACM AIES 2025*.](https://ojs.aaai.org/index.php/AIES/article/view/36663)
35. [U.S. FDA (2024). "Marketing Submission Recommendations for a Predetermined Change Control Plan." Final Guidance.](https://www.fda.gov/media/166704/download)
36. [IMDRF SaMD Working Group (2025). "Characterization Considerations for Medical Device Software." IMDRF/SaMD WG/N81 FINAL:2025.](https://www.imdrf.org/)

### Clinical AI Reporting

37. [Collins, G.S., Moons, K.G.M., Dhiman, P., et al. (2024). "TRIPOD+AI statement." *BMJ*, 385.](https://pubmed.ncbi.nlm.nih.gov/38626948/)
38. [Modise, L.M., Alborzi Avanaki, M., Ameen, S., et al. (2025). "Introducing the Team Card." *PLOS Digital Health*, 4(3), e0000495.](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000495)

### Tools and Registries

39. [Ozoani, E., Gerchick, M., & Mitchell, M. (2022). "Model Card Guidebook." Hugging Face.](https://huggingface.co/docs/hub/en/model-card-guidebook)
40. [TensorFlow Team (2020). "Model Card Toolkit." TensorFlow Responsible AI.](https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide)
41. [OWASP CycloneDX (2023). "Machine Learning Bill of Materials (AI/ML-BOM)." ECMA-424.](https://cyclonedx.org/capabilities/mlbom/)
42. [Microsoft (2019-present). "Transparency Notes for Azure AI Services."](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/)
43. [BigScience Workshop (2022). "BigScience BLOOM Responsible AI License (RAIL) 1.0."](https://huggingface.co/blog/open_rail)

### Meta-Analyses and Surveys

44. [Liang, W., Rajani, N., Yang, X., et al. (2024). "Systematic analysis of 32,111 AI model cards." *Nature Machine Intelligence*, 6, 744-753.](https://www.nature.com/articles/s42256-024-00857-z)
45. [Ozoani, E., Gerchick, M., & Mitchell, M. (2022). "The Landscape of ML Documentation Tools." Hugging Face.](https://huggingface.co/docs/hub/en/model-card-landscape-analysis)
46. [Bommasani, R., Klyman, K., Longpre, S., et al. (2023). "The Foundation Model Transparency Index." arXiv:2310.12941.](https://crfm.stanford.edu/fmti/)
47. [(2025). "From Reflection to Repair: A Scoping Review of Dataset Documentation Tools." arXiv:2602.15968.](https://arxiv.org/abs/2602.15968)
48. [Liu, J., Li, W., Jin, Z., & Diab, M. (2024). "Automatic Generation of Model and Data Cards." *Proc. NAACL 2024*.](https://aclanthology.org/2024.naacl-long.110/)
49. [(2025). "AI Transparency Atlas." arXiv:2512.12443.](https://arxiv.org/abs/2512.12443)

### Emerging

50. [OpenAI (2024). "GPT-4o System Card." arXiv:2410.21276.](https://openai.com/index/gpt-4o-system-card/)
51. [(2024). "AI product cards." *Data & Policy*, Cambridge University Press.](https://www.cambridge.org/core/journals/data-and-policy/article/ai-product-cards/07A9808C3495FD34B7A386507763E6F7)
