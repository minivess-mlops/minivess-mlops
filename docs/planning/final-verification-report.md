# Final Verification Report: PRD Gap Analysis and Integration

**Date**: 2026-02-24
**PRD Version**: 2.0.0 → 2.1.0
**Scope**: Cross-validation of bibliography.yaml against Zotero RDF (636 entries) and BibTeX (934 entries), DataLad alternative assessment, end-to-end biomedical ML systems survey

---

## 1. Methodology

Four parallel research agents analysed independent bibliographic sources to identify gaps in the PRD decision network:

| Agent | Source | Entries Analysed | Relevant Gaps |
|-------|--------|-----------------|---------------|
| RDF gap analysis | `vascular-mlops.rdf` (Zotero) | 636 | 101 (score ≥ 3) |
| BIB gap analysis | `vessops.bib` (BibTeX) | 934 | 312 (MLOps-relevant) |
| DataLad investigation | Web research + codebase | — | DVC vs DataLad |
| Fishing expedition | Web search | — | TOP 25 end-to-end systems |

Cross-validation identified entries flagged by multiple agents as highest priority.

---

## 2. Critical Findings

### 2.1 Missing Foundational Entry

**The MiniVess dataset paper (Poon et al., 2023)** — the foundational data asset for the entire project — was absent from `bibliography.yaml`. This has been corrected as `poon2023minivess`.

### 2.2 Strongest Gap Clusters

| Cluster | Gap Count (TOP 30) | Key Missing Papers |
|---------|--------------------|--------------------|
| Regulatory compliance | 7 entries (BIB) | Granlund 2021, Knoblauch 2023, Mallardi 2025 |
| Drift detection/monitoring | 5 entries (cross-agent) | CheXstray/MMC+, Rabanser 2024, Zamzmi 2024 |
| Reproducibility | 5 entries (BIB) | Kapoor 2022 (landmark), Colliot 2023 |
| Data management | 4 entries (DataLad) | Halchenko 2021, Wagner 2022, Markiewicz 2021 |

### 2.3 Cross-Agent Validated Entries (Highest Confidence)

Papers flagged by **2+ agents** independently:

1. **Granlund et al. (2021)** — Regulatory-compliant MLOps case study (RDF #11, BIB #3)
2. **Zamzmi et al. (2024)** — SPC-based OOD detection (RDF #20, BIB #2)
3. **Poon et al. (2023)** — MiniVess dataset paper (RDF #28, critical omission)

### 2.4 Closest Published Comparators to MinIVess

| System | Venue | Overlap |
|--------|-------|---------|
| MedMLOps | European Radiology 2025 | Four-pillar framework: availability, monitoring, privacy, usability |
| FetalMLOps | Med. Biol. Eng. Comput. 2026 | Medical imaging MLOps in regulated context |
| MLXOps4Medic | Preprint 2025 | Medical imaging MLOps + XAI operations |
| Oravizio | SN Computer Science 2021 | ML experiment → certified medical product |

### 2.5 Genuine Novelty Gaps (No Literature Found)

Five areas where MinIVess is genuinely novel with no strong published precedent:

1. **Vascular-specific MLOps pipelines** — no end-to-end system for micro-vascular segmentation
2. **Topology-aware drift detection** — no system monitors Betti numbers/connectivity as drift indicators
3. **Agent-orchestrated biomedical segmentation** — no LLM agent-orchestrated training pipeline
4. **Conformal prediction in deployed medical imaging** — no production deployment with coverage guarantees
5. **Local-first medical MLOps** — all production systems are cloud-first; `docker compose up` with zero API tokens is underrepresented

---

## 3. DataLad Assessment

### Recommendation: Add as Experimental Option (prior 0.10)

| Aspect | DataLad | DVC (current) |
|--------|---------|---------------|
| Provenance | Superior — `datalad run` with machine-readable JSON | Basic — pipeline stage deps/outs |
| ML pipelines | None built-in | Built-in stages, `dvc repro` |
| Container support | datalad-container (Docker/Singularity) | Not native |
| Ecosystem | Neuroscience (OpenNeuro, DANDI, CONP) | ML/data science |
| Storage backends | Broader (Dropbox, OSF, GIN, RIA) | Pre-implemented cloud remotes |
| Learning curve | Steeper (git-annex required) | Git-familiar |
| Regulatory fit | Stronger audit trail via commit metadata | Adequate with OpenLineage |

**Decision**: DataLad added as `datalad` option in `data_management_strategy` with prior 0.10 (experimental). DVC remains the resolved choice (0.35) due to ML pipeline management and lower migration cost (~320 lines `dvc_utils.py` already implemented). DataLad probability boosted in clinical_deployment archetype (0.15) due to provenance strengths.

**DVC Acquisition Note**: DVC was acquired by lakeFS in November 2025. Remains Apache 2.0 open source with backward compatibility guaranteed.

---

## 4. PRD Changes (v2.0.0 → v2.1.0)

### 4.1 Bibliography

**+20 entries** added under `# Verification Gap Analysis` section:

| Citation Key | Year | Primary PRD Node(s) |
|-------------|------|---------------------|
| `poon2023minivess` | 2023 | data_management_strategy, project_purpose |
| `halchenko2021datalad` | 2021 | data_management_strategy, reproducibility |
| `wagner2022fairlybig` | 2022 | data_management_strategy, containerization |
| `granlund2021regulatory` | 2021 | regulatory_compliance_approach, model_governance |
| `kapoor2022leakage` | 2022 | reproducibility_standard, data_validation |
| `feng2022chexstray` | 2022 | drift_detection_method, monitoring_stack |
| `rabanser2024drift` | 2024 | drift_detection_method, drift_response |
| `medmlops2025` | 2025 | monitoring_stack, drift_response, model_governance |
| `zamzmi2024spc` | 2024 | drift_detection_method, monitoring_stack |
| `roschewitz2026datasetshift` | 2026 | drift_detection_method, drift_response |
| `testi2026fetalmlops` | 2026 | pipeline_orchestration, regulatory_compliance |
| `huang2025mlxops4medic` | 2025 | monitoring_stack, xai_strategy, lineage |
| `liang2024modelcards` | 2024 | model_governance, documentation_standard |
| `knoblauch2023continuous` | 2023 | regulatory_compliance_approach, audit_trail |
| `mallardi2025responsible` | 2025 | regulatory_compliance_approach, compliance |
| `colliot2023reproducibility` | 2023 | reproducibility_standard, experiment_tracking |
| `kushibar2022layerensembles` | 2022 | uncertainty_quantification, ensemble_methods |
| `lewis2022augur` | 2022 | drift_detection_method, monitoring_stack |
| `barrak2021dvcevolution` | 2021 | data_management_strategy, gitops |
| `markiewicz2021openneuro` | 2021 | data_management_strategy, reproducibility |

### 4.2 Decision Node Updates

| Decision Node | Changes |
|--------------|---------|
| `data_management_strategy` (L2) | +1 option (datalad, 0.10), +7 references, updated conditional tables and archetypes |
| `drift_detection_method` (L3) | +6 references (CheXstray, Rabanser, Zamzmi, Roschewitz, Lewis, Augur) |
| `drift_response` (L5) | +5 references (CheXstray, Rabanser, Roschewitz, MedMLOps) |
| `regulatory_compliance_approach` (L5) | +4 references (Granlund, Knoblauch, Mallardi, FetalMLOps) |
| `monitoring_stack` (L5) | +3 references (MLXOps4Medic, MedMLOps, CheXstray) |
| `model_governance` (L5) | +2 references (Liang model cards, Granlund) |
| `reproducibility_standard` (L1) | +2 references (Kapoor leakage, Colliot medical imaging) |
| `uncertainty_quantification` (L2) | +1 reference (Kushibar Layer Ensembles) |

### 4.3 Network Topology

| Metric | v2.0.0 | v2.1.0 |
|--------|--------|--------|
| Nodes | 70 | 70 |
| Explicit edges | 130 | 131 |
| Bibliography entries | ~245 | ~265 |
| New edge | — | `compliance_depth → data_management_strategy` (moderate) |

---

## 5. Entries Considered but Not Added

The following high-scoring entries from agent results were **not** added to maintain signal-to-noise ratio, but are documented for future phases:

- **Uppuluri (2025)** — Foundation models in radiology (RDF #1, score 20). Broad survey, less actionable.
- **Babu et al. (2025)** — Kubernetes healthcare pipeline (RDF #4). Infrastructure-specific.
- **Ktena et al. (2024)** — Generative models for fairness (Nature Medicine, BIB #7). Tangential to core decisions.
- **Katz (2023)** — DVC scalability analysis (BIB #10). Covered by existing DVC references.
- **Sohn (2023) / Gibney (2022)** — Nature reproducibility commentaries (BIB #19-20). Kapoor 2022 is more actionable.
- **Stogiannos et al. (2023)** — AI governance frameworks in radiology (BIB #21). Covered by Schneider 2024.

---

## 6. Summary Statistics

| Metric | Before | After |
|--------|--------|-------|
| PRD version | 2.0.0 | 2.1.0 |
| Bibliography entries | ~245 | ~265 |
| Decision nodes | 70 | 70 |
| Explicit edges | 130 | 131 |
| Options in data_management_strategy | 4 | 5 |
| Decision nodes updated | — | 8 |
| Cross-agent validated entries added | — | 3 |
