# Phase 17: Data Engineering — Quality, ETL, and Interoperability for Vascular Imaging MLOps

> **Date**: 2026-02-24
> **PRD Version**: 1.6.0 → 1.7.0
> **Seed Papers**: 13 (data engineering cluster from vascular-tmp)
> **Web Research**: 7 topic searches (FHIR imaging, harmonization, KGs, multiplexed imaging, FL, LLM-assisted data prep, data quality)

---

## Executive Summary

This report synthesises 13 seed papers and post-January 2025 web research on data engineering for biomedical image segmentation MLOps. The analysis reveals **five actionable angles** for advancing the MinIVess vascular segmentation pipeline, each assessed for novelty against existing literature. The central finding is that **vascular imaging occupies a critical gap** in the data harmonization landscape — while brain MRI harmonization has mature tooling (ComBat, CycleGAN variants, traveling-phantom protocols), no dedicated harmonization methods exist for vascular MRI/MRA modalities. This gap, combined with emerging FHIR-to-knowledge-graph pipelines and multiplexed tissue imaging workflows, creates opportunities for novel integration at the intersection of clinical data interoperability and segmentation-aware data quality.

---

## 1. Seed Paper Synthesis

### 1.1 Clinical Data Interoperability Stack (Marfoglia Group)

Four papers from the Bologna/Paris group form a **vertically integrated clinical data pipeline**:

1. **Carbonaro et al. (2025)** — FHIR-based ETL for oncology. Converted 36,335 patients / 1,093,705 records into 1,151,559 RDF resources across 16 clinical entity types (mapped to ~8 distinct FHIR resource types) with perfect one-to-one mapping across all stages. Four-ontology integration (SNOMED CT, LOINC, NCIT, ATC). Quality assessed via Kahn DQA framework. Published in *Computers in Biology and Medicine*, 197, 111051.

2. **Marfoglia et al. (2025)** — FHIR R4 pipeline for rehabilitation data (MOTU dataset: 1,962 stays, 1,006 trans-femoral amputation patients). Achieved 98.7% field coverage, 91.1% direct mapping rate. FHIR Bundle transactions reported up to 50x faster than individual operations (cited from external literature, not empirically measured in this study). Processing time: 8 min 52 sec for full dataset. Published in *Computers in Biology and Medicine*, 187, 109745.

3. **Marfoglia et al. (2026a)** — MEDS-OWL extends the Medical Event Data Standard with OWL/RDF/SPARQL. 13 classes, 10 object properties, 20 data properties, and 24 OWL axioms. Demo on neurovasc (10,000 patients → 245,015 events → 1,330,351 triples) and MIMIC-IV (100 patients → 6,408,529 triples). Python library: `meds2rdf`. arXiv:2601.04164v2.

4. **Marfoglia et al. (2026b)** — CONNECTED framework for patient digital twins. Core: MIMO ontology + Ktor microservices (Kafka used in Source Layer for event streaming). 26.71 req/s at 20 concurrent threads. Manifest protocol enables AI models to declare input/output requirements as ontological specifications. Published in *Future Generation Computer Systems*, 180, 108380.

**Vascular relevance**: Trans-femoral amputation (MOTU) is directly caused by peripheral vascular disease. The neurovasc MEDS-OWL demo is explicitly vascular. The vertical stack (FHIR → MEDS-OWL → CONNECTED) provides a blueprint for coupling vascular imaging segmentation outputs with longitudinal patient records.

### 1.2 Multiplexed Tissue Imaging Workflows

Three papers define the state-of-the-art for high-content tissue imaging data engineering:

5. **Windhager et al. (2023)** — Nature Protocols end-to-end IMC workflow. steinbock pipeline (ilastik + DeepCell/Mesmer for segmentation), imcRtools (spatial statistics), cytomapper (visualisation). Docker containerised. 5-6 hours to complete. Applicable to IMC, CODEX, MIBI-TOF, mIF. Published in *Nature Protocols*, 18(11), 3565-3613.

6. **Eling et al. (2023)** — Companion Quarto/R workflow book implementing Windhager protocol. Batch correction via fastMNN, Harmony, Seurat. Cell phenotyping via Rphenograph, SNN, SOM. Full Docker reproducibility.

7. **Madler et al. (2025)** — scPortrait: scalable single-cell imaging toolkit. Novel `.h5sc` HDF5 format for out-of-core processing. Applied in a genome-scale autophagy screen (Schutze et al.) processing 120 million images from 40 million cells at >700 cells/sec. ViT-MAE self-supervised fine-tuning showed <5% neighbourhood overlap between image and transcriptome embeddings, quantifying morphological non-redundancy. bioRxiv 2025.09.22.677590.

**Novel connection**: The consensus-ML approach from SPACEMAP (Dawod et al., 2025) — where 3-of-4 method agreement generates training labels achieving 88% accuracy on independent validation vs. 69-81% for individual methods — could be adapted for **vascular cell phenotyping consensus** in atherosclerotic plaque imaging, where smooth muscle, macrophage, and endothelial classification is notoriously inconsistent across tools.

### 1.3 Phenotyping Discordance Resolution

8. **Dawod et al. (2025)** — SPACEMAP for multiplexed tissue imaging. Benchmarked 4 phenotyping methods (Leiden, SOM, SCIMAP, RESOLVE) on 2.6 million cells from INNATE trial CODEX biopsies. Pairwise agreement: 57.8% (SOM-SCIMAP) to 90.7% (SCIMAP-RESOLVE). Consensus-ML: 88% accuracy on Schürch CRC dataset, Cohen's Kappa +15-25% over individual methods (25% over SOM/RESOLVE, 23% over Leiden, 15% over SCIMAP). bioRxiv 2025.11.27.690631.

### 1.4 Data Foundations

9. **Cheng et al. (2024)** — General primer for data harmonization. Three-dimension model (syntax, structure, semantics). Retrospective vs. prospective archetypes. Harmonization-information tradeoff: "the level of granularity determines the amount of information lost" (Torres-Espin & Ferguson, 2022). Published in *Scientific Data*, 11, 152.

10. **Zhou et al. (2025)** — DATA x LLM survey. Bidirectional taxonomy: DATA4LLM (acquisition, deduplication, filtering, selection, mixing, synthesis) and LLM4DATA (cleaning, integration, discovery, analysis). RAG accuracy degradation: up to 12% at 100,000-page scale. Optimal synthetic data ratio: 20%. arXiv:2505.18458v3.

11. **Lin et al. (2025)** — LEAD: inference-free data selection for LLM instruction tuning. IDU utility function (current loss + temporal correction + exponential smoothing). 2.5% data outperforms full-dataset training. 5-10x training time reduction. Published in *VLDB*, Vol. 14.

12. **Ciupek et al. (2025)** — Systematic review of federated learning for medical imaging. 2,765 papers screened → 174 included. Eight-challenge framework. Real-world clinical deployment at ~5.2%. arXiv:2503.20107v2.

13. **Wasi & Ridoy (2025)** — "Messy Clinic" framework for clinical AI data. LLM harmonization at 78-92% precision. REPAR for temporal EHR modelling, MARIA for multimodal fusion. NeurIPS 2025 Workshop.

---

## 2. Web Research: Post-January 2025 Literature

### 2.1 FHIR for Medical Imaging

**University Hospital Erlangen (2025)** — Production DICOM-to-FHIR pipeline processing 150,000+ studies/year via Kafka streaming. Published in *Applied Clinical Informatics* (PMC12133321). **Critical gap identified**: FHIR `ImagingStudy` resource lacks fields for acquisition parameters (echo time, flip angle, contrast phase) essential for vascular MRA/CTA stratification. FHIR v6.0.0 ballot adds `bodyStructure` references but vascular-specific extensions remain community-developed.

### 2.2 Harmonization for Multi-Site Medical Imaging

**Comprehensive MRI Harmonization Survey (2025)** — arXiv:2507.16962v2. Taxonomises into acquisition-level (Pulseq, gammaSTAR), image-level (CycleGAN, IGUANe, PhyCHarm, BlindHarmony), and feature-level (ComBat-GAM, ComBatLS, DeepResBat). **Explicit gap statement**: "Existing efforts have primarily focused on quantitative MRI, diffusion MRI, and functional MRI" with "limited generalizability and cross-modality transferability." **No dedicated harmonization methods exist for vascular MRI.**

**ON-Harmony Dataset (2025)** — Traveling-phantom resource: 20 participants x 6 scanners x 5 sites. Published in *Scientific Data* (Nature). Gold-standard benchmark for validating harmonization algorithms.

### 2.3 Knowledge Graphs for Clinical Data

**MEDS Ecosystem (2025)** — Matured from ICLR 2024 workshop to production with MEDS-Tab (XGBoost AutoML), MEDS-Torch (deep learning), MIMIC-IV-MEDS reference implementation. The minimal 4-column schema (subject_id, time, code, numeric_value) maximises interoperability.

**DR.KNOWS (2025)** — Retrieves case-specific knowledge paths from medical KG and injects into LLM context for improved diagnostic prediction. Published in *JMIR AI* (PubMed 39993309).

### 2.4 Multiplexed Tissue Imaging

**IMMUcan Consortium (2025)** — 5-year workflow: ~10,000 samples, 2,500 patients, 5 cancer types. 350,000 manually labelled cells across 180 samples. Containerised Nextflow pipelines. Published in *Cell Reports Methods*.

**TRACERx-PHLEX (2024)** — Three Nextflow modules (deep-imcyto, TYPEx, Spatial-PHLEX). Each step is an independent containerised process for modular pipeline construction. Published in *Nature Communications*.

### 2.5 Federated Learning — Production Gap

**"From Data to Value" (MICCAI 2025)** — The "last-mile" gap paper. Current FL evaluation uses technical metrics (Dice, AUC) that don't map to value-based healthcare priorities. Recommends: clinical utility metrics, computational cost at inference, calibration under shift, auditability, regulatory documentation.

**Resource-Efficient FL (2025)** — PCA-based compression reduces energy consumption by 98% vs. standard FedAvg while maintaining accuracy. Published in *Medical Physics* (PMC12409104).

**FedNCA (MICCAI 2025)** — Neural Cellular Automata drastically reduce parameter count, enabling FL on low-cost edge devices in community hospitals.

### 2.6 LLM-Assisted Data Preparation

**Report-to-Label Pipeline (2025)** — Mixtral-8x7B-Instruct classifies radiology reports at 92% accuracy; CNN trained on LLM-generated labels reaches 89.5% accuracy on 15,896 images. Published in *Academic Radiology*. Directly applicable to bootstrapping vascular pathology labels from radiology report text.

**Weakly Supervised LLMs for Radiology (2025)** — Two-phase fine-tuning on 15,000 unlabeled Mayo Clinic reports using Mistral-class models. Published in *npj Digital Medicine*.

### 2.7 Data Quality Frameworks

**METRIC Framework (2024)** — Systematic review of 4,633 papers → 120 included → 15 awareness dimensions for medical training datasets (completeness, correctness, consistency, currency, uniqueness, validity, representativeness, balance, relevance, accessibility, security, privacy, traceability, interpretability, fairness). Published in *npj Digital Medicine* (PMC11297942).

**FDA Transparency Audit (2025)** — 93.3% of FDA-cleared AI/ML devices do not report training data source; 75.5% do not report test data source. Published in *npj Digital Medicine*.

**Clinical Data Lakehouse (2025)** — Head-to-head comparison: lakehouse (Delta Lake/Iceberg + medallion architecture) wins on FAIR compliance over data warehouse and data lake. Published in *JMIR* (PMC12357119).

---

## 3. Five Actionable Angles for MinIVess

### Angle 1: Vascular MRI Harmonization Gap — **NOVELTY: HIGH**

**The gap**: No dedicated harmonization methods exist for vascular MRI modalities (time-of-flight MRA, black-blood MRI, contrast-enhanced MRA). The comprehensive 2025 survey (arXiv:2507.16962v2) explicitly identifies this as a blind spot. Brain MRI harmonization has ComBat-GAM, CycleGAN, PhyCHarm, traveling-phantom protocols — vascular imaging has none of these.

**Why novel**: Cross-pollinating brain MRI harmonization tooling to vascular MRA requires domain-specific adaptations: (1) flow-dependent signal in TOF-MRA has no brain parenchyma analogue; (2) contrast timing in CE-MRA creates phase-dependent intensity distributions; (3) vessel calibre (0.5-25mm) spans a different scale than brain structures. A "VasculHarm" harmonization benchmark — analogous to ON-Harmony but with multi-site vascular MRA data — would be first-of-its-kind.

**Testable hypothesis**: ComBat-GAM applied naively to vascular MRA features reduces downstream segmentation Dice by >5% compared to a domain-adapted variant that accounts for flow-dependent signal physics.

**PRD integration**: New `data_harmonization_method` decision node.

### Angle 2: FHIR-Imaging-Segmentation Bridge — **NOVELTY: MEDIUM-HIGH**

**The gap**: The Erlangen DICOM-to-FHIR production pipeline (150,000+ studies/year) identified missing acquisition parameters in FHIR `ImagingStudy`. The Marfoglia FHIR → MEDS-OWL → CONNECTED vertical stack exists for tabular clinical data but **no implementation connects segmentation outputs to the FHIR interoperability layer**.

**Why novel**: Lifting vascular segmentation outputs (vessel diameter, plaque volume, stenosis grade) to FHIR `Observation` or `DiagnosticReport` resources — and linking them to MEDS-OWL knowledge graphs via the CONNECTED Manifest protocol — creates a novel feedback loop where downstream segmentation quality metrics propagate back to the clinical data layer. This "segmentation-aware interoperability" pattern has no published implementation.

**Testable hypothesis**: Encoding segmentation-derived vasculometric measurements as FHIR Observations with LOINC codes enables cross-institutional SPARQL queries that identify >3x more eligible patients for vascular imaging studies than text-based report search.

**PRD integration**: New `clinical_data_format` decision node. Connects to existing `annotation_platform`, `clinical_contract_schema`.

### Angle 3: Consensus-ML for Vascular Cell Phenotyping — **NOVELTY: MEDIUM**

**The concept**: SPACEMAP's consensus-ML approach (3-of-4 method agreement → training labels → supervised classifier) achieved 88% accuracy on independent validation vs. 69-81% for individual methods. Applying this to vascular tissue multiplexed imaging — where smooth muscle cell, macrophage, endothelial, and T cell classification is inconsistent across tools — would address a known reproducibility problem in atherosclerosis research.

**Why moderate novelty**: The consensus-ML pattern itself is published (Dawod et al., 2025), but its application to vascular tissue phenotyping is new. The scPortrait `.h5sc` format (Madler et al., 2025) provides the scalable data backend for storing the multi-method classification results.

**Testable hypothesis**: Consensus-ML applied to atherosclerotic plaque CODEX imaging improves macrophage subtype classification F1 by >15% over any single phenotyping method.

**PRD integration**: Strengthens existing `label_quality` decision node. Adds reference to SPACEMAP.

### Angle 4: Lakehouse Architecture for Vascular Imaging MLOps — **NOVELTY: MEDIUM**

**The concept**: The clinical data lakehouse comparison (JMIR 2025) demonstrates that Delta Lake/Iceberg medallion architecture (Bronze → Silver → Gold) wins on FAIR compliance for multimodal biomedical data. Mapping this to vascular imaging MLOps:
- **Bronze**: Raw DICOM + FHIR resources (immutable ingestion layer)
- **Silver**: Harmonised images (ComBat/CycleGAN-corrected) + MEDS-format EHR
- **Gold**: Feature stores for training (precomputed segmentation features, radiomic features, clinical covariates)

**Why moderate novelty**: Lakehouse architecture for healthcare AI is published, but the specific three-tier design for imaging MLOps with harmonization as the Bronze-to-Silver transformation and DVC-tracked dataset versions as the change-data-capture mechanism is not.

**Testable hypothesis**: Lakehouse medallion architecture with DVC integration reduces data preparation time for new model training by >40% compared to ad-hoc file-based pipelines.

**PRD integration**: New `data_storage_architecture` decision node. Connects to existing `containerization`, `data_versioning`.

### Angle 5: METRIC-Driven Data Quality Gates — **NOVELTY: MEDIUM-LOW**

**The concept**: The METRIC framework's 15 dimensions (Seyferth et al., 2024) provide a concrete checklist for pre-training dataset audits. Combined with the FDA transparency finding (93.3% of devices don't report data source), implementing automated METRIC quality gates in the training pipeline creates a regulatory compliance advantage.

**Why lower novelty**: The METRIC framework exists; the application is engineering rather than research. However, implementing it as a Pydantic-schema-driven quality gate in a MONAI training pipeline — where violations block training — is a concrete, publishable contribution.

**Testable hypothesis**: Automated METRIC quality gates catch >80% of dataset issues that would be identified in a manual audit at <5% of the time cost.

**PRD integration**: Strengthens existing `data_validation_tools` decision node. Adds METRIC reference.

---

## 4. PRD v1.7.0 Integration Recommendations

### 4.1 New Decision Nodes (3)

1. **`data_harmonization_method`** (L3-technology)
   - Options: combat_family (0.35), cyclegan_variants (0.25), physics_informed (0.20), none (0.20)
   - Conditional on: `data_versioning` (moderate), `containerization` (weak)
   - References: arXiv:2507.16962v2 harmonization survey, ON-Harmony dataset, Cheng et al. (2024)

2. **`clinical_data_format`** (L3-technology)
   - Options: fhir_meds_owl (0.40), fhir_only (0.25), meds_only (0.20), custom_csv (0.15)
   - Conditional on: `annotation_platform` (moderate), `clinical_contract_schema` (strong)
   - References: Marfoglia et al. (2025, 2026a, 2026b), Carbonaro et al. (2025), Erlangen pipeline (2025)

3. **`data_storage_architecture`** (L4-infrastructure)
   - Options: lakehouse_medallion (0.40), data_lake (0.25), feature_store (0.20), file_based (0.15)
   - Conditional on: `containerization` (strong), `data_versioning` (strong)
   - References: JMIR lakehouse comparison (2025), MEDS ecosystem

### 4.2 New Edges (8)

1. `data_harmonization_method` → `segmentation_models` (strong)
2. `data_harmonization_method` → `data_versioning` (moderate)
3. `clinical_data_format` → `annotation_platform` (moderate)
4. `clinical_data_format` → `clinical_contract_schema` (strong)
5. `data_storage_architecture` → `containerization` (strong)
6. `data_storage_architecture` → `data_versioning` (strong)
7. `data_storage_architecture` → `experiment_tracking` (moderate)
8. `label_quality` → `data_harmonization_method` (moderate)

### 4.3 Updated Existing Nodes

- **`label_quality`**: Add SPACEMAP consensus-ML reference (Dawod et al., 2025). Add METRIC framework reference (Seyferth et al., 2024). Increase `custom_iaa` prior from 0.30 to 0.35 based on evidence that multi-method consensus outperforms single tools.
- **`data_validation_tools`**: Add METRIC 15-dimension reference. Add FDA transparency audit finding.

### 4.4 New Bibliography Entries (15)

| citation_key | inline_citation | venue |
|---|---|---|
| carbonaro2025fhir | Carbonaro et al. (2025) | Comput Biol Med 197 |
| marfoglia2025fhir | Marfoglia et al. (2025) | Comput Biol Med 187 |
| marfoglia2026medsowl | Marfoglia et al. (2026) | arXiv:2601.04164 |
| marfoglia2026connected | Marfoglia et al. (2026) | Future Gen Comput Syst 180 |
| windhager2023imc | Windhager et al. (2023) | Nat Protoc 18(11) |
| madler2025scportrait | Madler et al. (2025) | bioRxiv 2025.09.22 |
| dawod2025spacemap | Dawod et al. (2025) | bioRxiv 2025.11.27 |
| cheng2024harmonization | Cheng et al. (2024) | Sci Data 11, 152 |
| zhou2025datallm | Zhou et al. (2025) | arXiv:2505.18458 |
| lin2025lead | Lin et al. (2025) | VLDB Vol 14 |
| ciupek2025fl | Ciupek et al. (2025) | arXiv:2503.20107 |
| wasi2025messyclinic | Wasi & Ridoy (2025) | NeurIPS 2025 WS |
| seyferth2024metric | Seyferth et al. (2024) | npj Digit Med |
| erlangen2025fhirimaging | Univ Hosp Erlangen (2025) | Appl Clin Inform |
| jmir2025lakehouse | JMIR (2025) | JMIR Med Inform |

---

## 5. Key References (Verified)

1. Carbonaro, A. et al. (2025). An ontology-based data engineering pipeline for FHIR-compliant oncological clinical data. *Computers in Biology and Medicine*, 197, 111051.
2. Marfoglia, A. et al. (2025). Towards FHIR-based clinical data interoperability: an ontological mapping approach. *Computers in Biology and Medicine*, 187, 109745.
3. Marfoglia, A., Jhee, J.H. & Coulet, A. (2026). MEDS-OWL: A Semantic Web Extension of the Medical Event Data Standard. arXiv:2601.04164v2.
4. Marfoglia, A. et al. (2026). Knowledge graph-based patient digital twins: The CONNECTED framework. *Future Generation Computer Systems*, 180, 108380.
5. Windhager, J. et al. (2023). An end-to-end workflow for multiplexed image processing and analysis. *Nature Protocols*, 18(11), 3565-3613.
6. Eling, N. et al. (2023). IMCDataAnalysis workflow. bodenmillergroup.github.io/IMCDataAnalysis.
7. Madler, S.C. et al. (2025). scPortrait: A scalable toolkit for processing, analyzing, and modeling single-cell imaging data. bioRxiv 2025.09.22.677590.
8. Dawod, A. et al. (2025). SPACEMAP: Resolving cell phenotyping discordance in multiplex tissue imaging. bioRxiv 2025.11.27.690631.
9. Cheng, C. et al. (2024). A General Primer for Data Harmonization. *Scientific Data*, 11, 152.
10. Zhou, X. et al. (2025). Data x LLM: From Principles to Practices. arXiv:2505.18458v3.
11. Lin, J. et al. (2025). LEAD: LEArning to Iteratively Select Data for LLM Instruction Tuning. *VLDB*, Vol. 14.
12. Ciupek, M. et al. (2025). Federated Learning: A new frontier in medical imaging data. arXiv:2503.20107v2.
13. Wasi, M.A. & Ridoy, M.A.R. (2025). Building Clinical AI: Challenges in Multimodal Corpora. NeurIPS 2025 Workshop.
14. Seyferth, S. et al. (2024). The METRIC-Framework for Assessing Data Quality for Trustworthy AI in Medicine. *npj Digital Medicine*. PMC11297942.
15. University Hospital Erlangen (2025). Large-Scale Integration of DICOM Metadata into HL7-FHIR. *Applied Clinical Informatics*. PMC12133321.
16. JMIR (2025). Enhancing Clinical Data Infrastructure: Comparative Evaluation. *JMIR Medical Informatics*. PMC12357119.
17. IMMUcan Consortium (2025). Multi-Modal Image Analysis for Large-Scale Cancer Tissue Studies. *Cell Reports Methods*.
18. ON-Harmony (2025). A Multi-Site Travelling-Heads Resource for Brain MRI Harmonisation. *Scientific Data*.

---

## 6. Architectural Implications

The data engineering research reveals a **five-layer data architecture** for vascular imaging MLOps:

```
Layer 5: TRAINING FEATURES
  Gold tier: precomputed features, radiomic signatures, clinical covariates
  Tools: Feature stores, DVC-tracked datasets, Pandera validation

Layer 4: HARMONISED DATA
  Silver tier: scanner-corrected images, MEDS-format EHR, unified annotations
  Tools: ComBat-adapted/CycleGAN, MEDS-OWL, SPACEMAP consensus-ML

Layer 3: STANDARDISED INTERCHANGE
  FHIR resources: ImagingStudy, Observation, DiagnosticReport
  DICOM-to-FHIR conversion: Kafka streaming + Matchbox FML engine
  Knowledge graph: SPARQL endpoint for cohort discovery

Layer 2: VALIDATED INGESTION
  Bronze tier: raw DICOM, clinical exports, annotation files
  Quality gates: METRIC 15-dimension audit, Great Expectations checks
  Tools: Pydantic v2 schemas, whylogs profiling

Layer 1: RAW SOURCES
  Hospital PACS, annotation platforms, EHR systems
  Federated: NVFlare/Flower for multi-site without data transfer
```

This architecture aligns with the existing MinIVess PRD stack (Pydantic v2, Pandera, Great Expectations, whylogs at Layer 2; DVC at Layer 4-5; MLflow at Layer 5) and extends it with interoperability (Layer 3) and harmonization (Layer 4) capabilities that are currently absent.

---

## 7. Cross-References to Existing PRD

| Existing Node | Connection | Evidence |
|---|---|---|
| `data_validation_tools` | METRIC 15-dimension audit extends Deepchecks/GE | Seyferth et al. (2024) |
| `label_quality` | SPACEMAP consensus-ML improves cleanlab | Dawod et al. (2025) |
| `annotation_platform` | FHIR bridge connects Label Studio to clinical data | Marfoglia et al. (2025) |
| `containerization` | Lakehouse + Docker Compose for Bronze/Silver/Gold | JMIR (2025) |
| `data_versioning` | DVC as change-data-capture for medallion tiers | Lin et al. (2025) adaptation |
| `clinical_contract_schema` | FHIR Observations encode TRIPOD+AI metrics | Phase 16 angle |
| `copilot_backend` | MEDS-OWL KG as copilot's structured retrieval source | Marfoglia et al. (2026a) |
| `segmentation_models` | Harmonization directly affects segmentation Dice | arXiv:2507.16962v2 |
| `experiment_tracking` | Lakehouse Gold tier feeds MLflow experiments | JMIR (2025) |
