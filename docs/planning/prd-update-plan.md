# PRD Update Plan — Phase 8: Paper Ingestion & Evidence Integration

**Date**: 2026-02-23
**Source**: 83 papers from `sci-llm-writer/biblio/biblio-vascular-tmp/`
**Status**: In Progress

## 1. Executive Summary

This plan captures evidence from 83 academic papers read across 8 review batches, mapping
findings to the 52 PRD decision nodes and generating prioritized implementation tasks. The papers
span: vessel segmentation architectures, foundation models (SAM/MedSAM/VISTA/vesselFM),
MLOps frameworks, regulatory compliance (IEC 62304, EU AI Act), drift detection, uncertainty
quantification, calibration, agent frameworks, reporting standards, and data quality tools.

## 2. Paper Classification Summary

### 2.1 Highly Relevant Papers (Relevance 4-5) — Direct PRD Impact

| Paper | Citation Key | Relevance | Primary Decision Nodes |
|-------|-------------|-----------|----------------------|
| vesselFM: Foundation Model for 3D Vessel Segmentation | wittmann2024vesselfm | 5 | segmentation_models, foundation_model_integration, augmentation_stack |
| COMMA: Mamba Network for 3D Vessel Segmentation | shi2025comma | 5 | segmentation_models, loss_functions |
| VessQC: Uncertainty-Guided Curation for 3D Vessels | terms2025vessqc | 5 | annotation_workflow, label_quality, uncertainty_quantification |
| VessShape: Few-Shot 2D Vessel Segmentation | galvao2025vessshape | 4 | segmentation_models, augmentation_stack, foundation_model_integration |
| AtlasSegFM: One-Shot Customization of Foundation Models | zhang2025atlassegfm | 4 | foundation_model_integration, segmentation_models |
| SAM2 Benchmark: Segment Anything in Medical Images | ma2024sam2bench | 4 | segmentation_models, foundation_model_integration |
| MedSAM2: Segment Anything in 3D Medical | ma2025medsam2 | 4 | segmentation_models, foundation_model_integration, annotation_workflow |
| Prediction-Powered Risk Monitoring | zhang2026pprm | 5 | drift_response, monitoring_stack, retraining_trigger |
| Diagnosing Harmful Data Shifts (Subasri) | subasri2023driftclinical | 5 | drift_response, monitoring_stack, data_validation_tools |
| Confidence Intervals for Medical Imaging (André) | andre2026ci | 5 | metrics_framework, uncertainty_quantification, testing_strategy |
| MLOps SLR (Nogare) | nogare2025mlopsslr | 4 | pipeline_orchestration, experiment_tracking, monitoring_stack |
| Automated Regulatory Docs for SaMD (Rosmarino) | rosmarino2025regdocs | 5 | compliance_depth, documentation_standard, documentation_generation |
| IEC 62304 DevOps (Martina) | martina2024iec62304devops | 5 | compliance_depth, ci_cd_platform, model_governance |
| EU AI Act Impact (Niemiec) | niemiec2025euaiact | 4 | compliance_depth, model_governance |
| CONSORT-AI Reporting (Kwong) | kwong2025consortai | 4 | documentation_standard, compliance_depth |
| MI-CLEAR-LLM Reporting (Park) | park2025miclearllm | 4 | documentation_standard, compliance_depth |
| RegOps Lifecycle (Lähteenmäki) | lahteenmaki2023regops | 4 | compliance_depth, ci_cd_platform, documentation_generation |
| CyclOps Toolkit (Krishnan) | krishnan2022cyclops | 4 | data_validation_tools, drift_response, monitoring_stack |
| Conformal Prediction for Medical (Shah-Mohammadi) | shahmohammadi2025conformal | 4 | uncertainty_quantification, calibration_tools |
| Topology-Aware Segmentation + UQ (Dhor/TUNE++) | dhor2026tunepp | 5 | loss_functions, uncertainty_quantification, segmentation_models |
| DATA-CARE Quality Tool (van Twist) | vantwist2026datacare | 4 | data_validation_tools, label_quality, data_profiling |
| MC Dropout for Image Quality (Bench) | bench2025mcdropout | 4 | uncertainty_quantification, model_diagnostics |
| Monitoring with Financial Risk Metrics (Chakraborty) | chakraborty2025finrisk | 4 | monitoring_stack, drift_response |
| Kim 2025 Monitoring Strategies | kim2025monitoring | 4 | monitoring_stack, drift_response |
| CC-MLOps Cell Counting (Testi) | testi2025ccmlops | 4 | pipeline_orchestration, experiment_tracking, xai_strategy |
| Calibration-Quantification Interconnections (Moreo) | moreo2025calibquant | 4 | calibration_tools, drift_response |
| nnQC: Diffusion-Based QC (Marciano) | marciano2025nnqc | 4 | model_diagnostics, data_validation_tools |
| SynthICL: Data Synthesis for ICL (Terms) | terms2025synthicl | 4 | augmentation_stack, foundation_model_integration |

### 2.2 Moderately Relevant Papers (Relevance 2-3)

| Paper | Citation Key | Relevance | Primary Decision Nodes |
|-------|-------------|-----------|----------------------|
| IBISAgent: Agentic MLLM for Segmentation | jiang2026ibisagent | 3 | agent_framework, segmentation_models |
| DeepRare: Multi-Agent Rare Disease | chen2026deeprare | 3 | agent_framework, pipeline_orchestration |
| Agentic Systems in Radiology (Blüthgen) | bluethgen2025agenticradiology | 3 | agent_framework, pipeline_orchestration |
| Agentic Systems + Regulation (Tzanis) | tzanis2026agenticregulation | 3 | agent_framework, compliance_depth |
| Agentic AI Survey (Abdollahi) | abdollahi2025agenticsurvey | 3 | agent_framework |
| DiLLS: LLM Multi-Agent Diagnostics | sheng2026dills | 3 | llm_observability, agent_framework |
| RE-MCDF: Multi-Expert LLM Clinical | shen2026remcdf | 2 | agent_framework |
| VISTA-PATH: Pathology Foundation Model | liang2026vistapath | 3 | foundation_model_integration, annotation_platform |
| MieDB-100k: Medical Image Editing | lai2026miedb | 2 | augmentation_stack |
| DaneelPath: Digital Pathology Tools | daneelpath2025 | 2 | annotation_platform |
| Beyond Benchmarks (Xie bone seg) | xie2026beyondbenchmarks | 3 | metrics_framework, testing_strategy |
| FID Critique (Wu) | wu2025fidcritique | 2 | metrics_framework |
| Gupta MONAI Deployment | gupta2025monaideploy | 3 | serving_architecture, monai_alignment |
| Oh Continual Learning | oh2025continuallearning | 3 | federated_learning, foundation_model_integration |
| MedKGI Knowledge Graphs | wang2025medkgi | 2 | agent_framework |
| Zhi Clinical Dialogue Survey | zhi2025agenticdialogue | 2 | agent_framework |
| MedSynth Synthetic Dialogues | rezaie2025medsynth | 1 | — |
| Seamless Transitions (VM migration) | attarkhorasani2025seamless | 1 | — |

### 2.3 Reporting Standards & Regulatory Papers

| Paper | Citation Key | Relevance | Primary Decision Nodes |
|-------|-------------|-----------|----------------------|
| ELEVATE-GenAI Framework | elevate2025genai | 3 | documentation_standard, compliance_depth |
| STARD Reporting | bossuyt2015stard | 3 | documentation_standard |
| TRIPOD+AI | tripodai2024 | 3 | documentation_standard |
| SPIRIT-AI/CONSORT-AI | spiritai2020 | 3 | documentation_standard |
| Five Years of CONSORT-AI (Han) | han2025fiveyears | 3 | documentation_standard |
| ACT Standards | act2015 | 2 | compliance_depth |

### 2.4 Copyright/Non-Relevant Papers (Relevance 1)

Several files were copyright notices, login pages, or unrelated content:
- `2022-copyright-b4776c.md`, `2023-copyright-be048e.md`, `2025-copyright-*.md` — Preprint copyright wrappers
- `2021-springer-b2993f.md` — Springer access page
- `2024-under-bc620f.md` — Unclear content

## 3. Bibliography Updates Required

### 3.1 New Entries for bibliography.yaml (~45 new entries)

#### Vessel Segmentation & Foundation Models
```yaml
- citation_key: wittmann2024vesselfm
  inline_citation: "Wittmann et al. (2024)"
  title: "vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation"
  year: 2024
  topics: [segmentation_models, foundation_model_integration, augmentation_stack]

- citation_key: shi2025comma
  inline_citation: "Shi et al. (2025)"
  title: "COMMA: Coordinate-aware Modulated Mamba for 3D Vessel Segmentation"
  year: 2025
  topics: [segmentation_models, loss_functions]

- citation_key: galvao2025vessshape
  inline_citation: "Galvão & Comin (2025)"
  title: "VessShape: Few-Shot 2D Blood Vessel Segmentation"
  year: 2025
  topics: [segmentation_models, augmentation_stack]

- citation_key: terms2025vessqc
  inline_citation: "Terms et al. (2025)"
  title: "VessQC: Uncertainty-Guided Curation for 3D Vessel Segmentation"
  year: 2025
  topics: [annotation_workflow, label_quality, uncertainty_quantification]

- citation_key: terms2025synthicl
  inline_citation: "Terms et al. (2025)"
  title: "SynthICL: Domain Randomization for ICL Medical Segmentation"
  year: 2025
  topics: [augmentation_stack, foundation_model_integration]

- citation_key: zhang2025atlassegfm
  inline_citation: "Zhang et al. (2025)"
  title: "Atlas is Your Perfect Context: One-Shot Customization for Foundation Models"
  year: 2025
  topics: [foundation_model_integration, segmentation_models]

- citation_key: ma2024sam2bench
  inline_citation: "Ma et al. (2024)"
  title: "Segment Anything in Medical Images and Videos: Benchmark"
  year: 2024
  topics: [segmentation_models, foundation_model_integration]

- citation_key: ma2025medsam2
  inline_citation: "Ma et al. (2025)"
  title: "MedSAM2: Segment Anything in 3D Medical Images and Videos"
  year: 2025
  topics: [segmentation_models, foundation_model_integration, annotation_workflow]

- citation_key: dhor2026tunepp
  inline_citation: "Dhor et al. (2026)"
  title: "TUNE++: Topology-Aware Uncertainty for 3D Segmentation"
  year: 2026
  topics: [loss_functions, uncertainty_quantification, segmentation_models]

- citation_key: marciano2025nnqc
  inline_citation: "Marciano et al. (2025)"
  title: "nnQC: Diffusion-Based Quality Control of Medical Image Segmentations"
  year: 2025
  topics: [model_diagnostics, data_validation_tools]

- citation_key: palaniappan2026vesselpose
  inline_citation: "Palaniappan et al. (2026)"
  title: "VesselPose: Topology-Aware 3D Vessel Segmentation"
  year: 2026
  topics: [segmentation_models, loss_functions]
```

#### Drift Detection & Monitoring
```yaml
- citation_key: zhang2026pprm
  inline_citation: "Zhang et al. (2026)"
  title: "Prediction-Powered Risk Monitoring of Deployed Models"
  year: 2026
  topics: [drift_response, monitoring_stack, retraining_trigger]

- citation_key: subasri2023driftclinical
  inline_citation: "Subasri et al. (2023)"
  title: "Diagnosing and Remediating Harmful Data Shifts for Clinical AI"
  year: 2023
  topics: [drift_response, monitoring_stack, data_validation_tools]

- citation_key: kim2025monitoring
  inline_citation: "Kim et al. (2025)"
  title: "Monitoring Strategies for Clinical AI Systems"
  year: 2025
  topics: [monitoring_stack, drift_response]

- citation_key: chakraborty2025finrisk
  inline_citation: "Chakraborty et al. (2025)"
  title: "Financial Risk Metrics for AI Monitoring"
  year: 2025
  topics: [monitoring_stack, drift_response]

- citation_key: krishnan2022cyclops
  inline_citation: "Krishnan et al. (2022)"
  title: "CyclOps: Data Processing and Model Evaluation Toolkit"
  year: 2022
  topics: [data_validation_tools, drift_response, monitoring_stack]
```

#### Uncertainty & Calibration
```yaml
- citation_key: andre2026ci
  inline_citation: "André et al. (2026)"
  title: "Confidence Intervals for Medical Image Analysis"
  year: 2026
  topics: [metrics_framework, uncertainty_quantification, testing_strategy]

- citation_key: shahmohammadi2025conformal
  inline_citation: "Shah-Mohammadi & Kain (2025)"
  title: "Conformal Prediction for Medical Image Segmentation"
  year: 2025
  topics: [uncertainty_quantification, calibration_tools]

- citation_key: bench2025mcdropout
  inline_citation: "Bench et al. (2025)"
  title: "Quantifying Uncertainty for Image Quality Assessment"
  year: 2025
  topics: [uncertainty_quantification, model_diagnostics]

- citation_key: moreo2025calibquant
  inline_citation: "Moreo et al. (2025)"
  title: "On the Interconnections of Calibration, Quantification, and Accuracy Prediction"
  year: 2025
  topics: [calibration_tools, drift_response]
```

#### MLOps & Compliance
```yaml
- citation_key: nogare2025mlopsslr
  inline_citation: "Nogare & Silveira (2025)"
  title: "MLOps for ML Model Lifecycle Automation: A Systematic Literature Review"
  year: 2025
  topics: [pipeline_orchestration, experiment_tracking, monitoring_stack]

- citation_key: rosmarino2025regdocs
  inline_citation: "Rosmarino et al. (2025)"
  title: "MLOps-Driven Automation of Regulatory Documentation for AI-Based SaMD"
  year: 2025
  topics: [compliance_depth, documentation_standard, documentation_generation]

- citation_key: martina2024iec62304devops
  inline_citation: "Martina et al. (2024)"
  title: "Software Medical Device Maintenance: DevOps-Based Approach"
  year: 2024
  topics: [compliance_depth, ci_cd_platform, model_governance]

- citation_key: niemiec2025euaiact
  inline_citation: "Niemiec (2025)"
  title: "EU AI Act: Compliance Implications for Medical AI"
  year: 2025
  topics: [compliance_depth, model_governance]

- citation_key: kwong2025consortai
  inline_citation: "Kwong et al. (2025)"
  title: "CONSORT-AI: Minimum Reporting Standards for Clinical AI"
  year: 2025
  topics: [documentation_standard, compliance_depth]

- citation_key: park2025miclearllm
  inline_citation: "Park et al. (2025)"
  title: "MI-CLEAR-LLM: Minimum Reporting for LLMs in Clinical Research"
  year: 2025
  topics: [documentation_standard, compliance_depth]

- citation_key: lahteenmaki2023regops
  inline_citation: "Lähteenmäki et al. (2023)"
  title: "RegOps Lifecycle for AI/ML Compliance"
  year: 2023
  topics: [compliance_depth, ci_cd_platform, documentation_generation]

- citation_key: testi2025ccmlops
  inline_citation: "Testi et al. (2025)"
  title: "CC-MLOps: Enhancing Cell Counting through MLOps"
  year: 2025
  topics: [pipeline_orchestration, experiment_tracking, xai_strategy]
```

#### Data Quality & Validation
```yaml
- citation_key: vantwist2026datacare
  inline_citation: "van Twist et al. (2026)"
  title: "DATA-CARE: Comprehensive Data Quality Assessment Tool"
  year: 2026
  topics: [data_validation_tools, label_quality, data_profiling]
```

#### Agent Frameworks
```yaml
- citation_key: abdollahi2025agenticsurvey
  inline_citation: "Abdollahi et al. (2025)"
  title: "Comprehensive Survey on Agentic AI"
  year: 2025
  topics: [agent_framework]

- citation_key: bluethgen2025agenticradiology
  inline_citation: "Blüthgen et al. (2025)"
  title: "Agentic Systems in Radiology"
  year: 2025
  topics: [agent_framework, pipeline_orchestration]

- citation_key: tzanis2026agenticregulation
  inline_citation: "Tzanis et al. (2026)"
  title: "Agentic AI Systems: Regulatory Implications"
  year: 2026
  topics: [agent_framework, compliance_depth]

- citation_key: jiang2026ibisagent
  inline_citation: "Jiang et al. (2026)"
  title: "IBISAgent: Reinforcing Visual Reasoning in MLLMs"
  year: 2026
  topics: [agent_framework, segmentation_models]

- citation_key: sheng2026dills
  inline_citation: "Sheng et al. (2026)"
  title: "DiLLS: Interactive Diagnosis of LLM Multi-Agent Systems"
  year: 2026
  topics: [llm_observability, agent_framework]
```

## 4. Decision File Updates Required

### 4.1 HIGH PRIORITY — Major Evidence Updates

#### L3: segmentation_models
**New options to add:**
- `vesselfm`: vesselFM foundation model (Wittmann et al., 2024) — first 3D vessel-specific FM. Zero-shot Dice 74.66 on SMILE-UHURA. Uses MONAI UNet. Includes MiniVess in training data.
- `comma_mamba`: COMMA Mamba architecture (Shi et al., 2025) — Dice 86.36, clDice 84.31 on KiPA. State-space model for vessels.

**Existing options to update:**
- `sam_variants`: Split into SAM2/MedSAM2 (Ma et al., 2025) — MedSAM2 achieves DSC 86-88% on organs but struggles with thin vessels.
- `vista3d`: Atlas-based customization (Zhang et al., 2025) shows VISTA3D drops from 88% to 27% Dice on small structures.

**Prior probability adjustments:**
- Increase `vista3d` → 0.20 (struggles with vessels)
- Add `vesselfm` at 0.20 (vessel-specific FM)
- Add `comma_mamba` at 0.10 (novel architecture)
- Reduce `sam_variants` → 0.10 (confirmed poor for thin vessels)
- Keep `segresnet` at 0.25, `swinunetr` at 0.15

**References to add**: 8+ new references

#### L2: uncertainty_quantification
**Evidence updates:**
- Conformal prediction: Shah-Mohammadi & Kain (2025) validate for medical segmentation → boost `conformal_prediction` to 0.30
- MC Dropout: Bench et al. (2025) show practical utility for image quality → keep `mc_dropout` at 0.20
- Confidence intervals: André et al. (2026) large-scale study recommends percentile bootstrap → affects `metrics_framework`
- TUNE++ (Dhor et al., 2026): topology-aware UQ for vessel segmentation → new UQ method for consideration

**Prior adjustments:**
- `conformal_prediction`: 0.25 → 0.30
- `temperature_scaling`: 0.30 → 0.25

#### L5: drift_response
**Major evidence:**
- Zhang et al. (2026) PPRM: semi-supervised risk monitoring with anytime-valid false alarm guarantees
- Subasri et al. (2023): unidirectional data shifts across hospitals, transfer learning for remediation, continual learning for temporal drift
- Kim et al. (2025): monitoring strategies for clinical AI
- Chakraborty et al. (2025): financial risk metrics adapted for AI monitoring

**Prior adjustments:**
- `automated_retrain`: 0.25 → 0.30 (more evidence for automated approaches)
- `alert_human`: 0.35 → 0.30

#### L1: compliance_depth
**Major evidence:**
- Rosmarino et al. (2025): Automated regulatory documentation via MLOps pipelines
- Martina et al. (2024): DevOps-based IEC 62304 compliance
- Niemiec (2025): EU AI Act classification impacts
- Lähteenmäki et al. (2023): RegOps lifecycle concept
- Kwong et al. (2025), Park et al. (2025): Reporting standards

**Prior adjustments:**
- `iec_62304_full`: 0.15 → 0.20 (automated compliance reduces barrier)
- `lightweight_audit`: 0.50 → 0.45
- Add new option: `regops_automated` at 0.15 (automated compliance via CI/CD)

### 4.2 MEDIUM PRIORITY — Evidence Additions

#### L5: monitoring_stack
- Add references: Subasri (2023), Zhang (2026), Kim (2025), Chakraborty (2025)
- Update rationale with label-agnostic monitoring evidence

#### L3: calibration_tools
- Add references: Moreo (2025), Shah-Mohammadi (2025)
- Update rationale with interconnections between calibration and quantification

#### L3: metrics_framework
- Add references: André (2026) for confidence interval methodology
- Update rationale with bootstrap CI recommendations

#### L3: loss_functions
- Add references: Dhor (2026) TUNE++ topology-aware loss, Shi (2025) COMMA losses
- Update rationale with topology-preserving evidence for vessels

#### L3: augmentation_stack
- Add references: Terms (2025) SynthICL domain randomization, Wittmann (2024) flow matching
- Update rationale with synthetic data generation evidence

#### L3: annotation_platform
- Add references: Terms (2025) VessQC napari plugin, Ma (2025) MedSAM2 3D Slicer
- Update rationale with uncertainty-guided annotation evidence

#### L3: data_validation_tools
- Add references: van Twist (2026) DATA-CARE, Krishnan (2022) CyclOps, Marciano (2025) nnQC
- Update rationale with comprehensive data quality tooling

#### L3: agent_framework
- Add references: Abdollahi (2025), Blüthgen (2025), Tzanis (2026), Jiang (2026), Sheng (2026)
- Update rationale with regulatory implications for agent systems

#### L3: documentation_standard
- Add references: Kwong (2025), Park (2025), Rosmarino (2025)
- Update rationale with reporting standards landscape

#### L5: documentation_generation
- Add references: Rosmarino (2025), Lähteenmäki (2023)
- Update rationale with automated regulatory doc generation evidence

### 4.3 LOW PRIORITY — Minor Additions

- L2: foundation_model_integration — add vesselFM, AtlasSegFM, SynthICL references
- L3: model_diagnostics — add nnQC, MC Dropout references
- L3: label_quality — add VessQC, DATA-CARE references
- L5: annotation_workflow — add VessQC napari, MedSAM2 references
- L5: retraining_trigger — add PPRM, Subasri continual learning references
- L3: llm_observability — add DiLLS references
- L3: testing_strategy — add André CI methodology references
- L4: federated_learning — add Oh (2025) continual learning references

## 5. GitHub Issues — Prioritized Backlog

### P0: Critical (Must Do Next)

1. **Integrate vesselFM as ModelAdapter** — vesselFM (Wittmann et al., 2024) is the first foundation model specifically for 3D vessel segmentation and includes MiniVess in its training data. Uses MONAI UNet internally.
   - Labels: `enhancement`, `models`
   - Size: L

2. **Implement Prediction-Powered Risk Monitoring** — Zhang et al. (2026) PPRM provides semi-supervised drift monitoring with formal false alarm guarantees. Directly applicable to deployed segmentation models.
   - Labels: `enhancement`, `monitoring`
   - Size: L

3. **Add Topology-Aware Loss Functions (clDice, cbDice)** — Multiple papers (Dhor 2026, Shi 2025, Palaniappan 2026) show topology-preserving losses critical for vessel segmentation. Currently only DiceCE implemented.
   - Labels: `enhancement`, `training`
   - Size: M

4. **Implement Confidence Interval Reporting** — André et al. (2026) provide definitive guidance on CI methods for medical segmentation. Percentile bootstrap recommended for most cases.
   - Labels: `enhancement`, `metrics`
   - Size: M

### P1: High Priority (Should Do Soon)

5. **Integrate Conformal Prediction (MAPIE)** — Shah-Mohammadi & Kain (2025) validate conformal prediction for medical segmentation. Already in deps but not integrated.
   - Labels: `enhancement`, `uncertainty`
   - Size: M

6. **Add Automated Regulatory Documentation** — Rosmarino et al. (2025) show MLOps pipelines can auto-generate IEC 62304 artifacts. Integrate into CI/CD.
   - Labels: `enhancement`, `compliance`
   - Size: L

7. **Add COMMA/Mamba Architecture** — Shi et al. (2025) COMMA achieves SOTA on vessel segmentation with Mamba SSM. New architectural paradigm.
   - Labels: `enhancement`, `models`
   - Size: L

8. **Implement VessQC-Style Uncertainty Annotation** — Terms et al. (2025) show uncertainty-guided curation improves recall 67%→94%. Integrate napari/3D Slicer plugin pattern.
   - Labels: `enhancement`, `annotation`
   - Size: M

9. **Add DATA-CARE Data Quality Assessment** — van Twist et al. (2026) provide comprehensive data quality tool. Complements existing Pandera/GE stack.
   - Labels: `enhancement`, `data-quality`
   - Size: M

10. **Add CyclOps Integration** — Krishnan et al. (2022) toolkit for clinical data processing, drift detection, and fairness evaluation. Python-native.
    - Labels: `enhancement`, `monitoring`
    - Size: M

### P2: Medium Priority (Nice to Have)

11. **Implement nnQC Segmentation Quality Control** — Marciano et al. (2025) diffusion-based QC framework. Automatic quality scoring of segmentation outputs.
    - Labels: `enhancement`, `validation`
    - Size: L

12. **Add Reporting Standard Templates** — CONSORT-AI (Kwong 2025), MI-CLEAR-LLM (Park 2025) provide structured reporting checklists for model documentation.
    - Labels: `documentation`
    - Size: S

13. **Explore AtlasSegFM One-Shot Approach** — Zhang et al. (2025) atlas-based foundation model customization. Relevant for new anatomical targets.
    - Labels: `research`
    - Size: L

14. **Add Agent Observability (DiLLS patterns)** — Sheng et al. (2026) layered summary approach for LLM agent diagnostics. Improve Langfuse integration.
    - Labels: `enhancement`, `observability`
    - Size: M

15. **Implement SynthICL Domain Randomization** — Terms et al. (2025) synthetic data generation for in-context learning. Alternative to traditional augmentation.
    - Labels: `research`
    - Size: L

16. **Add MC Dropout Uncertainty** — Bench et al. (2025) validate MC dropout for practical uncertainty estimation. Complement to temperature scaling.
    - Labels: `enhancement`, `uncertainty`
    - Size: M

17. **Calibration-Under-Shift Framework** — Moreo et al. (2025) prove equivalence of calibration/quantification/accuracy tasks. Implement cross-domain calibration.
    - Labels: `research`
    - Size: L

18. **EU AI Act Compliance Checklist** — Niemiec (2025) EU AI Act classification. Create compliance mapping for MinIVess as potential SaMD.
    - Labels: `documentation`, `compliance`
    - Size: S

19. **RegOps CI/CD Pipeline Extension** — Lähteenmäki et al. (2023) RegOps lifecycle. Extend GitHub Actions with regulatory artifact generation.
    - Labels: `enhancement`, `ci-cd`
    - Size: M

20. **MedSAM2 Interactive Adapter** — Ma et al. (2025) with 85%+ annotation cost reduction. Add as annotation tool option.
    - Labels: `enhancement`, `annotation`
    - Size: L

## 6. Execution Plan

### Step 1: Update bibliography.yaml (Task #12a)
Add ~45 new bibliography entries with full metadata.

### Step 2: Update HIGH PRIORITY decision files (Task #12b)
- `segmentation_models.decision.yaml` — Add vesselFM, COMMA, update SAM/VISTA priors
- `uncertainty_quantification.decision.yaml` — Add conformal evidence, TUNE++
- `drift_response.decision.yaml` — Add PPRM, Subasri, CyclOps
- `compliance_depth.decision.yaml` — Add RegOps, IEC 62304 DevOps, EU AI Act
- `monitoring_stack.decision.yaml` — Add drift monitoring evidence

### Step 3: Update MEDIUM PRIORITY decision files (Task #12c)
- calibration_tools, metrics_framework, loss_functions, augmentation_stack
- annotation_platform, data_validation_tools, agent_framework
- documentation_standard, documentation_generation

### Step 4: Update LOW PRIORITY decision files (Task #12d)
- foundation_model_integration, model_diagnostics, label_quality
- annotation_workflow, retraining_trigger, llm_observability, testing_strategy

### Step 5: Create GitHub issues (Task #13)
- Create 20 issues with P0/P1/P2 labels
- Add to org project PVT_kwDOCPpnGc4AYSAM
- Set Priority field values

### Step 6: Update Skills (Task #14)
- Update PRD-update Skill with project management instructions
- Create new planning/backlog Skill

### Step 7: Quality review (Task #15)
- Run validate protocol
- Verify no citation loss
- Verify all new citation_keys resolve
- Verify probability sums

## 7. Quality Checklist

- [ ] All new bibliography entries have complete metadata (doi/url, authors, venue)
- [ ] All decision file references use structured format (citation_key, relevance, sections, supports_options)
- [ ] All rationale fields contain author-year citations
- [ ] No existing references removed from any decision file
- [ ] All probability distributions still sum to 1.0
- [ ] All conditional tables updated if options added
- [ ] Pre-commit citation validation passes
- [ ] GitHub issues created with proper labels and project assignment
