---
title: "Synthetic Data, Data Quality Engineering, and Agentic Systems Report"
status: reference
created: "2026-03-12"
---

# Synthetic Data, Data Quality Engineering, Drift Detection, and Data-Savvy Agentic Systems for Scientific Discovery

**Literature Research Report for Issue #574**
**Date**: 2026-03-12
**Status**: Draft v2.0 — Citation-verified, reviewer-enriched (31 references, 0 hallucinated URLs)

---

## Executive Summary

This report synthesizes recent literature (2000-2026) across five interconnected domains that are critical to the MinIVess MLOps platform: (1) drift detection taxonomies and methods for medical imaging, (2) data quality engineering frameworks and pipelines for scientific AI, (3) synthetic data generation for drift simulation, (4) uncertainty quantification as a drift signal, and (5) data-savvy agentic systems for autonomous scientific discovery. The report provides the theoretical foundation for implementing a comprehensive drift detection suite (Evidently + Alibi-Detect + whylogs) in the Data Engineering Flow, and frames the MinIVess pipeline as one component of an emerging paradigm: **agentic science workflows** where data quality confidence enables hypothesis generation under quantified uncertainty.

**Changes from v1.0**: Removed 3 hallucinated arXiv IDs (placeholder `xxxxx` URLs) and 1 fabricated OpenReview slug. Corrected author initials (Miller, Muller), titles (Gupta, Nguyen), and added verified DOI for van Twist DATA-CARE. Added 10 high-priority papers from the provided bibliography (Zamzmi, Schwabe, Xiong, Singh, Gao, Ye, Luo, Biswas, Moreo, Specktor-Fadida). Corrected Appendix A drift type (topology corruption, not contrast change) and added `observability/drift.py` location.

---

## 1. Drift Detection: Taxonomy and Formal Definitions

### 1.1 The Drift Taxonomy

Distribution drift encompasses multiple distinct phenomena, each requiring different detection strategies and mitigation approaches. The formal taxonomy distinguishes:

**Data Drift (Covariate Shift)**: The input feature distribution changes -- P(X)_{t1} != P(X)_{t2} -- while the decision boundary remains static. This is the most common form in medical imaging, manifesting as changes in scanner parameters, acquisition protocols, patient demographics, or imaging conditions. [DeepChecks (2024). "Data Drift vs. Concept Drift: What Are the Main Differences?"](https://deepchecks.com/data-drift-vs-concept-drift-what-are-the-main-differences/) provides an accessible practitioner taxonomy; the academic foundation comes from [Shimodaira, H. (2000). "Improving Predictive Inference Under Covariate Shift by Weighting the Log-Likelihood Function." *J. Stat. Planning Inference* 90(2), 227-244.](https://doi.org/10.1016/S0378-3758(00)00115-4).

**Concept Drift (Real Drift)**: The relationship between inputs and outputs fundamentally changes -- P(Y|X)_{t1} != P(Y|X)_{t2}. In segmentation, this could mean that annotation conventions evolve (e.g., what counts as a "vessel" at the resolution boundary changes), or the biological ground truth itself shifts (e.g., disease progression alters vascular morphology). Concept drift is harder to detect because it requires label access.

**Label Drift (Prior Probability Shift)**: The output distribution P(Y) changes while P(X|Y) remains constant. For binary vessel segmentation, this manifests as changes in foreground ratio across volumes -- some imaging protocols capture denser vascular territories.

**Feature Drift**: Input feature distributions shift -- P(X) changes. In volumetric imaging, this includes intensity distribution shifts, contrast changes, resolution degradation, and noise level variations.

**Virtual Drift**: Data distribution changes but the model's decision boundary remains valid. This is the benign case -- detectable statistically but not performance-impacting. Distinguishing virtual from real drift is a key challenge.

### 1.2 Temporal Dynamics of Drift

The concept drift taxonomy (see attached taxonomy tree diagram) further classifies drift by temporal behavior:

- **Abrupt drift**: Sudden distribution change (e.g., scanner replacement)
- **Gradual drift**: Slow transition between distributions (e.g., tube aging in multiphoton systems)
- **Incremental drift**: Continuous small shifts accumulating over time
- **Recurring drift**: Periodic patterns (e.g., seasonal calibration cycles)
- **Blip drift**: Temporary anomaly that self-corrects

For multiphoton microscopy in the MinIVess context, the most relevant patterns are **gradual drift** (laser power degradation, objective lens aging, tissue preparation protocol variations) and **abrupt drift** (scanner change, software update, new tissue preparation protocol).

### 1.3 Drift in Medical Imaging: Empirical Evidence

[Kore, V., Abbasi Bavil, E., Subasri, V. et al. (2024). "Empirical data drift detection experiments on real-world medical imaging data." *Nat. Commun.* 15, 1887.](https://doi.org/10.1038/s41467-024-46142-w) provides the most comprehensive empirical study of drift detection in medical imaging. Using the TorchXRayVision classifier on chest X-rays, they demonstrated:

- COVID-19 induced a detectable distribution shift in chest X-ray data
- Both classifier-based and autoencoder-based drift detectors were effective
- Feature-space drift detection (using learned representations) outperformed pixel-space methods
- The key insight: **model embedding spaces are more informative for drift detection than raw image statistics**

This finding directly informs the MinIVess architecture: rather than monitoring only raw voxel statistics (mean, std, entropy), we should extract penultimate-layer embeddings from trained models and monitor those embedding distributions.

[Roschewitz, M., Khara, G., Yearsley, J. et al. (2023). "Automatic correction of performance drift under acquisition shift in medical image classification." *Nat. Commun.* 14, 6608.](https://doi.org/10.1038/s41467-023-42396-y) demonstrated an unsupervised alignment method that automatically corrects prediction drift caused by acquisition shifts (scanner changes) in mammography screening. Their prediction alignment method requires no labels from the new domain -- it uses only the reference distribution statistics and unlabeled target samples. This is directly applicable to the MinIVess scenario where new microscopy data arrives without immediate ground-truth annotations.

**Heterogeneous drift across subpopulations**: [Singh, H., Xia, F., Gossmann, A., Chuang, A., Hong, J.C., Feng, J. (2025). "Who Experiences Large Model Decay and Why? A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift." *Proc. Mach. Learn. Res.* 267, 55757-55787.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12747154/) introduces the SHIFT framework, a two-stage hierarchical hypothesis testing approach that identifies *which* subgroups experience disproportionate performance decay and *why* via variable-subset-specific covariate/outcome shifts. Unlike prior approaches that decompose average performance changes, SHIFT detects heterogeneous drift without requiring causal graphs or parametric assumptions. For MinIVess, this means drift detection should not only ask "has the distribution shifted?" but "which volume subtypes are most affected?"

**Proactive drift robustness**: [Xiong, X., Guo, Z., Zhu, H., Hong, C., Smoller, J.W., Cai, T., Liu, M. (2026). "Adversarial Drift-Aware Predictive Transfer: Toward Durable Clinical AI." *arXiv* 2601.11860.](https://arxiv.org/abs/2601.11860) introduces ADAPT, a framework that constructs an uncertainty set of plausible future models from historical source models and limited current data, then optimizes worst-case performance via adversarial (minimax) optimization. Validated on suicide risk prediction across ICD coding transitions and pandemic-induced shifts (2005-2021 EHR data). The key insight: rather than only detecting drift reactively, models can be trained to *anticipate* plausible future shifts.

### 1.4 Open-Source Drift Detection Tools

[Muller, R., Abdelaal, M., Stjelja, D. (2024). "Open-Source Drift Detection Tools in Action: Insights from Two Use Cases." *arXiv* 2404.18673v2.](https://arxiv.org/abs/2404.18673) provides a comparative analysis of the three major open-source drift detection libraries:

| Tool | Strengths | Best For |
|------|-----------|----------|
| **Evidently AI** | Rich dashboards, Grafana export, column-level drift summary, ML pipeline integration | General drift monitoring, stakeholder reporting |
| **NannyML** | Precise drift timing (CBPE, DLE for performance estimation without labels) | Pinpointing exact temporal onset of drift |
| **Alibi-Detect** | Statistically rigorous tests (MMD with permutation p-values), online detection, image data support | Hypothesis testing, high-dimensional data (embeddings) |

The recommended architecture is a **3-step pipeline**: (1) Assess accuracy impact with NannyML, (2) Confirm drift with Evidently, (3) Pinpoint timing with NannyML's temporal analysis. For the MinIVess pipeline, we adopt the **complementary pattern**: Alibi-Detect provides the statistical backbone (MMD, learned kernels), Evidently provides the visualization and reporting layer, and whylogs provides lightweight continuous profiling.

**Regulatory perspective**: [Zamzmi, G., Venkatesh, K., Nelson, B., Prathapan, S., Yi, P.H., Sahiner, B., Delfino, J.G. (2025). "Out-of-Distribution Detection and Data Drift Monitoring Using Statistical Process Control." *J. Imaging Inform. Med.* 38(2), 997-1015.](https://doi.org/10.1007/s10278-024-01212-9) brings the FDA's Center for Devices and Radiological Health perspective. They apply Statistical Process Control (SPC) methods -- control charts from manufacturing quality control -- to detect OOD data and monitor drift in ML-enabled clinical devices. SPC provides a well-understood regulatory framework: Shewhart charts for sudden shifts, CUSUM for gradual drift, EWMA for smoothed monitoring. For the MinIVess platform targeting eventual SaMD certification, the SPC approach provides both drift detection capability and regulatory alignment.

---

## 2. Data Quality Engineering for Scientific AI

### 2.1 Data Quality Dimensions

[Nguyen, T. et al. (2025). "Data quality management in big data: Strategies, tools, and educational implications." *J. Parallel Distrib. Comput.*](https://doi.org/10.1016/j.jpdc.2025.105034) surveys data quality dimensions and measurement methods for large-scale data systems. The classical dimensions -- completeness, accuracy, consistency, timeliness, uniqueness -- are necessary but insufficient for AI/ML workloads, where additional dimensions include:

- **Representativeness**: Does the training data cover the deployment distribution?
- **Label quality**: Are annotations consistent and correct?
- **Provenance**: Can we trace data lineage from acquisition to model input?

[Hiniduma, K.R., Ryan, M., Byna, S., Bez, J.L., Madduri, R. (2025). "AIDRIN 2.0: A Framework to Assess Data Readiness for AI." *arXiv* 2505.18213v2.](https://arxiv.org/abs/2505.18213) proposes a comprehensive framework with **6 pillars**: Data Quality, Understandability & Usability, Structure & Organization, Governance, Impact on AI, and Fairness. Their case study using the Flamby Heart Disease federated learning benchmark is instructive: excluding a single problematic data partition (client) improved classification accuracy from 70.6% to 74.7%. This demonstrates that **data readiness assessment is not overhead -- it is a performance lever**.

**Medical AI-specific quality dimensions**: [Schwabe, D., Becker, K., Seyferth, M., Klass, A., Schaeffter, T. (2024). "The METRIC-framework for assessing data quality for trustworthy AI in medicine: a systematic review." *npj Digit. Med.* 7, 203.](https://doi.org/10.1038/s41746-024-01196-4) conducted a PRISMA systematic review (62 records from 2362 studies) to synthesize the METRIC-framework, comprising **15 awareness dimensions** for evaluating medical training data quality. These dimensions span completeness, correctness, consistency, currentness, uniqueness, relevance, representativeness, balance, accessibility, traceability, annotation accuracy, annotation consistency, feature relevance, feature engineering, and data transformation quality. The METRIC framework is more granular than AIDRIN 2.0 and specifically targets the regulatory approval pathway for medical AI products -- directly relevant to the MinIVess SaMD certification goals.

The MinIVess platform already implements the DATA-CARE framework ([van Twist, E., van Winden, B., de Jonge, R. et al. (2026). "Data pipeline quality: development and validation of a quality assessment tool for data-driven algorithms and artificial intelligence in healthcare." *BMJ Health Care Inform.* 33(1), e101608.](https://doi.org/10.1136/bmjhci-2025-101608)) which scores datasets across completeness, correctness, consistency, uniqueness, timeliness, and representativeness. The drift detection suite extends this from static assessment to **continuous monitoring**.

### 2.2 Data Provenance and Fabrication Risks

[Gibson, J., White, N., Collins, G.S., Barnett, A.G. (2026). "Evidence of Unreliable Data and Poor Data Provenance in Clinical Prediction Model Research and Clinical Practice." *medRxiv*.](https://doi.org/10.64898/2026.02.24.26347028) delivers a sobering finding: two large Kaggle datasets (stroke prediction, diabetes prediction) that are likely **simulated or fabricated** were used in 124 clinical prediction model studies. Three models based on these datasets showed evidence of clinical deployment, and one was cited in a medical device patent. TRIPOD+AI assessment revealed major data provenance deficiencies across the field.

This finding reinforces a core MinIVess design principle: **every dataset must have verifiable provenance**. The acquisition flow (Flow 0) records acquisition metadata, checksums, and source provenance for every volume. The validation pipeline rejects data without complete provenance chains.

### 2.3 Label Leakage: A Pervasive Quality Threat

[Ramadan, M., Liu, S., Burkhart, A., Parker, T., Beaulieu-Jones, B.K. (2025). "Diagnostic Codes in AI Prediction Models and Label Leakage of Same-Admission Clinical Outcomes." *JAMA Netw. Open* 8(12), e2550454.](https://doi.org/10.1001/jamanetworkopen.2025.50454) found that 40.2% (37/92) of MIMIC-based prediction studies used ICD codes recorded after discharge to predict in-hospital events -- a form of temporal label leakage that inflates model performance to near-perfect AUROC (0.97-0.98).

[Matheny, M.E. & Davis, S.E. (2025). "Avoiding Label Leakage in AI Risk Models -- A Shared Responsibility for a Pervasive Problem." *JAMA Netw. Open* 8(12), e2550464.](https://doi.org/10.1001/jamanetworkopen.2025.50464) classifies three leakage types:
- **Time-based leakage**: Features from after prediction time
- **Aggregation-based leakage**: Patient/site patterns spanning train and test sets
- **Association-based leakage**: Surrogates deterministically related to outcomes

For segmentation, the analogous risks include: using post-processed annotations (label smoothing applied globally), overlapping patches between train/test volumes, and scanner-specific artifacts that correlate with annotation batches. The MinIVess 3-fold split design with volume-level (not patch-level) assignment explicitly prevents spatial leakage.

### 2.4 Cross-Validation Variability and Statistical Rigor

[Jafrasteh, B., Adeli, E., Pohl, K.M., Kuceyeski, A., Sabuncu, M.R., Zhao, Q. (2025). "Statistical variability in comparing accuracy of neuroimaging based classification models via cross validation." *Sci. Rep.* 15, 28745.](https://doi.org/10.1038/s41598-025-12026-2) demonstrates that statistical significance of model comparisons depends heavily on cross-validation setup (K folds, M repetitions). Testing 5 classifiers on 3 neuroimaging datasets, they found contradictory significance conclusions depending on CV configuration -- creating potential for p-hacking through "setup shopping."

[Sargsyan, K. (2026). "Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture." *arXiv* 2511.06701v2.](https://arxiv.org/abs/2511.06701) proposes a radical solution: enforce statistical rigor through type-safe programming. Their Haskell Research monad DSL + Lean 4 formalization of the LORD++ online FDR control algorithm (855 lines, zero `sorry`) achieves FDR control at 1.1% vs. 41% for naive sequential testing. This has direct implications for drift detection: sequential monitoring generates multiple hypothesis tests, and naive alerting will produce excessive false alarms. Online FDR control (LORD++, SAFFRON) should gate drift alarm escalation.

### 2.5 Segmentation Quality Control

[Specktor-Fadida, B., Ben-Sira, L., Ben-Bashat, D., Joskowicz, L. (2025). "SegQC: A segmentation network-based framework for multi-metric segmentation quality control and segmentation error detection in volumetric medical images." *Med. Image Anal.* 103, 103638.](https://doi.org/10.1016/j.media.2025.103638) presents SegQC, a framework for automated segmentation quality estimation and error detection. SegQCNet inputs a scan + segmentation mask and outputs voxel-wise error probabilities, enabling multi-metric quality estimation (DSC, IoU, ARVD) and error localization without ground-truth labels. For the MinIVess drift detection pipeline, SegQC-style quality estimation provides a **label-free Tier 3 signal**: if predicted segmentation quality degrades over time (as estimated by SegQCNet), this indicates either model drift or data drift affecting model performance.

---

## 3. Synthetic Data Generation for Drift Simulation

### 3.1 The Need for Simulated Drift

The MinIVess platform is an academic research system -- there is no continuous stream of new experimental data. To validate the drift detection pipeline, we need to **simulate** realistic distribution shifts. Three approaches exist, in order of increasing fidelity and complexity:

### 3.2 Approach 1: Parametric Drift Injection (Currently Implemented)

The existing `src/minivess/data/drift_synthetic.py` module implements four basic drift types:
- **Intensity shift**: Multiplicative + additive transform (simulates laser power changes)
- **Noise injection**: Gaussian noise with controllable severity (simulates detector degradation)
- **Resolution degradation**: Downsample + upsample blur (simulates optical defocus)
- **Topology corruption**: Structural distortion of vessel connectivity (simulates biological changes)

These are tested in `tests/v2/unit/test_drift.py` using KS tests to verify detectability. This approach is fast and controllable but lacks biological realism -- the synthetic drift is "mechanical" rather than reflecting actual microscopy physics.

### 3.3 Approach 2: External Dataset as Natural Drift (Backup Plan -- VesselNN)

The [VesselNN dataset](https://github.com/petteriTeikari/vesselNN_dataset) provides 12 volumes of multiphoton microscopy data from different tissue preparations and imaging conditions. Using VesselNN as a 4th dataset in the pipeline creates a natural covariate shift scenario:

- **MiniVess** (EBRAINS): 70 volumes, specific tissue preparation, specific microscope
- **VesselNN**: 12 volumes, different tissue preparations, potentially different microscopes
- The distribution shift between MiniVess and VesselNN is *real* -- it reflects actual acquisition differences

This is the recommended **backup plan** for drift simulation: no training of generative models required, just dataset registration and feature extraction. The drift detection pipeline can then compare MiniVess feature distributions against VesselNN feature distributions as a realistic out-of-distribution test.

### 3.4 Approach 3: Physics-Based Synthetic Generation (Future Work)

[Gupta, S., Kamal, S., Rahman, M.S., Rahman, A., Haque, A.K.M.B., Siddique, N. (2025). "Physics-Based Benchmarking Metrics for Multimodal Synthetic Images." *arXiv* 2511.15204v2.](https://arxiv.org/abs/2511.15204) introduces the PCMDE (Physics-Constrained Multimodal Data Evaluation) framework, which evaluates synthetic images against physical constraints rather than perceptual similarity. Their 3-stage pipeline -- multi-source component detection, rule-based physical constraint validation, and LLM-based contextual reasoning -- demonstrates that standard metrics (CLIPScore, VQAScore) fail to capture structural correctness in synthetic data.

For 3D vascular data, physics-based synthetic generation would involve:
1. **Procedural vessel tree generation**: L-systems or Murray's law branching (we already have graph-constrained models from PR #136)
2. **Point spread function (PSF) simulation**: Modeling the multiphoton excitation volume
3. **Noise model**: Shot noise + detector noise + background fluorescence
4. **Motion artifacts**: Tissue drift during z-stack acquisition

This is a significant implementation effort and is deferred to future work. The parametric drift injection (Approach 1) and external dataset (Approach 2) provide sufficient coverage for the current drift detection validation.

### 3.5 Domain Generalization as a Complementary Strategy

[Shi, Y., Zheng, S., Gottumukkala, R. et al. (2026). "CausalFund: Causality-Inspired Domain Generalization in Retinal Fundus Imaging for Low-Resource Screening." *medRxiv*.](https://doi.org/10.64898/2026.03.02.26347127) proposes disentangling disease-relevant features from spurious domain-dependent factors (device signatures, compression artifacts, illumination). Their CausalFund framework achieves domain generalization without target domain data by learning causal invariances during training.

[Gao, Z., Li, B., Salzmann, M., He, X. (2024). "Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts." *NeurIPS 2024*.](https://github.com/gaozhitong/MultiShiftSeg) addresses the dual challenge head-on: existing methods that achieve domain generalization (robustness to covariate shift) often fail at OOD detection (semantic shift), and vice versa. Their solution uses generative augmentation that coherently produces images with both anomaly objects and covariate shifts, plus a learnable semantic-exclusive uncertainty function that distinguishes between semantic shifts (should trigger high uncertainty) and domain shifts (should be generalized over). This is directly relevant to the MinIVess "detect vs. generalize" decision: **the platform needs both capabilities**, and they can be jointly optimized.

This connects to drift detection philosophically: if a model is domain-invariant (robust to acquisition shift), drift detection becomes less critical for performance maintenance. However, drift detection remains essential for **scientific validity** -- even if the model is robust, the researcher needs to know when the input distribution has shifted, because that shift may indicate biologically meaningful changes (e.g., different vascular phenotypes) that should inform hypothesis generation, not just model maintenance.

---

## 4. Uncertainty Quantification as a Drift Signal

### 4.1 The UQ-Drift Connection

Increased prediction uncertainty is often the earliest signal of distribution drift -- the model encounters inputs it is "less sure about." The MinIVess platform already implements conformal prediction (`src/minivess/conformal/`), deep ensembles, and uncertainty decomposition (aleatoric vs. epistemic). These UQ signals can serve as complementary drift indicators:

- **Rising epistemic uncertainty**: Indicates the model encounters novel inputs (data drift)
- **Rising aleatoric uncertainty**: May indicate noisier inputs or annotation ambiguity
- **Widening conformal prediction sets**: Indicates decreasing coverage reliability

[Moreo, A. (2025). "On the Interconnections of Calibration, Quantification, and Classifier Accuracy Prediction under Dataset Shift." *arXiv* 2505.11380.](https://arxiv.org/abs/2505.11380) proves formal equivalences via mutual reduction among three problems under dataset shift: calibration, quantification (class prevalence estimation), and classifier accuracy prediction. Specifically, access to an oracle for any one task enables solving the other two. This theoretical result has direct practical implications: **calibration degradation IS a drift signal**, and conversely, drift detection enables accuracy prediction without labels. For the MinIVess pipeline, this means monitoring calibration quality (via expected calibration error, ECE) provides a complementary drift detection channel.

**Uncertainty-driven adaptive acquisition**: [Tong Ye, C., Han, J., Liu, K., Angelopoulos, A., Griffith, L., Monakhova, K., You, S. (2025). "Learned, Uncertainty-driven Adaptive Acquisition for Photon-Efficient Multiphoton Microscopy." *Optics Express* 33(6).](https://arxiv.org/abs/2310.16102) demonstrates a direct application of conformal risk control for scanning microscopy denoising: the model quantifies pixel-wise uncertainty, then uses it to drive adaptive re-acquisition of only the most uncertain regions. They achieve up to 16x reduction in acquisition time and light dose on experimental confocal and multiphoton microscopy data -- directly in the MinIVess imaging modality. This is a real-world example of the "Lab-in-the-Loop" paradigm (Section 5.3): uncertainty-driven adaptive data acquisition in multiphoton microscopy.

### 4.2 Prediction Instability Diagnostics

[Miller, E.W. & Blume, J.D. (2026). "Diagnostics for Individual-Level Prediction Instability in Machine Learning for Healthcare." *arXiv* 2603.00192v1.](https://arxiv.org/abs/2603.00192) proposes the Empirical Prediction Interval Width (ePIW) for individual-level assessment of ML prediction instability. For segmentation, this translates to voxel-level or region-level prediction variance across ensemble members -- a direct measure of model confidence that can be monitored over time.

### 4.3 Model Retraining Science

[Ong, C., Struyven, R., Denniston, A., Merle, B., Engelmann, J. et al. (2026). "Considering the missing science of retraining and maintenance in medical artificial intelligence, using ophthalmology as an exemplar." *npj Digit. Med.*](https://doi.org/10.1038/s41746-026-02463-2) identifies three fundamental challenges constraining model retraining in medicine:

1. **Data standardization**: No consensus on minimum retraining dataset requirements
2. **Continual learning**: Catastrophic forgetting during fine-tuning
3. **Cost-aware retraining**: When is retraining worth the compute and validation cost?

They argue that academia must recognize model retraining as scholarship -- currently, publishing a new model earns academic credit, but maintaining a deployed model does not. This observation connects to the MinIVess project goals: by publishing the entire MLOps pipeline (including drift detection and retraining triggers), we demonstrate that model maintenance is a first-class research contribution.

---

## 5. Data-Savvy Agents and Agentic Science

### 5.1 The Data-Savvy Agent Framework

[Seedat, N., Liu, J., van der Schaar, M. (2025). "Position: What's the Next Frontier for Data-Centric AI? Data Savvy Agents!" *2nd DATA-FM Workshop @ ICLR 2025*.](https://arxiv.org/abs/2511.01015) argues that data-handling capabilities should be a top priority in agentic system design. They propose four key capabilities:

1. **Proactive Data Acquisition**: Agents autonomously gather task-critical knowledge or solicit human input to address data gaps. Beyond RAG's static retrieval, this involves navigating unstructured sources, negotiating access, and managing acquisition costs.

2. **Sophisticated Data Processing**: Context-aware handling of real-world data challenges -- not just applying standard preprocessing pipelines, but reasoning about *why* data has specific characteristics (e.g., missing blood pressure values correlating with patient severity).

3. **Interactive Test Data Synthesis**: Shifting from static benchmarks to dynamically generated evaluation data. Agents generate test cases tailored to specific domains and scenarios.

4. **Continual Adaptation**: Iteratively refining data, knowledge bases, and decision-making in response to non-stationary environments. This includes proactive change detection and plasticity-stability balancing.

**Agent autonomy taxonomy**: [Luo, Y., Li, G., Fan, J., Tang, N. (2026). "Data Agents: Levels, State of the Art, and Open Problems." *ACM SIGMOD/PODS 2026*.](https://arxiv.org/abs/2602.04261) proposes the first hierarchical taxonomy of data agents from **Level 0** (no autonomy, human executes all data tasks) to **Level 5** (full autonomy, agent proactively anticipates and executes data management, preparation, and analysis). Analogous to SAE J3016 for autonomous driving, this taxonomy provides a framework for positioning the MinIVess Drift Triage Agent (currently ~L2: supervised autonomy with human approval for retraining decisions) and setting a roadmap toward L3-L4 (conditional/high autonomy where the agent detects drift, triages severity, and initiates retraining with human oversight on final deployment).

The MinIVess pipeline implements aspects of all four Seedat capabilities:
- **Acquisition**: Flow 0 automates dataset downloading, format conversion, and provenance logging
- **Processing**: DATA-CARE quality assessment, Pandera schema validation, Great Expectations batch quality checks
- **Synthesis**: Parametric drift injection for testing drift detectors (Approach 1)
- **Adaptation**: Drift detection triggers retraining decisions (the pipeline being built for #574)

### 5.2 Autoresearch: Autonomous AI-Driven Experimentation

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) framework demonstrates a practical implementation of autonomous experimentation: an AI agent iteratively modifies training code, runs 5-minute experiments, evaluates results, and decides whether to keep or discard changes -- enabling ~100 experiments overnight on a single GPU.

[Schmid, P. (2026). "How Autoresearch Will Change Small Language Models Adoption."](https://www.philschmid.de/autoresearch) highlights the key architectural decisions:
- **Fixed 5-minute budgets** ensure experimental comparability
- **Single-file scope** (editing only `train.py`) maintains reproducibility
- **Git-based memory** allows agents to learn from commit history
- **Binary outcome metrics** (val_bpb) require no subjective judgment

Shopify's Tobi Lutke achieved a 0.8B model scoring 19% higher than a previous 1.6B model in 8 hours of autonomous experimentation. This validates the core thesis: **autonomous experimentation, when properly constrained, produces real research insights**.

**Operationalizing agentic AI**: [Biswas, S., Bhatt, H., Vaidhyanathan, K. (2026). "Architecting AgentOps Needs CHANGE." *arXiv* 2601.06456.](https://arxiv.org/abs/2601.06456) argues that DevOps/MLOps paradigms are insufficient for agentic AI because agent behavior evolves continuously at runtime (non-determinism). They propose the **CHANGE** framework with six capabilities: **C**ontextualize (track experiential state), **H**armonize (multi-agent alignment), **A**nticipate (behavioral drift prediction via digital twins), **N**egotiate (dynamic autonomy/oversight), **G**enerate (new tool/capability creation), **E**volve (long-term governance). For the MinIVess Drift Triage Agent, the "Anticipate" capability is most relevant: the agent should predict behavioral drift (model degradation) before it impacts downstream decisions, using digital twin simulations of the pipeline under synthetic drift scenarios.

The MinIVess parallel is the overnight runner skill (`.claude/skills/overnight-runner/`), which orchestrates autonomous test-fix-commit cycles. The drift detection pipeline extends this from code-level iteration to data-level iteration: autonomous monitoring -> drift detection -> retraining trigger -> model validation -> deployment decision.

### 5.3 The Lab-in-the-Loop Paradigm

Seedat et al. describe a "Lab-in-the-Loop" paradigm where data-savvy agents actively acquire experimental data, integrate findings, generate hypotheses, design experiments, and analyze results. Early efforts include [Boiko, D.A. et al. (2023). "Autonomous chemical research with large language models." *Nature* 624, 570-578.](https://doi.org/10.1038/s41586-023-06792-0) in chemistry and autonomous biology labs.

For multiphoton vascular imaging, the lab-in-the-loop paradigm would involve:
1. **Data-savvy agent** monitors incoming microscopy data for drift
2. When drift is detected, the agent assesses whether it is instrumental (scanner drift) or biological (new phenotype)
3. For instrumental drift: trigger model retraining with corrected data
4. For biological drift: flag as hypothesis-generating and route to the scientific analysis pipeline
5. The agent designs follow-up experiments to confirm the biological hypothesis

This vision connects the MinIVess MLOps pipeline to the broader goal of **agentic science** -- where data quality confidence enables autonomous hypothesis generation.

### 5.4 Agentic Knowledge Systems for Science

Several recent works demonstrate the trajectory toward agent-mediated scientific workflows:

[Rao, J. et al. (2026). "SciDataCopilot: An Agentic Data Preparation Framework for AGI-driven Scientific Discovery." *arXiv* 2602.09132.](https://arxiv.org/abs/2602.09132) -- Agents that autonomously process and prepare scientific data across domains, bridging the gap between raw experimental outputs and analysis-ready datasets. With 32 co-authors spanning multiple institutions, this represents a large-scale effort to standardize agentic data workflows.

[Wang, Z., Chen, Z., Yang, Z., Wang, X., Jin, Q., Peng, Y., Lu, Z., Sun, J. (2025). "DeepEvidence: Empowering Biomedical Discovery with Deep Knowledge Graph Research." *arXiv* 2601.11560.](https://arxiv.org/abs/2601.11560) -- Knowledge graph construction from biomedical literature, enabling evidence-based reasoning for drug discovery and clinical decision support. The knowledge graph approach is complementary to drift detection: it provides structured domain knowledge that agents can use to interpret *why* drift occurred (e.g., mapping scanner changes to known physics, or biological shifts to known phenotypes).

The common thread: **data quality is the bottleneck for agent autonomy**. An agent cannot generate reliable hypotheses from unreliable data. The drift detection pipeline is therefore not just an operational monitoring tool -- it is an **epistemic foundation** for agentic science.

---

## 6. Implementation Architecture for MinIVess

### 6.1 Three-Tier Drift Detection

Based on the literature review, the MinIVess drift detection suite should implement three tiers:

**Tier 1: Tabular Feature Drift (Evidently)**
- Extract statistical features from each volume (mean, std, p5, p95, SNR, contrast, entropy -- already in `src/minivess/data/feature_extraction.py`)
- Run Evidently `DataDriftPreset` comparing current batch features against reference distribution
- Export HTML reports and Grafana-compatible JSON
- Threshold: column-level drift detected in >=30% of features triggers Tier 2

**Tier 2: Embedding-Space Drift (Alibi-Detect)**
- Extract penultimate-layer embeddings from the trained model for each volume
- Run MMD test with permutation-based p-values (Alibi-Detect `MMDDrift`)
- Per-volume anomaly scoring via Mahalanobis distance (with Ledoit-Wolf shrinkage for the 70-volume reference set)
- Threshold: p < 0.01 on MMD test triggers drift alert

**Tier 3: Performance Drift (NannyML-style + SegQC)**
- When ground-truth labels are available (new annotations), compute actual metric drift
- Monitor DSC, clDice, HD95 distributions over time
- Conformal prediction set width as a label-free performance proxy
- SegQC-style segmentation quality estimation as additional label-free signal (Specktor-Fadida et al. 2025)

**Regulatory alignment**: All three tiers can be mapped to SPC control charts (Zamzmi et al. 2025): Tier 1 features as Shewhart charts, Tier 2 MMD as CUSUM, Tier 3 metrics as EWMA.

### 6.2 Drift Simulation Sources

1. **Parametric injection** (existing `drift_synthetic.py`): intensity shift, noise, resolution degradation, topology corruption
2. **External dataset** (VesselNN as 4th dataset): natural covariate shift from different acquisition conditions
3. **Cross-dataset features** (DeepVess, TubeNet-2PM): additional natural shift sources

### 6.3 Integration Points

- **Data Flow (Flow 1)**: Feature extraction + Evidently report generation after data loading
- **Post-Training Flow (Flow 3)**: Embedding-space drift check on validation data
- **Dashboard Flow (Flow 5)**: Drift report visualization, trend tracking, alert history
- **Drift Triage Agent** (existing `data_flow.py:370`): Pydantic AI agent that classifies drift severity and recommends action

### 6.4 Statistical Considerations for Small Datasets

With ~70 training volumes, classical drift detection faces power limitations:

- **MMD with kernel methods**: Does not require explicit covariance estimation -- preferred over Mahalanobis for population tests
- **Ledoit-Wolf shrinkage**: Required for any covariance-based methods (embedding dimension >> sample count)
- **PCA before Mahalanobis**: Retain 95% variance (typically 5-10 components) before computing distances
- **Multiple testing correction**: Online FDR control (LORD++) to manage false alarm rate across sequential monitoring windows
- **Conservative thresholds**: p < 0.01 rather than p < 0.05, with multiple consecutive alerts before triggering retraining

---

## 7. Connection to the MinIVess Academic Paper

The drift detection and data quality engineering story serves multiple roles in the planned academic paper:

1. **Methodological contribution**: A complete, reproducible drift detection pipeline for volumetric medical imaging -- filling a gap identified by [Kore et al. (2024)](https://doi.org/10.1038/s41467-024-46142-w) who noted the lack of standardized drift detection workflows for medical imaging.

2. **Discussion material**: The connection between data quality monitoring and scientific hypothesis generation (Section 5.3) provides rich discussion about how MLOps infrastructure enables new forms of scientific inquiry.

3. **Reproducibility demonstration**: By publishing the complete pipeline (including drift simulation, detection, and triage), we demonstrate that model maintenance is publishable research -- addressing the gap identified by [Ong et al. (2026)](https://doi.org/10.1038/s41746-026-02463-2).

4. **Future work**: The agentic science vision (Lab-in-the-Loop for microscopy) provides a compelling "north star" that motivates the current engineering work.

5. **Regulatory pathway**: The SPC-based drift monitoring framework (Zamzmi et al. 2025) provides a clear alignment with FDA expectations for ML-enabled medical devices, strengthening the SaMD certification narrative.

---

## References

1. [Biswas, S., Bhatt, H., Vaidhyanathan, K. (2026). "Architecting AgentOps Needs CHANGE." *arXiv* 2601.06456.](https://arxiv.org/abs/2601.06456)

2. [Boiko, D.A., MacKnight, R., Kline, B., Gomes, G. (2023). "Autonomous chemical research with large language models." *Nature* 624, 570-578.](https://doi.org/10.1038/s41586-023-06792-0)

3. [DeepChecks (2024). "Data Drift vs. Concept Drift: What Are the Main Differences?"](https://deepchecks.com/data-drift-vs-concept-drift-what-are-the-main-differences/)

4. [Gao, Z., Li, B., Salzmann, M., He, X. (2024). "Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts." *NeurIPS 2024*.](https://github.com/gaozhitong/MultiShiftSeg)

5. [Gibson, J., White, N., Collins, G.S., Barnett, A.G. (2026). "Evidence of Unreliable Data and Poor Data Provenance in Clinical Prediction Model Research and Clinical Practice." *medRxiv*.](https://doi.org/10.64898/2026.02.24.26347028)

6. [Gupta, S., Kamal, S., Rahman, M.S., Rahman, A., Haque, A.K.M.B., Siddique, N. (2025). "Physics-Based Benchmarking Metrics for Multimodal Synthetic Images." *arXiv* 2511.15204v2.](https://arxiv.org/abs/2511.15204)

7. [Hiniduma, K.R., Ryan, M., Byna, S., Bez, J.L., Madduri, R. (2025). "AIDRIN 2.0: A Framework to Assess Data Readiness for AI." *arXiv* 2505.18213v2.](https://arxiv.org/abs/2505.18213)

8. [Jafrasteh, B., Adeli, E., Pohl, K.M., Kuceyeski, A., Sabuncu, M.R., Zhao, Q. (2025). "Statistical variability in comparing accuracy of neuroimaging based classification models via cross validation." *Sci. Rep.* 15, 28745.](https://doi.org/10.1038/s41598-025-12026-2)

9. [Karpathy, A. (2026). "autoresearch: Autonomous AI-Driven LLM Research." GitHub.](https://github.com/karpathy/autoresearch)

10. [Kore, V., Abbasi Bavil, E., Subasri, V. et al. (2024). "Empirical data drift detection experiments on real-world medical imaging data." *Nat. Commun.* 15, 1887.](https://doi.org/10.1038/s41467-024-46142-w)

11. [Luo, Y., Li, G., Fan, J., Tang, N. (2026). "Data Agents: Levels, State of the Art, and Open Problems." *ACM SIGMOD/PODS 2026*.](https://arxiv.org/abs/2602.04261)

12. [Matheny, M.E. & Davis, S.E. (2025). "Avoiding Label Leakage in AI Risk Models -- A Shared Responsibility for a Pervasive Problem." *JAMA Netw. Open* 8(12), e2550464.](https://doi.org/10.1001/jamanetworkopen.2025.50464)

13. [Miller, E.W. & Blume, J.D. (2026). "Diagnostics for Individual-Level Prediction Instability in Machine Learning for Healthcare." *arXiv* 2603.00192v1.](https://arxiv.org/abs/2603.00192)

14. [Moreo, A. (2025). "On the Interconnections of Calibration, Quantification, and Classifier Accuracy Prediction under Dataset Shift." *arXiv* 2505.11380.](https://arxiv.org/abs/2505.11380)

15. [Muller, R., Abdelaal, M., Stjelja, D. (2024). "Open-Source Drift Detection Tools in Action: Insights from Two Use Cases." *arXiv* 2404.18673v2.](https://arxiv.org/abs/2404.18673)

16. [Nguyen, T. et al. (2025). "Data quality management in big data: Strategies, tools, and educational implications." *J. Parallel Distrib. Comput.*](https://doi.org/10.1016/j.jpdc.2025.105034)

17. [Ong, C., Struyven, R., Denniston, A., Merle, B., Engelmann, J. et al. (2026). "Considering the missing science of retraining and maintenance in medical artificial intelligence, using ophthalmology as an exemplar." *npj Digit. Med.*](https://doi.org/10.1038/s41746-026-02463-2)

18. [Ramadan, M., Liu, S., Burkhart, A., Parker, T., Beaulieu-Jones, B.K. (2025). "Diagnostic Codes in AI Prediction Models and Label Leakage of Same-Admission Clinical Outcomes." *JAMA Netw. Open* 8(12), e2550454.](https://doi.org/10.1001/jamanetworkopen.2025.50454)

19. [Rao, J. et al. (2026). "SciDataCopilot: An Agentic Data Preparation Framework for AGI-driven Scientific Discovery." *arXiv* 2602.09132.](https://arxiv.org/abs/2602.09132)

20. [Roschewitz, M., Khara, G., Yearsley, J. et al. (2023). "Automatic correction of performance drift under acquisition shift in medical image classification." *Nat. Commun.* 14, 6608.](https://doi.org/10.1038/s41467-023-42396-y)

21. [Sargsyan, K. (2026). "Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture." *arXiv* 2511.06701v2.](https://arxiv.org/abs/2511.06701)

22. [Schmid, P. (2026). "How Autoresearch Will Change Small Language Models Adoption." *Blog*.](https://www.philschmid.de/autoresearch)

23. [Schwabe, D., Becker, K., Seyferth, M., Klass, A., Schaeffter, T. (2024). "The METRIC-framework for assessing data quality for trustworthy AI in medicine: a systematic review." *npj Digit. Med.* 7, 203.](https://doi.org/10.1038/s41746-024-01196-4)

24. [Seedat, N., Liu, J., van der Schaar, M. (2025). "Position: What's the Next Frontier for Data-Centric AI? Data Savvy Agents!" *2nd DATA-FM Workshop @ ICLR 2025, Singapore*.](https://arxiv.org/abs/2511.01015)

25. [Shi, Y., Zheng, S., Gottumukkala, R. et al. (2026). "CausalFund: Causality-Inspired Domain Generalization in Retinal Fundus Imaging for Low-Resource Screening." *medRxiv*.](https://doi.org/10.64898/2026.03.02.26347127)

26. [Shimodaira, H. (2000). "Improving Predictive Inference Under Covariate Shift by Weighting the Log-Likelihood Function." *J. Stat. Planning Inference* 90(2), 227-244.](https://doi.org/10.1016/S0378-3758(00)00115-4)

27. [Singh, H., Xia, F., Gossmann, A., Chuang, A., Hong, J.C., Feng, J. (2025). "Who Experiences Large Model Decay and Why? A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift." *Proc. Mach. Learn. Res.* 267, 55757-55787.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12747154/)

28. [Specktor-Fadida, B., Ben-Sira, L., Ben-Bashat, D., Joskowicz, L. (2025). "SegQC: A segmentation network-based framework for multi-metric segmentation quality control and segmentation error detection in volumetric medical images." *Med. Image Anal.* 103, 103638.](https://doi.org/10.1016/j.media.2025.103638)

29. [Tong Ye, C., Han, J., Liu, K., Angelopoulos, A., Griffith, L., Monakhova, K., You, S. (2025). "Learned, Uncertainty-driven Adaptive Acquisition for Photon-Efficient Multiphoton Microscopy." *Optics Express* 33(6).](https://arxiv.org/abs/2310.16102)

30. [van Twist, E., van Winden, B., de Jonge, R. et al. (2026). "Data pipeline quality: development and validation of a quality assessment tool for data-driven algorithms and artificial intelligence in healthcare." *BMJ Health Care Inform.* 33(1), e101608.](https://doi.org/10.1136/bmjhci-2025-101608)

31. [Wang, Z., Chen, Z., Yang, Z., Wang, X., Jin, Q., Peng, Y., Lu, Z., Sun, J. (2025). "DeepEvidence: Empowering Biomedical Discovery with Deep Knowledge Graph Research." *arXiv* 2601.11560.](https://arxiv.org/abs/2601.11560)

32. [Xiong, X., Guo, Z., Zhu, H., Hong, C., Smoller, J.W., Cai, T., Liu, M. (2026). "Adversarial Drift-Aware Predictive Transfer: Toward Durable Clinical AI." *arXiv* 2601.11860.](https://arxiv.org/abs/2601.11860)

33. [Zamzmi, G., Venkatesh, K., Nelson, B., Prathapan, S., Yi, P.H., Sahiner, B., Delfino, J.G. (2025). "Out-of-Distribution Detection and Data Drift Monitoring Using Statistical Process Control." *J. Imaging Inform. Med.* 38(2), 997-1015.](https://doi.org/10.1007/s10278-024-01212-9)

---

## Appendix A: Existing MinIVess Infrastructure Relevant to #574

| Module | File | Status | Content |
|--------|------|--------|---------|
| Feature extraction | `src/minivess/data/feature_extraction.py` | Implemented | `extract_volume_features()` (9 features), `extract_batch_features()`, clDice proxy |
| Drift synthetic | `src/minivess/data/drift_synthetic.py` | Implemented | `DriftType` enum, `apply_drift()` with 4 types: intensity shift, noise injection, resolution degradation, topology corruption |
| Drift detection (legacy) | `src/minivess/validation/drift.py` | Implemented | `detect_prediction_drift()` with KS + PSI methods |
| Drift detection (full) | `src/minivess/observability/drift.py` | Implemented | `FeatureDriftDetector` (Tier 1, per-feature KS), `EmbeddingDriftDetector` (Tier 2, kernel MMD + permutation) |
| DATA-CARE quality | `src/minivess/validation/data_care.py` | Implemented | 6-dimension quality scoring (completeness, correctness, consistency, uniqueness, timeliness, representativeness) |
| Pandera schemas | `src/minivess/validation/schemas.py` | Implemented | NiftiMetadata, TrainingMetrics, AnnotationQuality |
| GE expectations | `src/minivess/validation/expectations.py` | Implemented | Nifti + training metrics suites |
| GE runner | `src/minivess/validation/ge_runner.py` | Implemented | `run_expectation_suite()`, `validate_nifti_batch()` |
| Quality gates | `src/minivess/validation/gates.py` | Implemented | `GateResult` dataclass |
| whylogs profiling | `src/minivess/validation/profiling.py` | Implemented | `profile_dataframe()`, `compare_profiles()`, `ProfileDriftReport` |
| Deepchecks config | `src/minivess/validation/deepchecks_vision.py` | Config stubs only | 2D-only library, incompatible with 3D volumetric data |
| Drift triage agent | `src/minivess/orchestration/flows/data_flow.py:370` | Implemented | Pydantic AI agent for drift classification (monitor/investigate/retrain) |
| Drift detection tests | `tests/v2/unit/test_drift_detection.py` | Implemented | 21 tests: feature extraction, Tier 1+2 detectors, topology metrics |
| Drift synthetic tests | `tests/v2/unit/test_drift.py` | Implemented | 6 tests: KS-based detectability for all 4 drift types |

### Missing for #574

1. **Evidently integration**: `DataDriftPreset` report generation with HTML + Grafana JSON export (Tier 1)
2. **Alibi-Detect integration**: MMD embedding-space drift with permutation p-values (Tier 2)
3. **VesselNN as drift source**: External dataset registration + feature comparison as natural covariate shift
4. **Drift report persistence**: Saving Evidently reports + drift metrics to MLflow artifacts
5. **End-to-end flow integration**: Wiring drift detection into Data Flow (post-loading) and Post-Training Flow (embedding drift)
6. **SPC control charts**: Shewhart + CUSUM + EWMA charts for regulatory alignment (Zamzmi et al. 2025)
7. **Grafana dashboard**: Drift metric panels with temporal trends and alert history

---

## Appendix B: Key Papers Not Yet Fully Integrated (For Future Iterations)

These papers from the provided bibliography are relevant but their full content was not deeply integrated into this report due to scope:

**Uncertainty & Calibration:**
- Villecroze et al. (2025) -- Empirical Bayes uncertainty
- Loaiza et al. (2025) -- Deep ensembles
- Stickland et al. (2020) -- Diverse ensembles and calibration
- Cheng et al. (2026) -- Spectral entropy calibration
- Ebrahimpour et al. (2025) -- Robust ensembles Lipschitz
- Aggarwal et al. (2024) -- Miscalibration in semi-supervised learning
- Korchagin et al. (2025) -- Confidence-aware training for uncertainty
- Kassapis et al. (2024) -- Calibrated adversarial refinement for stochastic segmentation

**Segmentation & Domain Generalization:**
- Zhang et al. (2026) -- Modality discrepancy segmentation
- Sengupta et al. (2026) -- SynthFM 3D zero-shot
- Moglia et al. (2026) -- Generalist models segmentation survey
- Nam et al. (2025) -- Test-time modality generalization
- Hu et al. (2025) -- SynthICL robust segmentation via data synthesis

**Data Quality & Annotation:**
- Mah et al. (2026) -- Annotation subjectivity and ground truth
- Schilling et al. (2022) -- Automated annotator variability inspection
- Kumari et al. (2025) -- Annotation ambiguity aware semi-supervised segmentation
- Marciano et al. (2025) -- Diffusion-based segmentation quality control

**Federated & Multi-Site:**
- Manthe et al. (2025) -- Federated neuroimage segmentation
- Rahman et al. (2026) -- FairVLM medical segmentation
- Ciupek et al. (2025) -- Federated learning in medical imaging

**Agentic & MLOps:**
- Abdollahi et al. (2025) -- Agentic AI in biomedicine survey
- Bade (2026) -- MLOps for production AI
- Andre et al. (2026) -- Performance uncertainty confidence intervals
