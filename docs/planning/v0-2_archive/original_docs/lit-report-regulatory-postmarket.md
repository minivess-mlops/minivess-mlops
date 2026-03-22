# Regulatory Compliance and Post-Market Surveillance for MLOps-Driven Medical AI

**Literature Research Report R4** | 31 papers | 2026-03-18
**Manuscript section**: Discussion: Regulatory-Native MLOps (future work)
**KG domains**: operations, infrastructure, testing
**Quality target**: MINOR_REVISION
**Dedup**: Papers already cited in R1, R2, or R3 are excluded.

---

## 1. Introduction: The Regulatory-MLOps Convergence

Medical AI faces a unique engineering challenge: the same model that must be continuously
improved (MLOps imperative) must also be locked, validated, and documented before each
deployment (regulatory imperative). These two imperatives appear contradictory — and until
recently, they were. The FDA's Predetermined Change Control Plan (PCCP) framework,
finalized in December 2024, and the EU AI Act (August 2024) represent the first regulatory
mechanisms that explicitly accommodate continuous-learning AI systems.

This report maps the regulatory landscape to the MinIVess MLOps architecture, identifying
where existing infrastructure already satisfies regulatory requirements and where gaps
remain. The platform already implements OpenLineage for audit trails, CycloneDX SBOM
generation, Evidently for drift detection, and a challenger-champion evaluation pattern.
The question is: are these sufficient, and what evidence from the literature supports
or challenges our architectural choices?

**Scope**: QMSR, PCCP, IEC 62304, SBOM, TRIPOD+AI, post-market surveillance, drift
detection, bias monitoring, locked-to-adaptive model lifecycle. **Excluded**: post-training
methods (R1), ensemble selection (R2), federated learning (R3 except privacy aspects).

---

## 2. The FDA Postmarket Surveillance Gap

### 2.1 Adverse Event Reporting Is Inadequate for AI

[Babic et al. (2025). "A general framework for governing marketed AI/ML medical devices."
*npj Digital Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12126487/) conducted
the first systematic analysis of the FDA's MAUDE adverse event database for AI/ML devices,
analyzing 943 reports. Their finding is alarming: the reporting system cannot capture
AI-specific failures such as algorithmic drift, demographic bias, or silent performance
degradation. The implication for MinIVess is clear — passive adverse event reporting is
insufficient; **proactive automated monitoring** must be the primary safety mechanism.

### 2.2 Most FDA-Cleared Devices Lack Rigorous Validation

[Singh, R. et al. (2025). "How AI is used in FDA-authorized medical devices: a taxonomy
across 1,016 authorizations." *npj Digital
Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12219150/) created a three-dimensional
taxonomy of 1,016 FDA authorizations: 84.4% are image-based, and quantitative image
analysis remains dominant. [Sivakumar et al. (2025). "FDA Approval of Artificial
Intelligence and Machine Learning Devices in Radiology." *JAMA Network
Open*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595527/) found that of 950
FDA-authorized AI/ML devices, **only 29% had clinical testing, and just 5% were
prospective**. 97% were cleared via the 510(k) pathway.

These findings validate the NEUROVEX platform's investment in rigorous validation
methodology: factorial experimental design, cross-validated evaluation, and TRIPOD+AI
reporting ([Collins et al. (2024). "TRIPOD+AI." *BMJ*,
385:e078378.](https://doi.org/10.1136/bmj-2023-078378)) are not merely academic best
practices — they represent a competitive regulatory advantage.

---

## 3. PCCPs: The Regulatory Bridge to Continuous Learning

### 3.1 What PCCPs Enable

[Carvalho et al. (2025). "Predetermined Change Control Plans: Guiding Principles for
Advancing Safe, Effective, and High-Quality AI-ML Technologies." *JMIR
AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/) outlines five PCCP principles:
transparency, risk management, robust protocols, stakeholder collaboration, and continuous
monitoring. PCCPs allow manufacturers to implement **preapproved modifications** to AI/ML
devices after market authorization — the regulatory mechanism that makes automated
retraining pipelines legal.

[DuPreez & McDermott (2025). "The use of predetermined change control plans to enable the
release of new versions of software as a medical device." *Expert Review of Medical
Devices*,
22(3):261-275.](https://www.tandfonline.com/doi/full/10.1080/17434440.2025.2468787)
surveyed regulatory practitioners across EU, UK, and US and found the current regulatory
approach **"not fit for purpose"** for continuous-learning AI SaMD. PCCPs are the proposed
solution, but implementation guidance remains sparse.

### 3.2 The Locked-to-Adaptive Lifecycle

[Gonzalez et al. (2024). "Regulating radiology AI medical devices that evolve in their
lifecycle." *arXiv:2412.20498*.](https://arxiv.org/abs/2412.20498) compared the locked-model
paradigm with emerging lifecycle-aware approaches under both EU AI Act and FDA PCCP
frameworks. The transition from locked to adaptive models is the central regulatory
challenge — and MinIVess's challenger-champion evaluation pattern with PCCP-compatible
documentation directly addresses it.

[Granlund et al. (2021). "Towards Regulatory-Compliant MLOps: Oravizio's Journey." *SN
Computer
Science*.](https://link.springer.com/article/10.1007/s42979-021-00726-1) documented
Oravizio, the first CE-certified medical device with an MLOps pipeline, demonstrating
that continuous training is possible but the model must be **locked after packaging**.
Their follow-up ([Granlund et al. (2024). "Towards regulatory compliant lifecycle for
AI-based medical devices in EU."
*arXiv:2409.08006*.](https://arxiv.org/abs/2409.08006)) extended this to the full EU
regulatory lifecycle with cross-stage dependencies between verification, development,
and monitoring.

### 3.3 So What? MinIVess as a PCCP Template

The MinIVess factorial experiment design IS a PCCP template. Each factor combination
(architecture × loss × post-training method) defines a preapproved modification space.
The validation protocol (cross-validated evaluation on held-out folds) defines the
evidence requirements. Drift-triggered retraining within this factor space would constitute
a PCCP-compliant modification — no new submission required.

---

## 4. IEC 62304 and the DevOps Mapping

### 4.1 Pull Requests as Design Controls

[Stirbu et al. (2023). "Continuous design control for machine learning in certified medical
systems." *Software Quality
Journal*.](https://link.springer.com/article/10.1007/s11219-022-09601-5) demonstrated
that the PR workflow maps directly to IEC 62304 design controls: requirements = design
input, committed artifacts = design output, PR review = verification/validation. Model
cards generated as PR artifacts satisfy Clause 8 traceability requirements.

MinIVess's branch model (feature → main → prod) with PR reviews and test tier gates
already implements this pattern. The existing OpenLineage integration provides the audit
trail that links each PR to its downstream effects on model performance.

### 4.2 DevOps for Medical Device Maintenance

[Martina et al. (2024). "Software medical device maintenance: DevOps based approach."
*J. Software: Evolution and Process*,
36(4):e2570.](https://onlinelibrary.wiley.com/doi/full/10.1002/smr.2570) implemented a
DevOps-based software maintenance architecture deployed at Quipu srl, demonstrating
cost-effective IEC 62304 compliance. This is one of the few papers with a **real deployed
implementation** rather than a proposal.

### 4.3 FDA Requirements Checklist

[Singh, V. et al. (2025). "United States Food and Drug Administration Regulation of
Clinical Software in the Era of AI/ML." *Mayo Clinic Proceedings: Digital
Health*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12264609/) enumerates what FDA
submissions require:

| FDA Requirement | MinIVess Component | Status |
|----------------|-------------------|--------|
| SBOM | CycloneDX via Docker | Implemented |
| Model registry + version history | MLflow | Implemented |
| Data lineage | DVC + OpenLineage | Implemented |
| Bias analysis | Evidently fairness slices | Partial |
| Post-market monitoring plan | Evidently + Alibi-Detect | Implemented |
| PCCP documentation | Factorial design as template | Planned |

---

## 5. Drift Detection: Dedicated Infrastructure Required

### 5.1 Performance Monitoring Alone Is Not Enough

[Kore et al. (2024). "Empirical data drift detection experiments on real-world medical
imaging data." *Nature Communications*,
15(1):1887.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10904813/) evaluated drift
detection on 239,235 chest radiographs and found that **monitoring performance alone is
not a good proxy for detecting data drift** — dedicated drift detection is required.
Detection sensitivity depends on sample size, which is critical for MinIVess's
small-dataset context (70 volumes).

### 5.2 Label-Agnostic Monitoring at Scale

[Subasri et al. (2025). "Detecting and Remediating Harmful Data Shifts." *JAMA Network
Open*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12138723/) demonstrated a label-agnostic
MMD-based monitoring pipeline across 143,049 hospital admissions at 7 hospitals. Transfer
learning and drift-triggered continual learning successfully mitigated performance
degradation during COVID-19. Their approach is directly implementable with Alibi-Detect
in the MinIVess stack.

### 5.3 Monitoring Implementation Gap

[Andersen et al. (2024). "Monitoring performance of clinical artificial intelligence in
health care." *JBI Evidence Synthesis*,
22(12):2423-2446.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11630661/) synthesized 39
sources and found a **"relative scarcity of evidence and guidance"** for continuous
monitoring implementation. MinIVess's monitoring architecture (Evidently embedding drift +
Alibi-Detect MMD + WhyLogs profiling) could serve as a reference implementation.

---

## 6. Bias Monitoring: A Regulatory Mandate

[Hasanzadeh et al. (2025). "Bias recognition and mitigation strategies in artificial
intelligence healthcare applications." *npj Digital Medicine*,
8:154.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897215/) provides a lifecycle-stage
bias taxonomy: bias can emerge from data collection, algorithm design, and deployment
decisions, requiring surveillance at each stage. Both FDA (January 2025 guidance) and EU
AI Act now **require** bias analysis in submissions and ongoing monitoring.

[Ng et al. (2024). "Scaling equitable artificial intelligence in healthcare with machine
learning operations." *BMJ Health & Care
Informatics*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11535661/) presents a framework
integrating health equity into MLOps infrastructure, outlining functional requirements
for fairness drift monitoring. For MinIVess, this means Evidently/WhyLogs reports must
include fairness slices — not as an optional feature but as a regulatory requirement.

---

## 7. The EU AI Act: Dual Regulatory Burden

[Aboy et al. (2024). "Navigating the EU AI Act: implications for regulated digital medical
products." *npj Digital
Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379845/) analyzes how the EU AI
Act applies to medical devices. Medical AI faces a **dual regulatory burden**: compliance
with both the Medical Device Regulation (MDR) and the AI Act simultaneously. High-risk
classification is automatic for diagnostic/therapeutic AI. The quality management system
requirements under both frameworks can be satisfied by a single MLOps platform with
comprehensive lineage tracking and automated documentation.

[Zhang, S. et al. (2025). "A decade of review in global regulation and research of
artificial intelligence medical devices (2015-2025)." *Frontiers in
Medicine*.](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1630408/full)
reviews regulatory frameworks across five jurisdictions, finding accelerating convergence.
This convergence means a single well-designed MLOps platform can satisfy multiple
jurisdictions — supporting MinIVess's goal as a reusable platform, not a single-use tool.

---

## 8. MLOps Maturity in Healthcare

### 8.1 Current State

[Rajagopal et al. (2024). "Machine Learning Operations in Health Care: A Scoping Review."
*Mayo Clinic Proceedings: Digital
Health*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11975983/) reviewed 148 articles and
identified 7 key MLOps topics, finding a paucity of prospective studies evaluating patient
outcomes. [Li, Y. et al. (2025). "Maturity Framework for Operationalizing Machine Learning
Applications in Health Care." *JMIR*,
27:e66559.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12448258/) proposed a three-stage
maturity model (low/partial/full), where full maturity requires continuous monitoring and
continual learning — capabilities MinIVess targets.

### 8.2 Explainability Operations

[Huang & Liu (2025). "MLXOps4Medic: A Service Framework for Machine Learning and
Explainability Operations in Medical Imaging AI Development." *IEEE
Access*.](https://ieeexplore.ieee.org/document/11151803/) demonstrates that XAI must be
operationalized as a pipeline stage, not an afterthought. Their microservices-based
framework integrates XAI generation and evaluation with MLflow-compatible provenance
tracking, achieving 73% operational overhead reduction. The EU AI Act's transparency
requirements make this integration mandatory for high-risk medical AI.

### 8.3 Clinical Deployment Framework

[You et al. (2025). "Clinical trials informed framework for real world clinical
implementation and deployment of artificial intelligence applications." *npj Digital
Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11832725/) adapts FDA clinical
trial phases as a deployment methodology: safety → efficacy → effectiveness → monitoring.
This maps to MinIVess's test tier system: staging (safety) → prod (efficacy) → GPU
(effectiveness) → post-deployment (monitoring).

---

## 9. Governance and Ethics

[Aldhafeeri (2025). "Governing Artificial Intelligence in Radiology: A Systematic Review
of Ethical, Legal, and Regulatory Frameworks." *Diagnostics*,
15(18):2300.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12468518/) reviewed 38 studies
on AI governance, identifying persistent challenges: algorithmic bias in imaging datasets,
opacity in diagnostic AI, and ambiguous liability frameworks. The emphasis on transparency
and audit trails validates MinIVess's OpenLineage integration and MLflow experiment
tracking as governance mechanisms.

---

## 10. MinIVess Regulatory Compliance Status

### Already Compliant

| Requirement | Mechanism | Evidence |
|-------------|-----------|----------|
| Software versioning (IEC 62304) | Git + PR workflow | Stirbu (2023) |
| Audit trail (IEC 62304 Cl.8) | OpenLineage + MLflow | Granlund (2024) |
| SBOM (FDA cybersecurity) | CycloneDX Docker SBOM | Singh V. (2025) |
| Data lineage | DVC versioning | de Almeida (2025) |
| Drift detection | Evidently + Alibi-Detect | Kore (2024), Subasri (2025) |
| Model registry | MLflow model versioning | Rajagopal (2024) |
| Test tiers | staging/prod/GPU gates | You (2025) |

### Gaps Identified

| Priority | Gap | Effort | Impact | Reference |
|----------|-----|--------|--------|-----------|
| P0 | PCCP documentation template | Medium | High | Carvalho (2025), DuPreez (2025) |
| P0 | Fairness slice monitoring | Low | High | Hasanzadeh (2025), Ng (2024) |
| P1 | Model cards as PR artifacts | Low | Medium | Stirbu (2023) |
| P1 | XAI pipeline stage | Medium | Medium | Huang & Liu (2025) |
| P2 | TRIPOD+AI reporting template | Medium | Medium | Collins (2024) |
| P2 | Challenger-champion evaluation docs | Medium | Medium | Gonzalez (2024) |
| P3 | EU AI Act technical documentation | High | Low | Aboy (2024) |

---

## 11. Discussion: Research Gaps

1. **PCCP implementation guidance for academic labs**: All published PCCP work targets
   industry. How academic research platforms translate factorial designs into PCCP
   templates is unexplored.

2. **Drift detection on small medical datasets**: Kore (2024) and Subasri (2025) used
   datasets of 200K+ samples. Whether MMD-based drift detection is statistically
   meaningful on 70 volumes is untested.

3. **Regulatory compliance automation**: While Stirbu (2023) maps PRs to design controls,
   no published system automatically generates IEC 62304-compliant documentation from
   CI/CD pipeline metadata.

4. **Preclinical-to-clinical regulatory pathway**: MinIVess operates in preclinical space.
   The regulatory pathway from preclinical research tool to clinical SaMD is undocumented
   for MLOps platforms.

5. **SBOM for ML models**: CycloneDX covers software dependencies, but there is no
   standard for documenting model provenance (training data, hyperparameters, hardware)
   in SBOM format.

---

## References

### Seed Papers (8)

1. [Collins, G.S. et al. (2024). "TRIPOD+AI: An updated reporting guideline." *BMJ*, 385:e078378.](https://doi.org/10.1136/bmj-2023-078378)
2. [Gallifant, J. et al. (2025). "TRIPOD-LLM: Reporting guideline for studies using LLMs." *Nature Medicine*.](https://doi.org/10.1038/s41591-024-03425-5)
3. [FDA (2021). "AI/ML-Based SaMD Action Plan."](https://www.fda.gov/media/145022/download)
4. [de Almeida, J.G. et al. (2025). "Medical machine learning operations." *European Radiology*.](https://link.springer.com/article/10.1007/s00330-025-11654-6)
5. [Vokinger, K.N. et al. (2021). "Mitigating bias in machine learning for medicine." *Communications Medicine*, 1:25.](https://doi.org/10.1038/s43856-021-00028-w)
6. [Muehlematter, U.J. et al. (2021). "Approval of AI-based medical devices." *Lancet Digital Health*, 3(3):e195-e203.](https://doi.org/10.1016/S2589-7500(20)30292-2)
7. [Pianykh, O.S. et al. (2020). "Continuous Learning AI in Radiology." *Radiology*, 297(1):6-14.](https://doi.org/10.1148/radiol.2020200038)
8. [Feng, J. et al. (2022). "Clinical AI Quality Improvement." *Nature Medicine*, 28:1423-1428.](https://doi.org/10.1038/s41591-022-01895-z)

### Discovered Papers (23)

9. [Babic, B. et al. (2025). "A general framework for governing marketed AI/ML medical devices." *npj Digital Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12126487/)
10. [Subasri, V. et al. (2025). "Detecting and Remediating Harmful Data Shifts for the Responsible Deployment of Clinical AI Models." *JAMA Network Open*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12138723/)
11. [Kore, A. et al. (2024). "Empirical data drift detection experiments on real-world medical imaging data." *Nature Communications*, 15(1):1887.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10904813/)
12. [Singh, R. et al. (2025). "How AI is used in FDA-authorized medical devices: a taxonomy across 1,016 authorizations." *npj Digital Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12219150/)
13. [Sivakumar, R. et al. (2025). "FDA Approval of AI/ML Devices in Radiology: A Systematic Review." *JAMA Network Open*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595527/)
14. [Granlund, T. et al. (2021). "Towards Regulatory-Compliant MLOps: Oravizio's Journey." *SN Computer Science*.](https://link.springer.com/article/10.1007/s42979-021-00726-1)
15. [Stirbu, V. et al. (2023). "Continuous design control for machine learning in certified medical systems." *Software Quality Journal*.](https://link.springer.com/article/10.1007/s11219-022-09601-5)
16. [Granlund, T. et al. (2024). "Towards regulatory compliant lifecycle for AI-based medical devices in EU." *arXiv:2409.08006*.](https://arxiv.org/abs/2409.08006)
17. [Gonzalez, C. et al. (2024). "Regulating radiology AI medical devices that evolve." *arXiv:2412.20498*.](https://arxiv.org/abs/2412.20498)
18. [Rajagopal, A. et al. (2024). "Machine Learning Operations in Health Care: A Scoping Review." *Mayo Clinic Proceedings: Digital Health*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11975983/)
19. [Andersen, E.S. et al. (2024). "Monitoring performance of clinical AI in health care." *JBI Evidence Synthesis*, 22(12):2423-2446.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11630661/)
20. [Hasanzadeh, F. et al. (2025). "Bias recognition and mitigation strategies in AI healthcare applications." *npj Digital Medicine*, 8:154.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897215/)
21. [Aldhafeeri, F.M. (2025). "Governing AI in Radiology: A Systematic Review." *Diagnostics*, 15(18):2300.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12468518/)
22. [Aboy, M. et al. (2024). "Navigating the EU AI Act." *npj Digital Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379845/)
23. [Li, Y. et al. (2025). "Maturity Framework for Operationalizing ML in Health Care." *JMIR*, 27:e66559.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12448258/)
24. [Ng, M.Y. et al. (2024). "Scaling equitable AI in healthcare with MLOps." *BMJ Health & Care Informatics*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11535661/)
25. [Martina, M.R. et al. (2024). "Software medical device maintenance: DevOps based approach." *J. Software: Evolution and Process*, 36(4):e2570.](https://onlinelibrary.wiley.com/doi/full/10.1002/smr.2570)
26. [Carvalho, E. et al. (2025). "Predetermined Change Control Plans: Guiding Principles." *JMIR AI*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577744/)
27. [DuPreez, J.A. & McDermott, J. (2025). "The use of PCCPs to enable new versions of SaMD." *Expert Review of Medical Devices*, 22(3):261-275.](https://www.tandfonline.com/doi/full/10.1080/17434440.2025.2468787)
28. [Singh, V. et al. (2025). "FDA Regulation of Clinical Software in the AI/ML Era." *Mayo Clinic Proceedings: Digital Health*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12264609/)
29. [Zhang, S. et al. (2025). "A decade of review in global regulation of AI medical devices (2015-2025)." *Frontiers in Medicine*.](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1630408/full)
30. [You, J.G. et al. (2025). "Clinical trials informed framework for AI deployment." *npj Digital Medicine*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11832725/)
31. [Huang, J. & Liu, Y. (2025). "MLXOps4Medic: Explainability Operations for Medical Imaging AI." *IEEE Access*.](https://ieeexplore.ieee.org/document/11151803/)
