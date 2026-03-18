# Comprehensive Reference: AI Cards and Structured Documentation Artifacts for ML/AI Pipelines

**Purpose**: Reference catalog for the Nature Protocols paper on 3D vascular segmentation MLOps.
**Date**: 2026-03-19
**Total references**: 55+

---

## Table of Contents

1. [Data-Focused Documentation](#1-data-focused-documentation)
2. [Model-and-Method-Focused Documentation](#2-model-and-method-focused-documentation)
3. [System-Focused Documentation](#3-system-focused-documentation)
4. [Use Case and Risk Documentation](#4-use-case-and-risk-documentation)
5. [Explainability and Evaluation Documentation](#5-explainability-and-evaluation-documentation)
6. [Sustainability and ESG Documentation](#6-sustainability-and-esg-documentation)
7. [Regulatory and Compliance Frameworks](#7-regulatory-and-compliance-frameworks)
8. [Medical Device and Clinical AI Documentation](#8-medical-device-and-clinical-ai-documentation)
9. [Card Generation Tools and Registries](#9-card-generation-tools-and-registries)
10. [Meta-Analyses and Surveys](#10-meta-analyses-and-surveys)
11. [Emerging and Proposed Card Types](#11-emerging-and-proposed-card-types)

---

## 1. Data-Focused Documentation

### 1.1 Datasheets for Datasets
- **Citation**: Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J.W., Wallach, H., Daume III, H., & Crawford, K. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92.
- **URL**: https://dl.acm.org/doi/10.1145/3458723
- **arXiv**: https://arxiv.org/abs/1803.09010
- **Description**: Proposes standardized datasheets (inspired by electronics industry) documenting a dataset's motivation, creation, composition, intended uses, distribution, and maintenance. First circulated as a draft in March 2018. Adopted by Microsoft, Google, and IBM internally.
- **Status**: Established (peer-reviewed, CACM)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable to documenting MiniVess volumetric datasets, annotation protocols, and demographic coverage.

### 1.2 Data Cards
- **Citation**: Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI." *Proc. ACM FAccT 2022*.
- **URL**: https://dl.acm.org/doi/10.1145/3531146.3533231
- **Playbook**: https://pair-code.github.io/datacardsplaybook/
- **Description**: Structured summaries of essential facts about ML datasets needed by stakeholders across a dataset's lifecycle. Includes explanations of processes and rationales that shape the data. Developed at Google with participatory design methods.
- **Status**: Established (peer-reviewed, ACM FAccT)
- **Biomedical imaging relevance**: **HIGH** -- template directly usable for documenting multiphoton imaging dataset quality, provenance, and biases.

### 1.3 Data Statements for NLP
- **Citation**: Bender, E.M. & Friedman, B. (2018). "Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science." *Transactions of the Association for Computational Linguistics*, 6, 587-604.
- **Updated**: Bender, E.M., Friedman, B., & McMillan-Major, A. (2021). Version 2 schema and guide.
- **URL**: https://aclanthology.org/Q18-1041/
- **Guide**: https://techpolicylab.uw.edu/data-statements/
- **Description**: Characterization of datasets providing context to understand how experimental results generalize, how software might be deployed, and what biases might be reflected. Originally NLP-focused but applicable to any domain.
- **Status**: Established (peer-reviewed, TACL)
- **Biomedical imaging relevance**: **MEDIUM** -- adaptable schema for documenting imaging protocol variations and population coverage.

### 1.4 Dataset Nutrition Labels
- **Citation**: Holland, S., Hosny, A., Newman, S., Joseph, J., & Chmielinski, K. (2018). "The Dataset Nutrition Label: A Framework To Drive Higher Data Quality Standards." arXiv:1805.03677.
- **Updated**: Chmielinski, K.S., Newman, S., Taylor, M., Joseph, J., Thomas, K., Yurkofsky, J., & Qiu, Y.C. (2022). "The Dataset Nutrition Label (2nd Gen): Leveraging Context to Mitigate Harms in Artificial Intelligence." arXiv:2201.03954.
- **URL**: https://datanutrition.org/
- **Label Maker**: https://labelmaker.datanutrition.org/dashboard
- **Description**: Diagnostic framework providing a distilled yet comprehensive overview of dataset "ingredients" before AI model development. Analogous to nutrition facts for food. The 2nd generation adds context-specific Use Cases & Alerts.
- **Status**: Established (arXiv + nonprofit implementation)
- **Biomedical imaging relevance**: **HIGH** -- "nutrition label" metaphor maps well to biomedical dataset quality communication to clinicians and reviewers.

### 1.5 Data Cards for NLP
- **Citation**: McMillan-Major, A., Auli, M., Barrault, L., et al. (2021). "Reusable Templates and Guides For Documenting Datasets and Models for Natural Language Processing and Generation." *Proc. GEM Workshop at ACL 2021*, 121-135.
- **URL**: https://aclanthology.org/2021.gem-1.11/
- **Description**: Case studies developing reusable documentation templates for the Hugging Face Hub and GEM benchmark. Establishes principles for creating data cards including stakeholder identification, guiding principles, and iterative revision.
- **Status**: Established (peer-reviewed workshop)
- **Biomedical imaging relevance**: **LOW** -- NLP-specific templates, but design process is transferable.

### 1.6 Dataset Development Lifecycle Documentation Framework
- **Citation**: Hutchinson, B., Smart, A., Hecht, B., et al. (2021). "Towards Accountability for Machine Learning Datasets: Practices from Software Engineering and Infrastructure." *Proc. ACM FAccT 2021*.
- **URL**: https://dl.acm.org/doi/pdf/10.1145/3442188.3445918
- **Description**: Rigorous framework for dataset development transparency drawing on software development lifecycle practices. Produces Data Requirements Specifications, Dataset Implementation Diaries, and Dataset Testing Reports at each lifecycle stage.
- **Status**: Established (peer-reviewed, ACM FAccT)
- **Biomedical imaging relevance**: **HIGH** -- lifecycle approach (conception -> creation -> evaluation -> maintenance) maps directly to biomedical dataset curation workflows.

### 1.7 CrowdWorkSheets
- **Citation**: Diaz, M., Kivlichan, I.D., Rosen, R., Baker, D.K., Amironesei, R., Prabhakaran, V., & Denton, E. (2022). "CrowdWorkSheets: Accounting for Individual and Collective Identities Underlying Crowdsourced Dataset Annotation." *Proc. ACM FAccT 2022*.
- **URL**: https://dl.acm.org/doi/abs/10.1145/3531146.3534647
- **arXiv**: https://arxiv.org/abs/2206.08931
- **Description**: Framework for transparent documentation of key decision points in the data annotation pipeline: task formulation, annotator selection, platform choices, dataset analysis, and maintenance. Addresses annotator positionality and identity.
- **Status**: Established (peer-reviewed, ACM FAccT)
- **Biomedical imaging relevance**: **HIGH** -- critical for documenting expert annotator qualifications, inter-rater agreement, and annotation protocols in medical imaging.

### 1.8 Croissant Metadata Format
- **Citation**: MLCommons (2024). "Croissant: A Metadata Format for ML-Ready Datasets." *Proc. DEEM Workshop at SIGMOD 2024*.
- **URL**: https://docs.mlcommons.org/croissant/
- **Spec**: https://docs.mlcommons.org/croissant/docs/croissant-spec.html
- **Description**: Standardized metadata vocabulary building on schema.org for ML datasets. Combines metadata, resource descriptions, data structure, and ML semantics in a single file. Supported by Kaggle, Hugging Face, and OpenML. Version 1.1 (Feb 2026) adds provenance, vocabulary interop, and usage policies.
- **Status**: Established (MLCommons standard, multi-platform adoption)
- **Biomedical imaging relevance**: **MEDIUM** -- useful for making MiniVess datasets discoverable and interoperable with broader ML ecosystem.

---

## 2. Model-and-Method-Focused Documentation

### 2.1 Model Cards
- **Citation**: Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., & Gebru, T. (2019). "Model Cards for Model Reporting." *Proc. FAT* 2019*, ACM.
- **URL**: https://dl.acm.org/doi/10.1145/3287560.3287596
- **arXiv**: https://arxiv.org/abs/1810.03993
- **Description**: Short documents accompanying trained ML models providing benchmarked evaluation across different demographic and application conditions. Documents intended use, performance evaluation procedures, ethical considerations, and limitations. The foundational "card" framework that inspired all others.
- **Status**: Established (peer-reviewed, ACM FAT*)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable for documenting DynUNet, SAM3, and ensemble model performance across vessel types, imaging protocols, and tissue conditions.

### 2.2 Model Cards -- Large-Scale Analysis
- **Citation**: Liang, W., Rajani, N., Yang, X., et al. (2024). "Systematic analysis of 32,111 AI model cards characterizes documentation practice in AI." *Nature Machine Intelligence*, 6, 744-753.
- **URL**: https://www.nature.com/articles/s42256-024-00857-z
- **Description**: Comprehensive analysis of 32,111 model cards on Hugging Face, characterizing what information they contain, temporal trends, and completeness gaps. Found that most cards lack safety, bias, and environmental impact information.
- **Status**: Established (peer-reviewed, Nature Machine Intelligence)
- **Biomedical imaging relevance**: **MEDIUM** -- provides evidence-based guidance on what sections matter most for model documentation.

### 2.3 Method Cards
- **Citation**: Adkins, D., Alsallakh, B., Cheema, A., Dupre, N., Goyal, J., Houben, S., McReynolds, E., Procope, C., Wang, E., & Zvitia, P. (2022). "Method Cards for Prescriptive Machine-Learning Transparency." *Proc. CAIN 2022 (1st Int. Conf. AI Engineering)*.
- **URL**: https://dl.acm.org/doi/10.1145/3522664.3528600
- **Description**: While Model Cards and FactSheets are descriptive (providing nutritional info about cooked meals), Method Cards are prescriptive (providing the recipes). Guide ML engineers through model development with actionable guidance on methodical and algorithmic choices.
- **Status**: Established (peer-reviewed, CAIN)
- **Biomedical imaging relevance**: **HIGH** -- prescriptive documentation of MONAI transform pipelines, loss function selection rationale, and augmentation strategies is exactly what Nature Protocols requires.

### 2.4 Value Cards
- **Citation**: Shen, H., Deng, W.H., Chattopadhyay, S., Wu, Z.S., Wang, X., & Zhu, H. (2021). "Value Cards: An Educational Toolkit for Teaching Social Impacts of Machine Learning through Deliberation." *Proc. ACM FAccT 2021*.
- **URL**: https://dl.acm.org/doi/abs/10.1145/3442188.3445971
- **Description**: Deliberation-driven toolkit using Model Cards, Persona Cards, and Checklist Cards to teach students and practitioners about ethical tradeoffs in ML metrics and deployment decisions.
- **Status**: Established (peer-reviewed, ACM FAccT)
- **Biomedical imaging relevance**: **LOW** -- primarily educational, but persona-based deliberation approach could inform clinical stakeholder engagement.

### 2.5 Consumer Labels for ML Models
- **Citation**: Seifert, C., Scherzinger, S., & Wiese, L. (2019). "Towards Generating Consumer Labels for Machine Learning Models." *Proc. IEEE CogMI 2019*.
- **URL**: https://ieeexplore.ieee.org/document/8998974
- **Description**: Proposes consumer-facing labels (analogous to product safety labels) for published ML models, targeting non-experts such as operators, executors of decisions, and decision subjects who need to understand model properties.
- **Status**: Established (peer-reviewed, IEEE)
- **Biomedical imaging relevance**: **MEDIUM** -- relevant for communicating model capabilities to clinicians who are ML non-experts.

---

## 3. System-Focused Documentation

### 3.1 System Cards
- **Citation**: Alsallakh, B., Cheema, A., Procope, C., Adkins, D., McReynolds, E., Wang, E., et al. (2022). "System-Level Transparency of Machine Learning." Technical Report, Meta.
- **URL**: https://ai.meta.com/research/publications/system-level-transparency-of-machine-learning/
- **Example**: https://ai.meta.com/tools/system-cards/ (Instagram Feed Ranking System Card)
- **Description**: Documents how multiple ML models, AI tools, and non-AI technologies work together as a system. Focuses on system architecture, component interactions, data flows, and protected information usage. Goes beyond individual model documentation.
- **Status**: Established (industry technical report + deployed at Meta)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable to documenting the multi-flow MLOps pipeline (data -> train -> evaluate -> deploy) as an integrated system.

### 3.2 GPT-4o System Card
- **Citation**: OpenAI (2024). "GPT-4o System Card." arXiv:2410.21276.
- **URL**: https://openai.com/index/gpt-4o-system-card/
- **Description**: Comprehensive documentation of GPT-4o's capabilities, limitations, and safety evaluations covering cybersecurity, CBRN, persuasion, and model autonomy risks. Follows OpenAI's Preparedness Framework.
- **Status**: Established (industry standard, widely referenced)
- **Biomedical imaging relevance**: **LOW** -- LLM-specific, but the risk categorization framework is transferable to medical AI safety assessment.

### 3.3 IBM AI FactSheets
- **Citation**: Arnold, M., Bellamy, R.K.E., Hind, M., Houde, S., Mehta, S., Mojsilovic, A., Nair, R., Ramamurthy, K.N., Reimer, D., Olteanu, A., Piorkowski, D., Tsay, J., & Varshney, K.R. (2019). "FactSheets: Increasing Trust in AI Services through Supplier's Declarations of Conformity." *IBM Journal of Research and Development*, 63(4/5).
- **URL**: https://aifs360.res.ibm.com/
- **IEEE**: https://ieeexplore.ieee.org/document/8843893
- **Description**: Supplier's declarations of conformity for AI services, documenting purpose, performance, safety, security, and provenance. More comprehensive than model cards because they cover entire AI services (potentially multiple models + APIs). Available as IBM AI FactSheets 360 toolkit.
- **Status**: Established (peer-reviewed, IBM JRD + deployed product)
- **Biomedical imaging relevance**: **HIGH** -- service-level documentation maps to MLOps platform documentation including MLflow tracking, Prefect orchestration, and BentoML serving.

### 3.4 Reward Reports
- **Citation**: Gilbert, T.K., Lambert, N., Dean, S., Zick, T., & Snoswell, A. (2022). "Reward Reports for Reinforcement Learning." arXiv:2204.10817. Published at *AAAI/ACM AIES 2023*.
- **URL**: https://dl.acm.org/doi/10.1145/3600211.3604698
- **Template**: https://rewardreports.github.io
- **Description**: Living documents that track updates to design choices and assumptions about what a system is optimizing for. Extends model cards to dynamic, post-deployment RL systems. Includes examples for BlenderBot 3, MuZero, MovieLens, and traffic control.
- **Status**: Established (peer-reviewed, AAAI/ACM AIES)
- **Biomedical imaging relevance**: **MEDIUM** -- the "living document" concept and optimization-target documentation are relevant to tracking loss function and metric evolution across HPO runs.

### 3.5 Robustness Gym
- **Citation**: Goel, K., Rajani, N.F., Vig, J., Taschdjian, Z., Bansal, M., & Re, C. (2021). "Robustness Gym: Unifying the NLP Evaluation Landscape." *Proc. NAACL-HLT 2021 Demonstrations*.
- **URL**: https://aclanthology.org/2021.naacl-demos.6/
- **GitHub**: https://github.com/robustness-gym/robustness-gym
- **Description**: Evaluation toolkit unifying 4 standard paradigms: subpopulations, transformations, evaluation sets, and adversarial attacks. While primarily a tool, its structured evaluation outputs serve as system-level documentation.
- **Status**: Established (peer-reviewed, NAACL)
- **Biomedical imaging relevance**: **MEDIUM** -- evaluation paradigm framework transferable to documenting model robustness across imaging conditions.

### 3.6 ABOUT ML
- **Citation**: Raji, I.D. & Yang, J. (2019). "ABOUT ML: Annotation and Benchmarking on Understanding and Transparency of Machine Learning Lifecycles." *NeurIPS 2019 Workshop on Human-Centric ML*.
- **URL**: https://partnershiponai.org/workstream/about-ml/
- **Description**: Multi-year, multi-stakeholder initiative by Partnership on AI to develop comprehensive, scalable documentation tools for ML systems at production scale. Provides process guides, quick guides, and reference documents.
- **Status**: Established (ongoing initiative, industry consortium)
- **Biomedical imaging relevance**: **MEDIUM** -- process-level guidance for implementing documentation practices at scale.

---

## 4. Use Case and Risk Documentation

### 4.1 Use Case Cards
- **Citation**: Hupont, I., Fernandez-Llorca, D., Baldassarri, S., & Gomez, E. (2024). "Use case cards: a use case reporting framework inspired by the European AI Act." *Ethics and Information Technology*, 26(2).
- **URL**: https://link.springer.com/article/10.1007/s10676-024-09757-7
- **Template**: https://gitlab.com/humaint-ec_public/use-case-cards
- **Description**: UML-based template for documenting AI system use cases, implicitly assessing risk levels aligned with the EU AI Act. Includes a UML diagram for system-user interactions. Co-designed with EU policy experts and validated with 11 domain experts.
- **Status**: Established (peer-reviewed, Springer)
- **Biomedical imaging relevance**: **HIGH** -- essential for documenting clinical use cases of vascular segmentation (surgical planning, drug delivery assessment, disease monitoring) and risk classification.

### 4.2 AI Cards (EU AI Act Framework)
- **Citation**: Golpayegani, D., Hupont, I., Panigutti, C., Pandit, H.J., Schade, S., O'Sullivan, D., & Lewis, D. (2024). "AI Cards: Towards an Applied Framework for Machine-Readable AI and Risk Documentation Inspired by the EU AI Act." *Proc. Annual Privacy Forum (APF) 2024*, Springer LNCS.
- **URL**: https://arxiv.org/abs/2406.18211
- **Description**: Holistic framework for representing AI system documentation in both human- and machine-readable formats. Encompasses technical specifications, context of use, and risk management aligned with EU AI Act Annex IV requirements.
- **Status**: Established (peer-reviewed, APF/Springer)
- **Biomedical imaging relevance**: **HIGH** -- machine-readable format enables automated compliance checking for medical AI deployments in the EU.

### 4.3 AI Usage Cards
- **Citation**: Wahle, J.P., Ruas, T., Mohammad, S.M., Meuschke, N., & Gipp, B. (2023). "AI Usage Cards: Responsibly Reporting AI-generated Content." *Proc. ACM/IEEE JCDL 2023*.
- **URL**: https://dl.acm.org/doi/10.1109/JCDL57899.2023.00060
- **arXiv**: https://arxiv.org/abs/2303.03886
- **Description**: Three-dimensional model (transparency, integrity, accountability) for reporting AI use in scientific research. Standardized approach for disclosing where, when, and how AI was used across research fields.
- **Status**: Established (peer-reviewed, ACM/IEEE JCDL)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable to Nature Protocols for disclosing AI-assisted components in the segmentation pipeline.

### 4.4 fAIlureNotes
- **Citation**: Moore, S., Liao, Q.V., & Subramonyam, H. (2023). "fAIlureNotes: Supporting Designers in Understanding the Limits of AI Models for Computer Vision Tasks." *Proc. CHI 2023*, ACM.
- **URL**: https://dl.acm.org/doi/10.1145/3544548.3581242
- **arXiv**: https://arxiv.org/abs/2302.11703
- **Description**: Designer-centered failure exploration and analysis tool featuring a taxonomy of failure modes for CV tasks across three levels: input-level, CV model-level, and response-level. Includes an automated failure engine.
- **Status**: Established (peer-reviewed, ACM CHI)
- **Biomedical imaging relevance**: **HIGH** -- failure mode documentation is critical for medical imaging where false negatives (missed vessels) have clinical consequences.

### 4.5 FeedbackLogs
- **Citation**: Barker, M., Kallina, E., Ashok, D., Collins, K.M., Casovan, A., Weller, A., Talwalkar, A., Chen, V., & Bhatt, U. (2023). "FeedbackLogs: Recording and Incorporating Stakeholder Feedback into Machine Learning Pipelines." arXiv:2307.15475.
- **URL**: https://arxiv.org/abs/2307.15475
- **Description**: Addenda to existing ML pipeline documentation that track stakeholder feedback collection, the feedback itself, and how it updates the pipeline. Designed as evidence for algorithmic auditing and iterative improvement documentation.
- **Status**: Emerging (arXiv preprint)
- **Biomedical imaging relevance**: **HIGH** -- captures clinician feedback loops, annotation corrections, and model improvement cycles that are central to clinical AI deployment.

---

## 5. Explainability and Evaluation Documentation

### 5.1 Saliency Cards
- **Citation**: Boggust, A., Suresh, H., Strobelt, H., Guttag, J.V., & Satyanarayan, A. (2023). "Saliency Cards: A Framework to Characterize and Compare Saliency Methods." *Proc. ACM FAccT 2023*.
- **URL**: https://vis.csail.mit.edu/pubs/saliency-cards/
- **PDF**: https://vis.csail.mit.edu/pubs/saliency-cards.pdf
- **Description**: Framework documenting saliency methods across three dimensions: methodology (how saliency is calculated), sensitivity (relationship with model and inputs), and perceptibility (how humans interpret it). Identifies 10 essential attributes from systematic review of 25 saliency methods.
- **Status**: Established (peer-reviewed, ACM FAccT)
- **Biomedical imaging relevance**: **HIGH** -- directly relevant for documenting Grad-CAM, SHAP, or Captum-based explanations of vascular segmentation predictions, critical for clinician trust.

### 5.2 Audit Cards
- **Citation**: Staufer, L., Yang, M., Reuel, A., & Casper, S. (2025). "Audit Cards: Contextualizing AI Evaluations." arXiv:2504.13839.
- **URL**: https://arxiv.org/abs/2504.13839
- **Description**: Structured format for contextualizing AI evaluation reports. Documents six key features: auditor identity, evaluation scope, methodology, resource access, process integrity, and review mechanisms. Found most existing evaluation reports omit crucial context.
- **Status**: Emerging (arXiv preprint, 2025)
- **Biomedical imaging relevance**: **MEDIUM** -- relevant for documenting third-party evaluation context of medical AI systems.

---

## 6. Sustainability and ESG Documentation

### 6.1 Carbon Emissions Estimation (ML CO2 Impact)
- **Citation**: Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). "Quantifying the Carbon Emissions of Machine Learning." arXiv:1910.09700.
- **URL**: https://arxiv.org/abs/1910.09700
- **Calculator**: https://mlco2.github.io/impact/
- **Description**: Identifies crucial aspects of neural network training that impact carbon emissions: server location, energy grid, training duration, and hardware type. Provides the ML Emissions Calculator tool for estimating CO2 impact.
- **Status**: Established (highly cited arXiv preprint + tool)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable to reporting carbon footprint of GPU training runs on RunPod/GCP for the Nature Protocols paper.

### 6.2 Energy and Policy Considerations for Deep Learning
- **Citation**: Strubell, E., Ganesh, A., & McCallum, A. (2019). "Energy and Policy Considerations for Deep Learning in NLP." *Proc. ACL 2019*.
- **URL**: https://aclanthology.org/P19-1355/
- **arXiv**: https://arxiv.org/abs/1906.02243
- **Description**: Quantifies financial and environmental costs of training large neural networks. Found training a single transformer with NAS produces CO2 equivalent to five cars' lifetimes. Proposes actionable recommendations for reducing costs and improving equity.
- **Status**: Established (peer-reviewed, ACL)
- **Biomedical imaging relevance**: **HIGH** -- provides methodological framework for reporting training costs in protocols paper.

### 6.3 CodeCarbon
- **Citation**: CodeCarbon contributors (2020-present). "CodeCarbon: Track emissions from Compute."
- **URL**: https://codecarbon.io/
- **GitHub**: https://github.com/mlco2/codecarbon
- **PyPI**: https://pypi.org/project/codecarbon/
- **Description**: Python package tracking carbon emissions from computing by measuring GPU + CPU + RAM power consumption and applying regional carbon intensity factors. Generates LaTeX snippets for research papers. Dashboard visualization for comparing emissions.
- **Status**: Established (open-source tool, wide adoption)
- **Biomedical imaging relevance**: **HIGH** -- can be directly integrated into training flows to report emissions in the protocol paper.

### 6.4 AI ESG Protocol
- **Citation**: Saetra, H.S. (2023). "The AI ESG Protocol: Evaluating and Disclosing the Environment, Social, and Governance Implications of Artificial Intelligence Capabilities, Assets and Activities." *Sustainable Development*, 31(2), 1027-1037.
- **URL**: https://onlinelibrary.wiley.com/doi/10.1002/sd.2438
- **Website**: https://www.aiesgprotocol.com/
- **Description**: Flexible high-level protocol for evaluating and disclosing ESG impacts of AI. Four steps: initial descriptive statement, main impact statement, risks and opportunities, and action plan. Covers micro to macro impact scopes.
- **Status**: Established (peer-reviewed, Wiley)
- **Biomedical imaging relevance**: **MEDIUM** -- governance and social impact dimensions relevant for documenting clinical AI deployment considerations.

---

## 7. Regulatory and Compliance Frameworks

### 7.1 EU AI Act Technical Documentation (Article 11 + Annex IV)
- **Citation**: European Parliament and Council (2024). "Regulation (EU) 2024/1689 (Artificial Intelligence Act)."
- **URL (Article 11)**: https://artificialintelligenceact.eu/article/11/
- **URL (Annex IV)**: https://artificialintelligenceact.eu/annex/4/
- **Description**: Legal requirement for high-risk AI systems to maintain comprehensive technical documentation covering system description, development process, data governance, risk management, testing, and post-market monitoring. Simplified forms available for SMEs.
- **Status**: Established (binding EU regulation)
- **Biomedical imaging relevance**: **HIGH** -- medical AI segmentation tools are likely "high-risk" under the Act, requiring full Annex IV documentation.

### 7.2 ALTAI (Assessment List for Trustworthy AI)
- **Citation**: EU High-Level Expert Group on AI (2020). "Assessment List for Trustworthy Artificial Intelligence (ALTAI) for self-assessment."
- **URL**: https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment
- **Tool**: https://altai.insight-centre.org/
- **Description**: Voluntary self-assessment checklist covering 7 requirements: human agency, technical robustness, privacy, transparency, non-discrimination, societal well-being, and accountability. Technology- and sector-neutral. Over 350 stakeholders participated in pilot.
- **Status**: Established (EU official document)
- **Biomedical imaging relevance**: **HIGH** -- directly applicable as a self-assessment tool for evaluating the MLOps platform's trustworthiness.

### 7.3 NIST AI Risk Management Framework (AI RMF 1.0)
- **Citation**: National Institute of Standards and Technology (2023). "Artificial Intelligence Risk Management Framework (AI RMF 1.0)." NIST AI 100-1.
- **URL**: https://www.nist.gov/itl/ai-risk-management-framework
- **PDF**: https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf
- **Description**: Voluntary framework for managing AI risks through four core functions: Govern, Map, Measure, and Manage. Accompanied by a Playbook, Roadmap, and Crosswalk documents. Released January 2023 through open, collaborative process.
- **Status**: Established (US government standard)
- **Biomedical imaging relevance**: **HIGH** -- provides comprehensive risk management vocabulary and structure applicable to medical imaging AI.

### 7.4 TechOps: Technical Documentation Templates for the AI Act
- **Citation**: Lucaj, L., Loosley, A., Jonsson, A., Gasser, U., & van der Smagt, P. (2025). "TechOps: Technical Documentation Templates for the AI Act." *Proc. AAAI/ACM AIES 2025*, 8(2), 1647-1660.
- **URL**: https://ojs.aaai.org/index.php/AIES/article/view/36663
- **GitHub**: https://github.com/aloosley/techops
- **Description**: Open-source templates and examples for documenting data, models, and applications across the entire AI lifecycle, specifically designed for EU AI Act compliance. Validated on real-world scenarios including image segmentation.
- **Status**: Established (peer-reviewed, AAAI/ACM AIES)
- **Biomedical imaging relevance**: **HIGH** -- validated on segmentation use case; directly reusable templates for AI Act compliance.

---

## 8. Medical Device and Clinical AI Documentation

### 8.1 TRIPOD+AI
- **Citation**: Collins, G.S., Moons, K.G.M., Dhiman, P., et al. (2024). "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods." *BMJ*, 385.
- **URL**: https://pubmed.ncbi.nlm.nih.gov/38626948/
- **Website**: https://www.tripod-statement.org/
- **Description**: 27-item checklist harmonizing reporting for prediction model studies regardless of whether regression or ML methods were used. Covers diagnostic, prognostic, monitoring, and screening purposes. Supersedes TRIPOD 2015.
- **Status**: Established (peer-reviewed, BMJ)
- **Biomedical imaging relevance**: **HIGH** -- mandatory reporting standard for clinical prediction/segmentation models in medical journals.

### 8.2 FDA Predetermined Change Control Plans (PCCPs)
- **Citation**: U.S. Food and Drug Administration (2024). "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions." Final Guidance.
- **URL**: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device
- **PDF**: https://www.fda.gov/media/166704/download
- **Description**: Framework allowing manufacturers of AI-enabled medical devices to pre-specify anticipated modifications at initial market authorization. Five guiding principles (developed with Health Canada and UK MHRA): focused, risk-based, evidence-based, transparent, and lifecycle-oriented.
- **Status**: Established (FDA final guidance, December 2024)
- **Biomedical imaging relevance**: **HIGH** -- essential for any path toward clinical deployment of the segmentation tool as SaMD.

### 8.3 IMDRF SaMD Characterization
- **Citation**: IMDRF SaMD Working Group (2025). "Characterization Considerations for Medical Device Software and Software-Specific Risk." IMDRF/SaMD WG/N81 FINAL:2025.
- **URL**: https://www.imdrf.org/
- **Description**: International framework for characterizing Software as a Medical Device (SaMD), including risk categorization, clinical evaluation guidance, and quality management system requirements. Adopted by FDA, EU, and other regulators.
- **Status**: Established (international regulatory framework)
- **Biomedical imaging relevance**: **HIGH** -- defines the regulatory classification pathway for medical imaging segmentation software.

### 8.4 Team Card
- **Citation**: Modise, L.M., Alborzi Avanaki, M., Ameen, S., Celi, L.A., Chen, V.X.Y., Cordes, A., et al. (2025). "Introducing the Team Card: Enhancing governance for medical Artificial Intelligence (AI) systems in the age of complexity." *PLOS Digital Health*, 4(3), e0000495.
- **URL**: https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000495
- **Prototype**: https://www.teamcard.io/team/demo
- **Description**: Protocol documenting researcher positionality -- worldviews, training, backgrounds, and experiences -- that shape decisions during clinical AI development. Promotes reflexivity to identify and address blind spots in development practices.
- **Status**: Established (peer-reviewed, PLOS Digital Health, 2025)
- **Biomedical imaging relevance**: **HIGH** -- directly relevant for disclosing team composition, expertise domains, and potential biases in biomedical imaging research.

---

## 9. Card Generation Tools and Registries

### 9.1 Hugging Face Model Card Infrastructure
- **Citation**: Ozoani, E., Gerchick, M., & Mitchell, M. (2022). "Model Card Guidebook." Hugging Face.
- **URL**: https://huggingface.co/docs/hub/en/model-card-guidebook
- **Landscape**: https://huggingface.co/docs/hub/en/model-card-landscape-analysis
- **Creator Tool**: https://huggingface.co/docs/huggingface_hub/en/guides/model-cards
- **Description**: Comprehensive ecosystem including a model card template (modelcard_template.md), Python API for programmatic card creation, interactive Model Card Creator Tool (no coding required), and landscape analysis of ML documentation tools. All models on the Hub have associated model cards.
- **Status**: Established (de facto industry standard)
- **Biomedical imaging relevance**: **MEDIUM** -- useful infrastructure for model distribution but may need domain-specific extensions.

### 9.2 Google Model Card Toolkit (MCT)
- **Citation**: TensorFlow Team (2020). "Model Card Toolkit." TensorFlow Responsible AI.
- **URL**: https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide
- **GitHub**: https://github.com/tensorflow/model-card-toolkit
- **Description**: Open-source toolkit that streamlines and automates generation of Model Cards. Integrates with TensorFlow Extended (TFX) and ML Metadata to pre-populate fields and generate training/evaluation graphics. Outputs HTML and Markdown formats.
- **Status**: Established (open-source, Google)
- **Biomedical imaging relevance**: **MEDIUM** -- automation capability valuable for generating cards at scale during HPO campaigns.

### 9.3 CycloneDX ML-BOM (Machine Learning Bill of Materials)
- **Citation**: OWASP CycloneDX (2023-present). "Machine Learning Bill of Materials (AI/ML-BOM)." ECMA-424.
- **URL**: https://cyclonedx.org/capabilities/mlbom/
- **Spec**: https://github.com/CycloneDX/specification
- **Description**: Extension of the SBOM standard to ML systems. Documents datasets, models, configurations, dependencies, provenance, and ethical considerations in machine-readable JSON/XML/Protocol Buffers. Part of the CycloneDX v1.5+ specification. Includes "modelCard" field.
- **Status**: Established (ECMA International standard)
- **Biomedical imaging relevance**: **HIGH** -- supply chain transparency is critical for medical device software; ML-BOM enables automated dependency and provenance tracking.

### 9.4 Microsoft Transparency Notes
- **Citation**: Microsoft (2019-present). "Transparency Notes for Azure AI Services."
- **URL**: https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/
- **Description**: Standardized documentation for each Azure AI service covering how the technology works, system owner choices that influence behavior, and the importance of whole-system thinking. Over 40 Transparency Notes published since 2019.
- **Status**: Established (industry practice, Microsoft)
- **Biomedical imaging relevance**: **MEDIUM** -- template structure transferable to documenting MLOps platform services.

### 9.5 Responsible AI Licenses (OpenRAIL)
- **Citation**: BigScience Workshop (2022). "BigScience BLOOM Responsible AI License (RAIL) 1.0."
- **URL**: https://huggingface.co/blog/open_rail
- **FAQ**: https://www.licenses.ai/faq-2
- **Description**: Open licenses combining royalty-free access and flexible downstream use with responsible use restrictions for identified critical scenarios. First OpenRAIL-M license created for BLOOM (176B parameter LLM). Now a general license framework.
- **Status**: Established (widely adopted, OECD-recognized)
- **Biomedical imaging relevance**: **MEDIUM** -- provides a model for licensing medical imaging AI with use restrictions (e.g., prohibiting use without clinical validation).

---

## 10. Meta-Analyses and Surveys

### 10.1 Landscape of ML Documentation Tools (Hugging Face)
- **Citation**: Ozoani, E., Gerchick, M., & Mitchell, M. (2022). "The Landscape of ML Documentation Tools." Hugging Face Hub Documentation.
- **URL**: https://huggingface.co/docs/hub/en/model-card-landscape-analysis
- **Description**: Comprehensive survey organizing documentation tools into data-focused, model-and-method-focused, and system-focused categories. Covers 15+ tools with examples, descriptions, and intended audiences.
- **Status**: Established (canonical reference)
- **Biomedical imaging relevance**: **HIGH** -- essential reference for selecting which card types to implement in the MLOps platform.

### 10.2 Foundation Model Transparency Index
- **Citation**: Bommasani, R., Klyman, K., Longpre, S., Kapoor, S., Maslej, N., Xiong, B., Zhang, D., & Liang, P. (2023). "The Foundation Model Transparency Index." arXiv:2310.12941. *Updated May 2024*.
- **URL**: https://crfm.stanford.edu/fmti/
- **Description**: 100 transparency indicators evaluating foundation model developers on upstream resources (data, labor, compute), model details (size, capabilities, risks), and downstream use (distribution, policies). Mean score improved from 37/100 (Oct 2023) to 58/100 (May 2024).
- **Status**: Established (Stanford CRFM, ongoing)
- **Biomedical imaging relevance**: **MEDIUM** -- provides a scoring framework adaptable to evaluating documentation completeness of medical AI models.

### 10.3 From Reflection to Repair: Scoping Review of Dataset Documentation Tools
- **Citation**: (2025). "From Reflection to Repair: A Scoping Review of Dataset Documentation Tools." arXiv:2602.15968.
- **URL**: https://arxiv.org/abs/2602.15968
- **Description**: Mixed-methods analysis of 59 dataset documentation publications. Identifies three conceptualizations (eliciting reflection, enabling scrutiny, repairing infrastructure) and four persistent patterns impeding adoption: unclear value operationalization, decontextualized designs, unaddressed labor demands, and deferred integration.
- **Status**: Emerging (arXiv preprint, 2025)
- **Biomedical imaging relevance**: **MEDIUM** -- provides evidence on documentation adoption barriers relevant to protocol design.

### 10.4 Automatic Generation of Model and Data Cards
- **Citation**: Liu, J., Li, W., Jin, Z., & Diab, M. (2024). "Automatic Generation of Model and Data Cards: A Step Towards Responsible AI." *Proc. NAACL 2024*.
- **URL**: https://aclanthology.org/2024.naacl-long.110/
- **GitHub**: https://github.com/jiarui-liu/AutomatedModelCardGeneration
- **Description**: Uses LLMs to automatically generate model and data cards, addressing information incompleteness in human-generated cards. Introduces CardBench (4.8k model cards + 1.4k data cards) and CardGen pipeline with two-step retrieval.
- **Status**: Established (peer-reviewed, NAACL)
- **Biomedical imaging relevance**: **MEDIUM** -- automation approach could accelerate documentation of large model registries.

### 10.5 AI Transparency Atlas
- **Citation**: (2025). "AI Transparency Atlas: Framework, Scoring, and Real-Time Model Card Evaluation Pipeline." arXiv:2512.12443.
- **URL**: https://arxiv.org/abs/2512.12443
- **Description**: Automated multi-agent pipeline evaluating documentation completeness using EU AI Act Annex IV and Stanford Transparency Index as baselines. 8 sections, 23 subsections prioritizing safety (25%) and critical risk (20%). Evaluated 50+ models at <$0.06/model.
- **Status**: Emerging (arXiv preprint, 2025)
- **Biomedical imaging relevance**: **MEDIUM** -- scoring framework applicable for benchmarking our documentation completeness against the field.

---

## 11. Emerging and Proposed Card Types

### 11.1 AI Product Cards
- **Citation**: (2024). "AI product cards: a framework for code-bound formal documentation cards in the public administration." *Data & Policy*, Cambridge University Press.
- **URL**: https://www.cambridge.org/core/journals/data-and-policy/article/ai-product-cards/07A9808C3495FD34B7A386507763E6F7
- **Description**: Framework for code-bound formal documentation of AI solutions in public administration. Includes a catalog of Domain Problems enabling cross-administration knowledge sharing and validation of target data for model reproducibility.
- **Status**: Established (peer-reviewed, Cambridge)
- **Biomedical imaging relevance**: **LOW** -- public administration focus, but code-bound documentation concept is transferable.

### 11.2 MCP Server Cards
- **Citation**: Model Context Protocol Community (2025). "SEP-1649: MCP Server Cards - HTTP Server Discovery via .well-known."
- **URL**: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1649
- **Spec**: https://modelcontextprotocol.io/specification/2025-11-25
- **Description**: Standardized discovery mechanism for HTTP-based MCP servers providing metadata (name, version, protocol version, capabilities, transport config) via .well-known/mcp.json endpoint. Enables client-side discovery before connection.
- **Status**: Emerging (specification proposal)
- **Biomedical imaging relevance**: **LOW** -- infrastructure-level protocol, not directly relevant to biomedical documentation.

---

## Summary Statistics

| Category | Count | Established | Emerging |
|----------|-------|------------|----------|
| Data-focused | 8 | 7 | 1 |
| Model/Method-focused | 5 | 5 | 0 |
| System-focused | 6 | 6 | 0 |
| Use case/Risk | 5 | 4 | 1 |
| Explainability/Evaluation | 2 | 1 | 1 |
| Sustainability/ESG | 4 | 4 | 0 |
| Regulatory/Compliance | 4 | 4 | 0 |
| Medical Device/Clinical | 4 | 4 | 0 |
| Tools/Registries | 5 | 5 | 0 |
| Meta-analyses/Surveys | 5 | 3 | 2 |
| Emerging/Proposed | 2 | 1 | 1 |
| **Total** | **50** | **44** | **6** |

---

## High-Relevance References for Biomedical Imaging MLOps

The following are rated **HIGH** relevance for the Nature Protocols paper:

**Must-cite (foundational)**:
1. Mitchell et al. (2019) -- Model Cards
2. Gebru et al. (2021) -- Datasheets for Datasets
3. Pushkarna et al. (2022) -- Data Cards
4. Collins et al. (2024) -- TRIPOD+AI
5. FDA (2024) -- PCCPs for AI/ML SaMD

**Strongly recommended**:
6. Holland et al. (2018) / Chmielinski et al. (2022) -- Dataset Nutrition Labels
7. Boggust et al. (2023) -- Saliency Cards
8. Moore et al. (2023) -- fAIlureNotes
9. Hupont et al. (2024) -- Use Case Cards
10. Golpayegani et al. (2024) -- AI Cards (EU AI Act)
11. Lacoste et al. (2019) -- Carbon emissions
12. CodeCarbon -- Carbon tracking tool
13. CycloneDX ML-BOM -- Supply chain transparency
14. Arnold et al. (2019) -- IBM FactSheets
15. Adkins et al. (2022) -- Method Cards
16. Hutchinson et al. (2021) -- Dataset Lifecycle Documentation
17. Diaz et al. (2022) -- CrowdWorkSheets
18. Barker et al. (2023) -- FeedbackLogs
19. Modise et al. (2025) -- Team Card
20. Lucaj et al. (2025) -- TechOps templates
21. Alsallakh et al. (2022) -- System Cards (Meta)
22. NIST (2023) -- AI RMF 1.0
23. EU HLEG (2020) -- ALTAI
24. EU (2024) -- AI Act Article 11 / Annex IV
25. Wahle et al. (2023) -- AI Usage Cards
