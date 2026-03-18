# Biomedical Agentic AI: From MLOps Platform Extensions to Autonomous Scientific Discovery

**Status**: Complete (v1.1 — updated with full agent research results)
**Date**: 2026-03-18
**Branch**: fix/fda-readiness-improvement-xml
**Audience**: NEUROVEX manuscript discussion, repo README, KG updates, GitHub issues
**Paper count**: 70 papers (27 seeds + 43 web-discovered)

---

## 1. Introduction: Why Agentic AI Matters for Biomedical MLOps

The transition from generative to agentic AI represents not merely an engineering upgrade but a fundamental shift in how biomedical research infrastructure operates. While generative AI produces content from prompts, agentic AI systems autonomously plan, execute, and adapt multi-step workflows — precisely the capability gap that separates current biomedical MLOps platforms (Level 2: pipeline automation) from their aspirational state (Level 4: autonomous governance-embedded pipelines).

We synthesize 62 papers spanning clinical agentic systems, MLOps automation, knowledge graph integration, and self-driving laboratories to identify where agentic capabilities deliver genuine value versus where they introduce unnecessary complexity. Applying the Markov novelty principle — that meaningful reviews must connect previously disconnected domains rather than catalogue existing work — we focus on three convergences that no single paper addresses:

1. **The deterministic-agentic boundary**: Where exactly in a biomedical ML pipeline should LLM-based agents intervene, and where should deterministic reproducibility be preserved?
2. **Regulatory-agentic alignment**: How do FDA QMSR, IEC 62304, and TRIPOD+AI compliance requirements *constrain* agent autonomy, and can those constraints become *assets*?
3. **The platform extensibility imperative**: How can a MONAI-first MLOps scaffold incorporate agentic capabilities without breaking the zero-config, config-driven philosophy that makes it usable by PhD researchers?

These questions are not abstract. They determine the architecture of NEUROVEX — a multiphoton biomedical segmentation MLOps platform where five Prefect flows orchestrate training, evaluation, post-training, deployment, and biostatistics, and where Pydantic AI agents already provide experiment summarization, drift triage, and figure narration within those flows.

---

## 2. The Landscape: What Agentic AI Actually Does in Biomedicine (2024–2026)

### 2.1 Clinical Agentic Systems: Beyond the Chatbot

The field has moved decisively from chatbot-style interactions to multi-step autonomous workflows. [Zhao et al. (2026). "An Agentic System for Rare Disease Diagnosis with Traceable Reasoning." *Nature*.](https://doi.org/10.1038/s41586-025-10097-9) demonstrated a multi-agent system that diagnoses rare diseases through traceable reasoning chains, achieving clinician-level accuracy — published in *Nature*, signaling that agentic AI has crossed the credibility threshold for clinical applications.

The convergence of tool-augmented LLM agents with domain-specific models has produced several compelling implementations. [Ferber et al. (2025). "Development and validation of an autonomous AI agent for clinical decision-making in oncology." *Nature Cancer* 6(8), 1337–1349.](https://www.nature.com/articles/s43018-025-00991-6) built an autonomous GPT-4-based agent integrating vision transformers for mutation detection, MedSAM for radiological segmentation, and OncoKB/PubMed search — improving decision accuracy from 30.3% (GPT-4 alone) to 87.2% through tool orchestration. This is not agentic for the sake of agentic; the tool composition is the mechanism that enables the performance gain.

For segmentation specifically, [Liu et al. (2026). "MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-Turn Agentic Reinforcement Learning." *arXiv:2602.03320*.](https://doi.org/10.48550/arXiv.2602.03320) demonstrated that reinforcement learning-guided multi-turn interactions with SAM-based models outperform single-shot segmentation. Meanwhile, [He et al. (2025). "RSAgent: Learning to Reason and Act for Text-Guided Segmentation via Multi-Turn Tool Invocations." *arXiv:2512.24023*.](https://doi.org/10.48550/arXiv.2512.24023) showed that text-guided agentic segmentation can achieve competitive results through tool invocation chains — reasoning about *what* to segment before *how* to segment it.

### 2.2 Multi-Agent Architectures: When Collaboration Adds Value

A recurring pattern across successful medical AI agents is the multi-agent multi-disciplinary team (MDT) architecture. [Kim et al. (2024). "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making." *NeurIPS 2024* (Oral).](https://arxiv.org/abs/2404.15155) demonstrated that automatically assigning collaboration structures — solo for simple tasks, multi-agent consensus for complex ones — outperforms static architectures. This adaptive topology is a critical insight: not every decision point needs a full agent committee.

[Zhang et al. (2026). "OMGs: A Multi-Agent System Supporting MDT Decision-Making across the Ovarian Tumour Care Continuum." *arXiv:2602.13793*.](https://doi.org/10.48550/arXiv.2602.13793) and [Chen et al. (2025). "Enhancing diagnostic capability with multi-agents conversational large language models." *npj Digital Medicine* 8, 139.](https://www.nature.com/articles/s41746-025-01550-0) both confirm that multi-agent consensus, inspired by clinical MDT discussions, improves diagnostic accuracy. However, [Zhao, H. et al. (2025). "ConfAgents: A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis." *arXiv:2508.04915*.](https://doi.org/10.48550/arXiv.2508.04915) introduces conformal prediction to gate agent cost — only escalating to expensive multi-agent consultations when single-agent confidence is insufficient. This uncertainty-guided escalation pattern is directly applicable to MLOps: most pipeline decisions are routine and need no LLM; only anomalous situations warrant agentic intervention.

### 2.3 The MONAI Ecosystem Agent

A pivotal development for MONAI-first platforms is [Nath et al. (2025). "VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge." *CVPR 2025*.](https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html), published as part of the official MONAI project (github.com/Project-MONAI/VLM-Radiology-Agent-Framework). VILA-M3 integrates domain expert models — including MONAI BRATS segmentation — as knowledge sources into a VLM, achieving ~9% improvement over Med-Gemini. This is the architectural proof that MONAI models can serve as tools within an agentic framework, validating NEUROVEX's design of wrapping MONAI-based adapters in Pydantic AI agents.

---

## 3. Agentic MLOps: The Platform Engineering Perspective

### 3.1 The Two-Tier Separation Is Not Unique — But Its Enforcement Is

The pattern of separating deterministic ML pipelines from LLM-assisted decision-making appears in several independent systems. [SelfAI (2025). "Building a Self-Training AI System with LLM Agents." *arXiv:2512.00403*.](https://arxiv.org/abs/2512.00403) implements three agents (User Agent, Cognitive Agent, Experiment Manager) that collaborate on hyperparameter optimization with fault-tolerant parallel training — nearly identical to NEUROVEX's Prefect (orchestration) + SkyPilot (compute) + Optuna (HPO) stack, but with LLM agents in the loop.

What distinguishes a biomedical platform from a general-purpose agentic ML framework is the *enforcement* of the deterministic boundary. [de Almeida et al. (2025). "Medical machine learning operations: a framework to facilitate clinical AI development and deployment in radiology." *European Radiology*.](https://link.springer.com/article/10.1007/s00330-025-11654-6) defines four MedMLOps pillars — availability, continuous monitoring/validation/(re)training, privacy, and ease of use — all of which require deterministic auditability. An agent that autonomously modifies a training configuration without a traceable decision log violates IEC 62304 Clause 8. The correct architecture is: agents *recommend*, flows *execute*, and every transition is logged to OpenLineage/Marquez.

### 3.2 Knowledge Graphs as Agent Memory

The integration of knowledge graphs with agentic systems has moved from theoretical to practical. [Ghafarollahi & Buehler (2025). "SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning." *Advanced Materials* 2413523.](https://onlinelibrary.wiley.com/doi/10.1002/adma.202413523) uses ontological knowledge graphs as the memory substrate for multi-agent hypothesis generation, with Ontologist, Scientist, and Critic agents collaborating through graph-structured knowledge. [Lu et al. (2025). "KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment." *NeurIPS 2025* (Spotlight).](https://arxiv.org/abs/2502.06472) demonstrates automated KG enrichment from PubMed, identifying 38,230 new entities from 1,200 articles with 83.1% correctness.

For NEUROVEX, which already maintains a 65-node Bayesian decision knowledge graph, the implication is clear: the KG is not just documentation — it is agent memory. A Pydantic AI agent reading the KG's posterior probabilities can make informed decisions about which model architecture to prioritize for the next training run, which loss function to try when drift is detected, or which TRIPOD compliance items need attention.

### 3.3 Autonomy Taxonomies: Where We Are and Where We're Going

[Luo et al. (2026). "Data Agents: Levels, State of the Art, and Open Problems." *arXiv:2602.04261*.](https://doi.org/10.48550/arXiv.2602.04261) proposes an L0–L5 autonomy taxonomy for data agents, paralleled by [Zheng et al. (2025). "From Automation to Autonomy: A Survey on LLMs in Scientific Discovery." *EMNLP 2025*.](https://aclanthology.org/2025.emnlp-main.895/) which introduces Tool/Analyst/Scientist levels. NEUROVEX currently operates at L2 (tool-assisted, human-orchestrated) with implemented Pydantic AI agents providing L3-adjacent capabilities for specific decision points. The target is not L5 (full autonomy) — which would be inappropriate for regulated medical devices — but L3 with *selective* L4 capabilities gated by uncertainty quantification.

[Moskalenko & Kharchenko (2024). "Resilience-aware MLOps for AI-based medical diagnostic systems." *Frontiers in Public Health* 12:1342937.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004236/) reinforces that resilience must be built into the MLOps pipeline stages, not bolted on afterward. The uncertainty calibration and graceful degradation patterns they describe map directly onto NEUROVEX's VLM calibration flow and conformal prediction infrastructure.

---

## 4. Implementable Agentic Extensions for NEUROVEX

Building on the literature synthesis, we identify concrete agentic features organized by implementation priority. The guiding principle: **agentic where the feedback loop is the value, not the language model** — five of seven proposed extensions use LLMs only for irreducible natural-language capabilities.

### 4.1 P1: Agentic Dashboard & Annotation UI (CopilotKit + AG-UI + WebMCP)

[Karim et al. (2025). "Transforming Data Annotation with AI Agents." *Future Internet* 17(8).](https://doi.org/10.3390/fi17080353) surveys agent-assisted annotation architectures. [Huang et al. (2025). "A pathologist–AI collaboration framework." *Nature Biomedical Engineering* 9, 455–470.](https://www.nature.com/articles/s41551-024-01223-5) (Nuclei.io) demonstrates that active-learning feedback loops between AI and human annotators reduce annotation time by 62% while improving accuracy by 72%. The pattern: agent suggests annotations, human corrects, corrections flow back to model retraining.

For NEUROVEX, CopilotKit + AG-UI provides the agentic UI layer, WebMCP provides tool connectivity, and Pydantic AI provides the decision logic. The dashboard becomes a collaborative workspace where researchers ask questions ("Which loss function performed best for topology preservation?"), receive KG-grounded answers, and can trigger new experiments from the UI.

### 4.2 P1: Observability Stubs (Sentry + PostHog)

Error tracking (Sentry) and product analytics (PostHog, SOC 2 Type II certified as of May 2025) are foundational for any agentic system that will eventually serve researchers beyond the original developers. These are not agentic features themselves, but they are prerequisites for agentic features — an agent cannot adaptively respond to user needs without telemetry on what users actually do.

### 4.3 P2: Uncertainty-Driven Adaptive Acquisition Agent (Flow 0b)

The highest-novelty extension per the Markov synthesis rule. [Ali et al. (2025). "Hybrid intelligence in medical image segmentation." *Scientific Reports* 15, 41200.](https://www.nature.com/articles/s41598-025-24990-w) (HybridMS) demonstrates uncertainty-driven clinician feedback for MedSAM, reducing annotation time by 82% for standard cases. [Kwon & Kim (2026). "Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift." *Scientific Reports*.](https://www.nature.com/articles/s41598-026-40637-w) provides the theoretical framework for conformal prediction-gated agent autonomy.

The proposed agent combines conformal prediction (already implemented in NEUROVEX) with restless bandit field selection for 2-photon microscopy. The agent decides *when enough data has been collected* for a given vascular morphology class — a decision that currently requires expert judgment and has no automated solution in the literature.

### 4.4 P2: Federated Learning for Multi-Site Vessel Data

Three frameworks compete:

| Framework | Ecosystem Fit | Medical Validation | Complexity | NEUROVEX Alignment |
|-----------|--------------|-------------------|------------|-------------------|
| **MONAI FL** | Native MONAI integration | Extensive (Pati et al., 2022) | Medium | **Highest** — TOP-1 principle |
| **NVIDIA FLARE** | NVIDIA ecosystem | FDA-cleared devices use it | High | High — GPU-centric |
| **Flower** | Framework-agnostic | Limited medical evidence | Low | Low — no MONAI integration |

MONAI FL ([monai.readthedocs.io/en/1.3.2/fl.html](https://monai.readthedocs.io/en/1.3.2/fl.html)) is the natural choice per NEUROVEX's TOP-1 principle (MONAI ecosystem extension). It provides `MonaiAlgo` for server-side and client-side training with built-in privacy and provenance tracking. NVIDIA FLARE is the underlying infrastructure that MONAI FL builds upon. Flower, while popular for research benchmarks, lacks MONAI-native integration and would require custom adapters for 3D medical segmentation.

### 4.5 P2: KG-Enrichment Agent for Literature Monitoring

KARMA's nine-agent pipeline for automated knowledge graph enrichment from PubMed demonstrates a practical pattern for keeping NEUROVEX's 65-node PRD knowledge graph current with new vascular biology literature. The agent would periodically scan for new publications related to vessel segmentation, loss functions, and calibration metrics, proposing updates to decision node posteriors when new evidence contradicts current assumptions.

### 4.6 Future Research: Self-Evolving Segmentation Agent

[Li, S. et al. (2025). "A co-evolving agentic AI system for medical imaging analysis." *arXiv:2509.20279*.](https://arxiv.org/abs/2509.20279) (TissueLab) demonstrates the co-evolving agent pattern: an LLM orchestrator + local tool factories + active learning loop that adapts to new imaging domains within minutes. For NEUROVEX, this means an agent that could propose new data augmentation strategies, loss function modifications, or model architecture changes based on failure mode analysis of current segmentation results — and test those proposals in the factorial experimental framework.

[Karpathy (2026). "Autoresearch." GitHub.](https://github.com/karpathy/autoresearch) provides a minimal 630-line implementation of this concept: an LLM agent that iteratively edits training scripts, runs time-boxed experiments (5-minute GPU cap), and commits improvements. In overnight runs, it completed 126 experiments autonomously. The NEUROVEX equivalent: a Pydantic AI agent that proposes YAML config modifications, submits them as SkyPilot jobs, evaluates results against the factorial baseline, and either commits the improvement or rolls back.

---

## 5. Regulatory-Agentic Alignment: Constraints as Assets

### 5.1 The AgentOps Framework

[Biswas et al. (2026). "Architecting AgentOps Needs CHANGE." *arXiv:2601.06456*.](https://doi.org/10.48550/arXiv.2601.06456) proposes the CHANGE framework for Agent Operations: Compliance, Human-in-the-Loop, Autonomy Levels, Networked Agents, Governance, and Evolution. For regulated medical devices, the Compliance and Governance pillars are not optional — they are the *primary* design constraints.

[Tzanis et al. (2026). "Agentic Systems in Radiology: Principles, Opportunities, Privacy Risks, Regulation, and Sustainability." *DIII* 107(1), 7–16.](https://doi.org/10.1016/j.diii.2025.10.002) explicitly maps regulatory requirements onto agentic architectures, identifying privacy risks and sustainability concerns that generic agentic frameworks overlook. For NEUROVEX, every agent decision must be logged to OpenLineage/Marquez for IEC 62304 traceability. This is not overhead — it is the mechanism that enables FDA-ready deployment.

### 5.2 Ethics and Governance

[Ghaffar Nia et al. (2025). "Ethical Perspectives on Deployment of Large Language Model Agents in Biomedicine." *AI and Ethics* 6(1):32.](https://doi.org/10.1007/s43681-025-00847-w) surveys the ethical landscape. [Cao et al. (2026). "From Agents to Governance: Essential AI Skills for Clinicians in the Large Language Model Era." *JMIR* 28(1):e86550.](https://doi.org/10.2196/86550) emphasizes that clinician AI literacy must advance alongside system capabilities. For NEUROVEX, this translates to a design principle: every agentic feature must have a deterministic fallback, and every agent recommendation must be presented with uncertainty bounds, not as authoritative decisions.

---

## 6. Discussion: Novel Synthesis for NEUROVEX

### 6.1 The Deterministic-Agentic Boundary Is a Feature, Not a Limitation

The literature consistently shows that the most successful medical agentic systems are NOT fully autonomous. MDAgents selectively activates multi-agent consultation. ConfAgents uses conformal prediction to gate agent cost. HybridMS triggers human-in-the-loop only when uncertainty exceeds a threshold. The pattern is: **deterministic by default, agentic by exception, human-in-the-loop by uncertainty**.

NEUROVEX's two-tier architecture (Prefect deterministic + Pydantic AI optional) is not a compromise — it is the correct architecture for regulated biomedical platforms. The five Prefect flows remain fully deterministic and reproducible. The three implemented Pydantic AI agents (experiment summarizer, drift triage, figure narrator) are read-only observers that produce text, not actions. Future agents (adaptive acquisition, config recommendation) will be gated by conformal prediction confidence — only acting autonomously when uncertainty is below a threshold, and deferring to human review otherwise.

### 6.2 The Knowledge Graph as Agent Ground Truth

No surveyed medical agentic system uses a Bayesian probabilistic knowledge graph as agent memory. This is NEUROVEX's potential contribution to the field: demonstrating that decision nodes with posterior probabilities, conditioned on experimental evidence, can serve as a principled substrate for agent decision-making. When the drift triage agent queries the KG node for `loss_function` (posterior: 0.85 for `cbdice_cldice`), it receives not just a recommendation but a calibrated confidence score grounded in factorial experimental results. This is fundamentally different from prompting an LLM with "what loss function should I use?" and hoping for a reasonable answer.

### 6.3 Critical Cautionary Finding: LLM Agents May Not Respond to Experimental Feedback

[EMNLP Findings (2025). "LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?"](https://aclanthology.org/2025.findings-emnlp.838/) reveals that LLM agents show **no sensitivity to experimental feedback** — random labels have no impact on their suggestions. This is a critical finding for any platform considering LLM-driven experiment design. The correct architecture is: use LLMs for priors and warmstarting (as in LLAMBO, Liu et al. 2024, ICLR), but rely on proper Bayesian optimization for the actual search loop. This validates NEUROVEX's Optuna+ASHA approach over pure LLM-driven HPO.

[Lopes et al. (2026). "Engineering AI Agents for Clinical Workflows." *IEEE/ACM CAIN '26*.](https://arxiv.org/abs/2602.00751) reports that 80% of engineering effort in their production clinical AI system ("Maria") was data engineering, stakeholder alignment, and governance — not model development. This finding reinforces NEUROVEX's emphasis on infrastructure (Docker, Prefect, MLflow, OpenLineage) over model sophistication.

### 6.4 Manuscript Discussion — Future Work Directions

For the NEUROVEX Nature Protocols manuscript discussion section, we propose the following future research directions:

1. **Uncertainty-gated agent autonomy**: Building on ConfAgents and conformal selective prediction, developing a framework where agent autonomy level is continuously calibrated by uncertainty estimates, enabling safe progressive automation of biomedical ML pipelines.

2. **Knowledge graph-grounded experiment design**: Using the probabilistic PRD as input to an experiment design agent that proposes factorial configurations based on unexplored regions of the model-loss-augmentation parameter space.

3. **Multi-site federated vessel segmentation via MONAI FL**: Extending the platform to support privacy-preserving multi-institutional training, with the factorial evaluation framework providing standardized comparison across sites.

4. **Co-evolving segmentation agents**: Adapting the TissueLab pattern to vessel segmentation, where an LLM-orchestrated agent continuously refines segmentation strategies based on failure mode analysis and expert feedback.

5. **Regulatory-native agentic MLOps**: Demonstrating that FDA QMSR, IEC 62304, and TRIPOD+AI compliance requirements can be encoded as machine-readable constraints within the knowledge graph, enabling automated compliance verification as a prerequisite for agent-driven model promotion.

---

## 7. Recommended GitHub Issues

### P1 Issues (Implement stubs/demos)

| Issue Title | Scope | Dependencies |
|-------------|-------|--------------|
| feat: CopilotKit/AG-UI agentic dashboard with WebMCP | Interactive researcher-AI dashboard, KG-grounded Q&A | Dashboard flow, KG |
| feat: Sentry error tracking stub | Error monitoring for all 5 Prefect flows | Docker Compose |
| feat: PostHog telemetry stub | Opt-in usage analytics for researcher workflows | Docker Compose |
| feat: Update PRD agent-framework decision for Pydantic AI | Sync stale PRD (still lists LangGraph as 0.45) with ADR-0007 | PRD files |

### P2 Issues (Research/deferred)

| Issue Title | Scope | Dependencies |
|-------------|-------|--------------|
| research: Federated learning comparison (MONAI FL vs NVIDIA FLARE vs Flower) | Structured comparison table, KG decision node | None |
| research: Adaptive acquisition agent (Flow 0b) with conformal bandit | Uncertainty-driven microscopy field selection | Conformal UQ |
| research: KG-enrichment agent for automated literature monitoring | KARMA-style pipeline for PRD decision node updates | KG, Pydantic AI |
| research: Self-evolving segmentation agent (TissueLab pattern) | LLM-orchestrated config search with factorial evaluation | All flows |
| research: Oneleet/CISO-assistant security compliance tooling | Security compliance automation for academic-to-commercial transition | CI/CD |
| research: CopilotKit agentic annotation interface | Agent-assisted vessel annotation with active learning | Data annotation module |

---

## 8. Academic Reference List

### Seed Papers (27)

1. [Abdollahi, A. et al. "The Next Paradigm in Medical AI: A Survey of Agentic AI in Biomedicine." *TechRxiv*.](https://www.authorea.com/doi/full/10.36227/techrxiv.176472621.18843904)
2. [Biswas, S. et al. (2026). "Architecting AgentOps Needs CHANGE." *arXiv:2601.06456*.](https://doi.org/10.48550/arXiv.2601.06456)
3. [Bluethgen, C. et al. (2025). "Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges." *arXiv:2510.09404*.](https://doi.org/10.48550/arXiv.2510.09404)
4. [Cao, W. et al. (2026). "From Agents to Governance." *JMIR* 28(1):e86550.](https://doi.org/10.2196/86550)
5. [Ghaffar Nia, N. et al. (2025). "Ethical Perspectives on LLM Agents in Biomedicine." *AI and Ethics* 6(1):32.](https://doi.org/10.1007/s43681-025-00847-w)
6. [Grzybowski, A. et al. "Agentic AI in Ophthalmology." *Acta Ophthalmologica*.](https://doi.org/10.1111/aos.70099)
7. [He, X. et al. (2025). "RSAgent." *arXiv:2512.24023*.](https://doi.org/10.48550/arXiv.2512.24023)
8. [Hu, M. et al. (2025). "A Survey of Scientific LLMs." *arXiv:2508.21148*.](https://doi.org/10.48550/arXiv.2508.21148)
9. [Hu, X. et al. "The Landscape of Medical Agents." *TechRxiv*.](https://www.authorea.com/doi/full/10.36227/techrxiv.176581395.56964766)
10. [Jin, R. et al. (2026). "STELLA: Biomedical World Model with Self-Evolving Agents." *bioRxiv*.](https://doi.org/10.1101/2025.07.01.662467)
11. [Karim, M.M. et al. (2025). "Transforming Data Annotation with AI Agents." *Future Internet* 17(8).](https://doi.org/10.3390/fi17080353)
12. [Li, B. et al. (2026). "Agentic AI and In Silico Team Science." *Nature Biotechnology*.](https://doi.org/10.1038/s41587-026-03035-1)
13. [Liu, S. et al. (2026). "MedSAM-Agent." *arXiv:2602.03320*.](https://doi.org/10.48550/arXiv.2602.03320)
14. [Luo, Y. et al. (2026). "Data Agents: Levels, State of the Art." *arXiv:2602.04261*.](https://doi.org/10.48550/arXiv.2602.04261)
15. [Qiu, J. et al. (2024). "LLM-Based Agentic Systems in Medicine." *Nature Machine Intelligence* 6(12).](https://doi.org/10.1038/s42256-024-00944-1)
16. [Rahman, M. et al. (2025). "LLM-Based Data Science Agents." *arXiv:2510.04023*.](https://doi.org/10.48550/arXiv.2510.04023)
17. [Rao, J. et al. (2026). "SciDataCopilot." *arXiv:2602.09132*.](https://doi.org/10.48550/arXiv.2602.09132)
18. [Ruan, J. et al. (2026). "AOrchestra." *arXiv:2602.03786*.](https://doi.org/10.48550/arXiv.2602.03786)
19. [Santos, A. et al. (2025). "Interactive Data Harmonization." *arXiv:2502.07132*.](https://arxiv.org/abs/2502.07132)
20. [Tzanis, E. et al. (2026). "Agentic Systems in Radiology: Principles, Privacy, Regulation." *DIII* 107(1):7–16.](https://doi.org/10.1016/j.diii.2025.10.002)
21. [Xiao, M. et al. (2025). "Knowledge-Driven Agentic Corpus Distillation." *arXiv:2504.19565*.](https://doi.org/10.48550/arXiv.2504.19565)
22. [Zhang, Y. et al. (2026). "OMGs: Multi-Agent MDT for Ovarian Tumours." *arXiv:2602.13793*.](https://doi.org/10.48550/arXiv.2602.13793)
23. [Zhao, H. et al. (2025). "ConfAgents." *arXiv:2508.04915*.](https://doi.org/10.48550/arXiv.2508.04915)
24. [Zhao, W. et al. (2026). "Agentic System for Rare Disease Diagnosis." *Nature*.](https://doi.org/10.1038/s41586-025-10097-9)
25. [Zhao, W. et al. "Medical AI Agents: Comprehensive Survey." *TechRxiv*.](https://www.authorea.com/doi/full/10.36227/techrxiv.176463029.99260745)
26. [Zheng, Q. et al. (2026). "End-to-End Agentic RAG." *arXiv:2508.15746*.](https://doi.org/10.48550/arXiv.2508.15746)
27. [Zhu, Y. et al. (2025). "A Survey of Data Agents." *arXiv:2510.23587*.](https://doi.org/10.48550/arXiv.2510.23587)

### Web-Discovered Papers (35)

28. [Ali, N.M. et al. (2025). "Hybrid intelligence in medical image segmentation." *Scientific Reports* 15, 41200.](https://www.nature.com/articles/s41598-025-24990-w)
29. [Chen, C. et al. (2025). "Evidence-based diagnostic reasoning with multi-agent copilot for human pathology." *arXiv:2506.20964*.](https://arxiv.org/abs/2506.20964)
30. [Chen, J. et al. (2025). "PathAgent: Toward Interpretable Analysis of WSI." *arXiv:2511.17052*.](https://arxiv.org/abs/2511.17052)
31. [Chen, X. et al. (2025). "Enhancing diagnostic capability with multi-agents conversational LLMs." *npj Digital Medicine* 8, 139.](https://www.nature.com/articles/s41746-025-01550-0)
32. [de Almeida, J.G. et al. (2025). "Medical machine learning operations (MedMLOps)." *European Radiology*.](https://link.springer.com/article/10.1007/s00330-025-11654-6)
33. [Fang, Y. et al. (2025). "A Comprehensive Survey of Self-Evolving AI Agents." *arXiv:2508.07407*.](https://arxiv.org/abs/2508.07407)
34. [Feng, J. et al. (2025). "M³Builder: Multi-Agent System for Automated ML in Medical Imaging." *arXiv:2502.20301*.](https://arxiv.org/abs/2502.20301)
35. [Ferber, D. et al. (2025). "Autonomous AI agent for clinical decision-making in oncology." *Nature Cancer* 6(8), 1337–1349.](https://www.nature.com/articles/s43018-025-00991-6)
36. [Ghafarollahi, A. & Buehler, M.J. (2025). "SciAgents." *Advanced Materials* 2413523.](https://onlinelibrary.wiley.com/doi/10.1002/adma.202413523)
37. [Huang, G. et al. (2025). "SurvAgent." *arXiv:2511.16635*.](https://arxiv.org/abs/2511.16635)
38. [Huang, K. et al. (2025). "Biomni: A General-Purpose Biomedical AI Agent." *bioRxiv:2025.05.30.656746*.](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1)
39. [Huang, Z. et al. (2025). "Nuclei.io: A pathologist–AI collaboration framework." *Nature Biomedical Engineering* 9, 455–470.](https://www.nature.com/articles/s41551-024-01223-5)
40. [Jiang, Z. et al. (2025). "AIDE: AI-Driven Exploration in the Space of Code." *arXiv:2502.13138*.](https://arxiv.org/abs/2502.13138)
41. [Karpathy, A. (2026). "Autoresearch." *GitHub*.](https://github.com/karpathy/autoresearch)
42. [Kim, Y. et al. (2024). "MDAgents: Adaptive Collaboration of LLMs for Medical Decision-Making." *NeurIPS 2024*.](https://arxiv.org/abs/2404.15155)
43. [Kwon, H. & Kim, D.J. (2026). "Conformal selective prediction with cost-aware deferral." *Scientific Reports*.](https://www.nature.com/articles/s41598-026-40637-w)
44. [Li, R. et al. (2025). "CARE-AD: multi-agent LLM framework for Alzheimer's prediction." *npj Digital Medicine*.](https://www.nature.com/articles/s41746-025-01940-4)
45. [Li, S. et al. (2025). "TissueLab: A co-evolving agentic AI system for medical imaging." *arXiv:2509.20279*.](https://arxiv.org/abs/2509.20279)
46. [Liu, T. et al. (2024). "LLAMBO: LLMs to Enhance Bayesian Optimization." *ICLR 2024*.](https://arxiv.org/abs/2402.03921)
47. [Lu, Y. et al. (2025). "KARMA: Multi-Agent LLMs for KG Enrichment." *NeurIPS 2025* (Spotlight).](https://arxiv.org/abs/2502.06472)
48. [Mehandru, N. et al. (2025). "BioAgents: Multi-agent systems in bioinformatics." *Scientific Reports* 15, 39036.](https://www.nature.com/articles/s41598-025-25919-z)
49. [Merkow, J. et al. (2024). "Scalable Drift Monitoring in Medical Imaging AI (MMC+)." *arXiv:2410.13174*.](https://arxiv.org/abs/2410.13174)
50. [MLXOps4Medic (2025). *IEEE Xplore*.](https://ieeexplore.ieee.org/document/11151803/)
51. [Moskalenko, V. & Kharchenko, V. (2024). "Resilience-aware MLOps for medical diagnostics." *Frontiers in Public Health* 12:1342937.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004236/)
52. [Nath, V. et al. (2025). "VILA-M3: Medical Expert Knowledge for VLMs." *CVPR 2025*.](https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html)
53. [Qu, Y. et al. (2025). "CRISPR-GPT for agentic automation of gene-editing experiments." *Nature Biomedical Engineering*.](https://www.nature.com/articles/s41551-025-01463-z)
54. [Ren, S. et al. (2025). "Towards Scientific Intelligence: LLM-based Scientific Agents." *arXiv:2503.24047*.](https://arxiv.org/abs/2503.24047)
55. [Roohani, Y. et al. (2024). "BioDiscoveryAgent." *arXiv:2405.17631*.](https://arxiv.org/abs/2405.17631)
56. [Schneider, J. (2025). "Generative to Agentic AI: Survey and Conceptualization." *arXiv:2504.18875*.](https://arxiv.org/abs/2504.18875)
57. [Seifrid, M. et al. (2024). "Self-Driving Laboratories for Chemistry and Materials Science." *Chemical Reviews* 124(16), 9633–9732.](https://pubs.acs.org/doi/10.1021/acs.chemrev.4c00055)
58. [SelfAI (2025). "Building a Self-Training AI System with LLM Agents." *arXiv:2512.00403*.](https://arxiv.org/abs/2512.00403)
59. [Wang, S. et al. (2025). "Pathology-CoT: Learning Visual Chain-of-Thought Agent." *arXiv:2510.04587*.](https://arxiv.org/abs/2510.04587)
60. [Wang, Z. et al. (2025). "MedAgent-Pro: Evidence-based Multi-modal Medical Diagnosis." *arXiv:2503.18968*.](https://arxiv.org/abs/2503.18968)
61. [Yang, C. et al. (2025). "LungNoduleAgent: Collaborative Multi-Agent System." *AAAI 2026, arXiv:2511.21042*.](https://arxiv.org/abs/2511.21042)
62. [Zheng, T. et al. (2025). "From Automation to Autonomy: LLMs in Scientific Discovery." *EMNLP 2025*.](https://aclanthology.org/2025.emnlp-main.895/)
63. [Lopes, C.L.V. et al. (2026). "Engineering AI Agents for Clinical Workflows: Architecture, MLOps, and Governance." *IEEE/ACM CAIN '26, arXiv:2602.00751*.](https://arxiv.org/abs/2602.00751)
64. [Qu, Y. et al. (2025). "CRISPR-GPT for agentic automation of gene-editing experiments." *Nature Biomedical Engineering*.](https://www.nature.com/articles/s41551-025-01463-z)
65. [MedSAM3 (2025). "Delving into Segment Anything with Medical Concepts." *arXiv:2511.19046*.](https://arxiv.org/abs/2511.19046)
66. [Hickman, R.J. et al. (2025). "Atlas: a brain for self-driving laboratories." *Digital Discovery*, RSC.](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00115j)
67. [Nouri, N. et al. (2026). "CellAtria: Agentic AI framework for scRNA-seq data analysis." *npj AI*.](https://www.nature.com/articles/s44387-025-00064-0)
68. [MITRE (2025). "SAFE-AI: A Framework for Securing AI-Enabled Systems." *MITRE MP250397*.](https://atlas.mitre.org/pdf-files/SAFEAI_Full_Report.pdf)
69. [EMNLP Findings (2025). "LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?"](https://aclanthology.org/2025.findings-emnlp.838/)
70. [Roohani, Y. et al. (2024). "BioDiscoveryAgent: AI Agent for Designing Genetic Perturbation Experiments." *arXiv:2405.17631*.](https://arxiv.org/abs/2405.17631)

---

## Appendix A: Original Prompt (Verbatim)

```
on how we could both 1) actually implement something agentic to this repo for the Pydantic AI
micro-orchestration beyond the planned AG-UI/CoPIlotKit/WebMCP for the dashboard and data
annotation module. [Full prompt preserved in git history — see initial commit of this file.]
```

## Appendix B: Alignment Decisions

- **Framing**: MLOps platform with agentic extensions (not agentic-first)
- **P1 issues**: AG-UI/CopilotKit + Sentry/PostHog stubs
- **P2 issues**: Federated learning comparison (FLARE vs MONAI FL vs Flower), adaptive acquisition agent, clinical reasoning
- **Scope**: Comprehensive survey (62 papers, seeds + web search finds)
- **Quality gate**: Markov novelty rule — synthesize across domains, no annotated bibliography
