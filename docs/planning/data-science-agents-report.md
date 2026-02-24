# Phase 18: Data Science Practices with Agents for Vascular Imaging MLOps

> **Date**: 2026-02-24
> **PRD Version**: 1.7.0 → 1.8.0
> **Seed Papers**: 11 (data science / agents cluster from vascular-tmp)
> **Web Research**: 6 topic searches (LLM health agents, RAG biomedical, human-AI co-creation, autonomy levels, LLM data prep, Bayesian UQ)

---

## Executive Summary

This report synthesises 11 seed papers and post-January 2025 web research on agentic data science for biomedical imaging. The central framework is the **L0-L5 data agent autonomy taxonomy** (Luo et al., 2026; Zhu et al., 2025) which maps current vascular segmentation MLOps at **L2 (tool-assisted, human-orchestrated)** and identifies **L3 (autonomous pipeline orchestration)** as the next critical transition. Four actionable angles emerge, the strongest being the **co-evolving segmentation agent** pattern (TissueLab architecture) where LLM orchestration + local tool factories + active learning from annotator corrections creates a self-improving pipeline without massive pre-labeled datasets.

---

## 1. Seed Paper Synthesis

### 1.1 Data Agent Taxonomies (L0-L5)

Two companion papers establish the definitive autonomy framework:

1. **Luo et al. (2026)** — "Data Agents: Levels, State of the Art, and Open Problems." SIGMOD 2026 tutorial. L0-L5 taxonomy modeled on SAE J3016 driving automation. L2→L3 transition (procedural executor → autonomous orchestrator) identified as the critical unsolved leap. Most production systems (Databricks, Snowflake Cortex) sit at L2-L3. arXiv:2602.04261v1.

2. **Zhu et al. (2025)** — "A Survey of Data Agents: Emerging Paradigm or Overstated Hype?" arXiv:2510.23587. Companion survey reviewing a broad sample of representative data agents across the literature. Current state: predominantly L2, with emerging proto-L3 systems. Bottlenecks toward L3: limited pipeline orchestration beyond predefined operators, incomplete lifecycle coverage, deficient strategic reasoning. Governance risks from terminological ambiguity between L1 and L3.

**Vascular MLOps mapping**: Current minivess-mlops pipeline = L2 (Hydra-zen configs + manual orchestration). Target = L3 (LangGraph agent autonomously orchestrating data prep → training → evaluation → deployment based on high-level performance goals).

### 1.2 LLM-Based Data Science Agents

3. **Rahman et al. (2025)** — "LLM-Based Data Science Agents: A Survey." arXiv:2510.04023. First lifecycle-aligned taxonomy mapping 45 systems across 6 stages. Critical finding: **>90% of surveyed agents lack explicit trust/safety mechanisms**. Deployment/monitoring (Stage 6) is the least-represented stage. Submitted to POMACS.

4. **Nie et al. (2026)** — "DSGym: A Holistic Framework for Evaluating and Training Data Science Agents." arXiv:2601.16344 (Stanford/Together AI). Revealed the **shortcut phenomenon**: on QRData, withholding data files caused only a 40.5% accuracy drop (agents retained ~60% performance via shortcuts); DAEval showed 86.8% drop (data-dependent); DiscoveryBench 44.4%. DSBio: 90 expert-derived bioinformatics tasks. Fine-tuned 4B model attains competitive performance with GPT-4o on standardised analysis benchmarks (mixed results: beats GPT-4o on DABStep-hard but falls behind on DAEval-Verified).

5. **Zhou et al. (2026)** — "Can LLMs Clean Up Your Mess? Survey of Application-Ready Data Preparation with LLMs." arXiv:2601.17058. Paradigm shift from rule-based to prompt-driven preparation. Convergence toward **hybrid LLM-ML approaches** generating executable programs rather than direct LLM inference. 20-30% of enterprise revenue lost to data inefficiencies.

### 1.3 RAG and Knowledge-Grounded Agents

6. **Khan et al. (2025)** — "RAG: What is There for Data Management Researchers?" SIGMOD Record, Vol. 54(4). Panel from ICDE 2025. Tensions: long-context LLMs (Gemini 1.5 Pro, 2M tokens) vs. traditional RAG chunking. Emerging variants: GraphRAG, KG-RAG, AgenticRAG, multimodal RAG. Vector index scalability: index-building stretches to hours at >1 billion vectors.

### 1.4 Interactive and Human-AI Data Science

7. **York et al. (2025)** — "Interactive Data Harmonization with LLM Agents." arXiv:2502.07132 (NYU). Harmonia system: ReAct-loop agent + bdi-kit Python library + Provenance DB for reproducibility. Demo: harmonizing 179-column endometrial cancer dataset to GDC vocabulary. Demonstrated LLM-corrected schema mapping errors interactively.

8. **Zhao et al. (2025)** — "GASF: Generative Agile Symbiosis Framework." SSRN 6150047. Reframes AI from tool to symbiont. Hypothesis Canvas replaces user stories. AI Trust Scorecard: W'(S) := F(S) AND (T(S) >= T_min). Identified "Phantom Done" anti-pattern where features pass technical tests but fail safety.

### 1.5 Specialised Data Science Methods

9. **Chakraborty (2026)** — "Progressive Bayesian Confidence Architectures for Cold-Start Health Analytics." arXiv:2601.03299. Three-tier framework (clues at 70% posterior mass, patterns at 85%, correlations at 95% CI). In a single N-of-1 experiment, mean time-to-first-insight: **5.3 +/- 1.2 days vs. 31.7 days** for fixed-threshold baselines (83% reduction). Monte Carlo across 100 datasets: 5.8 +/- 1.4 days. False discovery rate below 6% (5.9% at day 30; Monte Carlo mean 5.3%, 95% CI 3.1-7.8%).

10. **Falcao et al. (2025)** — "Evaluating LLM-based interoperability." arXiv:2510.23893, accepted ICSE 2026. DIRECT vs. CODEGEN strategies. qwen2.5-coder:32b dominated: DIRECT pass@1 >= 0.99 on simple tasks; CODEGEN achieved 0.75 on unit-conversion task where all models failed with DIRECT.

11. **Hoseini et al. (2025)** — "End-to-End ML with LLMs and Semantic Data Management." DEEM@SIGMOD 2025. Ontology-based semantic data lake (SEDAR/PLASMA) + LLM code generation for Chemistry 4.0. Only frontier models (GPT-4o, Gemini 2 Pro, Grok 3) succeeded on data wrangling; smaller models failed. ML modeling competitive with human baseline (MSE 0.0062 vs. 0.0056).

---

## 2. Web Research: Post-January 2025 Literature

### 2.1 Healthcare Agent Benchmarks

**MedAgentBench (NEJM AI, 2025)** — 300 clinically derived tasks across 10 categories from 100 synthetic patient profiles (>700,000 data elements). All agent systems outperformed non-agentic LLM baselines.

**7-Dimensional Agent Taxonomy (arXiv, Feb 2026)** — Survey of 49 studies finding **event-triggered activation absent in ~92%** and **drift detection absent in ~98%** of healthcare agent systems.

**npj Digital Medicine Benchmark (2026)** — Median improvement of **53 percentage points** in single-agent tool-calling studies, confirming tool use (not model size) as primary performance driver.

### 2.2 Biomedical RAG Architectures

**Medical Graph RAG (ACL 2025)** — Triple Graph Construction + U-Retrieval (top-down precision + bottom-up refinement). Outperforms vanilla RAG on evidence grounding with private medical data.

**MedRAG (WWW 2025)** — KG-integrated RAG achieving **71% improvement** in LLaMA-2 clinical QA via SPOKE biomedical knowledge engine.

**Self-Correcting Graph RAG (PMC, 2025)** — Agent retrieves graph-based clinical knowledge, evaluates its own output against evidence nodes, and re-queries when confidence is low.

### 2.3 Co-Evolving Medical Imaging Agents

**TissueLab (arXiv, Sep 2025)** — Open-source agentic system spanning pathology, radiology, and spatial omics. LLM orchestrator + local tool factories + active learning from clinician feedback. Achieves SOTA across diverse clinical tasks without massive datasets. **Closest architectural precedent for vascular MLOps agent.**

**mAIstro (GitHub, 2025)** — Multi-agent framework with specialist subagents for EDA, radiomics extraction, segmentation (nnU-Net + TotalSegmentator). Vendor-neutral (GPT-4, Claude, DeepSeek, LLaMA, Qwen).

### 2.4 Bayesian Uncertainty for Agents

**MedBayes-Lite (arXiv, Nov 2025)** — Plug-in Bayesian enhancement for clinical LLMs. Reduces overconfidence by **32-48%** on MedQA/PubMedQA/MIMIC-III. Prevents up to **41% of diagnostic errors** by flagging for human review. Parameter overhead <3%.

**UQ Survey (arXiv, Mar 2025)** — Conformal prediction identified as most practically deployable UQ paradigm for LLMs (distribution-free coverage guarantees).

---

## 3. Four Actionable Angles for MinIVess

### Angle 1: Co-Evolving Vascular Segmentation Agent — **NOVELTY: HIGH**

**The concept**: Adapt the TissueLab architecture (LLM orchestrator + local tool factories + active learning) specifically for vascular segmentation. The agent autonomously: (a) receives new imaging data, (b) selects appropriate preprocessing based on scanner metadata, (c) runs segmentation (MONAI/nnU-Net), (d) estimates uncertainty (conformal prediction), (e) routes low-confidence cases to radiologist review, (f) incorporates corrections as active learning signals.

**Why novel**: While TissueLab covers pathology/radiology/spatial omics, no implementation targets vascular-specific workflows where flow-dependent signals, contrast timing, and multi-scale vessel hierarchies (0.5-25mm) require domain-specific tool selection. The integration of Phase 16's conformal prediction acquisition agent with a co-evolving data science agent creates a novel feedback loop.

**L0-L5 positioning**: Moves minivess from L2 (human-orchestrated) to L3 (conditionally autonomous pipeline orchestration).

**Testable hypothesis**: A co-evolving agent with active learning reduces annotation effort by >50% for new vascular imaging sites while maintaining segmentation Dice within 2% of fully-supervised baselines.

**PRD integration**: Strengthen existing `agent_framework` node; add reference to TissueLab architecture.

### Angle 2: Graph-RAG for Vascular Protocol Knowledge — **NOVELTY: MEDIUM-HIGH**

**The concept**: Build a knowledge graph connecting vascular imaging protocols, scanner specifications, annotation guidelines, and segmentation quality records. Deploy Medical Graph RAG (ACL 2025) over this graph so that the vascular segmentation agent can ground its data decisions in retrievable evidence rather than hallucination.

**Why novel**: Graph-RAG exists for clinical QA (MedRAG: 71% improvement) and hepatology decision support. Applying it to the **data engineering layer** of a segmentation pipeline — where the agent retrieves protocol norms to validate preprocessing decisions — is novel. The self-correcting loop (evidence retrieval → confidence check → re-query) adds principled uncertainty handling.

**Testable hypothesis**: Graph-RAG-grounded preprocessing decisions reduce data preparation errors by >30% compared to LLM-only decisions without retrieval.

**PRD integration**: Strengthen existing `copilot_backend` node; add graph_rag option evidence.

### Angle 3: Bayesian Confidence-Flagged Human-in-the-Loop — **NOVELTY: MEDIUM**

**The concept**: Wrap the vascular segmentation quality-control agent with MedBayes-Lite-style Bayesian confidence calibration. The progressive framework (Chakraborty, 2026) provides tiered uncertainty communication: clues (70% posterior) → patterns (85%) → confirmed (95% CI). Low-confidence segmentations are routed to human review; high-confidence auto-approved.

**Why novel**: MedBayes-Lite exists for clinical LLMs; progressive Bayesian confidence exists for wearable health analytics. Combining both in a **segmentation quality triage pipeline** — where confidence level determines whether a case is auto-approved, flagged for review, or rejected — is the novel integration.

**Testable hypothesis**: Bayesian confidence routing reduces radiologist review load by >60% while maintaining label quality within 2% of full manual review.

**PRD integration**: Strengthen existing `uncertainty_quantification` node; connect to `agent_framework`.

### Angle 4: Harmonia-Style Interactive Harmonization — **NOVELTY: MEDIUM-LOW**

**The concept**: Adapt the Harmonia interactive harmonization system (York et al., 2025) for vascular imaging metadata. ReAct-loop agent + bdi-kit primitives + Provenance DB for reproducibility. The agent interactively resolves schema mismatches between multi-site vascular annotation conventions.

**Why lower novelty**: Harmonia exists and the adaptation to imaging metadata is engineering rather than research. However, the Provenance DB ensuring reproducibility of all harmonization decisions is a novel governance contribution.

**Testable hypothesis**: Interactive LLM harmonization resolves >80% of multi-site schema conflicts with <10 minutes of human interaction per dataset.

**PRD integration**: Connect to Phase 17's `data_harmonization_method` and `clinical_data_format` nodes.

---

## 4. PRD v1.8.0 Integration Recommendations

### 4.1 Updated Existing Nodes (3)

1. **`agent_framework`**: Add TissueLab and mAIstro as references. Increase `langgraph` prior from 0.40 to 0.45 based on LangGraph's alignment with the co-evolving agent pattern. Note L2→L3 transition as target.

2. **`copilot_backend`**: Add Medical Graph RAG and MedRAG references. Strengthen `graph_rag` option evidence (71% improvement in clinical QA).

3. **`uncertainty_quantification`**: Add MedBayes-Lite reference (32-48% overconfidence reduction). Add progressive Bayesian confidence reference (Chakraborty, 2026).

### 4.2 New Edges (3)

1. `agent_framework` → `data_harmonization_method` (moderate): Agent orchestrates harmonization pipeline selection
2. `agent_framework` → `clinical_data_format` (moderate): Agent manages FHIR-to-MEDS-OWL transformations
3. `uncertainty_quantification` → `agent_framework` (strong): UQ confidence levels gate agent autonomy (high-confidence = auto, low = human-in-loop)

### 4.3 New Bibliography Entries (12)

| citation_key | inline_citation | venue |
|---|---|---|
| luo2026dataagents | Luo et al. (2026) | SIGMOD 2026 Tutorial |
| zhu2025dataagents | Zhu et al. (2025) | arXiv:2510.23587 |
| rahman2025dsagents | Rahman et al. (2025) | arXiv:2510.04023 |
| nie2026dsgym | Nie et al. (2026) | arXiv:2601.16344 |
| zhou2026llmclean | Zhou et al. (2026) | arXiv:2601.17058 |
| khan2025rag | Khan et al. (2025) | SIGMOD Rec 54(4) |
| york2025harmonia | York et al. (2025) | arXiv:2502.07132 |
| chakraborty2026bayesian | Chakraborty (2026) | arXiv:2601.03299 |
| falcao2025interop | Falcao et al. (2025) | ICSE 2026 |
| hoseini2025e2eml | Hoseini et al. (2025) | DEEM@SIGMOD 2025 |
| tissuelab2025 | TissueLab (2025) | arXiv:2509.20279 |
| medbayes2025 | MedBayes-Lite (2025) | arXiv:2511.16625 |

---

## 5. Key References (Verified)

1. Luo, Y. et al. (2026). Data Agents: Levels, State of the Art, and Open Problems. SIGMOD 2026 Tutorial. arXiv:2602.04261.
2. Zhu, Y. et al. (2025). A Survey of Data Agents: Emerging Paradigm or Overstated Hype? arXiv:2510.23587.
3. Rahman, M. et al. (2025). LLM-Based Data Science Agents: A Survey. arXiv:2510.04023.
4. Nie, F. et al. (2026). DSGym: A Holistic Framework for Data Science Agents. arXiv:2601.16344.
5. Zhou, W. et al. (2026). Can LLMs Clean Up Your Mess? arXiv:2601.17058.
6. Khan, A. et al. (2025). RAG: What is There for Data Management Researchers? SIGMOD Record 54(4), 33-42.
7. York, A. et al. (2025). Interactive Data Harmonization with LLM Agents. arXiv:2502.07132.
8. Zhao, X. et al. (2025). GASF for Human-AI Co-Creation in SE. SSRN 6150047.
9. Chakraborty, R. (2026). Progressive Bayesian Confidence Architectures. arXiv:2601.03299.
10. Falcao, R. et al. (2025). Evaluating LLM-based interoperability. ICSE 2026. arXiv:2510.23893.
11. Hoseini, S. et al. (2025). End-to-End ML with LLMs and Semantic Data Management. DEEM@SIGMOD 2025.
12. TissueLab (2025). A Co-Evolving Agentic AI System for Medical Imaging. arXiv:2509.20279.
13. MedBayes-Lite (2025). Bayesian UQ for Safe Clinical Decision Support. arXiv:2511.16625.
14. Medical Graph RAG (2025). Evidence-Based Medical LLM via Graph RAG. ACL 2025.
15. MedRAG (2025). KG-Elicited Reasoning for Healthcare Copilot. WWW 2025.

---

## 6. Cross-References to Existing PRD

| Existing Node | Connection | Evidence |
|---|---|---|
| `agent_framework` | L2→L3 transition roadmap; TissueLab co-evolution | Luo et al. (2026), TissueLab (2025) |
| `copilot_backend` | Graph-RAG grounding for data decisions | Medical Graph RAG (2025), MedRAG (2025) |
| `uncertainty_quantification` | Bayesian confidence for agent autonomy gating | MedBayes-Lite (2025), Chakraborty (2026) |
| `data_harmonization_method` | Interactive LLM harmonization | York et al. (2025) Harmonia |
| `clinical_data_format` | Semantic data lake + LLM code generation | Hoseini et al. (2025) |
| `label_quality` | >90% agents lack trust/safety mechanisms | Rahman et al. (2025) |
| `acquisition_agent` | Co-evolving agent + conformal prediction loop | Phase 16 + TissueLab |
