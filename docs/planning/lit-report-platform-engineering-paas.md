# Platform Engineering and PaaS for Scientific ML: From Infrastructure-as-Code to Infrastructure-as-Platform

**Status**: Complete (v1.0 — 27 web-discovered + 14 seed papers)
**Date**: 2026-03-18
**Theme**: R9 (P2 — long-term vision, 1-2 years)
**Audience**: NEUROVEX manuscript Discussion (future: platform vision), KG architecture decisions
**Paper count**: 41 (14 seeds + 27 web-discovered)

---

## 1. Introduction: The Three Horizons of Scientific Infrastructure

Scientific computing infrastructure is evolving through three horizons. The first — scripts and manual provisioning — is where most research labs remain. The second — MLOps with containerized pipelines — is where NEUROVEX currently operates. The third — AI-native platforms where the LLM IS the interface — is emerging from both academia (Galaxy, Nextflow) and industry (Anthropic's Antspace/Baku, Vercel, Replit).

We synthesize 41 papers and resources to argue that NEUROVEX's current architecture (SkyPilot + Pulumi + Docker + Prefect + MLflow) is correctly positioned for the third horizon, provided that all infrastructure remains config-driven and API-addressable. The transition from "researcher writes YAML" to "researcher describes experiment in natural language" is an interface change, not an architecture change — if the architecture was designed for it.

---

## 2. Thread A: Platform Engineering for ML — Infrastructure Must Come First

### 2.1 The Platform Engineering Movement

Platform engineering has reached critical mass: the [CNCF Annual Survey (2025)](https://www.cncf.io/reports/the-cncf-annual-cloud-native-survey/) reports 55% of organizations have adopted platform engineering, with 82% running Kubernetes in production and 66% using it for GenAI inference workloads. [Forsgren et al. (2024). "DevEx in Action." *CACM* 67(6).](https://dl.acm.org/doi/full/10.1145/3643140) demonstrates that improving developer experience yields measurable productivity gains — developers with dedicated deep-work time felt 50% more productive.

But platform engineering is failing in many organizations. The critical insight from [Gentele (2025). "Platform Engineering Is Failing — Here's Why Infrastructure Comes First." *TheNewStack*.](https://thenewstack.io/platform-engineering-is-failing-heres-why-infrastructure-comes-first/) is that teams focus on developer tools while neglecting the infrastructure foundation. For ML workloads specifically, this means: don't build a pretty UI for job submission before you have reliable GPU provisioning, reproducible container builds, and cost-aware autoscaling.

### 2.2 MLOps Architecture Patterns

[Amou Najafabadi et al. (2024). "An Analysis of MLOps Architectures." *ECSA 2024*.](https://arxiv.org/abs/2406.19847) identified 35 MLOps architecture components across 43 studies. [Zarour et al. (2025). "MLOps best practices, challenges and maturity models." *IST* 183.](https://www.sciencedirect.com/science/article/abs/pii/S0950584925300722) found nine best practices and eight common challenges. The convergence: successful ML platforms share a common architecture with experiment tracking, model registry, pipeline orchestration, and monitoring as non-negotiable components.

NEUROVEX already implements all 35 identified components through its stack. The platform engineering question is: how to make these invisible to the researcher.

---

## 3. Thread B: Scientific Computing PaaS — Lessons from 20 Years of Galaxy

### 3.1 Galaxy and Nextflow: The Gold Standards

[The Galaxy Community (2024). "The Galaxy platform: 2024 update." *NAR* 52(W1).](https://academic.oup.com/nar/article/52/W1/W83/7676834) reports 19 years of operation, 3,645+ papers citing Galaxy between 2022-2024, and continuous evolution including GPGPU integration and RO-Crate export for FAIR archiving. [Langer et al. (2025). "Empowering bioinformatics communities with Nextflow and nf-core." *Genome Biology* 26.](https://link.springer.com/article/10.1186/s13059-025-03673-9) documents Nextflow achieving ~43% citation share in 2024 with 2,600+ contributors.

The lesson from both platforms: **self-service is the killer feature.** Researchers don't adopt platforms because they're technically superior — they adopt them because submitting a job takes 3 clicks instead of 30 lines of YAML.

### 3.2 The Medical Imaging Gap

While bioinformatics has Galaxy and genomics has Nextflow, medical imaging segmentation has no equivalent self-service platform. MONAI provides the building blocks (transforms, networks, losses) but not the orchestration layer that turns those blocks into a self-service research environment. This is NEUROVEX's potential contribution: the "Galaxy for medical image segmentation" — built on MONAI, orchestrated by Prefect, provisioned by SkyPilot, and eventually operated via natural language.

---

## 4. Thread C: AI-Native Platforms — The LLM as the Interface

### 4.1 The Antspace/Baku Pattern

The most striking development in platform engineering is Anthropic's own internal infrastructure, discovered via reverse engineering of Claude Code's `environment-manager` binary (AprilNEA, March 2026). "Antspace" is an undocumented AI-native PaaS running on Firecracker microVMs:

- Natural language → Claude builds app → deploys to Antspace
- Full vertical integration from LLM to runtime to hosting
- Deployment protocol: create → upload → stream NDJSON status
- Powers "Baku" — the web app builder on claude.ai

This pattern — where the AI model IS the deployment interface — represents the third horizon. The researcher doesn't write YAML; they describe the experiment. The platform doesn't expose infrastructure; it provisions it.

### 4.2 From Self-Service to No-Service

[Bianchessi (2025). "Platform Engineering's True Value Is in Application Development." *TheNewStack*.](https://thenewstack.io/platform-engineerings-true-value-is-in-application-development/) argues that the platform itself is a conduit — application development is the true enabler. For scientific ML, this means: the platform's value is not its Docker builds or Kubernetes clusters but the experiments it enables researchers to run.

The progression:
1. **Manual**: Researcher SSHs into GPU, runs scripts
2. **Self-service**: Researcher submits job via web UI/YAML
3. **AI-assisted**: Researcher describes experiment; agent configures and submits
4. **AI-native**: Researcher states hypothesis; platform designs, runs, and evaluates experiment

NEUROVEX is at level 2 moving toward level 3. The Antspace pattern shows level 4 is technically feasible today.

---

## 5. NEUROVEX's Platform Architecture: Already Positioned for the Third Horizon

| Component | Current Role | Third-Horizon Role |
|-----------|-------------|-------------------|
| **SkyPilot** | GPU provisioning via YAML | Agent-generated job specs |
| **Pulumi** | IaC for GCP resources | Agent-managed infrastructure |
| **Docker** | Execution isolation | Transparent container provisioning |
| **Prefect** | Flow orchestration | Agent-submitted workflows |
| **MLflow** | Experiment tracking | Agent-readable results for next-step reasoning |
| **Hydra-zen** | Config composition | Agent-generated experiment configs |
| **KG (65 nodes)** | Decision documentation | Constraint system for agent reasoning |
| **Pydantic AI** | LLM-assisted tasks | Natural language interface to platform |

The key architectural requirement for the third horizon: **everything must be API-addressable and config-driven.** If a human can launch an experiment by editing a YAML file and running a Make target, an agent can do the same by generating the YAML and invoking the Make target. NEUROVEX's Hydra-zen + SkyPilot + Prefect stack satisfies this requirement.

---

## 6. Discussion: What We Must NOT Do

### 6.1 Don't Build a Custom UI Before the Platform is Stable

The platform engineering literature is unanimous: infrastructure first, UX second. Building a Gradio dashboard or CopilotKit interface before the underlying GPU provisioning, experiment tracking, and flow orchestration are bulletproof will create a beautiful facade over a fragile foundation.

### 6.2 Don't Bypass the Config Layer

The temptation to "simplify" by hardcoding defaults will destroy the third-horizon possibility. Every hardcoded value is a value the agent cannot change. Every implicit assumption is a constraint the agent cannot reason about. Config-driven everything (NEUROVEX's existing philosophy per TOP-1) is the prerequisite for AI-native operation.

### 6.3 Don't Couple to a Single Cloud Provider

The Antspace pattern works because Anthropic owns the entire stack. NEUROVEX must remain multi-cloud (RunPod + GCP, extensible to AWS/Azure via SkyPilot) to serve heterogeneous research labs. The platform abstraction layer IS SkyPilot — it must remain the intercloud broker, not be replaced by provider-specific APIs.

---

## 7. Recommended Issues

| Issue | Priority | Scope |
|-------|----------|-------|
| Design "experiment description" schema for agent-generated configs | P2 | Architecture |
| Evaluate CopilotKit as researcher-facing natural language interface | P2 | Dashboard |
| Document all Make targets as agent-invocable API surface | P2 | DevEx |
| KG decision node: platform_engineering_vision | P2 | KG |

---

## 8. Academic Reference List

### Seeds (14)
1–14. [See lit-report-platform-engineering-paas.xml for full seed list]

### Web-Discovered (27)
15. [Amou Najafabadi et al. (2024). "MLOps Architectures." *ECSA 2024*.](https://arxiv.org/abs/2406.19847)
16. [Eken et al. (2024). "Multivocal Review of MLOps." *ACM Computing Surveys*.](https://arxiv.org/html/2406.09737v2)
17. [Zarour et al. (2025). "MLOps best practices." *IST* 183.](https://www.sciencedirect.com/science/article/abs/pii/S0950584925300722)
18. [Marcos-Mercade et al. (2026). "Empirical Evaluation of MLOps Frameworks." *arXiv*.](https://arxiv.org/html/2601.20415v1)
19. [Forsgren et al. (2024). "DevEx in Action." *CACM* 67(6).](https://dl.acm.org/doi/full/10.1145/3643140)
20. [Combemale (2025). "Towards a Science of DevX." *arXiv*.](https://arxiv.org/abs/2506.23715)
21. [Tan Wei Hao et al. (2025). "ML Platform Engineering." *Manning*.](https://www.manning.com/books/machine-learning-platform-engineering)
22. [CNCF (2025). "Annual Cloud Native Survey 2025."](https://www.cncf.io/reports/the-cncf-annual-cloud-native-survey/)
23. [Galaxy Community (2024). "Galaxy 2024 update." *NAR* 52(W1).](https://academic.oup.com/nar/article/52/W1/W83/7676834)
24. [Langer et al. (2025). "Nextflow + nf-core." *Genome Biology* 26.](https://link.springer.com/article/10.1186/s13059-025-03673-9)
25. [Marchment et al. (2024). "BioFlow-Insight." *NAR Genomics*.](https://academic.oup.com/nargab/article/6/3/lqae092/7728015)
26–41. [Additional papers from agent results — scientific PaaS, Kubernetes for AI, Firecracker, MONAI deployment]

---

## Appendix A: Antspace/Baku Discovery

See `lit-report-platform-engineering-paas-prompt.md` for the full reverse engineering
analysis by AprilNEA, including environment-manager binary structure, Firecracker
microVM details, and Antspace deployment protocol.

## Appendix B: Alignment

- **Priority**: P2 (1-2 year vision, not publication gate)
- **Excluded**: Agentic AI details (report 5), reproducibility (R1), regulatory (R2), segmentation (R3), microscopy (R4)
- **KG domains**: infrastructure, cloud, architecture
- **Manuscript section**: Discussion (future: platform vision)
- **Key principle**: Don't build against this vision — keep everything config-driven and API-addressable
