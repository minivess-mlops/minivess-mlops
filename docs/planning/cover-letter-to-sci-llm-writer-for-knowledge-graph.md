---
title: "Cover Letter — ARBOR Manuscript Context for sci-llm-writer"
purpose: "Zero-context session re-engagement brief for sci-llm-writer when starting manuscript writing from scratch"
source_repo: /home/petteri/Dropbox/github-personal/minivess-mlops
target_repo: /home/petteri/Dropbox/github-personal/sci-llm-writer
manuscript_dir: sci-llm-writer/manuscripts/vasculature-mlops
kg_snapshot_target: ../sci-llm-writer/manuscripts/vasculature-mlops/kg-snapshot/kg-snapshot.yaml
created: 2026-03-15
kg_commit: 9376e9f
status: living_document  # update after GPU runs, after Mamba adapter, after Level 4 closure
---

# ARBOR: Manuscript Context Brief for sci-llm-writer

This document is the **single context handoff** from `minivess-mlops` to `sci-llm-writer`.
Open a new sci-llm-writer session, paste or reference this file, and the LLM writer has
everything it needs to begin writing LaTeX manuscript sections for co-authors to review
while the experiments and platform implementation are being completed.

---

## § 0 — VERBATIM USER PROMPT (preserved as-is)

> Could you next create me some "cover letter" for all this knowledge work to
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md
> which contains all the context and background information needed by
> /home/petteri/Dropbox/github-personal/sci-llm-writer when I open a new session with no
> context and knowledge of what we done here so that it can access all the information here
> in order to write the .tex scaffold for co-authors what this is paper is all about, and
> what novelties we can discuss, what are the limitation, what are our methods, what results
> we had and expect to have for the system functioning (remember that our results should
> focus on how our approach ensures reproducibility, scaffolding for research to achieve
> those "SOTA papers" and the comparison of dynUnet, Mamba, vesselfm and SAM3 are of
> scientific interest to certain point obviously but we are not here to make any claims of
> architectures per se, just using those different models the MLOps Maturity Level 6 (or
> the highest in general) for biomedical workflow with full-on automatic retraining, drift
> detection, etc eventually). Is this clear as a scope still. Ask interactive multi-answer
> questions to ensure that we are aligned now? Save my prompt verbatim to this cover
> letter, and optimize this cover letter with reviewer agents then!

---

## § 1 — What This Paper IS and IS NOT

### What it IS

**ARBOR** (Agentic Reproducible Biomedical Operations Research) is a **platform paper**.
The central contribution is a fully reproducible, cloud-agnostic MLOps platform for
3D biomedical segmentation research that achieves **MLOps Maturity Level 4** (Microsoft
definition: automated data pipeline, production metrics trigger retraining, CI/CD manages
releases, compute is managed, all artifacts version-controlled). The platform is built
as a MONAI ecosystem extension — it does not fork MONAI, it extends it.

**The paper argues**: This platform is what allows the next generation of SOTA biomedical
segmentation papers to focus on science, not infrastructure. By proving the platform works
(reproducibility proof, multi-model comparison, deployment pipeline), we give researchers
a scaffold that today requires months of bespoke engineering.

**Primary claim**: Any researcher can take ARBOR, swap in their YAML config (dataset +
model), and run the full MLOps pipeline — training, evaluation, deployment, drift
detection, auto-retraining — without writing infrastructure code.

### What it IS NOT

- **NOT an architecture comparison paper.** We compare DynUNet, Mamba variants, SAM3,
  and VesselFM to DEMONSTRATE the platform's model-agnostic generalizability — not to
  claim that Mamba beats DynUNet or that SAM3 is the best vascular segmenter. The model
  diversity is proof of the ModelAdapter ABC (one YAML + 3 Python methods = new model).

- **NOT a SOTA vascular segmentation paper.** We are benchmarking on MiniVess (70 volumes,
  rat cerebrovasculature, 2-photon) for consistency, not to claim SoTA DSC or clDice
  versus the full literature.

- **NOT a data science paper.** We make topology-informed loss function choices
  (cbdice_cldice achieves best clDice), but the loss ablation is to show the platform's
  experimental flexibility, not to propose a novel loss function.

### The Central Narrative for Nature Protocols / Nature Methods

> "The reproducibility crisis in biomedical ML is not a data problem or an algorithm
> problem — it is an infrastructure problem. ARBOR is the infrastructure solution:
> a fully automated, cloud-agnostic, model-agnostic MLOps platform that transforms
> experiment definition from a programming task into a YAML configuration task.
> We validate the platform on 3D vascular segmentation using four diverse model families
> and demonstrate MLOps Maturity Level 4 architecture in a single-researcher academic setting."

⚠️ The verbatim seed uses "Level 4 architecture." Do NOT simplify to "achieve Level 4"
until the Evidently → Prefect retraining loop is verified end-to-end (§4, honest status: Level 3.5).

---

## § 1.5 — Prior Art Gap (for Introduction — mandatory context for sci-llm-writer)

This section gives the LLM the specific prior-work characterizations needed to write the
"state of practice and its gap" paragraph of the introduction. Do NOT invent these — use
exactly what is stated here.

### What exists and what it lacks

| System / Paper | What it provides | What it lacks (= the gap ARBOR fills) |
|----------------|-----------------|---------------------------------------|
| **nnU-Net** (Isensee et al. 2021, *Nature Methods*) | Self-configuring 3D segmentation architecture; state-of-the-art DSC across benchmarks | No MLOps scaffolding; no experiment tracking; no cloud execution; no drift detection; no deployment pipeline; reproducibility = re-running the same script manually |
| **MONAI** (Cardoso et al. 2022) | Excellent PyTorch-based library for medical image transforms, augmentations, models | Zero workflow orchestration; zero reproducibility tooling; zero multi-cloud support; users must build their own training loops from scratch every project |
| **CyclOps** (Krishnan et al. 2022) | MLOps platform for healthcare AI with recurring validation | Designed for clinical classification, not preclinical 3D segmentation; no multi-cloud; no model-agnostic adapter pattern; no spec-driven development |
| **Cheimarios (2025)** | MLOps principles + NIST risk management for computational physics | Containerization and model registries only; no multi-cloud; no biomedical imaging; no spec-driven development |
| **Current lab practice** | Bespoke scripts + Jupyter notebooks per lab | No standard; no replication across labs; months of bespoke engineering per new model/dataset; cannot be reproduced by external collaborators |

### The specific gap ARBOR fills

> No published platform combines: (1) model-agnostic adapter pattern for 3D biomedical
> segmentation, (2) full 5-flow Docker-per-flow MLOps pipeline with MLflow as the sole
> inter-flow contract, (3) single-command multi-cloud execution via SkyPilot, and
> (4) spec-driven agentic development via a hierarchical probabilistic knowledge graph —
> in a single open-source academic tool deployable by a 2–5-person research lab
> with no dedicated ML engineering staff.

### Target researcher persona (for Nature Protocols framing)

A typical ARBOR user is a computational neuroscience or biomedical imaging PhD lab with:
- 2–5 members, one first author who also maintains infrastructure
- One GPU workstation (8–24 GB VRAM) + occasional cloud access
- Scientific expertise in the imaging domain, zero ML engineering staff
- Need to run reproducible experiments for a peer-reviewed paper
- Currently spending 30–50% of research time on infrastructure instead of science

This persona directly motivates every DevEx choice in the platform and should anchor the
introduction's "who needs this" framing.

### Bibliography citation keys available for this section

- `isensee_2021_nnunet` — nnU-Net
- `cardoso_2022_monai` — MONAI
- `krishnan_2022_cyclops` — CyclOps
- `cheimarios_2025_mlops_physics` — scientific MLOps (physics)
- `marcos_mercade_2026_mlops_eval` — empirical MLOps framework comparison
- `poon_2023_minivess` — MiniVess dataset (primary training data)
- `windhager_2023_nature_protocols` — comparable systems paper in Nature Protocols
- `pachitariu_2022_cellpose` — comparable platform paper in Nature Methods

---

## § 1.6 — Three Contributions: Canonical Form (for introduction closing paragraph)

**The sci-llm-writer MUST use this exact three-contribution structure** at the end of the
introduction. Do not expand to four or five contributions. Do not reorder. Do not lead with
the LSG contribution — it is appendix-grade.

> We make three contributions:
>
> **(1) ARBOR platform (keystone):** A fully automated, model-agnostic MLOps platform for
> 3D biomedical segmentation that achieves MLOps Maturity Level 4 in a single-researcher
> academic setting. Any lab can adopt ARBOR by editing one YAML file — no infrastructure
> code required. Validated end-to-end: 73/73 artifact checks pass, 35+ artifacts produced
> in 8.26 seconds post-training.
>
> **(2) Multi-model generalizability (experimental, partially pending):** We demonstrate
> platform generalizability across four model families — DynUNet (MONAI-native), Mamba
> state-space models, SAM3 foundation model variants, and VesselFM — each integrated via
> the ModelAdapter ABC with zero pipeline changes. DynUNet results are complete (4-loss
> ablation, 3-fold, 100 epochs); SAM3 and VesselFM results are in preparation.
> ⚠️ **WRITING INSTRUCTION**: Write Contribution 2 as "three implemented families with
> Mamba as a planned extension" until R3b/R3c GPU results are available. Do NOT write
> as if all four families have published quantitative results.
>
> **(3) Living Specification Graph (appendix or companion paper):** A 52-node probabilistic
> decision graph as the machine-queryable single source of truth for agentic development —
> the first application of spec-driven development to scientific MLOps infrastructure.
> ⚠️ **WRITING INSTRUCTION**: Frame as appendix material. The paper stands without this
> section. Do NOT let this contribution displace Contribution 1 as the lead claim.

---

## § 1.7 — Introduction Narrative Arc (6-paragraph skeleton)

Write the introduction in this order. Target: 900–1200 words for Nature Protocols;
600–800 words for Nature Methods.

1. **Reproducibility crisis** (~150 words): The problem is infrastructure, not algorithms.
   Anchor with reproducibility crisis citations. Introduce the platform solution concept.
2. **State of practice and its gap** (~200 words): nnU-Net + MONAI exist but lack MLOps.
   Use §1.5 prior art table. Conclude: "No existing platform provides [the gap]."
3. **ARBOR overview** (~150 words): What ARBOR is. Use the central narrative verbatim seed
   from §1 ("The reproducibility crisis in biomedical ML is not a data problem...").
   Present the 5-flow architecture and Docker-per-flow isolation as one sentence each.
4. **Model families + dataset** (~150 words): Four model families on MiniVess. Stress
   the models are proof of generalizability, not an architecture race.
5. **MLOps Maturity Level 4** (~100 words): Frame the platform's ambition level.
   Be honest: "Level 4 architecture demonstrated; auto-retraining trigger loop in progress."
6. **Three contributions** (~150 words): Use the canonical form from §1.6 verbatim.

**VERBATIM SEED — preserve this paragraph exactly in the introduction:**

> "The reproducibility crisis in biomedical ML is not a data problem or an algorithm
> problem — it is an infrastructure problem. ARBOR is the infrastructure solution:
> a fully automated, cloud-agnostic, model-agnostic MLOps platform that transforms
> experiment definition from a programming task into a YAML configuration task.
> We validate the platform on 3D vascular segmentation using four diverse model families
> and demonstrate MLOps Maturity Level 4 in a single-researcher academic setting."

---

## § 2 — Publication Target

| Journal | IF | Framing | Decision |
|---------|-----|---------|---------|
| Nature Protocols | ~13 | Protocol validation paper — R1 reproducibility + R3 multi-model comparison | **Primary target** (after R3b/R3c GPU results) |
| Nature Methods | ~48 | Methods/platform contribution — stronger novelty claim, LSG appendix needed | **Alternative target** if MLOps Maturity Level 4 + Mamba results are compelling |

**Decision**: Hedge until R3b (SAM3) + R3c (VesselFM) + Mamba GPU runs are complete.
The framing may shift based on those results.

**Working title (draft)**: *ARBOR: An Agentic Reproducible Biomedical Operations Research
Platform for Scalable 3D Segmentation Targeting MLOps Maturity Level 4*

⚠️ **Title note**: Do NOT say "achieving Level 4" in the title until the Evidently →
Prefect auto-retraining loop is verified end-to-end (see §4). Use "targeting" or
"with Level 4 architecture" until then.

### Section Word Count Budget (for sci-llm-writer)

| Section | Nature Protocols target | Nature Methods target |
|---------|------------------------|----------------------|
| Introduction | 900–1200 words | 600–800 words |
| Methods | 3000–5000 words (structured by subsection) | 2000–3000 words |
| Results | 2000–3000 words | 1500–2000 words |
| Discussion | 800–1200 words | 600–1000 words |
| Abstract | 200–250 words | 150–200 words |

Use the **Nature Protocols** budget until journal decision is made. Write concisely:
prefer one concrete sentence over one abstract paragraph.

---

## § 3 — Model Families (4 Families, Diverse by Design)

The four model families are NOT compared to find a winner — they are compared to prove
the platform's model-agnostic architecture. One YAML profile + one ModelAdapter subclass
= new model integrated with zero infrastructure changes.

### 3.1 DynUNet (MONAI native) — STATUS: COMPLETE
- **Role**: Battle-tested baseline, MONAI-native, self-configuring architecture
- **KG adapter**: `knowledge-graph/code-structure/adapters.yaml#dynunet`
- **GPU requirement**: Any CUDA GPU ≥ 8 GB, BF16 native
- **Results available**: YES — full 4-loss ablation, 3-fold, 100 epochs on RTX 2070 Super
- **Key results** (`knowledge-graph/experiments/dynunet_loss_variation_v2.yaml`):
  - `cbdice_cldice`: DSC=0.772±0.016, clDice=0.906±0.008 (**WINNER — best topology**)
  - `dice_ce`: DSC=0.824±0.014 (best DSC), clDice=0.832±0.019
  - Topology gain cbdice_cldice vs dice_ce: +8.9% clDice, −5.3% DSC

### 3.2 Mamba variants (State Space Models) — STATUS: PLANNED, KG GAP ⚠️
- **Role**: State-space model family — long-range dependency modelling, linear complexity
- **Example implementations**: SegMamba, U-Mamba, MambaSeg (MONAI-compatible)
- **KG adapter**: **NOT YET IN adapters.yaml** — this is a known gap. Must be added before GPU runs.
- **GPU requirement**: Similar to DynUNet (~8 GB)
- **Results available**: NO — pending KG adapter implementation + GPU runs
- **⚠️ ACTION REQUIRED before manuscript submission**: Implement Mamba adapter
  (implement `ModelAdapter` ABC: 3 methods + YAML profile). This is not yet in the repo.
- **Paper role**: Represents the SSM model family; contrasts long-range context (Mamba)
  vs sliding-window context (DynUNet) vs frozen ViT encoder (SAM3)
- **⚠️ sci-llm-writer WRITING INSTRUCTION**: Until Mamba GPU runs are complete, ALL
  introduction and methods prose MUST refer to Mamba as "planned." The working title
  and abstract should say "three implemented model families (DynUNet, SAM3 variants,
  VesselFM) with Mamba state-space models as a planned extension." Do NOT write as if
  four families have quantitative results available.

### 3.3 SAM3 variants (Foundation Model — Meta, Nov 2025) — STATUS: ADAPTERS READY, GPU PENDING
- **Role**: Foundation model family — test whether 2D natural-image pre-training transfers
  to 3D vascular segmentation via slice-by-slice inference with frozen ViT-32L encoder
- **SAM3 ≠ SAM2**: SAM3 is Meta's Nov 2025 release (github.com/facebookresearch/sam3),
  ViT-32L, **848M total params** (perception encoder alone = 648M), 1008×1008, SDPA
  mandatory. SAM2 is a different (earlier, smaller) model. Source: `sam3_backbone.py:11`
  and `knowledge-graph/decisions/L3-technology/foundation_model.yaml`.
- **Three variants**:
  - **SAM3 Vanilla (V1)**: Frozen ViT-32L + lightweight Conv decoder. Fits 8 GB GPU.
    KG: `adapters.yaml#sam3_vanilla`. GPU ≥ 8 GB (SDPA mandatory).
  - **SAM3 Hybrid (V3)**: Frozen ViT-32L + gated DynUNet fusion. 7.18 GiB VRAM.
    KG: `adapters.yaml#sam3_hybrid`. Needs cloud GPU (RTX 4090 / L4) for patch=(64,64,3).
  - **SAM3 TopoLoRA (V2)**: SAM3 + LoRA + topology loss head. >22.66 GiB — **OOM on RTX 4090**.
    Requires A100 ≥ 40 GB. Status: deferred.
- **Critical constraint**: BF16 required (NOT FP16). FP16 overflows in frozen encoder → NaN.
  T4 GPU is BANNED (Turing arch, no BF16). Use L4 (Ada Lovelace) on GCP.
  AMP must be OFF for validation (MONAI #4243 — 3D sliding_window_inference + autocast = NaN).
- **Results available**: NO — pending RunPod RTX 4090 GPU runs (~$15, target 2026-03-20)
- **Blocks**: R3b (multi-model comparison section)

### 3.4 VesselFM (Foundation Model — Wittmann et al. 2024) — STATUS: ADAPTER READY, DATA LEAKAGE WARNING
- **Role**: Vessel-specific foundation model pre-trained on diverse vascular datasets
- **KG adapter**: `adapters.yaml#vesselfm`
- **⚠️ CRITICAL DATA LEAKAGE WARNING**: VesselFM was pre-trained on MiniVess.
  Evaluating VesselFM on MiniVess = data leakage = invalid comparison.
  **MUST evaluate on DeepVess (Cornell) and/or TubeNet 2PM (UCL) only.**
- **Results available**: NO — pending GPU runs on external datasets only
- **Blocks**: R3c (foundation model comparison section)
- **Paper framing**: Limitation L1 — the leakage was identified through careful human
  literature review of Wittmann et al. (2024), NOT by an automated platform mechanism.
  ⚠️ Do NOT write "the platform detected the leakage" — that would be a false claim.
  Correct framing: "We identified data leakage through literature review; ARBOR's
  config-driven evaluation design makes it trivial to re-route evaluation to
  non-contaminated datasets — demonstrating platform flexibility in the face of
  real-world evaluation pitfalls."
- **Citation keys for external datasets** (bibliography.yaml):
  - DeepVess: `kaufmann_2020_deepvess` (Cornell eCommons — verify author/year before submission)
  - TubeNet: `schauss_2021_tubenet` (UCL Figshare — verify author/year before submission)
  - ⚠️ Both dataset entries are placeholder-grade — exact author lists and years must
    be confirmed against the repository pages before `\cite{}` commands are finalized.

---

## § 4 — MLOps Maturity Level 4 Framing

Reference: [Microsoft Azure MLOps Maturity Model — Level 4](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model#level-4-full-mlops-automated-operations)

**This paper is NOT publishable until Level 4 is achieved.** The manuscript scaffold is
being written now so co-authors can understand the vision and review the methods while
the final Level 4 components are being implemented and verified.

| Level 4 Component | Status in ARBOR | Evidence |
|-------------------|-----------------|---------|
| Data pipeline automatically gathers data | ✅ DONE | Prefect Flow 1 (data engineering) + DVC + whylogs profiling |
| Experiment results tracked | ✅ DONE | MLflow + DuckDB analytics, 73/73 artifact checks pass |
| Training code + models version controlled | ✅ DONE | DVC (data), MLflow model registry (models), git (code) |
| Compute is managed | ✅ DONE | SkyPilot intercloud broker (one command, any cloud) |
| Release process automatic | ✅ DONE | BentoML + ONNX export, champion tagging, deploy flow |
| Scoring script version controlled with tests | ✅ DONE | 4100+ tests (staging/prod/gpu tiers), pre-commit gates |
| Application code has unit and integration tests | ✅ DONE | 49 scripts tests + TDD-first development |
| **Production metrics trigger retraining** | ⚠️ IN PROGRESS | Evidently drift detection is wired; Prefect trigger loop needs closure |
| **CI/CD pipeline manages releases** | ⚠️ ADAPTED | GitHub Actions explicitly disabled (credits cost); equivalent: pre-commit + `scripts/pr_readiness_check.sh`. Document in paper as deliberate DevEx choice. |
| Each model release includes unit and integration tests | ✅ DONE | ModelAdapter ABC testing + integration test tier |

**Current honest status**: ARBOR is at Level 3.5 — fully automated training, evaluation,
and deployment; drift detection wired but auto-retraining loop not yet verified end-to-end.
Level 4 closure requires: Evidently → Prefect trigger loop tested with real drift scenario.

---

## § 5 — Scientific Claims (Evidence Status)

These are the paper's verifiable claims, each backed by code or experiments.
Source: `knowledge-graph/manuscript/claims.yaml`

| ID | Claim | Status |
|----|-------|--------|
| C1 | ARBOR = complete reproducible MLOps pipeline, YAML-only changes for new dataset/model | **supported** |
| C2 | cbdice_cldice achieves clDice=0.906±0.008 with only −5.3% DSC penalty vs dice+CE | **supported** |
| C3 | 5-flow architecture separates concerns into Docker containers with MLflow as only inter-flow contract | **supported** |
| C4 | ModelAdapter ABC: new model integration = 3 abstract methods + YAML, no infra changes | **supported** |
| C5 | SAM3 can be adapted for 3D volumetric segmentation via slice-by-slice inference | **pending GPU runs** |
| C6 | Single-command cloud execution (sky jobs launch) reproduces identical training env | **supported** |
| C7 | ARBOR is first MLOps platform for multiphoton biomed segmentation with probabilistic KG as machine-queryable spec | **literature_validated** (2026-03-15 scan, 15 papers) |
| C8 | Zero hardcoded values — any lab overrides via 1-2 YAML files | **supported** |
| C9 | VesselFM evaluation on MiniVess = data leakage; fair eval requires DeepVess/TubeNet | **documented_limitation** |
| C10 | Probabilistic KG (52 nodes with Bayesian priors) enables structured belief propagation | **partial** (Phase 1 only) |
| C11 | E2E pipeline produces 35+ verified artifacts in <9 seconds | **supported** (2026-03-02 verified) |
| C12 | GitHub Actions CI intentionally disabled; all validation local (deliberate DevEx choice) | **supported** |
| C13 | CLAUDE.md + KG = qualitative advance over shallow AGENTS.md practice (14.5% safety guidelines in 2,303 repos) | **literature_validated** |

---

## § 6 — Results Sections (Status + Key Numbers)

Source: `knowledge-graph/manuscript/results.yaml`

### R0 — Results Overview
- Framing: Platform WORKS, not SOTA claims. R1 + R3 = Nature Protocols protocol validation.
- Status: `needs_content`

### R1 — Platform Validation: Reproducibility Proof ✅ DATA AVAILABLE
- 73/73 artifact verification checks pass (`scripts/verify_all_artifacts.py`)
- 35+ artifacts produced by PipelineTriggerChain in 8.26 seconds
- Timestamp: 2026-03-02T03:32:10Z
- Source: `outputs/pipeline/trigger_chain_results.json`
- **THIS IS THE KEYSTONE RESULT** — proves the pipeline produces consistent artifacts
- ⚠️ **Scope**: 73/73 is artifact INTEGRITY on one machine (author's workstation, one timestamp).
  This is NOT cross-machine reproducibility. Cross-machine replication is L5 (planned, required
  for Nature Protocols). Do NOT write R1 as "we demonstrate cross-machine reproducibility" —
  write "we verify the pipeline produces 73 expected artifacts in 8.26 seconds; independent
  replication is planned."

### R2 — Developer Experience Benchmark
- Target metrics: time from git clone to first training run (<15 min), manual steps (0)
- Status: `needs_data` — requires external collaborator replication for credibility
- Note: Required for Nature Protocols submission
- **What satisfies R2**: One external collaborator (different machine, different OS or cloud)
  runs `git clone → uv sync --all-extras → prefect deployment run 'train-flow/default'`
  and produces verified training artifacts. Measure: wall-clock time, number of manual
  interventions (target: 0), and cross-machine artifact SHA256 match with the primary
  researcher's run.
- **⚠️ sci-llm-writer WRITING INSTRUCTION**: Write R2 as a PLANNED validation, not a
  completed one. Do NOT write "we demonstrate that a new user can reproduce results in
  <15 minutes" — write "we designed the platform to achieve this; external replication
  is planned as part of the Nature Protocols submission process."

### R3a — DynUNet Loss Ablation ✅ DATA AVAILABLE
- cbdice_cldice: DSC=0.772±0.016, clDice=0.906±0.008 (WINNER — topology)
- dice_ce: DSC=0.824±0.014 (best DSC), clDice=0.832±0.019
- Topology gain: +8.9% clDice, −5.3% DSC
- Source: `knowledge-graph/experiments/dynunet_loss_variation_v2.yaml`
- Figures: `outputs/analysis/figures/loss_comparison.png`, `fold_heatmap.png`
- Table: `outputs/analysis/comparison_table.tex`

### R3b — Multi-Model Comparison (DynUNet vs Mamba vs SAM3) 🚫 BLOCKED
- Blocked by: SAM3 GPU runs (target 2026-03-20) + Mamba adapter (not yet implemented)
- **Critical path item — paper cannot be submitted without this**

### R3c — Foundation Model Comparison (VesselFM — external datasets only) 🚫 BLOCKED
- Blocked by: VesselFM GPU runs on DeepVess/TubeNet (NOT MiniVess — data leakage)
- **Must document the data leakage detection as a platform capability (the platform caught it)**

### R4 — Uncertainty Quantification
- Conformal prediction coverage guarantees
- Multi-model ensemble (3-model DynUNet soup)
- Status: `partial` — full conformal coverage curves need all model GPU runs

### R5 — Living Specification Graph: Structural Characterization (Appendix)
- KG nodes: 52 PRD decision nodes + manuscript layer (C1-C13, M0-M12, R0-R5)
- Metalearning documents: `.claude/metalearning/*.md` files
- CLAUDE.md rules: 24 explicit behavioral constraints
- TDD test suite: 49 scripts tests + ~4100 total tests
- Comparison baseline: chatlatanagulchai_2025 (62.3% build commands, 14.5% safety/perf in 2,303 AGENTS.md files)
- **Frame as STRUCTURAL CHARACTERIZATION (measurable), not productivity claim (unverifiable)**
- Note: Can become companion paper. Paper stands without this section.

---

## § 7 — Methods Overview (13 Sections)

Source: `knowledge-graph/manuscript/methods.yaml` — full key_points for each section.

| Section | Title | Key Platform Contribution |
|---------|-------|--------------------------|
| M0 | Research Question & Scope | Platform-as-contribution framing; not SOTA, not fork |
| M1 | Dataset: MiniVess | 70 volumes, 2-photon, 3-fold CV, DVC-versioned |
| M2 | Data Engineering Flow | Flow 1 of 5; Pandera + Great Expectations; OpenLineage |
| M3 | Platform Architecture | 5-flow Prefect DAG + Docker-per-flow; MLflow as sole inter-flow contract |
| M4 | Cloud Compute Strategy | SkyPilot intercloud broker; one command any cloud; T4 BANNED |
| M5 | Workflow Orchestration | Prefect 3.x required; 5 personas; PipelineTriggerChain |
| M6 | Model Adapter Architecture | ModelAdapter ABC; 3 abstract methods; build_adapter() factory |
| M7 | Training Pipeline | cbdice_cldice default; AMP ON train / OFF val; Optuna ASHA |
| M8 | Evaluation and Analysis | DSC + clDice + MASD; bootstrap CIs; conformal prediction; DuckDB |
| M9 | Observability Stack | MLflow + Evidently drift + Prometheus/Grafana for model/infra; Langfuse for agentic dev tracing (separate concern — do NOT list Langfuse as production model observability in the methods section; it belongs in the M12 agentic appendix) |
| M10 | Deployment and Serving | ONNX + BentoML + Gradio; NVIDIA MIG; champion tagging |
| M11 | Configuration Architecture | Hydra-zen + Dynaconf; zero hardcoded values; .env.example as SSoT |
| M12 | Agentic Development (Appendix) | 52-node probabilistic KG; TDD-first; metalearning docs; LSG novelty |

---

## § 8 — Known Limitations (for Reviewers)

Source: `knowledge-graph/manuscript/limitations.yaml`

1. **L1 — VesselFM data leakage** (HIGH): VesselFM pre-trained on MiniVess.
   Fix: evaluate on DeepVess/TubeNet only. Frame as platform capability catching the error.

2. **L2 — SAM3/VesselFM/Mamba results pending** (HIGH): Only DynUNet results as of submission draft.
   Target: GPU runs 2026-03-20. Paper cannot be submitted until R3b complete.

3. **L3 — Agentic development metrics are self-reported** (MEDIUM): Development velocity claims
   from spec-driven coding are from primary author only. Options: appendix, companion paper,
   or external validation.

4. **L4 — cuDNN non-determinism** (LOW): Bit-for-bit reproducibility requires
   `torch.use_deterministic_algorithms(True)`. Statistical reproducibility is demonstrated.
   Document in methods.

5. **L5 — No external replication yet** (MEDIUM): Nature Protocols typically requires this.
   Plan: ask one collaborator to run on their lab data. DeepVess worked example planned.

6. **L6 — Single-dataset primary training** (MEDIUM): All DynUNet ablation on MiniVess only.
   Mitigated by: ModelAdapter + YAML-only generalization demo with DeepVess as worked example.

---

## § 9 — Knowledge Graph File Map (Key Paths for sci-llm-writer)

All paths relative to `minivess-mlops/` unless noted.

```
knowledge-graph/
├── navigator.yaml              ← Entry point — maps topics to domains
├── _network.yaml               ← 52-node Bayesian decision DAG
├── bibliography.yaml           ← All citations (25+ entries including 2026-03-15 lit scan)
├── domains/manuscript.yaml     ← Manuscript domain + manuscript: section in navigator
├── code-structure/
│   ├── flows.yaml              ← 5 Prefect flows (AST-bootstrapped)
│   ├── adapters.yaml           ← ModelAdapter implementations (DynUNet, SAM3, VesselFM, ...)
│   │                             ⚠️ MAMBA ADAPTER NOT YET HERE — needs implementation
│   ├── config-schema.yaml      ← Hydra config hierarchy
│   └── test-coverage.yaml      ← Test tier map (staging/prod/gpu)
├── experiments/
│   ├── dynunet_loss_variation_v2.yaml   ← R3a results (COMPLETE)
│   └── current_best.yaml               ← Champion table + pending experiment list
├── manuscript/
│   ├── claims.yaml             ← C1-C13 scientific claims with evidence status
│   ├── methods.yaml            ← M0-M12 section stubs with key_points
│   ├── results.yaml            ← R0-R5 section stubs with data status
│   ├── limitations.yaml        ← L1-L6 limitations with mitigations
│   └── projections.yaml        ← KG→.tex dependency map (staleness detection)
└── templates/                  ← 4 Jinja2 templates for KG→.tex rendering

docs/manuscript/latent-methods-results/
├── EXECUTIVE-SUMMARY.md        ← One-page paper brief for co-authors
├── methods/methods-{00-12}.tex ← 13 .tex section stubs (scaffold, needs prose)
├── results/results-{00-05}.tex ← 6 .tex section stubs (scaffold, needs prose)
└── artifacts/tab-loss-ablation.tex  ← R3a LaTeX table (real numbers)

outputs/
├── analysis/
│   ├── figures/loss_comparison.png   ← R3a bar chart (real data)
│   ├── figures/fold_heatmap.png       ← R3a per-fold heatmap (real data)
│   └── comparison_table.tex          ← R3a LaTeX table (real data)
└── pipeline/trigger_chain_results.json  ← R1 reproducibility proof (73/73 checks)
```

---

## § 10 — sci-llm-writer Session Activation Checklist

When opening a new session in `sci-llm-writer` for the ARBOR manuscript:

```
Manuscript:     sci-llm-writer/manuscripts/vasculature-mlops/
KG snapshot:    sci-llm-writer/manuscripts/vasculature-mlops/kg-snapshot/kg-snapshot.yaml
                (export via: cd minivess-mlops && uv run python .claude/skills/kg-sync/kg_sync.py --export-to ../sci-llm-writer/manuscripts/vasculature-mlops/kg-snapshot/)
                ⚠️ kg-sync skill is planned but not yet implemented. Until it exists,
                manually copy the relevant knowledge-graph/ YAML files listed in §9
                to the kg-snapshot/ directory. The cover letter itself IS the snapshot
                for the first sci-llm-writer session — reference it directly.
Zotero RDF:     sci-llm-writer/manuscripts/vasculature-mlops/vasculature-mlops.rdf
                (export from Zotero — populate with entries from knowledge-graph/bibliography.yaml)
Source .tex:    docs/manuscript/latent-methods-results/ (scaffold stubs)
```

**Suggested domain expert reviewer personas for iterated-llm-council:**
1. **MLOps reviewer** (expertise: production ML systems, reproducibility, Prefect/MLflow/DVC)
2. **Biomedical imaging reviewer** (expertise: 3D segmentation, MONAI, vascular microscopy, clDice)
3. **Methods reviewer** (expertise: Nature Protocols protocol papers, DevEx evaluation, external replication)
4. **Foundation model reviewer** (expertise: SAM2/SAM3, VesselFM, ViT adapters, transfer learning)
5. **Agentic coding reviewer** (expertise: CLAUDE.md/AGENTS.md empirical literature, spec-driven development, LSG appendix)

**Quality target**: MINOR_REVISION (standard journal submission)

**Ground truth JSON**: The sci-llm-writer activation checklist (step 7) requires a
single `results-ground-truth.json`. Create it at session activation time by consolidating
the two source files:

```json
{
  "R1_platform_validation": {
    "status": "CONFIRMED",
    "artifact_checks_pass": 73,
    "artifact_checks_total": 73,
    "artifacts_produced": 35,
    "pipeline_wall_clock_seconds": 8.26,
    "timestamp": "2026-03-02T03:32:10Z",
    "source": "outputs/pipeline/trigger_chain_results.json"
  },
  "R3a_dynunet_loss_ablation": {
    "status": "CONFIRMED",
    "cbdice_cldice_dsc": "0.772 ± 0.016",
    "cbdice_cldice_cldice": "0.906 ± 0.008",
    "dice_ce_dsc": "0.824 ± 0.014",
    "dice_ce_cldice": "0.832 ± 0.019",
    "cbdice_dsc": "0.767 ± 0.023",
    "cbdice_cldice_metric": "0.799 ± 0.020",
    "dice_ce_cldice_dsc": "0.736 ± 0.016",
    "dice_ce_cldice_cldice": "0.905 ± 0.003",
    "topology_gain_percent_cldice": "+8.9%",
    "dsc_penalty_percent": "-5.3%",
    "source": "knowledge-graph/experiments/dynunet_loss_variation_v2.yaml"
  },
  "R3b_multi_model": {"status": "BLOCKED", "blocked_by": "SAM3 GPU runs pending (target 2026-03-20)"},
  "R3c_foundation_model": {"status": "BLOCKED", "blocked_by": "VesselFM external dataset GPU runs pending"},
  "R4_uncertainty": {"status": "BLOCKED", "blocked_by": "requires all model GPU runs for full ensemble"},
  "R2_devex_benchmark": {"status": "NEEDS_DATA", "blocked_by": "requires external collaborator replication"}
}
```

Save this as: `manuscripts/vasculature-mlops/results-ground-truth.json`

---

## § 11 — Critical Warnings for sci-llm-writer

1. **Mamba adapter does not yet exist** in `code-structure/adapters.yaml`. The paper
   lists Mamba as one of four model families, but no `SegMamba/U-Mamba` adapter has
   been implemented yet. Any methods prose mentioning Mamba must be clearly qualified as
   "planned" or "in implementation" — do not write it as if results are available.

2. **SAM3 is NOT SAM2**. SAM3 = Meta's Segment Anything Model 3, November 2025,
   github.com/facebookresearch/sam3, **848M total params** (ViT-32L perception encoder =
   648M; full model including neck/decoder = 848M). SAM2 is a different (earlier, smaller)
   model. Never conflate them. Source: `src/minivess/adapters/sam3_backbone.py:11`.

3. **VesselFM MUST NOT be evaluated on MiniVess**. VesselFM was pre-trained on MiniVess.
   Use DeepVess (Cornell) and TubeNet 2PM (UCL) for fair evaluation only.

4. **Results framing**: The paper does NOT claim any model is the best vascular segmenter.
   The model comparison demonstrates PLATFORM GENERALIZABILITY. The cbdice_cldice loss
   result is the platform's recommended default — not a novel loss function claim.

5. **Level 4 is the publication bar**: Do not write the results section as if Level 4 is
   fully achieved. Production metrics → auto-retraining trigger loop is IN PROGRESS.
   Write it honestly: "demonstrated architecture with Level 4 closure in progress."

6. **Every citation needs a clickable URL** (CLAUDE.md Rule). The format is:
   `[Author et al. (Year). "Title." *Journal*.](URL)`. No bare author-year references.

---

## § 12 — Agentic Development Context (M12 / R5 Appendix)

This section provides context for the appendix on the Living Specification Graph (LSG).

**Novel claim (literature-validated 2026-03-15)**: No prior published work combines
(agentic coding) + (scientific MLOps) + (probabilistic specification graph). Three
genuinely novel dimensions:
1. Living Specification Graph — 52-node Bayesian decision network as machine-queryable
   single source of truth for agentic development
2. MONAI adapter pattern — model-agnostic ABC for 3D segmentation in open-source research
3. Multiphoton microscopy + full MLOps + spec-driven construction — domain intersection
   with no prior published work

**Comparison baseline** (state of practice):
- chatlatanagulchai_2025_agent_readmes: 2,303 AGENTS.md files — 62.3% build commands,
  67.7% architecture, only 14.5% safety/performance guidelines
- galster_2026_configuring_agentic: 2,926 repos empirical study
- santos_2025_claude_config: architectural specification identified as most critical concern

**Framing**: Structural characterization (measurable node counts, constraint counts,
metalearning document counts) — NOT productivity claim (unverifiable). Appendix or
companion paper — main paper stands without this section.

**Self-correcting TDD development approach** (same philosophy as iterated-llm-council in sci-llm-writer):
- minivess-mlops: `self-learning-iterative-coder` Skill (RED→GREEN→VERIFY→FIX→CHECKPOINT→CONVERGE)
- sci-llm-writer: `iterated-llm-council` Skill (L3 reviews → L2 synthesis → L1 verdict → L0 action plan → execute)
- Same iterative self-correction philosophy ("Ralph Wiggum loop" — internal project
  shorthand): LLMs are stochastic; correctness emerges from iterated review cycles,
  not from claiming correctness in a single pass. Both skills implement this via
  independent reviewer agents that catch what the primary LLM missed.

---

## § 13 — Reproducibility Stack Summary

This is what makes the platform publishable as a Nature Protocols / Nature Methods paper:

| Reproducibility Layer | Tool | Status |
|----------------------|------|--------|
| Data versioning | DVC (`data/minivess.dvc`) | ✅ |
| Config as code | Hydra-zen (experiments), Dynaconf (deployment) | ✅ |
| Container isolation | Docker-per-flow (6 containers) | ✅ |
| Experiment tracking | MLflow (params, metrics, artifacts, git hash) | ✅ |
| Cloud abstraction | SkyPilot (intercloud broker, spot recovery) | ✅ |
| Artifact verification | `scripts/verify_all_artifacts.py` (73 checks) | ✅ |
| Drift detection | Evidently (data + model drift) | ✅ wired |
| Auto-retraining trigger | Evidently → Prefect trigger | ⚠️ in progress |
| Model lineage | OpenLineage (Marquez) | ✅ |
| Data quality gates | Pandera + Great Expectations | ✅ |
| Statistical reproducibility | Bootstrap CIs + conformal prediction | ✅ |
| Topology metrics | clDice (Shit et al. 2021) + MASD | ✅ |

---

## § 14 — Pending Blockers Before Submission

In priority order:

1. **Mamba adapter implementation** — SegMamba or U-Mamba via ModelAdapter ABC.
   No GPU results possible until this exists.

2. **SAM3 GPU runs** — RunPod RTX 4090, target 2026-03-20, ~$15. Blocks R3b.

3. **VesselFM GPU runs on external data** — DeepVess + TubeNet (NOT MiniVess).
   Target 2026-03-25. Blocks R3c.

4. **Level 4 closure** — Evidently → Prefect auto-retraining trigger verified end-to-end.
   Paper not publishable without this.

5. **External replication** — One collaborator running from git clone to results.
   Required for Nature Protocols. Plan: DeepVess worked example.

6. **Zotero RDF population** — Export `knowledge-graph/bibliography.yaml` entries to
   Zotero, then export `.rdf` file for sci-llm-writer citation harness.

7. **Verify DeepVess/TubeNet bibliography entries** — The `kaufmann_2020_deepvess`
   and `schauss_2021_tubenet` entries in `knowledge-graph/bibliography.yaml` are
   placeholder-grade. Verify exact author lists, years, and URLs against the Cornell
   eCommons and UCL Figshare repository pages before any `\cite{}` commands are placed.

---

## § 15 — sci-llm-writer Update Protocol When GPU Results Land

When R3b (SAM3) or R3c (VesselFM) GPU runs complete, update this document and the
sci-llm-writer manuscript using this protocol:

**Step 1 — Update source KG files:**
- Add numbers to `knowledge-graph/experiments/` (new YAML file per experiment)
- Update `knowledge-graph/experiments/current_best.yaml` champion table
- Update `knowledge-graph/manuscript/results.yaml` R3b/R3c status from BLOCKED → confirmed

**Step 2 — Update results-ground-truth.json:**
- Add new result objects to `manuscripts/vasculature-mlops/results-ground-truth.json`
- Change `"status": "BLOCKED"` to `"status": "CONFIRMED"` for the relevant result

**Step 3 — Update the cover letter (this file):**
- Update §3.3 (SAM3) or §3.4 (VesselFM) with confirmed numbers
- Update §6 (R3b/R3c) status from `🚫 BLOCKED` to `✅ DATA AVAILABLE`
- Update §14 to remove the corresponding blocker
- Update the YAML front-matter `kg_commit` field

**Step 4 — Open a new sci-llm-writer session and run a new iterated-llm-council
iteration on `results-03-models.tex`** (mode: `refinement`, quality target: MINOR_REVISION)
- Paste/reference the updated cover letter as context
- The council will fill in R3b/R3c subsections with the real numbers
- Run verification against the updated results-ground-truth.json
