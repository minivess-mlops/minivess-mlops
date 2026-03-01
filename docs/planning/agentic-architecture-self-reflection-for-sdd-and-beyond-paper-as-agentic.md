# Second-Pass Analysis: The Paper IS the Agentic Process

**Date:** 2026-03-01
**Branch:** `feat/agentic-architecture`
**Companion:** `agentic-architecture-self-reflection-for-sdd-and-beyond.md` (first-pass)
**Status:** Paper strategy and architecture implications when the novelty IS the agentic process

---

## Abstract

This second-pass analysis reframes the MinIVess MLOps repository from a portfolio artifact
into a peer-reviewed contribution where the *process of agentic development* is co-equal
with the *system produced*. We introduce the concept of **Probabilistic SDD** — a Bayesian
decision network that extends Spec-Driven Development with explicit uncertainty quantification
over architectural choices. For implementation, we collapse the probabilistic specification
to its **MAP (Maximum A Posteriori) scenario** — selecting the highest-probability option at
each decision node — and discuss the full probabilistic framework as a reusable methodology
in the paper's Discussion section. This MAP-collapsed specification, combined with an agentic
coding agent (Claude Code), constitutes a new paradigm for reproducible scientific software:
**spec-as-transfer**, where the transferable artifact is neither code nor documentation but
a machine-readable specification from which domain-adapted systems can be generated. We
target **Nature Methods** with a single paper that presents both the usable platform
(self-configuring, topology-aware vascular segmentation in the nnU-Net tradition) and the
agentic development blueprint (how we built it and how others can replicate the process
for their own domains), and identify the architectural changes required to make both
contributions defensible within a single manuscript.

---

## 1. The Thesis: Spec-as-Transfer

### 1.1 The Reuse Crisis in Research Software

The dominant paradigm for sharing research software is **clone-and-modify**:

1. Researcher A publishes a paper with a GitHub repository
2. Researcher B clones the repository
3. Researcher B spends 4–8 weeks fighting dependency conflicts, understanding implicit
   assumptions, adapting hardcoded parameters, and rebuilding infrastructure
4. Researcher B either succeeds (rare) or abandons the attempt (common)

This paradigm fails because **code is an implementation artifact** — it is coupled to its
environment, its author's implicit assumptions, and the specific hardware it was tested on.
Transferring code across domains requires understanding all of these coupling points, which
is often harder than building from scratch.

The Fowler–Boeckeler (2025) three-paradigm taxonomy (spec-first, spec-anchored,
spec-as-source) describes the specification–code relationship within a single project.
We propose a fourth paradigm:

### 1.2 Spec-as-Transfer: The Fourth Paradigm

| Paradigm | Primary Artifact | Transfer Unit | Reuse Mechanism |
|----------|-----------------|---------------|----------------|
| Spec-First | Code (spec discarded) | Code | Clone and modify |
| Spec-Anchored | Code + spec (co-evolving) | Code + spec | Fork both |
| Spec-as-Source | Spec (code generated) | Spec | Regenerate code |
| **Spec-as-Transfer** | **MAP-collapsed spec** | **Spec template + domain overlay** | **Populate spec → generate system** |

In spec-as-transfer, the paper's supplementary material includes not the code repository
but the **specification template** from which the code was generated. The probabilistic
PRD (70 decision nodes with conditional probability tables) is collapsed to its MAP
scenario — the highest-probability option at each node — producing a deterministic
specification that an agentic coding tool can execute directly. A researcher in a
different domain:

1. Adopts the generic template (schema, protocols, backbone defaults) — minutes
2. Populates domain-specific decisions (L1 strategic, L3 technology, domain overlay) — days
3. Runs Claude Code with the populated MAP spec + CLAUDE.md constitution — days
4. Obtains a working, tested, documented system adapted to their domain — total: ~5–10 days

Compare this to 4–8 weeks of clone-and-modify, with no guarantee of success.

The full probabilistic specification — with alternative options, conditional dependencies,
and archetype modifiers — remains available as a supplementary artifact. Researchers who
want to explore non-MAP paths (e.g., choosing a different orchestration framework because
their lab already runs Airflow) can consult the probability distributions and rationale
for each decision node.

### 1.3 Why This Is Novel

No existing SDD framework (SpecKit, OpenSpec, BMAD, Kiro, Tessl) encodes **uncertainty**
over design decisions. All produce deterministic specifications: "use PostgreSQL" or "use
SQLite." The MinIVess/music-attribution PRD system encodes probabilistic alternatives:
"PostgreSQL at 0.60, SQLite at 0.25, Supabase at 0.15, conditional on team size." This
allows downstream researchers to collapse probabilities according to their context rather
than accepting a hardcoded choice.

**For the paper and implementation**, we adopt the **MAP (Maximum A Posteriori) approach**:
at each decision node, select the highest-probability option. This produces a single,
deterministic, executable specification — the "winner" at each node. The full probabilistic
framework (Bayesian network, archetype modifiers, domain overlays) is discussed in the
paper's Discussion section as a methodological contribution for the community.

Formally:

```
Probabilistic SDD = SDD (descriptions, rationale, references)
                  + Bayesian Network (prior probabilities, conditional tables, DAG)
                  + Archetype System (team-profile probability modulation)
                  + Domain Overlay System (domain-specific probability adjustments)
                  + Temporal Model (volatility classification, review scheduling)
                  + Validation Engine (17 automated invariants)

MAP Scenario     = argmax_option P(option | evidence, archetype, domain)
                   for each decision node
                 → Deterministic specification (directly executable by Claude Code)
```

The knowledge-to-code ratio is approximately 24.2% (Vasilopoulos, 2026), confirming that
non-trivial context infrastructure is required for effective agentic development.

---

## 2. Publication Strategy

### 2.1 One Paper, Two Contributions

This is **one paper** with two intertwined contributions that are inseparable:

1. **The output:** A self-configuring, topology-aware vascular segmentation platform
   that researchers can use immediately (the system)
2. **The blueprint:** The agentic development process (probabilistic SDD, MAP-collapsed
   spec, CLAUDE.md constitution, TDD mandate) that describes HOW to get there — and how
   others can replicate the process for their own domains

These are not orthogonal — they are co-dependent. The platform validates the process;
the process makes the platform reproducible and transferable. The paper presents both.

**Title concept:** "MinIVess: A Self-Configuring Platform for Topology-Aware Vascular
Segmentation, Built and Transferable via Agentic Spec-Driven Development"
(or shorter: "MinIVess: Agentic Spec-Driven Development of a Self-Configuring Vascular
Segmentation Platform")

**Target:** Nature Methods (presubmission enquiry — see Section 2.4)
- **Fallback:** Nature Computational Science → PLOS Computational Biology (Software Section)

**Paper structure:**
- **Results:** Platform capabilities — adaptive compute, topology-aware metrics, conformal
  UQ, multi-dataset validation (the usable output)
- **Methods:** The agentic development process — probabilistic SDD with MAP collapse,
  CLAUDE.md constitution, TDD skill, Prefect orchestration (the blueprint)
- **Discussion:** Spec-as-transfer paradigm, probabilistic SDD as a general framework,
  cross-domain transferability evidence, limitations

**Why Nature Methods:**
1. nnU-Net (7,263 citations), Cellpose, ilastik, Fiji, SciPy all set precedent for
   tool/engineering papers where automation IS the method
2. Omega (2024) and GeneAgent (2025) confirm Nature Methods actively publishes
   LLM-agent tool papers
3. No MLOps-for-medical-segmentation paper exists there — MONAI itself is arXiv-only
4. VesSAP (2020) and VascuViz (2021) confirm vascular imaging is within scope
5. The 2022 editorial explicitly values "high immediate practical value"
6. **No paper anywhere in the Nature family describes a complete research software system
   built using agentic AI tools** — this is a genuine first

**Framing:** "nnU-Net automated model configuration; MinIVess automates the entire research
lifecycle — and we show you how to do the same for your domain."

### 2.2 Journal Assessment

| Journal | IF | Platform Fit | Process Fit | Recommendation |
|---------|---|-------------|------------|----------------|
| **Nature Methods** | **32.1** | **High** | **High** (in Methods + Discussion) | **Primary target** |
| **Nature Protocols** | **13.1** | **High** | **Very High** | **Strong alternative** (see below) |
| Nature Computational Science | ~12 | **High** | Medium | Fallback #1 |
| PLOS Computational Biology | 3.8 | **High** | Medium | Fallback #2 (LabOps precedent) |
| Medical Image Analysis | 11.8 | **High** | Low | Fallback #3 (segmentation focus) |
| MELBA | Growing | **High** | Low | Resource Track alternative |

**Nature Protocols case:**
- **IF:** ~13.1 (5-year: ~14.8), top-tier for reproducible methodology
- **Article types:** (1) **Protocol** — step-by-step, requires Introduction, Materials,
  Procedure, Timing, Troubleshooting, Anticipated Results; (2) **Tutorial** (introduced 2018)
  — focuses on the *thought process* behind a method and how to design experiments, rather
  than rigid step-by-step instructions
- **Precedent for computational protocols:** ColabFold/AlphaFold protocol (2024), ML
  interpretability in neuroimaging (2020), scRNA-seq analysis guidelines, genomics QC
  pipelines — Nature Protocols routinely publishes computational step-by-step workflows
- **Why it fits our "blueprint" angle:** The agentic development process (PRD template →
  MAP collapse → CLAUDE.md → Claude Code → working system) is inherently a **protocol**.
  The Tutorial format specifically asks for the "thought process behind an experiment" —
  which maps directly to our spec-as-transfer methodology
- **Advantage over Nature Methods:** Nature Protocols explicitly expects processes/protocols
  as the contribution, so the "is this a method or software engineering?" scope question
  disappears. The blueprint IS the protocol.
- **Critical distinction:** Nature Protocols publishes protocols for **using** tools, not
  **building** them. Every computational protocol there (ColabFold, scRNA-seq, neuroimaging
  ML) describes how to use a tool to answer a biological question. A "how to build MLOps
  with Claude Code" paper would fall outside scope.
- **Two viable framings:**
  1. **Reframe as usage protocol** (stronger fit): "A protocol for building, validating,
     and deploying 3D vascular segmentation pipelines using MinIVess MLOps" — numbered
     steps: prepare data → configure adaptive compute → run sweep → evaluate with
     conformal UQ → deploy → generate figures. User = vascular biology researcher.
  2. **Tutorial format** (thought process): Focus on *why* decisions were made, how to
     adapt the approach — maps to the spec-as-transfer blueprint angle.
- **Best strategy:** Nature Protocols as a **follow-up** after the Nature Methods paper,
  following the established pattern (ColabFold: Nature Methods 2022 → Nature Protocols
  2024; MCMICRO: Nature Methods 2021 → Nature Protocols 2023). This is the most natural
  path and creates a two-publication arc from the same work.
- **Primarily commissioned:** Unsolicited submissions are unusual. Presubmission enquiry
  required. Most content is invited by editors.
- **For computational protocols:** "Materials" = software + hardware, "Procedure" = numbered
  steps with CRITICAL STEP and TIMING callouts, "Troubleshooting" = mandatory table,
  "Anticipated Results" = example output figures

**Nature Methods case in detail:**
- **IF:** 32.1 (5-year: 51.7), #1 in Biochemical Research Methods
- **Acceptance rate:** ~8–10%, ~50–60% desk rejected
- **Precedent:** nnU-Net (2021, 7,263 cit.), Cellpose (2021, 3,000+ cit.), Fiji, ilastik,
  SciPy — all tool/engineering papers where automation IS the method
- **LLM-agent precedent:** Omega (2024, LLM napari agent), GeneAgent (2025, self-verification
  language agent) — Nature Methods is actively accepting agent papers NOW
- **Gap we fill:** No MLOps platform paper exists in Nature Methods; MONAI is arXiv-only
- **Key risk:** Single-dataset validation (MiniVess, 70 volumes) vs. nnU-Net's 23 datasets.
  **Mitigation:** validate on 3–5 vascular/tubular datasets before submission (see §3.3.4)
- **Article type:** Article (not Brief Communication). 3,000–5,000 words, 6 figs, 50 refs
- **Presubmission enquiry:** YES, required (see §2.4)

### 2.3 Precedent Papers

| Paper | Venue | Relevance |
|-------|-------|-----------|
| nnU-Net (Isensee et al., 2021) "Self-configuring method" | **Nature Methods** | **Primary framing model** — "no new net," automation IS the method, 7,263 citations |
| Cellpose (Stringer et al., 2021) "Generalist segmentation" | **Nature Methods** | Generalist tool, standard architecture, contribution is pipeline/tooling |
| Omega (Royer et al., 2024) "LLM for bioimage analysis" | **Nature Methods** | **LLM-agent tool paper accepted in Nature Methods** |
| GeneAgent (Wang et al., 2025) "Self-verification language agent" | **Nature Methods** | **Agent paper accepted July 2025** — self-verification parallels our TDD |
| VesSAP (Todorov et al., 2020) "Whole mouse brain vasculature" | **Nature Methods** | Vascular segmentation + graph analysis — confirms domain is in scope |
| VascuViz (2021) "Multimodality vascular visualization" | **Nature Methods** | Vascular imaging pipeline — confirms domain interest |
| Fiji (Schindelin et al., 2012) "Open-source platform" | **Nature Methods** | Pure software platform paper — engineering IS the contribution |
| The Virtual Lab (Swanson et al., 2025) "AI agents design nanobodies" | **Nature** | Multi-agent system in Nature main — agents for science at highest venue |
| Einecke (2026) "Conversational AI for Rapid Scientific Prototyping" | arXiv | Process-as-contribution framing |
| LabOps (2025) "Self-hosted workflow of open-source tools" | PLOS CompBio | Infrastructure composition as contribution |
| Watanabe et al. (2025) "Agentic Coding PRs on GitHub" | arXiv | Empirical analysis of agentic development patterns |
| METR RCT (Becker et al., 2025) "AI tools on experienced developers" | arXiv | Quantitative evidence that process matters |

### 2.4 Nature Methods Presubmission Enquiry Strategy

Nature Methods accepts presubmission enquiries via the online Manuscript Tracking System.
Response typically within 2 working days. This is **strongly recommended** given the
novelty of framing MLOps as a "method."

**Cover letter structure (referenced abstract + brief letter):**

1. **The nnU-Net parallel** (opening): "Just as nnU-Net (Isensee et al., 2021, Nature
   Methods) automated model configuration to democratize segmentation, MinIVess automates
   the entire research lifecycle — from data engineering through deployment — eliminating
   infrastructure wrangling for biomedical imaging researchers."

2. **The method** (what is novel): Self-configuring adaptive compute profiles (analogous to
   nnU-Net's dataset fingerprint → heuristic rules), topology-aware multi-metric evaluation
   framework (18 losses, 8 metrics including clDice, Betti errors, junction F1), conformal
   prediction sets for segmentation uncertainty, and full lifecycle orchestration via 5
   persona-based flows.

3. **LLM-agent precedent** (scope fit): Reference Omega (2024) and GeneAgent (2025) as
   evidence Nature Methods is actively publishing agent/LLM tool papers.

4. **Validation evidence**: "2,231 automated tests, validated on N vascular/tubular
   segmentation datasets [after multi-dataset expansion], model-agnostic across DynUNet,
   SegResNet, and VISTA-3D architectures."

5. **Broad applicability**: Model-agnostic design, multi-environment operation
   (local/cloud/CI), specification template enabling zero-config adoption for new domains.

**What NOT to emphasize:**
- Do NOT lead with "vibe coding" or "built by Claude Code" — risks scope mismatch
- Do NOT lead with the MiniVess dataset — lead with the generalizable platform
- Do NOT mention IEC 62304 or clinical compliance — Nature Methods excludes clinical focus
- Keep "MLOps" in Methods section only; use "reproducible lifecycle management" in title/abstract

**Recommended article type:** Article (not Brief Communication or Resource).

### 2.5 The VibeX 2026 Opportunity (Companion Venue)

The 1st International Workshop on Vibe Coding and Vibe Researching, co-located with
EASE 2026, is the first academic venue explicitly targeting agentic development as a
contribution type. A short companion paper or extended abstract here would:
- Establish early presence in a forming field
- Provide peer-reviewed validation of the process methodology
- Create a citable reference complementing the Nature Methods submission
- Focus on the spec-as-transfer blueprint while the journal paper leads with the platform

---

## 3. Architecture Implications: What Must Change

The first-pass report recommended architecture changes for portfolio purposes. When the
novelty IS the agentic process, the requirements shift significantly:

### 3.1 What Becomes Critical

| Component | First-Pass Priority | Paper Priority | Why |
|-----------|-------------------|----------------|-----|
| **Process metrics** | Not mentioned | **Critical** | Must quantify sessions, prompts, tokens, agent invocations, test counts, LOC per session |
| **PRD template extraction** | Phase 5 (ongoing) | **Phase 0** | The reusable spec IS the blueprint contribution |
| **External dataset validation** | Not mentioned | **Critical** | Nature Methods requires generalizability ("two distinct applications minimum") |
| **Ablation: with vs. without spec** | Not mentioned | **High** | Must demonstrate spec-driven > ad-hoc |
| **Cross-domain demonstration** | Not mentioned | **High** | Validate spec-as-transfer on a second domain |
| **Development diary/log** | Not mentioned | **High** | Structured record of agentic development sessions |
| **Three-gate CI** | Phase 2 | **Earlier — Phase 1** | Demonstrates L3 maturity for the paper |
| **MCP server** | Phase 3 | **Phase 2** | Demonstrates protocol-native infrastructure |

### 3.2 What Becomes Less Important

| Component | First-Pass Priority | Paper Priority | Why |
|-----------|-------------------|----------------|-----|
| Agent teams demonstration | Phase 4 | **Deferred** | CooperBench ~50% degradation; risky to demonstrate |
| SpecKit evaluation | Phase 1 | **Skip** | We propose our own framework; evaluating competitors is secondary |
| CLAUDE.md redundancy audit | Phase 0 | **Lower** | The CLAUDE.md IS evidence of context engineering |

### 3.3 New Required Components

#### 3.3.1 Development Process Instrumentation

To make the process a publishable contribution, we need **quantitative process metrics**:

| Metric | Source | Comparable to |
|--------|--------|--------------|
| Total Claude Code sessions | Git log + session logs | Vasilopoulos: 283 sessions |
| Human prompts | Session transcripts | Vasilopoulos: 2,801 prompts |
| Agent turns | Session transcripts | Vasilopoulos: 16,522 turns |
| Knowledge-to-code ratio | LOC in `.claude/` + `docs/prd/` vs `src/` | Vasilopoulos: 24.2% |
| Token consumption | LiteLLM proxy logs | BMAD: ~230M/week |
| Test count progression | Git log + pytest output | Currently: 2,282 tests |
| Time per feature | Git timestamps | METR RCT: AI 19% slower |
| Spec drift incidents | Manual tracking | Vasilopoulos: 2 incidents |
| Agent invocations by skill | Session logs | Vasilopoulos: 1,197 invocations |

**Implementation:** Add a `scripts/collect_development_metrics.py` that parses git log,
Claude Code session logs (if accessible), and pytest output to produce a structured
development diary.

#### 3.3.2 SDD Framework Choice: OpenSpec-Inspired Lightweight Approach

Per the first-pass analysis (Section 8), we do **not** adopt any SDD framework wholesale.
Instead, we adopt practices incrementally, with **OpenSpec's delta-spec pattern** as the
primary influence for our brownfield codebase:

| Practice | Source | Status |
|----------|--------|--------|
| Delta specs for new features | OpenSpec (brownfield pattern) | Markdown in `specs/` |
| Constitution file | SpecKit | Already have (CLAUDE.md) |
| Trigger tables | Vasilopoulos G3 | Add to CLAUDE.md |
| Self-spec pattern | Piskala (2026) | Agent drafts spec → human reviews |
| Three-gate CI | Course material | Technical + scientific + compliance |

**Why not full OpenSpec or SpecKit:**
- OpenSpec (TypeScript/Node) — language mismatch with our Python-only stack
- SpecKit (Python CLI, 5-phase gates) — assumes greenfield; our 2,282-test brownfield needs
  delta specs, not full system specs
- BMAD — ~230M tokens/week, incompatible with academic budgets
- Kiro — AWS vendor lock-in
- Tessl — private beta, 1:1 spec-to-code too rigid for ML pipelines

**For the paper:** We describe this as a "lightweight, OpenSpec-inspired delta-spec
approach" and contrast it with the full frameworks in a comparison table. The PRD template
(below) IS our spec framework — domain-specific, probabilistic, and purpose-built.

#### 3.3.3 Probabilistic SDD Template Package

Extract the PRD system into a standalone, domain-generic template:

```
probabilistic-sdd-template/
  _schema.yaml                     # GENERIC: decision node JSON Schema
  _network-template.yaml           # GENERIC: L1-L5 skeleton
  backbone-defaults.yaml           # GENERIC: common tech decisions
  templates/
    decision-node.yaml             # GENERIC: CHANGE_ME template
    scenario.yaml                  # GENERIC: CHANGE_ME template
    archetype.yaml                 # GENERIC: CHANGE_ME template
    domain-overlay.yaml            # GENERIC: CHANGE_ME template
  protocols/
    add-decision.md                # GENERIC: maintenance protocol
    update-priors.md               # GENERIC
    ingest-paper.md                # GENERIC
    validate.md                    # GENERIC
    ...
  examples/
    minivess/                      # DOMAIN: vascular segmentation (70 nodes)
    music-attribution/             # DOMAIN: music metadata (85 nodes)
  GUIDE.md                         # NEW: step-by-step population guide
  CLAUDE-TEMPLATE.md               # NEW: constitution template
```

**The critical finding from cross-project analysis:** approximately 60–70% of the PRD
system is already domain-generic. The schema, protocols, Bayesian semantics, archetype
mechanism, and domain overlay structure are identical across MinIVess and
music-attribution-scaffold.

#### 3.3.4 Cross-Domain Validation Experiment

To demonstrate spec-as-transfer, we need at least one additional domain instantiation.
Options (in order of feasibility):

1. **Retinal OCT layer segmentation** — 2D, different topology (layers vs. trees),
   well-known public datasets (Duke, AROI), different model preferences (2D U-Net vs.
   3D DynUNet). Demonstrates that the same PRD template produces a working system for a
   fundamentally different imaging modality.

2. **Cardiac MRI segmentation** — 3D-ish (2D+t), ACDC dataset freely available, different
   metrics (ejection fraction, wall thickness). Demonstrates temporal extension.

3. **Brain lesion segmentation** — 3D, BraTS dataset freely available, different topology
   (blob-like vs. tubular). Demonstrates the overlay system handles different topology
   preferences.

The experiment protocol:
1. Start with the generic PRD template
2. Populate for the target domain (record time)
3. Generate system with Claude Code (record sessions, tokens, time)
4. Run training and evaluation (record metrics)
5. Compare against baseline from the domain's standard benchmark
6. Compare effort against clone-and-modify of MinIVess

#### 3.3.5 External Dataset Validation

For Nature Methods, the 2022 editorial requires "two distinct applications typically being
the minimum" for validation. We need multi-dataset evidence beyond MiniVess:

| Dataset | Modality | Structures | Public | Relevance |
|---------|----------|-----------|--------|-----------|
| MiniVess | 2P microscopy | Cerebral vasculature | Yes | Primary |
| VesselMNIST3D | Synthetic | 3D vessel trees | Yes | Topology benchmark |
| TubeTK | MRA | Brain vessels | Yes | Clinical relevance |
| DRIVE/STARE | Fundus | Retinal vessels | Yes | 2D → 3D generalization |
| BraTS | MRI | Brain tumors | Yes | Non-vascular control |

### 3.4 Revised Phase Plan

| Phase | Focus | Timeline | Paper Section |
|-------|-------|----------|---------------|
| **0** | Extract PRD template + write GUIDE.md | 2–3 days | Methods (blueprint contribution) |
| **1** | Process instrumentation + development diary | 1 week | Methods (process metrics) |
| **2** | Three-gate CI + Langfuse observability | 1 week | Methods (L3 maturity evidence) |
| **3** | MCP server for MLflow | 1 week | Results (protocol-native infra) |
| **4** | External dataset validation (3–5 datasets) | 2 weeks | Results (generalizability — **Nature Methods requirement**) |
| **5** | Cross-domain experiment (retinal OCT) | 2 weeks | Discussion (spec-as-transfer proof) |
| **6** | Paper writing + figures | 2 weeks | Full manuscript |

---

## 4. The Probabilistic SDD Contribution

### 4.1 Why This Is Publishable (Discussion Section Material)

The probabilistic SDD framework is discussed in the paper's Discussion section as a
methodological contribution, while the implementation uses the MAP (highest-probability)
scenario at each decision node. The PRD system occupies a genuinely novel position:

| Dimension | Traditional SDD | Our PRD (Discussion) | MAP Scenario (Implementation) |
|-----------|----------------|---------------------|-------------------------------|
| Decision representation | Single path | Probability distribution | Highest-probability path |
| Dependencies | Implicit (prose) | Explicit (conditional tables, DAG) | Resolved to deterministic |
| Team adaptation | "Consider your team" | Archetype weights (quantified) | Solo-researcher archetype applied |
| Domain adaptation | "For your domain..." | Domain overlay (structured YAML) | Vascular segmentation overlay |
| Evidence tracking | Footnotes | Append-only bibliography | Same (preserved) |
| Validation | Manual review | 17 automated invariants | Same (preserved) |
| Machine readability | No | Yes (YAML, JSON Schema) | Yes (collapsed to executable) |

**Key framing for the paper:** The implementation picks winners. The Discussion explains
the full probabilistic framework and argues that downstream researchers can pick *different*
winners based on their context (different archetype, different domain overlay). This is
the spec-as-transfer mechanism.

### 4.2 Connection to Bayesian Decision Theory

The PRD's mathematical foundation maps to well-established decision theory:

- **Noisy-OR posterior computation** — standard Bayesian network inference
- **Conditional probability tables** — standard probabilistic graphical model
- **Archetype modifiers** — equivalent to Bayesian hierarchical priors
- **Domain overlays** — equivalent to likelihood functions from domain evidence
- **Scenario composition** — equivalent to MAP (Maximum A Posteriori) estimation
- **Paper ingestion** — equivalent to Bayesian updating with new evidence

This grounding in established theory strengthens the academic contribution. We are not
inventing a new mathematical framework — we are applying Bayesian decision networks to a
new domain (software architecture specification).

### 4.3 The "If Your Implementation Doesn't Help" Escape Hatch

A critical feature of spec-as-transfer: even if the MinIVess code is useless to a
researcher (wrong language, wrong framework, wrong assumptions), the **specification
remains valuable**. The researcher can:

1. Read the PRD to understand what decisions were made and why
2. Adopt the decisions that match their context
3. Generate a completely new implementation from the spec using Claude Code
4. Never touch a single line of MinIVess code

This is fundamentally different from the clone-and-modify paradigm, where the code IS the
only artifact. The spec is an independent, transferable, human-readable decision record
that outlives any particular implementation.

---

## 5. The "Agentic Development" Story Arc

### 5.1 MinIVess as a Living Case Study

The repository's git history encodes the complete development narrative:

| Phase | Commits | Tests | Key Milestone |
|-------|---------|-------|--------------|
| Foundation (v0.1-alpha → v2) | ~50 | 103 | Adaptive compute, model profiles |
| Training infrastructure | ~30 | 400+ | 18 losses, cross-fold validation |
| Evaluation & ensemble | ~40 | 600+ | Bootstrap CI, conformal UQ |
| Graph topology | ~25 | 800+ | P0–P2 issues, 25 GitHub issues |
| Conformal prediction | ~20 | 1559 | MAPIE, morphological CP |
| Visualization & dashboard | ~10 | 2282 | Prefect Flow 5, Observable |
| **Agentic transition** | **TBD** | **TBD** | **Spec-as-transfer, MCP, observability** |

Each phase was developed entirely with Claude Code. The git log, commit messages, and
issue tracking encode the agentic development process. This is evidence for the Methods
section (the blueprint).

### 5.2 What We Can Measure (Retrospectively and Prospectively)

**Retrospective (from existing git history):**
- Commits per phase
- Tests per phase (test count progression)
- Lines of code per phase
- Issue-to-commit ratio
- Time between issue creation and closure

**Prospective (for the agentic transition phase):**
- Claude Code sessions (count, duration)
- Human prompts per session
- Tokens consumed per feature
- Spec-to-implementation time
- Spec drift incidents
- Three-gate CI pass rate

### 5.3 Alignment with the Slide Deck

The paper maps to the following slide concepts from the in-preparation course:

| Slide Concept | Paper Demonstration |
|--------------|-------------------|
| Agentic MLOps maturity L0–L5 (fig-rnd-39) | MinIVess L2 → L3 transition with metrics |
| Seven-layer R&D stack (fig-rnd-25, 36) | All 7 layers mapped with implementation status |
| SDD three mitigation levels (fig-fundamentals-29) | Level 2 → Level 3 (delta specs) → Probabilistic SDD |
| Brownfield delta specs (fig-context-30) | Working example on 2,282-test codebase |
| MCP for scientific infrastructure (fig-rnd-32) | mlflow-mcp-server as demonstration |
| Agentic observability (fig-rnd-33) | Langfuse traces in Prefect flows |
| Fowler three paradigms (fig-context-27) | Extended with spec-as-transfer (fourth paradigm) |
| MinIVess agentic transition (fig-rnd-26) | The paper IS the documentation of this transition |
| Data flywheel (fig-ent-35) | PRD as compounding knowledge asset |
| Agent XAI for scientific workflows (fig-rnd-34) | Three-level explainability in agentic decisions |

---

## 6. Reviewer Anticipation

### 6.1 Expected Objections and Preemptive Responses

**Objection 1: "This is just another ML pipeline paper."**
- Response: The contribution is not the pipeline but the *specification system* that enables
  pipeline generation. The PRD template + Claude Code produces working systems for new
  domains in days, not weeks. We demonstrate this with a cross-domain experiment.

**Objection 2: "The probabilistic PRD has no formal guarantees."**
- Response: We explicitly do NOT claim formal guarantees. The Bayesian network uses standard
  noisy-OR inference (Pearl, 1988). The contribution is the *application* of established
  decision theory to software specification, not novel mathematics.

**Objection 3: "How do you know the generated system is correct?"**
- Response: Three mechanisms: (1) TDD mandate — all features have tests before
  implementation, (2) three-gate CI — technical + scientific + compliance, (3) the
  specification itself encodes acceptance criteria that are programmatically verifiable.

**Objection 4: "This only works with Claude Code / Anthropic."**
- Response: The PRD template is agent-agnostic YAML. The CLAUDE.md could be adapted to
  any context file format (.cursorrules, AGENTS.md, Kiro steering). The protocols are
  natural language. Vendor lock-in is a valid concern; we discuss it in limitations.

**Objection 5: "N=2 (MinIVess + music-attribution) is not enough to claim generalizability."**
- Response: N=2 projects + 1 cross-domain experiment (retinal OCT) provides 3 instantiations
  across fundamentally different domains (3D vascular, music metadata, 2D retinal layers).
  We acknowledge this is early evidence, not definitive proof.

**Objection 6: "You're measuring your own productivity — there's no controlled experiment."**
- Response: Fair. We cite the METR RCT (Becker et al., 2025) for controlled evidence.
  Our contribution is the *specification system*, demonstrated through case studies.
  A controlled experiment (same spec, with vs. without Claude Code) is future work.

**Objection 7: "The 60–70% domain-generic claim needs validation."**
- Response: We validate by file count and line count across two independent instantiations.
  The schema, protocols, and validation engine are byte-identical. The archetype and
  overlay structures are structurally identical with different content.

### 6.2 Nature Methods-Specific Requirements

From the 2022 editorial ("What makes a Nature Methods paper," Nat Methods 19, 771):

- **"Strong validation data to demonstrate performance, reproducibility, general
  applicability"** — the "general applicability" clause is our biggest challenge.
  **Mitigation:** validate on 3–5 vascular/tubular datasets (§3.3.4), demonstrate
  model-agnostic operation across DynUNet/SegResNet/VISTA-3D.
- **"Two distinct applications typically being the minimum"** — we need at minimum
  2 imaging modalities (2-photon microscopy + MRA or fundus photography).
- **"Methods should be useable by others"** — the spec-as-transfer template + zero-config
  defaults directly addresses this.
- **"Focus on the method, NOT new biological findings"** — our paper focuses entirely on
  the platform/method, not biological discoveries. Perfect alignment.
- **Code availability:** Required with DOI — GitHub + Zenodo. Already open source.
- **Data availability:** Required — MiniVess dataset is public + all external datasets.
- **Article limits:** 3,000–5,000 words (excl. abstract/methods/refs/legends), 6 figs, 50 refs.
- **Presubmission enquiry:** Strongly recommended (see §2.4).

**Framing guidance (from nnU-Net precedent):**
- nnU-Net's name ("no new net") telegraphed that engineering IS the contribution.
  Consider: "MinIVess" already suggests "minimal" vascular infrastructure — lean into this.
- nnU-Net structured the paper around dataset fingerprint → heuristic rules → empirical
  validation. Our analog: dataset profiling → adaptive compute profiles → topology-aware
  multi-metric evaluation.
- The 2022 editorial explicitly says Nature Methods publishes "a mix of papers with high
  conceptual novelty and high immediate practical value." We aim for practical value.

### 6.3 Fallback Journal Requirements

If Nature Methods desk-rejects (likely on scope grounds — "this is more software
engineering than biological methods"), the same paper can be submitted with minimal
reframing to:

**Nature Computational Science (IF ~12):**
- Emphasize computational methodology and reproducibility
- Broader audience for "spec-driven scientific software"
- Presubmission enquiry recommended

**PLOS CompBio Software Section (IF 3.8):**
- Open source required (BSD/LGPL/MIT) — need to verify current license
- Long-term availability required — GitHub + Zenodo DOI
- LabOps (2025) precedent: infrastructure composition as valid contribution
- Most realistic fallback if Nature-family rejects

---

## 7. Concrete Next Steps (Ordered by Dependency)

### Immediate (This Week)

1. **Extract PRD template** — create `probabilistic-sdd-template/` as a standalone
   directory with generic schema, protocols, backbone defaults, and two worked examples.
   This is the blueprint contribution (Methods section).

2. **Start development diary** — begin structured logging of Claude Code sessions for the
   agentic transition phase. Even rough notes become valuable evidence.

3. **Verify license** — ensure the repository license is compatible with Nature
   requirements (MIT/BSD preferred).

### Short-Term (2 Weeks)

4. **Implement three-gate CI** — demonstrates L3 maturity. Technical (pytest), scientific
   (Dice > 0.80 baseline), compliance (ruff + mypy + pre-commit).

5. **Build mlflow-mcp-server** — demonstrates protocol-native infrastructure. 5 tools
   (list_experiments, query_runs, get_metrics, compare_runs, get_champion).

6. **Instrument Langfuse tracing** — activate in analysis_flow and dashboard_flow.

### Medium-Term (4 Weeks)

7. **External dataset validation** — run MinIVess pipeline on 3–5 external vascular/tubular
   datasets. This is **non-negotiable for Nature Methods** ("two distinct applications
   minimum"). Record multi-dataset metrics for Results section.

8. **Cross-domain experiment** — populate PRD template for retinal OCT. Generate system
   with Claude Code. Run on Duke/AROI dataset. Record all process metrics. This is the
   critical evidence for spec-as-transfer (Discussion section).

### Paper Writing (2 Weeks)

9. **Submit presubmission enquiry** to Nature Methods (see §2.4 for strategy)
10. **Draft manuscript** — Results (platform + validation), Methods (agentic blueprint),
    Discussion (spec-as-transfer, probabilistic SDD framework)
11. **Prepare supplementary** — PRD template, CLAUDE.md, development diary, process metrics

---

## 8. Risk Assessment for the "Process as Novelty" Angle

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Reviewers reject "process" as a contribution | Medium | Medium | Process is in Methods/Discussion; platform stands alone. Fallback: Nature Protocols Tutorial format where process IS the expected contribution |
| Cross-domain experiment fails to produce competitive results | Medium | High | Choose a well-studied domain (retinal OCT) with established baselines |
| Claude Code dependency perceived as vendor lock-in | High | Medium | Emphasize agent-agnostic spec; discuss in limitations |
| PRD template too complex for adoption | Medium | Medium | GUIDE.md with step-by-step walkthrough; worked examples |
| Nature Methods desk rejection | Medium–High | Medium | Presubmission enquiry first; Nature Protocols Tutorial → Nat Comp Sci → PLOS CompBio as ready fallbacks |
| Single-dataset validation insufficient | **High** | **High** | **Non-negotiable: validate on 3–5 datasets before submission** (see §3.3.4) |
| Process metrics are not impressive enough | Medium | Medium | Frame as "first measurement" not "optimal result" |
| The field moves too fast — SDD landscape changes before publication | High | Low | Frame as snapshot + principles, not tool recommendations |

---

## 9. The "One Paper, Dual Contribution" Framing

The MinIVess repository produces a single paper with two inseparable contributions:

```
                    MinIVess Repository
                          │
            ┌─────────────┴─────────────┐
            │                           │
    THE OUTPUT (Results)         THE BLUEPRINT (Methods)
    ────────────────────         ──────────────────────
    What: Self-configuring       What: Probabilistic SDD
    topology-aware vascular      (MAP-collapsed) +
    segmentation platform        spec-as-transfer process

    Evidence: Multi-dataset      Evidence: Cross-domain
    validation, 18 losses,       experiment, process metrics,
    8 metrics, conformal UQ      development diary

    "Here is the tool you        "Here is how we built it
     can use right now"           and how YOU can build
                                  your own for your domain"
            │                           │
            └─────────────┬─────────────┘
                          │
              Nature Methods Submission
              ──────────────────────────
              Framing: "nnU-Net automated model config;
              MinIVess automates the entire lifecycle —
              and we show you how to do the same"

              Discussion: Probabilistic SDD as a
              general framework for spec-driven
              scientific software engineering

              Companion: VibeX 2026 extended abstract
              + JOSS citability record
```

The slide deck (Teikari, 2026) provides the pedagogical framework that contextualizes the
paper, but is not itself a publication target.

---

## References (Additional to First-Pass Report)

- Becker, A. et al. (2025). Measuring the impact of AI tools on developer productivity:
  A randomized controlled trial. *METR*.
- Berg, S. et al. (2019). ilastik: Interactive machine learning for (bio)image analysis.
  *Nature Methods*, 16, 1226–1232.
- Einecke, N. (2026). Conversational AI for rapid scientific prototyping: A case study on
  ESA's ELOPE competition. arXiv:2601.04920.
- Isensee, F. et al. (2021). nnU-Net: A self-configuring method for deep learning-based
  biomedical image segmentation. *Nature Methods*, 18, 203–211.
- Lu, C. et al. (2024). The AI Scientist: Towards fully automated open-ended scientific
  discovery. arXiv:2408.06292.
- Mallardi, G. et al. (2026). MLOps in the healthcare domain: A systematic literature
  review. *Springer LNCS*.
- Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: How to train your own model.
  *Nature Methods*, 19, 1634–1641.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
- Royer, L.A. et al. (2024). Omega — harnessing the power of large language models for
  bioimage analysis. *Nature Methods*. DOI: 10.1038/s41592-024-02310-w.
- Sapkota, N. et al. (2025). Vibe coding vs. agentic coding: Fundamentals and practical
  implications. arXiv:2505.19443.
- Schindelin, J. et al. (2012). Fiji: An open-source platform for biological-image analysis.
  *Nature Methods*, 9, 676–682.
- Stringer, C. et al. (2021). Cellpose: A generalist algorithm for cellular segmentation.
  *Nature Methods*, 18, 100–106.
- Stringer, C. & Pachitariu, M. (2025). Cellpose3: One-click image restoration for improved
  cellular segmentation. *Nature Methods*. DOI: 10.1038/s41592-025-02595-5.
- Swanson, K. et al. (2025). The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies.
  *Nature*. DOI: 10.1038/s41586-025-09442-9.
- Todorov, M. et al. (2020). Machine learning analysis of whole mouse brain vasculature.
  *Nature Methods*, 17, 442–449. (VesSAP)
- Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing
  in Python. *Nature Methods*, 17, 261–272.
- Wang, Z. et al. (2025). GeneAgent: Self-verification language agent for gene-set analysis
  using domain databases. *Nature Methods*, 22, 1677–1685.
- Watanabe, T. et al. (2025). On the use of agentic coding: An empirical study of pull
  requests on GitHub. arXiv:2509.14745.
- Wratten, L. et al. (2021). Reproducible, scalable, and shareable analysis pipelines with
  bioinformatics workflow managers. *Nature Methods*, 18, 1161–1168.

---

*This document is a companion to the first-pass analysis. Together, they define the
strategic direction for the MinIVess Nature Methods submission: a single paper presenting
both the usable platform (output) and the agentic spec-driven development process
(blueprint), targeting a genuine first in the Nature family — a complete research
software system built and transferable via agentic AI tools.*
