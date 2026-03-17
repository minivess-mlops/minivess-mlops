# Metalearning: Model Lineup Ignorance — Did Not Know Paper's Own Models

**Date:** 2026-03-17
**Severity:** P0 — CRITICAL institutional knowledge failure
**Trigger:** User asked "Why do we need both SAM3 Vanilla and Hybrid?" and Claude
could not answer without researching 8 planning docs + 16 adapter files

---

## What Happened

1. Claude was planning GPU experiment runs (#734) for the paper
2. Claude listed "SAM3 Vanilla + SAM3 Hybrid + VesselFM" without knowing WHY
3. Claude suggested running on RunPod (dev-only) instead of GCP (staging/prod)
4. User asked what SAM3 Vanilla trains — Claude could not answer
5. User had to list the correct model lineup from memory:
   - DynUNet (baseline)
   - MambaVesselNet++ (new architecture)
   - SAM3 Vanilla / Hybrid / TopoLoRA (foundation model variants)
   - VesselFM (zero-shot + fine-tuned, external data only)
6. Claude did not know that SAM3 Vanilla could be zero-shot (no training needed)

## Root Causes

### RC1: No "Paper Model Lineup" Decision Node in KG
The knowledge graph has individual model decisions (primary_3d_model, foundation_model,
mamba_architecture) but NO centralized node listing "these are the N models compared
in the paper." The information exists scattered across results.yaml, methods.yaml,
and adapter docstrings — but no single authoritative source.

### RC2: Did Not Read KG Before Planning GPU Runs
CLAUDE.md Rule #13: "Read Context Before Implementing." The KG has models.yaml,
manuscript/results.yaml, and 16 adapter files. None were read before planning #734.
This violates the metalearning from 2026-03-16 (kg-exists-read-it-first.md).

### RC3: Copied Issue Text Verbatim Instead of Verifying
Issue #734 was auto-generated in a previous session. Claude copied its model list
without cross-checking against the KG, adapter code, or planning docs. This violates
Rule #11: "Plans Are Not Infallible."

### RC4: RunPod vs GCP Confusion (AGAIN)
metalearning/2026-03-16-runpod-dev-not-primary-recurring-confusion.md documents
this EXACT failure pattern. Claude suggested RunPod for paper results despite the
architecture clearly stating GCP = staging + prod.

## The Correct Model Lineup (Authoritative)

| # | Model | Family | Training Strategy | Paper Role |
|---|-------|--------|-------------------|------------|
| 1 | **DynUNet** | CNN baseline | Full training (100 ep × 3 fold) | R3a: loss ablation baseline |
| 2 | **MambaVesselNet++** | SSM hybrid | Full training | R3b: architecture comparison |
| 3 | **SAM3 Vanilla** | Foundation (frozen) | Zero-shot (no training!) OR decoder-only fine-tune | R3b: domain gap baseline |
| 4 | **SAM3 TopoLoRA** | Foundation (LoRA) | LoRA fine-tune on FFN layers | R3b: topology-aware adaptation |
| 5 | **SAM3 Hybrid** | Foundation (fusion) | DynUNet + SAM3 features | R3b: 3D context fusion |
| 6 | **VesselFM** | Foundation (pre-trained) | Zero-shot AND fine-tuned | R3c: external eval only (data leakage) |

Key insight user provided: SAM3 Vanilla zero-shot (no training) IS a valid baseline
showing the domain gap. We don't need to spend GPU hours fine-tuning it if zero-shot
evaluation demonstrates SAM3's poor performance on microvasculature.

## Corrective Actions

1. **Create `paper_model_comparison.yaml` KG decision node** — single source of truth
   for which models are in the paper, their training strategy, and GPU requirements
2. **Add to CLAUDE.md** — a brief model lineup reference or pointer to the KG node
3. **Always read models.yaml + manuscript/results.yaml** before any GPU planning
4. **GCP for paper results** — NEVER suggest RunPod for production/paper runs

## Rule Addition

**Before planning ANY GPU experiment run, read:**
1. `knowledge-graph/domains/models.yaml`
2. `knowledge-graph/manuscript/results.yaml`
3. `src/minivess/adapters/CLAUDE.md`
4. The specific adapter file for each model being discussed

**If you cannot answer "what does this model train?" from memory, READ FIRST.**
