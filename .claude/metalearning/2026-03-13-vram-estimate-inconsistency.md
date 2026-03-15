# 2026-03-13 — VRAM Estimate Inconsistency (Recurring Failure)

## The Problem

Claude Code repeatedly provides **different VRAM estimates for the same model** across
conversations and even within a single conversation. The user has observed this pattern
multiple times, causing justified frustration and eroding trust.

## Specific Incident (2026-03-13)

A reviewer agent "corrected" VesselFM VRAM from 10 GB → 6 GB in the RunPod smoke test
plan, claiming the original was wrong. This was itself wrong:

- `configs/model_profiles/vesselfm.yaml:18` states: **~6 GB inference, ~10 GB fine-tuning**
- The smoke test does **fine-tuning**, not inference
- The reviewer confused inference with fine-tuning — halving the estimate
- The plan author (also Claude) accepted this "correction" without verifying

This is the same model, in the same repo, with the VRAM number written in a YAML file —
and Claude still got it wrong.

## Root Cause Analysis

### Why This Keeps Happening

1. **No single source of truth**: VRAM numbers are scattered across 9+ files:
   - `configs/model_profiles/*.yaml` (per-model hardware profiles)
   - `src/minivess/adapters/CLAUDE.md` (adapter context doc)
   - `CLAUDE.md` (root project doc — "Models Fitting 8GB GPU" table)
   - `LEARNINGS.md` (accumulated discoveries)
   - `docs/planning/*.md` (planning docs with estimates)
   - `.claude/metalearning/*.md` (failure analysis)
   - `.claude/projects/.../memory/MEMORY.md` (auto-memory)
   - `src/minivess/adapters/sam3_vram_check.py` (enforcement code)

   Each file has a different subset of models, different numbers, and different dates.

2. **No distinction between MEASURED vs ESTIMATED**: The word "estimated" appears in
   `vesselfm.yaml:18` but not in any standardized way. There's no field like
   `vram_measured: true/false` or `vram_source: "RTX 2070 Super benchmark 2026-03-07"`.

3. **Inference vs training conflated**: VesselFM has 6 GB (inference) and 10 GB
   (fine-tuning) — a 67% difference. Without reading the full YAML comment, Claude
   picks whichever number seems "right" for the context.

4. **Stale estimates propagate**: LEARNINGS.md line 49 says SAM3 V1 Vanilla needs
   "≥16 GB" — an initial estimate from 2026-03-07 that was corrected by actual
   measurement (2.9 GB). But the stale 16 GB number still exists in the file.

5. **LLM stochastic recall**: Claude doesn't deterministically remember which number
   is authoritative. Different conversation turns, different context windows, different
   token sampling → different numbers produced with equal confidence.

## What Must Change

### Single Source of Truth: `configs/model_profiles/*.yaml`

The model profile YAML files MUST be the **authoritative, canonical source** for all
VRAM requirements. Every other file that mentions VRAM must either:
- **Reference** the profile: "See `configs/model_profiles/vesselfm.yaml`"
- **Be deleted** (if it's a stale duplicate)

### Required Fields in Model Profiles

Each `configs/model_profiles/*.yaml` MUST include:

```yaml
vram:
  inference_gb: 6.0      # GPU memory for inference only
  training_gb: 10.0      # GPU memory for fine-tuning (batch=1)
  measured: false         # true = actual benchmark, false = estimate
  measured_gpu: null      # e.g., "RTX 2070 Super" if measured
  measured_date: null     # e.g., "2026-03-07" if measured
  measured_config:        # patch size, batch size, etc. if measured
    patch_size: null
    batch_size: null
  notes: "Estimated based on DynUNet architecture with wider filters"
```

### Current Measured vs Estimated Status

| Model | Inference | Training | MEASURED? | GPU | Date |
|-------|-----------|----------|-----------|-----|------|
| DynUNet | ~3.5 GB | ~3.5 GB | **YES** | RTX 2070 Super | 2026-03-07 |
| SAM3 V1 Vanilla | 2.9 GB | 3.5 GB | **YES** | RTX 2070 Super | 2026-03-07 |
| SAM3 Hybrid | 7.18 GiB (OOM@64) | TBD | **PARTIAL** | RTX 2070 Super | 2026-03-09 |
| SAM3 V2 TopoLoRA | TBD | ≥16 GB est | **NO** | — | — |
| VesselFM | ~6 GB est | ~10 GB est | **NO** | — | — |

### Rule for Claude Code

**NEVER estimate VRAM from architecture description.** The ONLY valid sources are:
1. `configs/model_profiles/*.yaml` — read the file, quote the number
2. Actual benchmark results — cite the log file or measurement session

If asked "how much VRAM does VesselFM need?", the answer is:
> "configs/model_profiles/vesselfm.yaml says ~6 GB inference, ~10 GB fine-tuning.
> These are ESTIMATES, not measured. No benchmark has been run."

NOT: "VesselFM uses DynUNet with wider filters, so probably ~6 GB" — this is the
exact stochastic guessing that caused this incident.

## Prior Incidents

- **2026-03-02**: SAM3 confused with SAM2, wildly wrong VRAM estimates
  (see `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md`)
- **2026-03-09**: SAM3 Hybrid OOM at patch=(64,64,3) discovered during actual training
  (see `configs/model_profiles/sam3_hybrid.yaml` comments)
- **2026-03-13**: VesselFM 10 GB → 6 GB "correction" by reviewer agent (this incident)

## Action Items

1. Add structured `vram:` section to all model profiles (T-immediate)
2. Delete stale VRAM numbers from LEARNINGS.md, CLAUDE.md, MEMORY.md — replace with
   references to model profiles
3. Run actual VesselFM VRAM benchmark on RTX 4090 during smoke test (T4.1)
4. Add pre-commit hook or test that validates VRAM numbers are only in model profiles
