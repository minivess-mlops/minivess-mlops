# Metalearning: Factual Errors Introduced During Manuscript Writing from Cover Letter

**Date**: 2026-03-16
**Session**: sci-llm-writer PHASE 4 content fill for NEUROVEX co-author teaser
**Model**: Claude Sonnet 4.6
**Severity**: HIGH — three factual errors reached the committed .tex files

---

## What Happened

During PHASE 4 filling of the NEUROVEX co-author teaser scaffold, three distinct
categories of error were introduced. Two were caught by the user mid-session; one
required the user to explicitly flag the error after reading the generated diff.

---

## Error 1: Wrong Citation Key (CITATION RULE VIOLATION)

### What went wrong
Used `\citep{teikari_2023_minivess}` throughout the manuscript. Petteri Teikari is
the **second** author on this paper. The first author is Charissa Poon.

**Correct key**: `poon_2023_minivess`
**Paper**: Poon, Charissa, Petteri Teikari, Muhammad Febrian Rachmadi, Henrik Skibbe,
and Kullervo Hynynen. 2023. "A Dataset of Rodent Cerebrovasculature from in Vivo
Multiphoton Fluorescence Microscopy Imaging." Scientific Data 10 (1): 1.

### Root cause
The wrong key was **already present** in the scaffold from the previous session (PHASE 2).
During PHASE 4, the key was propagated into new content (abstract, intro) without
checking it against the Zotero RDF.

**CLAUDE.md Rule #2 says**: "BEFORE writing ANY `\cite{}` command, search Zotero RDF
first. Never invent citation keys."

The rule was violated in two ways:
1. The pre-existing wrong key in the PHASE 2 stub was not caught when the file was read
2. New `\citep{}` commands were added by copying the same wrong key without RDF verification

### Fix applied
`sed -i` replace-all: `teikari_2023_minivess` → `poon_2023_minivess` across all .tex files

### Prevention
- **Whenever reading an existing .tex file before editing it**: scan for ALL `\citep{}`
  commands and verify each against the Zotero RDF before proceeding.
- Never assume a pre-existing citation key is correct — verify it on first read.

---

## Error 2: Wrong Platform Scope ("single-researcher academic setting")

### What went wrong
Wrote "NEUROVEX targets MLOps Maturity Level 4 in a **single-researcher academic setting**"
in the abstract, introduction, and conclusions.

**Correct framing**: The platform targets academic research groups ranging from **solo
researchers to multi-lab collaborations** (2-5 person labs up to multi-institutional groups).

### Root cause
The cover letter (§1) contains the phrase "MLOps Maturity Level 4 in a single-researcher
academic setting" to describe the **current development context** (the primary researcher's
workstation), NOT the intended user base.

The cover letter §1.5 explicitly describes the target persona as "A typical NEUROVEX user
is a computational neuroscience or biomedical imaging PhD lab with 2–5 members." This
directly contradicts "single-researcher."

The error was **misinterpreting a contextual phrase** (describing current state) as a
**design target** (describing intended scope).

### Fix applied
Changed to: "for academic research groups ranging from solo researchers to multi-lab
collaborations"

### Prevention
- When the cover letter describes a "current state" (what exists on the author's machine),
  do NOT use it to describe the "target audience" (who the platform is for).
- Before writing any scope claim, cross-check against:
  1. The target persona section (§1.5 of cover letter)
  2. The claims.yaml (C1: "any lab can adopt NEUROVEX")
  3. The platform's explicit design goals

---

## Error 3: Invented Future Plan for SAM3 TopoLoRA

### What went wrong
Wrote: "SAM3 TopoLoRA is deferred: it requires >22.66 GiB VRAM and exceeds the capacity
of the RTX 4090 used for development; **a cloud A100 run is planned**."

The words "a cloud A100 run is planned" were **not in any source**. The cover letter says
"Status: deferred" with no planned timeline.

### Root cause
Extrapolation beyond the source material. The fact that GCP and RunPod are available
(true) combined with the OOM reason (true) led to generating a plausible-sounding
conclusion ("therefore a cloud run is planned") that was not stated anywhere.

**The actual situation** (confirmed by user): GCP and RunPod ARE available with A100-class
instances. So the claim was actually directionally correct — but it was invented, not
sourced. The correct phrasing is "a full run on A100-class cloud GPU (GCP or RunPod)
is scheduled after the core model family results are complete" — now sourced from the
user's explicit correction.

### Fix applied
Changed to: "a full run on A100-class cloud GPU (GCP or RunPod) is scheduled after
the core model family results are complete"

### Prevention
- **Only write future plans that are explicitly stated in source documents**
- When a source says "deferred" without a plan, write "deferred" — do not extrapolate
  a timeline or a plan that wasn't stated
- If the user provides additional context that clarifies a plan exists, incorporate it
  immediately and note the source

---

## Systemic Issue: Cover Letter Is NOT a Reliable Single Source of Truth

The cover letter (`docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md`)
was written as a summary briefing document. It contains:
- Accurate numbers (clDice, DSC, MASD) sourced from KG YAML files ✅
- Accurate flow architecture taxonomy ✅
- **Imprecise scope descriptions** that describe current state, not design intent ❌
- **Missing explicit plans** that exist but weren't documented at cover letter write time ❌

### Priority order for source verification

When the cover letter conflicts with other sources, use this priority order:
1. **Primary source files** (flows.yaml, claims.yaml, dynunet_loss_variation_v2.yaml) — authoritative
2. **Zotero RDF** — authoritative for citations
3. **User's real-time corrections** — authoritative for intent
4. **Cover letter** — useful context but must be cross-checked

### Action item for next session
The cover letter should be audited against primary KG sources (claims.yaml, flows.yaml)
and corrected where it contains scope descriptions or status claims that are imprecise.
In particular: the "single-researcher academic setting" phrase should be updated to
reflect the actual multi-audience target.

---

## Quality Check: What Was Actually Correct

To be fair: the technical content was largely correct:
- All four loss condition numbers (DSC, clDice, MASD) match the KG YAML ✅
- clDice std = 0.0075 (not 0.008) correctly maintained ✅
- R1 scope guard correctly maintained (single machine, NOT cross-machine) ✅
- 73/73 artifact checks and 8.26s timing correct ✅
- hpo_flow correctly excluded per D4 guard ✅
- Writing guards correctly applied (no "Langfuse in M9", no KG IDs in contributions) ✅
- Level 4 overclaim avoided throughout (all occurrences say "targeting") ✅

The errors were in citation authorship, platform scope framing, and one extrapolated plan.

---

## Recommendation for Opus 4.6

If using Opus 4.6 for the next manuscript writing session:
1. Provide the primary KG YAML files as context (not just the cover letter)
2. Run `biblio check manuscript.tex` before writing any citations
3. Have the model explicitly state the source for every factual claim it writes
4. Ask the model to flag any claim it derives by inference rather than direct source lookup
