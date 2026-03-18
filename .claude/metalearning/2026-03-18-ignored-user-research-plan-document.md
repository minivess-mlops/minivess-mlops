# Metalearning: Ignored User's Existing Research Plan Document

**Date:** 2026-03-18
**Severity:** MODERATE — Produced useful but misaligned deliverables
**Category:** Context blindness / "I know better" anti-pattern

## What Happened

The user asked to create batch literature research reports. They referenced an
existing plan document at:
`docs/planning/research-reports-general-plan-for-manuscript-writing.md`

This document had **4 numbered themes** with detailed descriptions, seed papers,
and reference documents:

1. **Computational Reproducibility** — reproducibility crisis, Jupyter notebooks,
   Docker vulnerabilities, pinned deps, DVC, MLflow, code smells
2. **Medical MLOps** — preclinical to clinical, QMSR, PCCP, SaMD, regulatory
3. **Biomedical Segmentation** — tubular segmentation, MONAI, foundation models,
   calibration/UQ, biostatistics, factorial design, sample size
4. **Multiphoton Microscopy for Neuroscience** — 2-photon acquisition, closed-loop
   feedback, real-time algorithms, in vivo mouse/rat studies

Claude Code instead invented its own R1-R4:
- R1: Post-Training Methods (not in user's list)
- R2: Ensemble & Uncertainty (not in user's list)
- R3: Federated Multi-Site (not in user's list)
- R4: Regulatory Post-Market (partial overlap with user's #2)

## Root Cause

1. **Did not read the user's referenced document.** The user said "these 4 themes"
   but Claude Code did not open `research-reports-general-plan-for-manuscript-writing.md`
   to see what the actual 4 themes were.

2. **Assumed the themes from context.** Instead of reading the source document,
   Claude Code inferred themes from the surrounding conversation (which was about
   agentic AI, FDA compliance, etc.) and fabricated R1-R4 from recent session context.

3. **The "I know better" anti-pattern.** Claude Code generated themes it thought
   were useful rather than following the user's explicit specification.

## Impact

- R1-R4 plans are still useful — they cover legitimate gaps
- But they are NOT what the user asked for
- 2 of the user's 4 themes (Computational Reproducibility, Multiphoton Microscopy)
  were completely ignored
- Time wasted: ~10 minutes creating plans for wrong themes

## Correct Behavior

When the user references a document with "these themes" or "these numbered items":
1. **READ THE DOCUMENT FIRST** — before planning anything
2. **Match the user's numbering** — if they have 1-4, create R1-R4 matching 1-4
3. **If additional themes seem valuable**, propose them as R5-R8, don't replace

## Prevention

- CLAUDE.md Rule #11 (Tokens Upfront): "Read ALL relevant source files before writing"
- Rule #13: "Read Context Before Implementing"
- This failure is a violation of both rules
