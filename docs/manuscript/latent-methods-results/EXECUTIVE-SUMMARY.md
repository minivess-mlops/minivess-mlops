# ARBOR: Executive Summary for Co-Authors

**Cold-resume document** — read this in 10 minutes to understand what this paper is about.
Then read `latent-methods-results.tex` for the full scaffold.

---

## What is this paper?

**One sentence:** ARBOR is an open-source MLOps platform that makes reproducible
multi-model benchmarking of 3D vascular segmentation a matter of configuration, not
infrastructure engineering.

**What it is NOT claiming:**
- NOT SOTA segmentation — the *platform* is the contribution
- NOT the only MLOps solution for biomedical imaging
- NOT specific to multiphoton microscopy (works for any MONAI 3D segmentation)

**Target venue:** Nature Protocols or Nature Methods (IF ~15--48)

**Comparable papers:**
- Pachitariu & Stringer (2022) "Cellpose" — Nature Methods
- Windhager et al. (2023) "End-to-end workflow" — Nature Protocols

---

## The 5-Line Summary

1. We built an MLOps platform (ARBOR) for 3D biomedical segmentation using MONAI
2. One config file = new model or new dataset — no code changes
3. Five Prefect flows + Docker-per-flow = complete reproducible pipeline
4. cbdice_cldice loss wins on topology (clDice=0.906) with -5.3% DSC penalty
5. Spec-driven agentic development (Claude Code + SDD) accelerated the build

---

## Status as of 2026-03-15

| Section | Status | Blocker |
|---------|--------|---------|
| M1-M12 (Methods) | Scaffold (bullets + Mermaid) | Need co-author narrative |
| R1 (Reproducibility) | **Complete** | — |
| R2 (DevEx benchmark) | Missing | External replication needed |
| R3a (DynUNet loss) | **Complete** | — |
| R3b (SAM3 comparison) | **BLOCKED** | GPU runs (target 2026-03-20) |
| R3c (VesselFM) | **BLOCKED** | GPU runs on DeepVess/TubeNet |
| R4 (Ensemble/UQ) | Partial | Full conformal after GPU runs |
| R5 (Agentic) | Self-reported | External validation or move to appendix |

---

## Key Technical Decisions (for co-authors to review)

1. **cbdice_cldice as default loss** — topology > voxel overlap for vasculature
2. **SAM3 (Nov 2025) not SAM2** — completely different model, 648M params
3. **T4/Turing GPUs BANNED** — no BF16, causes NaN in SAM3 validation
4. **VesselFM on external data only** — data leakage on MiniVess
5. **GitHub Actions CI disabled** — intentional (credits), local validation only

---

## Files to Read Next

- `methods/methods-03-architecture.tex` — Full system architecture
- `methods/methods-06-model-adapters.tex` — ModelAdapter ABC
- `results/results-01-reproducibility.tex` — Verification proof
- `results/results-03-models.tex` — Loss comparison numbers
- `knowledge-graph/manuscript/claims.yaml` — 12 scientific claims + evidence
- `knowledge-graph/manuscript/limitations.yaml` — Known limitations

---

## Contact

Primary author: Petteri Teikari
Repo: github.com/petteriTeikari/minivess-mlops
License: MIT
