# Cold-Start: Pre-GCP Training QA — 4 XML Plans + 2-Epoch Debug

**Created**: 2026-03-17
**Branch**: `docs/pre-gcp-qa-plan` (on top of `main`)
**Plan**: `docs/planning/pre-full-gcp-training-qa-plan.md`

## Paste this to resume:

```
Continue the pre-GCP training QA planning session. Read the plan at
docs/planning/pre-full-gcp-training-qa-plan.md for full Q&A context.

## What's DECIDED (from previous session Q&A):

1. **6-factor factorial design confirmed**:
   - n=4 models (DynUNet, Mamba++, SAM3 TopoLoRA, SAM3 Hybrid)
   - m=3 losses (dice_ce, cbdice_cldice, dice_ce_cldice)
   - c=2 aux_calib (None, hL1-ACE from Barfoot et al. 2025 IEEE TMI)
   - k=3 post-training (None, SWA, Multi-SWA)
   - p=2 post-hoc_recalib (None, temperature scaling)
   - l=5 ensemble strategies (None, within-family, cross-family, all-trainable, all+zero-shot)
   - + 2 zero-shot baselines (SAM3 Vanilla, VesselFM external-only)
   - Fold strategies: single-best-fold + CV-average (NOT called "ensemble")

2. **4 XML execution plans needed** (PR-A through PR-D):
   - PR-A: Biostatistics Flow — ANOVA (pingouin+statsmodels), calibration metrics, DCA, Riley plot
   - PR-B: Evals/Analysis Flow — hierarchical ensemble evaluation, UQ logging
   - PR-C: Post-Training Flow — mL1-ACE loss, SWA factorial config
   - PR-D: Deploy Flow — MLflow Registry → BentoML verification chain

3. **mL1-ACE implemented BEFORE debug** (all 24 conditions in one run)

4. **Debug run**: 24 conditions × fold-0 × 2 epochs (~$6-8 GCP L4 spot)
   Executed one-by-one: Train → verify → Post-Training → verify → Evals → verify → Biostatistics → verify → Deploy

5. **TRIPOD compliance**: Read and incorporate TRIPOD+AI, TRIPOD+CODE, TRIPOD-LLM:
   - /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/collins-2025-tripod-code-reporting-guideline-protocol.md
   - /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/collins-2024-tripod-ai-reporting-guidelines-clinical-prediction-models.md
   - /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/gallifant-2025-tripod-llm-reporting-guideline.md

6. **Biostatistics Flow is 85% done**. Gaps: ANOVA η²/ω², calibration slope/O:E/Brier/IPA, DCA, Riley plot, interaction plots.

7. **Metrics**: clDice (trusted), MASD (trusted, needs scaling), DSC (foil)

8. **Cloud**: GCP L4 spot (NOT RunPod). ~$98 total for full run.

## NEXT STEPS:

1. Read TRIPOD papers (3 files above)
2. Read foundation-PLR statistical methodology for comparison
3. Write 4 executable XML plans (PR-A through PR-D)
4. Execute with self-learning TDD skill
5. Run 2-epoch debug on GCP
6. Verify end-to-end pipeline
7. Full 50-epoch production runs

## Key files to read:
- docs/planning/pre-full-gcp-training-qa-plan.md (Q&A + factorial design)
- docs/planning/factorial-design-demo-experiment-plan.md (biostatistics detail)
- docs/planning/mlflow-metrics-and-params-double-check-before-large-scale-gcp-work.md (slash-prefix)
- knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml (6 models)
- knowledge-graph/decisions/L3-technology/primary_metrics.yaml (trusted/foil metrics)
```
