# Pre-Full GCP Training QA Plan: Post-Training + Evals + Biostatistics

**Created**: 2026-03-17
**Purpose**: Plan 3 PRs (Post-Training, Evals, Biostatistics) + 2-epoch debug run before full GCP factorial

## Original User Prompt (verbatim)

> Well I would next create lite version of "#734 (GPU experiment runs) — now unblocked, needs GCP L4 spot + ralph-loop" next Run all the models on GCP, but instead for full 50 epochs, let's run for 2 epochs to have a last "debug run", and then debug+improve all the subsequent flows "Post-Training", "Evals" and "Biostatistics" implemented "fully" to ensure that the actual GCP runs for full epochs produce the experiment artifacts that we truly for the "publication gate". Let's start planning to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/pre-full-gcp-training-qa-plan.md that will plan how we get plans for three PRs 1) Post-training, 2) Evals, 3) Biostatistics (see e.g. /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/biostatistics-prefect-flow.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/biostatistics-prefect-flow-plan.xml
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/biostatistics-prefect-flow-plan-double-check.xml), and optimize with reviewer agents until converging into an optimal grand plan, which will be followed by design of executable .xml plans for each of these PRs. And let's use this as an excuse to improve our kg as we have not worked on those Flows after creating the kg at all. We need to expand our "Minimum Scientific Research" concept (remember the pairwise design, and factorial design, e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/vasculature-mlops/neurovex-methods.tex) with the post-training module (see /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/post-training-plugins-and-swa-planning.md, e.g. multi-SWA? not multi-SWAG!) that creates another factor (n trainable models x m losses * k post-training methods), and the Evals Flow creates then Ensembles (see /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/advanced-ensembling-bootstrapping-report.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/evaluation-and-ensemble-execution-plan.xml
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-flow-evaluation-and-ensemble-planning.md) in few different ways and then you have l different ways for ensembling (in addition to not ensembling the "ensembled 3 folds, cross-validation case" and using only single fold). So factorial design of n trainable models * m models * k post-training methods * l ensembling methods. You can think a bit if model merging is analogous to "traditional deep ensembles" (https://arxiv.org/abs/1612.01474) and it should be moved to Evals, and the post-training should have the model soups like all the SWA variants that actually involve some training? And remember that the Flows are decoupled also in terms of flexible compute so that the training flow always benefit from GPU, as does post-training, and so does Evals (we need to run inference on the MLflow (and optionally BentoML-served) models from previous flows with the MLflow as contract mode, see e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/ahola-2023-ahmed-agile-medical-software.md). Start by saving this prompt verbatim to that .md file, and let's start planning those 3 PRs and look our previous planning docs with fresh eyes and read them line-by-line and analyse if their optimal for our whole ML pipeline (See also our previous repo that should be improved in all those aspects: /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR). And ask me a lot of clarifying questions with your interactive multi-answer questionnaire as I feel like we have a lot of misunderstood items on our kg, as well as nodes with suboptimal information? Be thorough, as better plan well and then execute the well-planned XML rather than having to fix a lot of issues after coding with our self-learning TDD when our visions for the tasks were not aligned!

---

## Q&A Session Log

### User Clarification #1 (verbatim — factorial factor structure)

> I did not quite understand this description of yours as to me the temperature scaling is a
> post-training method itself? And the temperature scaling is not an alternative to SWA and
> Multi-SWA but rather a "chained choice" as in we could train with model A, use multi-SWA
> and with the multi-SWA use or not use temperature scaling? And not like you have just one
> way to do post hoc re-calibration so we would k post-training methods and p different
> re-calibration methods, right? And as we have such a tiny dataset so not sure if the
> temperature recalibration itself is that useful? We could provide quantitative support on
> how the "best post hoc calibration" for such a tiny dataset will affect the downstream
> statistics, so the p = 2? either post-hoc calibration, or not at all would seem the best
> for me! Obviously conformal prediction has multiple variants too, but does conformal
> prediction for UQ really affect much the evals? so should it be considered a factor in our
> design? Every algorithmic decision which has alternatives and affecting the final results
> to me is an additional factor in our paper. And don't afraid too much adding new factors
> as to me it adds scientific credibility to the manuscript as we are describing a "real
> scientific experiment" as we are submitting to a scientific journal (this is not some Show
> HackerNews type of hackathon project!). See the screenshot and the actual code of the paper
> "Deep neural networks for medical image segmentation are often overconfident..." in
> https://github.com/cai4cai/Average-Calibration-Losses which I just discovered now, so the
> ablation study would get even heavier (and scientifically more interesting, and still not so
> expensive in the end as our L4 costs are quite decent). So we would have m losses "for
> segmentation" and each loss would come with either this auxiliary loss or not (so m losses
> x 2 options for the auxiliary loss use (binary)).

**Key insight**: Factors are CHAINED, not alternatives. The full pipeline is:

```
Training: model(n) × seg_loss(m) × aux_calib_loss(binary) → checkpoint
Post-Training: SWA_method(k) → modified checkpoint
Post-hoc Calibration: recalibration(p) → calibrated model
Evaluation: ensemble_method(l) → predictions + uncertainty
```

**New paper discovered**: Barfoot et al. (2025). "Average Calibration Losses for
Improved Segmentation Calibration." *IEEE TMI*. arXiv:2506.03942. MONAI-based,
SegResNet, mL1-ACE auxiliary loss (hard-binned and soft-binned variants).
CC BY 4.0. https://github.com/cai4cai/Average-Calibration-Losses

### User Clarification #2 (verbatim — ensemble understanding wrong)

> Your understanding of the ensemble method is plain wrong

> read more carefully what we have planned in advanced-ensembling-bootstrapping-report.md,
> evaluation-and-ensemble-execution-plan.xml, prefect-flow-evaluation-and-ensemble-planning.md
> and ask better clarification questions on the ensembling

### User Clarification #3 (verbatim — ensemble factor structure explained)

> I am not sure if I understand your explanation here [...] As to me it is the single-fold
> baseline, the 3 folds per training averaged (which technically should be defined via our
> custom Class for MLflow so that MLflow Serving can create this average as we cannot average
> the model weights but we need to average; but I would not call this an ensemble in scientific
> sense), and then we have several "real ensembling" combinations which I guess you were
> describing. And given my previous explanation of what the cross-validation experiment means,
> to me the ensemble creation scientifically actually becomes even heavier (which I am totally
> fine as this is computationally as heavy as the full training): with 2 fold strategies
> (single fold, CV average) and m ensembling strategies (and as you have 3 folds there for
> the single-fold, you pick the fold that gave you the best compound metric 0.5*clDice +
> 0.5*MASD, which you hopefully also find from kg, and mention that the MASD needs some
> scaling factor as it is larger than clDice). The ensembling then would for for either the
> best single fold of each mlrun, and then with the CV average, it would be technically 2
> MLflow ensembles in succession so we probably need to create some new variant of the same
> Class? And then the most brute force ensemble would be to simple ensemble all the 4 models
> with trainable models, trained with all the 3 models, with and without including the
> zero-shot foundation models. And then ensembling the different losses for the same model
> family and not mixing mambas with dynunets for example. Is this clearer now?

**Key corrections to my understanding:**

1. **CV average is NOT an "ensemble"** scientifically — it's just the standard CV inference
   mode (average predictions from 3 fold-specific models). It needs a custom MLflow class
   that runs 3 forward passes and averages predictions.

2. **Two fold strategies** (orthogonal to ensembling):
   - Single-fold: pick the fold with best compound metric (0.5*clDice + 0.5*MASD, with
     MASD scaling since it's on a different scale than clDice)
   - CV-average: average predictions from all 3 folds (standard practice, not "ensemble")

3. **"Real" ensembling** operates ON TOP of the fold strategy:
   - Within-family ensembles: ensemble different losses for same model family
     (e.g., DynUNet×dice_ce + DynUNet×cbdice_cldice + DynUNet×dice_ce_cldice)
   - Cross-family ensembles: ensemble different model families
     (DynUNet + Mamba++ + SAM3_Hybrid, all with same loss)
   - Brute-force: ensemble all 4 trainable models × 3 losses
   - With/without zero-shot foundation models (SAM3 Vanilla, VesselFM)

4. **Hierarchical MLflow serving**: CV-average is one MLflow pyfunc class.
   Ensembling on top of CV-average would be a second MLflow pyfunc wrapping
   multiple CV-average models — "2 MLflow ensembles in succession."

5. **Compound metric for fold selection**: 0.5*clDice + 0.5*normalize(MASD)
   where MASD needs scaling to [0,1] range since raw MASD is in voxel units.

### Round 1 Answers

**Q1: Ensemble levels?**
- **Answer**: Yes — within-family + cross-family + brute-force + brute-force-with-zero-shot.
  Applied to BOTH single-fold and CV-average fold strategies.

**Q2: mL1-ACE auxiliary calibration loss?**
- **Answer**: Yes — binary factor, hard-binned hL1-ACE only. Doubles training to 72 runs (~$36).

**Q3: 2-epoch debug scope?**
- **Answer**: All 24 conditions (4 models × 3 losses × 2 aux_calib), fold-0 only. ~$6-8.

---

## Confirmed Factorial Design

### Training Factors (require GPU, produce checkpoints)

| Factor | Levels | Count |
|--------|--------|-------|
| **n: Model** | DynUNet, MambaVesselNet++, SAM3 TopoLoRA, SAM3 Hybrid | 4 |
| **m: Segmentation loss** | dice_ce, cbdice_cldice, dice_ce_cldice | 3 |
| **c: Auxiliary calibration loss** | None, hL1-ACE (hard-binned mL1-ACE) | 2 |
| **Training conditions** | n × m × c = 4 × 3 × 2 = **24** | |
| **× 3 folds** | = **72 training runs** | |

### Post-Training Factors (GPU, modify checkpoints)

| Factor | Levels | Count |
|--------|--------|-------|
| **k: Weight method** | None, SWA, Multi-SWA | 3 |
| **Post-training conditions** | 72 runs × 3 = **216 checkpoints** | |

### Evaluation Factors (GPU inference, no training)

| Factor | Levels | Count |
|--------|--------|-------|
| **f: Fold strategy** | Best single fold, CV-average (3-fold prediction avg) | 2 |
| **p: Post-hoc recalibration** | None, Temperature scaling | 2 |
| **l: Ensemble strategy** | None, Within-family, Cross-family, All-trainable, All+zero-shot | 5 |
| **Evaluation conditions** | 216 × 2 × 2 × 5 = **4,320 evaluation points** | |

### Zero-shot Baselines (separate, no training)

| Model | Dataset | Purpose |
|-------|---------|---------|
| SAM3 Vanilla | MiniVess | Domain gap baseline |
| VesselFM | DeepVess/TubeNet ONLY | External evaluation (data leakage) |

### Metrics (MetricsReloaded)

| Metric | Role | Direction |
|--------|------|-----------|
| clDice | TRUSTED (topology) | Higher = better |
| MASD | TRUSTED (surface) | Lower = better |
| DSC | FOIL (shows wrong rankings) | Higher = better |

### Cost Estimate

| Phase | Runs | GPU Hours | Cost |
|-------|------|-----------|------|
| Debug (2-epoch, fold-0) | 24 | ~48 | ~$10 |
| Full training (50-epoch, 3-fold) | 72 | ~200 | ~$44 |
| Post-training (SWA, Multi-SWA) | ~144 | ~50 | ~$11 |
| Evaluation (inference) | ~4,320 points | ~100 | ~$22 |
| Biostatistics (local CPU) | — | 0 | $0 |
| Buffer (preemption) | — | ~50 | ~$11 |
| **Total** | | **~448** | **~$98** |

### Round 2 Answers

**Q4: ANOVA library?** Both — pingouin for main report, statsmodels for supplementary interactions.

**Q5: Deploy Flow (4th plan)?** Verify full MLflow Model Registry → BentoML fetch chain.

**Q6: mL1-ACE timing?** Implement BEFORE debug. All 24 conditions in one run.

**Q7: Debug execution?** One-by-one with checkpoints. Verify MLflow artifacts between flows.

**Additional user requirement**: Read TRIPOD papers and ensure adherence:
- Collins (2025) TRIPOD+CODE reporting guideline
- Collins (2024) TRIPOD+AI reporting guidelines
- Gallifant (2025) TRIPOD-LLM reporting guideline

---

## 4 XML Plans Needed

| PR | Flow | Key Gaps | Est. Tasks |
|----|------|----------|------------|
| PR-A: Biostatistics | Flow 5 | ANOVA, calibration metrics, DCA, Riley plot | ~8 |
| PR-B: Evals/Analysis | Flow 3 | Hierarchical ensemble, UQ logging, consistency | ~10 |
| PR-C: Post-Training | Flow 2.5 | mL1-ACE loss, SWA factorial config | ~6 |
| PR-D: Deploy | Flow 6 | MLflow Registry → BentoML chain verification | ~4 |

## Biostatistics: Existing (85%) vs Missing (15%)

Already: BCa bootstrap, Wilcoxon+Holm, Bayesian signed-rank+ROPE, Friedman+Nemenyi,
ICC, Cohen's d/Cliff's delta/VDA, spec curve, ECE, temp scaling, 4 conformal variants,
calibration curves, DuckDB, 11 figures, 5 LaTeX tables, TRIPOD lineage.

Missing: Two-way ANOVA (η²/ω²), calibration slope/O:E/Brier/IPA, DCA, Riley plot,
interaction plots.

### User Clarification #4 (verbatim — cost transparency)

> And we should report in the appendix (at least) of the financial cost (debug vs real
> training) that it took to develop this repo and run the results, and if it does not
> sound like an advertisement to Skypilot, also talk how much money was saved using
> Skypilot and the use of spots instead of on-demand instances, and compared to Runpod
> pay-as-you-go instances for the "dev". Update the plan and kg accordingly, and create
> a P1 reporting Issue on this for the academic manuscript, and ideally this information
> would be saved to MLflow as well (if we can programmatically get the instance cost at
> the time of the execution and the time it took to run the instance, then we could
> simple sum all the costs together from the MLflow artifact storage with some filters
> for debug and real runs)

**Issue created**: #795 (P1: Financial cost reporting for manuscript appendix)

**MLflow integration**: `cost/total_usd` already logged per run (slash-prefix from PR #793).
Need to add: `cost/instance_type` param, `cost/spot_vs_ondemand_savings_pct` metric.
Biostatistics flow generates cost appendix table by querying MLflow.

### 5th Plan Added: Cost Reporting

| PR | Flow | Key Work |
|----|------|----------|
| PR-E: Cost Reporting | Biostatistics + manuscript | Cost appendix table, spot vs on-demand analysis, SkyPilot savings |
