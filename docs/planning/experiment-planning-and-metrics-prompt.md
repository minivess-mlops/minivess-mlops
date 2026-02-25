# Experiment Planning & Metrics — Original Prompt

> Verbatim user prompt captured on 2026-02-25

---

Next create a branch feat/experiment-pipeline-variation and plan how to run the actual experiment now. How well are you able to run the whole MONAI-based baseline experiment yourself? Do we need any Optuna for this even? Basic grid search with the hyperparameters handled by hydra? We could start with the dynUnet, and only controllable training parameters to start with is the number of epochs for convergence, early stopping criteria (?), dynamic U-Net handling learning rate schedule (?). Let's keep the training parameters as vanilla and default and see what we get off-the shelf. The parameter that we could play with would be the loss used for optimizations and linear combinations of different losses (compound losses). Metrics Reloaded (https://metrics-reloaded.dkfz.de/metric-selection-mapping?category=Semantic_Segmentation https://www.nature.com/articles/s41592-023-02151-z https://arxiv.org/abs/2206.01653 https://github.com/Project-MONAI/MetricsReloaded) should be treated as the ground truth for this and it gives this interactive selection tool for the correct metrics (/home/petteri/Dropbox/github-personal/minivess-mlops/docs/MetricsReloaded.html): 1) ✔️ Selecting an overlap-based segmentation metric completed
As overlap-based segmentation metric, Center Line Dice* has been added to the metric candidates pool. You are ready for the next step!

*If the center line of structures should be assessed simultaneously with general overlap agreement, Center Line Dice can be combined with DSC, IoU or Fᵦ Score.

ℹ️ If problem fingerprints differ between classes (e.g., simultaneous segmentation of convex and tubular structures), a class-specific metric pool must be generated (background class: optional). Add the selected metric(s) to class-specific metric pool. (so centerline dice with and without DSC), 2) "Selecting boundary-based segmentation metrics (if any; S7)
Key weaknesses of overlap-based metrics include shape unawareness and limitations when dealing with small structures or high size variability. Our general recommendation is therefore to complement an overlap-based metric with a boundary-based metric.

Repeat for each class: If problem fingerprints differ between classes (e.g., simultaneous segmentation of convex and tubular structures), a class-specific metric pool must be generated (background class: optional)." -> "✔️ Selecting a boundary-based segmentation metric completed
As boundary-based segmentation metric, Mean Average Surface Distance (MASD) has been added to the metric candidates pool.

If problem fingerprints differ between classes (e.g., simultaneous segmentation of convex and tubular structures), a class-specific metric pool must be generated (background class: optional). Add the selected metric(s) to class-specific metric pool.

You are ready for the next step!" -> Additional metrics for your study
In addition to the recommended metrics, we encourage you to select further additional metrics specific to your problem if needed.
Application-specific metrics:
The pool of standard metrics can be complemented with custom metrics to address application-specific complementary properties. For example, in liver segmentation, the absolute volume of the segmented liver is an important clinical parameter, so the Absolute Volume Error could be added to the metric pool.
Non-reference-based metrics:
Validation and evaluation of algorithms could go far beyond purely technical performance. Thus, non-reference-based metrics assessing speed, memory consumption or carbon footprint, for example, can be added to the metric pool. 3) clDice from https://arxiv.org/abs/2003.07311 (https://github.com/jocpae/clDice), . And let's use the application specific losses found from the literature then a) https://arxiv.org/abs/2407.01517 "centerline boundary Dice (cbDice) loss function" https://github.com/PengchengShi1220/cbDice /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/shi-2024-centerline-boundary.md; b) Smooth clDice from /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/morand-2025-smooth-cldice-vascular-segmentation.md; c)   ; extra) explore this as well as extra approaches https://arxiv.org/html/2601.18045v1 (https://github.com/NatsuGao7/TopoUnet.git) - https://arxiv.org/html/2506.12395v1 - https://arxiv.org/html/2509.03938v1 (https://github.com/Puzzled-Hui/TopoSculpt) - https://arxiv.org/html/2503.05541v1 (https://github.com/jmlipman/CoLeTra) - https://arxiv.org/html/2506.16556v1 - https://arxiv.org/abs/2203.06425 - https://pubmed.ncbi.nlm.nih.gov/38320370/ (https://github.com/rmaphoh/feature-loss) - https://doi.org/10.1007/978-3-031-72111-3_67 (https://github.com/cesaracebes/centerline_CE) - https://arxiv.org/html/2503.07369v1 for skeletonization of the centerline - https://link.springer.com/chapter/10.1007/978-3-032-05472-2_22 for how to synthesize 3D vascular images https://github.com/pshavela/tubular-aware-lfm . So 2 metrics from Metrics Reloaded and 2-3 main application specific metrics (clDice and cbDice are obligatory, and assess if the "student work" on smooth clDice is of good quality and can we even implement it fast as no code was shared. All the extra metrics that come with code should be integrated to this repo, and we can later decide if we want to use them in the experiment (depending basically how long does this all take to train). Save my prompt verbatim to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/experiment-planning-and-metrics-prompt.md and then let's start planning the next phase with actual training of the dynamic U-Net with our MlOps system with few parameters and validating that the MLflow part is correct with the chosen metrics and all. Save this plan to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/experiment-run-to-mlflow-plan.md and make sure that the experiment artifacts are saved correctly, DVC versioning for data, hydra configs are versioned, git hash is stored correctly, etc. No need yet to deploy anything and use the MLflow-as-contract for the BentoMl part with the dynaconf version control. Does this seem clear? Ask questions, please!


 4 agents finished (ctrl+o to expand)
   ├─ Explore (Explore existing training pipeline) · 25 tool uses · 86.2k tokens
   │  ⎿  Done
   ├─ Task (Research loss papers and repos) · 40 tool uses · 79.7k tokens
   │  ⎿  Done

   ├─ Task (Research MetricsReloaded and MONAI metrics) · 39 tool uses · 91.7k tokens
   │  ⎿  Done
   └─ Explore (Research MiniVess dataset details) · 27 tool uses · 68.1k tokens
      ⎿  Done

● All 4 research agents returned. Here's my synthesis and questions.

  Research Summary

  Good news: The pipeline is real, not a scaffold. The trainer has a working training
  loop with AMP, gradient clipping, early stopping, and MLflow tracking. All 8 loss
  functions have real PyTorch forward passes. The DynUNet adapter, data loader, and
  Hydra-zen configs are functional.

  Metrics landscape:
  - MetricsReloaded is already a dependency and provides exact clDice + MASD (the two
  metrics your toolkit selected)
  - MONAI has MetricsReloadedBinary wrapper for seamless integration
  - Training-time: fast TorchMetrics (Dice, F1) on GPU
  - Evaluation-time: MetricsReloaded (clDice, MASD, DSC, HD95, NSD) on CPU

  Loss function landscape for the experiment grid:

  Loss: DiceCELoss
  Status: Ready
  Source: MONAI built-in
  ────────────────────────────────────────
  Loss: SoftclDiceLoss
  Status: Ready
  Source: MONAI built-in
  ────────────────────────────────────────
  Loss: SoftDiceclDiceLoss
  Status: Ready
  Source: MONAI built-in (Dice+clDice)
  ────────────────────────────────────────
  Loss: VesselCompoundLoss
  Status: Ready
  Source: Already in repo (DiceCE+clDice)
  ────────────────────────────────────────
  Loss: TopologyCompoundLoss
  Status: Ready
  Source: Already in repo (DiceCE+clDice+Betti)
  ────────────────────────────────────────
  Loss: BettiLoss
  Status: Ready
  Source: Already in repo
  ────────────────────────────────────────
  Loss: cbDice
  Status: Needs integration
  Source: PengchengShi1220/cbDice (Apache-2.0, 3D, already uses MONAI)
  ────────────────────────────────────────
  Loss: Smooth clDice
  Status: No code exists
  Source: Student paper, 2D only — skip for now
  ────────────────────────────────────────
  Loss: Centerline CE
  Status: ~100 LOC to port
  Source: cesaracebes/centerline_CE (Apache-2.0, 3D)

  MiniVess dataset: 70 volumes, ~939 MB, download script ready, patch size 128×128×32.

  Now, here are my clarifying questions:

● User answered Claude's questions:
  ⎿  · What GPU hardware is available for training? This determines batch size and
     training time. → Could we implement some ready-made profiles as we are using
     Hydra with the composable architecture? So we could have 1) CPU (16-128 GB),
     2) low-cost GPU (My 2070 Super with 8 GB of RAM), 3) high-end consumer GPU
     (RTX 4090, 24 GB), 4) Local NVIDIA DGX Spark server, 5) High-end Cloud,
     single-instance, 6) High-end, multi-gpu instance (e.g. 8 GPUs). So that the
     dynamic U-Net batch sizes and other parameters would be chosen accordingly?
     The researcher could simple provide an input argument to "make_train" or "uvx
     train.py" --compute-profile CPU? Would this make sense and provide better
     DevEx, and obviously we can provide instruction how people can modify the
     composable configs as they wish, but provide people a quickstart without
     having to figure the low-level details if they do not want (progressive
     disclosure in UX)
     · For the first experiment run, use 5-fold cross-validation or a simpler
     train/val split? 5-fold on 7 losses would be 35 training runs. → Remember that
      one of the design goals of the v0.1 release was to have the CV folds as
     flexible as possible and easy tweaking of the hard-coded splits (for
     repeatable experiments). So we need to debug that the CV folding works
     properly, so the 3-fold CV is the lightest option when we want to test
     ensembling with different models (from artifact store) with winner-takes it
     all, averaging, any cool method emerging, etc.
     · Should I integrate cbDice loss (from PengchengShi1220/cbDice) now, or start
     with the 6 losses already in the repo and add cbDice as a second experiment
     batch? → Can you integrate all the losses that I gave you know, including the
     extra ones as well if they come with github repos. Let's not integrate losses
     from papers. And then let's start with two application-specific losses in the
     actual training (clDice and cbDice)
     · Should I attempt to run the actual training myself (download data + execute
     training loop), or create the plan and scripts for you to run manually? → Well
      let's start implementing everything and training for 1 epoch. See the v0.1
     for some train tests which are quite rudimentary. We should implement some
     train tests that use both REAL DATA and synthetic data. So we should make sure
      that the data is correct (with no NaNs from data augmentation or anything
     other weird), and that the training script works (which can be tested with
     synthetic data as well in addition to the real data) so that there is no bug
     in the training code. Using REAL data is very important now especially for
     local tests! Let's think of efficient Github Actions CI tests as we need to
     make the data available there as well, with caching and everything. So the
     Github Actions CI is another PR then after this

● Excellent answers. Let me now write the comprehensive experiment plan incorporating
  all your decisions, then we'll enter plan mode for implementation.


The ensembling requirement is also mandatory! Is this clear! We should be         
  flexibly ensemble easily 1) different hyperparameter runs of the same model (e.g. 
   dyNUnet of same size optimized with different loss functions), 2) different      
  folds of the same model (3-fold CV), 3) totally different models (dynUnet of      
  different loss functions with some SAMv3 variants). Remember that this is should  
  be a tool for researchers allowing them to do quick experimentation easily and    
  fast! Researchers should be just be doing "I found this interesting new loss from 
   arxiv and would like to see how this performs", "this new paper showed that if   
  combine MAMBA, Transformer and CNN models as ensemble they complement each other      
  and provide more robust segmentation than each of the single models". So we don't 
   want to be too opinionated how the models should be ensembled, but create as         
  flexible architecture allowing different kinds of ensembles done while running     
  the experiment, and EVEN better, allowing flexible post-training ensemble         
  construction so that all the models used are saved to the artifact store and      
  there is a MLflow deploy built for those so that you can run inference on every   
  mdoel saved without having to re-run the experiments. Is this clear? Plan how to  
  make sure that you don't build with the self-learning TDD coding skill anything   
  that makes this vision hard to achieve                                            



