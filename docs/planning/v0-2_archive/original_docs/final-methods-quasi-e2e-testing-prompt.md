---
title: "User Prompt: Final Methods Quasi-E2E Testing"
status: reference
created: ""
---

# User Prompt (Verbatim) — Final Methods Quasi-E2E Testing

> Let's next create test/final-quasi-e2e-testing branch which should test all the permutations that we have implemented. Is it possible to create some test_all_permutations.py test that knows how to dynamically query the repo for all implemented models, all losses, all the metrics in multi-metrics validation, all the post-training / post hoc methods, all the Analysis ensemble methods, all the deployment methods. And this is more or less the algorithm and method selection that we have for the academic manuscript submission, so part of this task definitely should be about making sure that ALL the tools implemented have some mechanism, schema, whatever, etc. that allows dynamic discovery and permutation combinationarial setup defined on-the-fly by a deterministic .py file while creating a data-driven .yaml on the actually picked combinations, so we could later also re-run this test run if needed! But in real-world maintainable repo I would like the all combinations to be discovered dynamically as the DevEx would be abysmal if someone have to be manually defining this combinations. Do some deep web search on possible ways to make this happen! Anb obviously there could be two variants: 1) the hardcore full brute force combinations of all possible combinations (number of combinations explode quite fast), 2) the more practical one if let's say most of the post-hoc methods are model-agnostic so they don't need to be run after each every model and every possible loss on every different model, right? The post-hoc methods need to be run only on model and one loss. Identify all these and encode this reproducibly in some schema, .yaml, etc. for each of the implemented method/model. Does this make sense to you? Then ensembling, model merging, model soups do not necessarily work the same way for all the types of models. Like if dynUnet has different 6 loss function options, whereas SAMv3 might just have one option? (when finetuning) so we cannot do brute force grid search and simplification of it as there are conditional operations in the execution DAG based on what is being computed, right? So this should be some conditional, decision tree type of structure? When dynunet has been run, these methods are now possible in post-hoc, and this type of ensembling is possible, and this type of deployment should be done in BentoML. And then in Ensembling part in the Prefect Analysis Flow we should still have an easy task to ensemble across very heterogeneous model architectures as the deployment (both MLflow and BentoML) API is standardized in the sense that we always get the safe dict schema back with binary mask, probablities per volume, dict for various uncertainty quantifications, etc.? And then this schema is shared by all the methods and only defined in one place. And if you have some probabilistic methods, they can produce very different keys in the dict compared to deterministic models, but then you should have None on some key? Or how do you think that the missing keys should be handled to have harmonized schema for serving output? And finally after your fixed, you should create a .XML plan on how to actually run some bash script .sh file that can reproducibly do all these. Output by default to MLFlow and everything the same as previously implemented. This should now test EVERYTHING that we have implemented so far and hunt for any remaining bugs when running this with real data. Save my prompt verbatim, and then start planning with reviewer agents until convergence of excellent plan to be run with our self-learning TDD skill: /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/final-methods-quasi-e2e-testing.md

←  ☒ Schema  ☒ Reduction  ☒ Capabilities  ☒ Cross-arch  ✔ Submit  →

Review your answers

 ● For the harmonized serving output schema: should missing UQ keys (e.g., uncertainty_map for
 deterministic models) be `None` (explicit absence) or should we use Pydantic discriminated unions
 where each model type declares exactly which fields it produces?
   → Optional[T] = None (Recommended)
 ● For combinatorial test reduction: `allpairspy` can reduce e.g. 720 full combinations to ~20
 pairwise-covering cases. Should we use this for the 'practical' variant, or just manual
 expert-defined reduction based on model-agnostic properties?
   → I don't know when I would have the time to run the full brute force 720 combination run :D So
   we should definitely be implementing the full run, but now for debugging this repo, we
   definitely need the practical variant, which hopefully in the future should be easy to
   automatically be constructed, if Claude Code guardrails (CLAUDE.md) and test suite enforces
   certain type of schemes for all the existing methods in ALL THE FLOWS so that the deterministic
   .py function has an easier job figuring out what methods go with what others. HIghest priority
   should be total automation of this for all the existing methods and excellent guardrails that
   help both Claude Code generate new method as efficiently and having test suite that test for
   these things. So what would your preferred way of automatically creating this schema? Like
   Claude Code should figure out that these are always needed when adding new methods and it should
    enforce the user's prompt to go through some hoops even though the user did not think of
   explicitly asking this schema creation. E.g. if user wants to add new loss that only works with
   SAMv3, then the loss should have the schema auto-generated. ANd the test should pick the lack of
    schema with tests (and maybe by precommit already) as this is really crucial for
   reproducibility and maintainability that all the new methods are added to match the old logic,
   and policing any new methods that go against the rules
 ● Per-model loss compatibility: should we define valid losses per model in a YAML capability file
 (e.g., `model_capabilities.yaml`) or derive it programmatically from the model adapter's
 properties?
   → This is a bit tricky task as you could by default think that all the losses work everywhere
   that we have added, and there are millions of losses out there! Should the exceptions be rather
   said, if for example LORA does not work with something default?
 ● For cross-architecture ensembling (DynUNet + SAM3 outputs together): should this be a
 first-class feature tested in this round, or deferred since it requires all models to actually
 produce compatible checkpoint outputs?
   → Why do we need to mock something now? I mean the ensembling should run on every time when you
   run Prefect Flow with all the possible ensemble combination that we have defined in .yaml config
    for the Ensembling Task inside the Prefect Analysis Flow, right? So in the MLflow there is
   always some data? (well as in now when we have run the experiments). And we should test that it
   works with various combos of data there. What is the point of using mockup from the start as it
   is static and does not catch any glithces in the real data that we are getting? I mean even if
   we train the dynunet just for one epoch, we still get a model that we can run inference on. The
   performance will be awful but those numbers will still be real and enable ensembling, right? And
    then prior submission we should create some golden dataset for ensemble from real training runs
    for enough epochs so that people have realistic golden dataset for evaluation ensembling! Does
   this make sense

So now the .sh script that we create should read some test_practical_combos.yaml that is based
  on the defaults.yaml and only changing few config keys to "debug" ones, like train for 1 epoch,
  whether to use the full dataset or using let's say 2 train volumes and 2 val volumes, and
  testing then in the Prefect Analysis Flow the 2 (?) external test datasets that we have DVC
  versioned. And whether a similar 4 volume subset should be chosen for each of those two
  external test datasets. Remember that now the focus in this branch should be on the training
  mechanics testing and not the model performance! No point training for days if the architecture
  is riddled with bugs!
