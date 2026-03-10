# E2E Testing Plan — Original User Prompt (Verbatim)

**Date**: 2026-03-10
**Context**: After completing flow implementations, inter-flow connectivity, Docker tier optimization (#537, #568-571), and SecOps hardening.

---

## Interactive Alignment Q&A — User Responses (Verbatim)

### Q1: Models
> Yes all the 5 models that run on my local GPU, and then the heavy SAM3 with the vesselfm have to be verified later (update the Issue: https://github.com/minivess-mlops/minivess-mlops/issues/366)

### Q2: Training Duration
> 2 epochs, 3 folds (Recommended)

### Q3: Train→Post-Training Seam Fix
> Phase 0 of e2e plan (Recommended)

### Q4: Annotation Flow
> No-op + BentoML health check

### Q5: Data Engineering Flow
> Verify MiniVess exists, run Pandera/GE validation gates, download DeepVess/TubeNet, BUT DO not use the original vesselNN dataset which was from my 2016 paper

### Q6: Ensemble Scope
> Per-model best + cross-model (Recommended)

### Q7: Biostatistics Output
> Markdown + DuckDB + Parquet (Recommended)

### Q8: Deployment / Champion Metric
> You don't remember what we have converged as the fucking best metric even? Should be blasted all across the repo in the CLAUDE.md that we used Metric Reloaded tool for this so if we choose only one best model (either single family or cross-family), then use 0.5*clDice + 0.5*MASD (with proper scaling so that they are on same range): /home/petteri/Dropbox/github-personal/minivess-mlops/docs/MetricsReloaded.html

### Q9: BentoML Response
> Full with calibration check

### Q10: Monitoring
> Full monitoring pipeline (Grafana dashboards + Evidently drift detection + Prometheus scraping BentoML metrics)

### Q11: Infrastructure
> pytest-docker plugin

### Q12: Test Location
> Makefile target + pytest (Recommended)

### Q13: Drift P0 Issue
> Data + model + concept drift (Recommended)

### Q14: Post-Training Plugins
> SWA + calibration + conformal (Recommended)

### Q15: MLflow Storage Verification
> Verify + also test artifact retrieval (full round-trip proof)

### Q16: Runtime
> Accept it, mark @slow (Recommended)

### Q17: Grafana Depth
> Create real dashboards (Recommended) — model performance panel, inference latency, data drift (Evidently), BentoML request rate

### Q18: Execution Mode
> Docker via Prefect (Recommended)

### Q19: VesselNN Exclusion
> Not sure of the evaluation bias as I am not totally sure if the vesselNN volumes are also in the minivess :D So this is one way to mitigate the data leakage and keeping the test evaluation meaningful :D

### Q20: Prefect Verification
> Prefect + MLflow + logs, and also check what gets logged to Prefect logs in the end, as it would be nice to see at least when every Flow gets run, new Task gets run, and how about the Pydantic AI events as probably logger.INFO level does not get there?

### Q21: Compose Files for pytest-docker
> Reuse existing files (Recommended)

### Q22: Agent Logging
> Structured JSONL events (Recommended) — flow start/end, task start/end, agent decisions as JSONL to Prefect logger

### Q23: Plan Phases
> 5 phases as described (Recommended) — Phase 0: fix seams, Phase 1: individual flows, Phase 2: MLflow/Prefect contracts, Phase 3: BentoML+Grafana+Evidently, Phase 4: pytest-docker harness

### Q24: Drift Issue Timing
> Create now, reference in plan (Recommended)

---

## Original Prompt (Verbatim)

Now as we have all the flows now implemented, inter-flow connectivity working (/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-flow-connectivity-execution-plan.xml
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/prefect-flow-connectivity.md) and docker images optimized with different tiers (#537, #568-571) and SecOps (/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/docker-security-hardening-mlsecops-plan-2nd-pass.xml/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/docker-security-hardening-mlsecops-plan.xml/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/docker-improvements-for-debug-training.md), we are ready to design a truly e2e testing plan to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/e2e-testing-1st-pass.xml. The annotation Flow should now be tested with some mockup approach. As annotation flow should have a queue for new un-DVC-versional volumes, and as we don't have any non-DVC data (acquisition flow should download or have downloaded minivess and the 2 external datasets that will be used as test datasets in the Analysis Flow) then the Annotation Flow just could read this empty queue and return OK and data engineering Flow either is not really doing anything as we have not implemented any Prefect tasks or micro-orchestration with Pydantic AI. Then the modeling should run the debug config for  all the models, post-training should do the model merging and post-hoc calibration, analysis should create ensembles, and the biostatistics part should create a summary of all those different models and mlflow run names ran. And then after we should deploy the best champion model (pick the best performance even though all models have garbage performance) and deploy in the deployment Flow and have BentoML serve this then, and test it by some individual volume from the external dataset and make sure that we get the dict response back (binary mask, probabilities for each voxel, dict for uncertainties, etc.). Then monitor the deployment with Grafana and Evidently. For this e2e task we don't need to be yet simulating data and model drift with synthetic samples (and open P0 Issue on this as this needs to be addressed later then when this plan is executed successfully). There are probably a lot of errors now which you will rigorously then address with plan and self-learning TDD skill implementation combo. Absolutely no panic reactive fixing! Save my prompt verbatim first, and let's start then optimizing this plan to be executed with self-learning TDD Skill. Be thorough and rigorous with planning and execution with no sloppy work and shortcuts, no tech debt introduced! As the name implies this is an end-to-end testing so we need to test the whole pipeline and improve our integration tests and make sure that the MLflow contract works, BentoML can access BentoML, grafana integration works, etc. as you probably realized ;) ask extensive interactive multi-answer questions to make sure that we are aligned with the goals of this debug plan
