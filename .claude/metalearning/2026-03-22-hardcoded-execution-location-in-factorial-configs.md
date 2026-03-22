# Metalearning: Hardcoding Execution Location in Factorial Configs

**Date**: 2026-03-22
**Severity**: P0-CRITICAL — recurring anti-pattern, 9th+ occurrence of rigidity
**Rule violated**: TOP-1 (Flexible MLOps), Design Goal #1 (DevEx)

## What Happened

Claude Code repeatedly labels factorial design layers with hardcoded execution
locations: "Layer A (Training, **cloud GPU**)", "Layer B (Post-training, **CPU, local**)".
This bakes deployment choices into the factorial STRUCTURE, which should be
agnostic to where flows execute.

The user corrected: "We should NEVER hard-code on where a specific Flow is executed.
If a person wants to execute the Train Flow on one's local CPU and wait for 180 days,
it is their choice. The whole idea with the Prefect Flow design was that we are as
flexible as possible where things are executed."

## Why This Is Wrong

1. **TOP-1 violation**: The platform must support arbitrary deployment topologies.
   A lab with 8×A100 runs everything on-premise. A student runs on their laptop.
   A cloud user uses SkyPilot. The YAML describes WHAT to sweep, not WHERE.

2. **Analysis Flow also needs GPU**: Claude keeps saying "analysis = CPU" but the
   Analysis Flow runs **inference on served models from MLflow** — that requires GPU.
   The user pointed this out explicitly.

3. **Prefect = location-agnostic orchestration**: Prefect flows can run anywhere —
   Docker container, SkyPilot job, local Python process. The factorial config
   defines the experiment DESIGN (factors × levels), not the deployment STRATEGY.

## Correct Mental Model

```
Factorial Config (WHAT):          Deployment Config (WHERE):
  factors:                           deploy:
    model_family: [...]                training: skypilot/gcp
    loss_name: [...]                   analysis: skypilot/gcp
    aux_calibration: [...]             biostatistics: local
    post_training_method: [...]        dashboard: local
    recalibration: [...]
    ensemble_strategy: [...]
```

The factorial config has NO mention of GPU, CPU, cloud, local, SkyPilot, Docker.
Those are deployment concerns, configured separately.

## Prevention Rule

**NEVER label a factorial layer with execution location.**
- WRONG: `Layer A (Training, cloud GPU)`
- CORRECT: `Layer A (Training)` — where it runs is the user's choice
- WRONG: `Layer B (Post-training, CPU, local)`
- CORRECT: `Layer B (Post-training)` — currently runs in same flow as training

**NEVER assume a flow's execution location is fixed.**
Analysis Flow can run on GPU (inference) or CPU (if using pre-computed predictions).
The Prefect deployment config, not the factorial YAML, determines WHERE.

## Occurrence Count

This is at minimum the 9th time execution location was hardcoded into factorial
design documentation or configs.

## Cross-References

- `configs/factorial/debug.yaml` — "Layer B: Post-training factors (CPU, local)" (WRONG)
- `configs/factorial/paper_full.yaml` — same label (WRONG)
- `.claude/metalearning/2026-03-22-wrong-metalearning-doc-failure-mode.md`
- `docs/planning/context-compounding-and-learning-repo-plan.md` (prevention plan)
