---
title: "Intent Summary — All Verbatim User Prompts"
date: 2026-03-16
status: living_document
purpose: >
  Ground-truth intent index for the MinIVess MLOps repository. Every plan, report,
  and implementation is a derivative of these user instructions. This document is
  the authoritative source for "what the user actually asked for."
---

# Intent Summary

## Prompt Index

All verbatim user prompts preserved in `docs/planning/`. These are the primary
sources — plans and implementations are secondary derivatives.

| # | File | Date | Intent (1-line) |
|---|------|------|-----------------|
| P1 | `modernize-minivess-mlops-plan-prompt.md` | 2026-02-23 | Refactor v0.1-alpha to production MLOps, MONAI-first, Nature Protocols paper, agentic/FMOps |
| P2 | `experiment-planning-and-metrics-prompt.md` | 2026-02-25 | DynUNet training pipeline, compound losses, clDice/cbDice, 3-fold CV, ensembling, MLflow |
| P3 | `agentic-architecture-self-reflection-for-sdd-and-beyond-paper-as-agentic-PROMPT.md` | 2026-03-01 | Second-pass: paper novelty = agentic dev with Claude Code, SDD + probabilistic PRD |
| P4 | `advanced-segmentation-double-check-prompt.md` | 2026-03-04 | Review 40+ SOTA papers: segmentation, UQ, foundation models, Mamba, synthetic data |
| P5 | `e2e-testing-user-prompt.md` | 2026-03-10 | Design E2E testing: all flows, inter-flow contracts, 24-point interactive Q&A |
| P6 | `final-methods-quasi-e2e-testing-prompt.md` | 2026-03-10 | Dynamic model/loss/metric discovery, conditional DAG schema, combinatorial reduction |
| P7 | `prompt-574-synthetic-data-drift-detection.md` | 2026-03-12 | Synthetic data, drift detection, data quality pipelines, agentic science workflows |
| P8 | `profiler-benchmarking-user-prompt.md` | 2026-03-13 | PyTorch profiling for GPU/CPU, MLflow logging, default ON, 17 reference links |
| P9 | `s3-mounting-testing-user-prompt.md` | 2026-03-13 | Pulumi-based S3 provisioning, multi-cloud abstraction, access tests |
| P10 | `repo-to-manuscript-prompt.md` | 2026-03-15 | Manuscript scaffold, KG bridge, OpenSpec alignment, intent expression principle |
| P11 | `drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md` §0 | 2026-03-16 | Drift detection e2e: Grafana+Evidently+VesselNN+synthetic generators+deploy monitoring, Level 4 gate |

## Cross-Reference: Prompt → Plan → Implementation

| Prompt | Plans Generated | Key Implementation |
|--------|----------------|-------------------|
| P1 | `modernize-minivess-mlops-plan.md` | Full v2 rewrite, ModelAdapter ABC, Prefect flows |
| P2 | `dynunet-ablation-plan.md`, experiment configs | `dynunet_loss_variation_v2` experiment, 4 losses x 3 folds |
| P3 | `repo-to-manuscript.md` | PRD system, 52 decision nodes, knowledge graph |
| P4 | `advanced-segmentation-double-check.md` | SAM3 adapter, VesselFM adapter, Mamba adapter |
| P5 | `e2e-testing-phase-*.xml` (3 phases) | `PipelineTriggerChain`, 73 validation checks |
| P6 | `final-methods-quasi-e2e-testing-plan.xml` | `capability_discovery.py`, `quasi_e2e_runner.py` |
| P7 | `synthetic-data-drift-detection-plan.xml` | Evidently, whylogs integration (partial, superseded by P11) |
| P8 | `profiler-benchmarking-plan.xml` | Profiler module (partial) |
| P9 | `s3-mounting-testing-plan.xml` | Pulumi DVC bucket (archived with UpCloud) |
| P10 | `knowledge-management-upgrade.md` | KG navigator, domains, manuscript scaffold |
| P11 | `drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md` | Full Level 4 drift monitoring: 6th+7th Prefect flows, 4 synthetic generators, Evidently Docker service, Alertmanager, Grafana timeline dashboard, whylogs continuous profiling, VesselNN DVC batching, multi-family champion evaluation |

## Intent Themes (Synthesized from P1-P10)

1. **Production MLOps for biomedical imaging** — not a toy repo, Nature Protocols-grade
2. **MONAI ecosystem extension** — adapt 3rd-party models, never fork MONAI
3. **Agentic development as paper novelty** — the process IS the contribution
4. **Zero manual work** — everything automatic, one-command reproducibility
5. **Heterogeneous lab support** — any cloud, any GPU, config-only changes
6. **Scientific rigor** — compound losses, topology metrics, proper CV, UQ
7. **Spec-driven development** — PRD as evidence base, decisions as YAML nodes

## Files with Contextual User Guidance (Not Full Verbatim Prompts)

These files contain embedded user direction without a full verbatim prompt section:

| File | Topic |
|------|-------|
| `compound-loss-double-check.md` | Compound loss decisions |
| `sam3-installation-issues-and-synthesis.md` | SAM3 troubleshooting path |
| `sam3-nan-loss-fix.md` | val_loss=NaN root cause |
| `runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md` | Cloud provider selection |
| `multi-metric-downstream-double-check.md` | Metric validation decisions |
| `cloud-architecture-decisions-2026-03-14.md` | Cloud choices (RunPod, GCP) |
| `mlflow-deployment-storage-analysis.md` | MLflow deployment options |
| `cover-letter-to-sci-llm-writer-for-knowledge-graph.md` | KG handoff to manuscript |

## How to Use This Document

- **Before implementing**: check which user prompt(s) relate to your task
- **When plans conflict**: the verbatim prompt is the source of truth (CLAUDE.md Rule #11)
- **For the manuscript**: these prompts document the agentic development methodology
- **For new sessions**: cross-reference this with `knowledge-graph/navigator.yaml` for domain routing

## Statistics

| Metric | Count |
|--------|-------|
| Explicit verbatim prompts | 11 |
| Files with contextual guidance | 8+ |
| Total planning docs (.md + .xml) | 207 |
| Plans with XML execution format | 64 |
| Date range | 2026-02-23 to 2026-03-16 |
