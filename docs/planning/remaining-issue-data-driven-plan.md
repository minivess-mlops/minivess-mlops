# Remaining Issues: Data-Driven Execution Plan

**Created**: 2026-03-17
**Open issues at time of analysis**: 29
**Methodology**: Codebase audit of partial implementations + semantic clustering + publication-gate sequencing

---

## Executive Summary

Of 29 open issues, **3 can be closed immediately** (already implemented or behavioral),
**12 should be deferred post-publication** (label: `post-publication`), and **14 form
the pre-publication critical path** organized into 5 semantic PR clusters.

| Category | Count | Action |
|----------|-------|--------|
| Close immediately | 3 | Already done or behavioral rule |
| Pre-publication PRs | 14 | 5 PR clusters, ordered by dependency |
| Post-publication defer | 12 | Label + close with explanation |

**Total estimated PRs**: 5 pre-publication + 0-3 post-publication batches

---

## Section 1: Close Immediately (3 issues)

These are already implemented or not actionable as code:

| # | Title | Reason to close |
|---|-------|-----------------|
| **#690** | P2: Makefile targets for RunPod dev | **Fully implemented.** All 8 targets exist in Makefile (lines 116-144): `dev-gpu`, `dev-gpu-heavy`, `dev-gpu-stop`, `dev-gpu-ssh`, `dev-gpu-sync`, `dev-gpu-upload-data`, `dev-gpu-cleanup`, `dev-gpu-status`. |
| **#716** | P1: RunPod → local mlruns sync | **Implemented as `make dev-gpu-sync`** (Makefile line 132). Uses `sky rsync down` from `/opt/vol/mlruns/`. Also `src/minivess/compute/mlruns_sync.py` exists. |
| **#750** | P0: Claude asks humans cloud state | **Behavioral rule, not code.** Already enforced via `.claude/rules/no-state-questions.md` + metalearning doc. Cannot be "fixed" in code — it's an agent instruction. Close as "enforced via rules". |

---

## Section 2: Post-Publication Deferrals (12 issues → label `post-publication`)

These are valuable but not required for the Nature Protocols submission gate.
Apply label `post-publication`, close with explanation, and revisit after R1 submission.

### Cluster A: Multi-Cloud Expansion (3 issues)
Not needed when we have RunPod + GCP working.

| # | Title | Why defer |
|---|-------|-----------|
| **#709** | P2: AWS + Azure recipe YAMLs | No implementation exists. Academic labs with AWS/Azure credits can wait until post-publication community adoption. CLAUDE.md says exactly 2 providers. |
| **#627** | P2: RunPod Pulumi provider | RunPod works fine via SkyPilot CLI. Pulumi IaC for RunPod is a nice-to-have for persistent infra management, not a submission gate. |
| **#682** | P2: dstack as SkyPilot alternative | Pure research/exploration. SkyPilot works. Evaluating alternatives is post-publication scope. |

### Cluster B: Production Hardening (3 issues)
Production-grade deployment beyond what's needed for paper results.

| # | Title | Why defer |
|---|-------|-----------|
| **#615** | P2: MLflow nginx + TLS | Password auth works for current scale. TLS + reverse proxy is production hardening for multi-user deployments. |
| **#612** | P2: DagsHub MLflow migration | Optional migration to managed MLflow. Not needed — file-based + self-hosted MLflow is sufficient for paper. |
| **#611** | P2: HPO completion barrier | SEQUENTIAL HPO strategy works. PARALLEL requires PostgreSQL worker pool — complex orchestration not needed for paper results (we run 1 HPO trial at a time). |

### Cluster C: Research & Benchmarking (2 issues)
Research tasks with no code deliverable needed for submission.

| # | Title | Why defer |
|---|-------|-----------|
| **#652** | P2: CodSpeed CI alternatives | Benchmark tooling research. No CI runs anyway (GitHub Actions disabled). Post-publication when CI is re-enabled. |
| **#679** | P1: Platform Engineering plan | Planning/design task for internal researcher platform. The platform IS this repo — the plan is the paper itself. Post-publication community outreach. |

### Cluster D: Nice-to-Have Enhancements (4 issues)

| # | Title | Why defer |
|---|-------|-----------|
| **#653** | P2: Grafana temporal perf dashboard | Regression detection dashboard. We have drift monitoring dashboard + model performance dashboard. Temporal regression is post-publication polish. |
| **#752** | P2: RunPod Mode B (file_mounts) | Mode A (Network Volume) works. Mode B (ephemeral data) is an optimization for labs without persistent volumes. Defer until community requests. |
| **#713** | P3: VesselFM leakage warning severity | 5-line config change. Not blocking anything. |
| **#712** | P3: SAM3 encoder dtype configurable | BF16 enforcement works. Making it configurable is a post-publication flexibility enhancement. |

---

## Section 3: Pre-Publication Critical Path (14 issues → 5 PRs)

Ordered by dependency chain. Each PR is a semantically coherent unit.

### PR-1: FinOps & Infrastructure Timing (consolidate 5 issues)
**Issues**: #683, #747, #717, #735, #751
**Estimated tests**: 15-20
**Dependency**: None (standalone)
**Why first**: Cost visibility is needed before running expensive GPU experiments (#734).

These 5 issues share one theme: **understanding and optimizing infrastructure cost**.
Significant code already exists (`infrastructure_timing.py`, `compute_cost_analysis()`).

| # | Title | What's needed |
|---|-------|---------------|
| **#683** | Infrastructure timing + cost pipeline | Add Grafana dashboard panels for cost_* metrics from `infrastructure_timing.py`. Wire MLflow metrics → Prometheus → Grafana. |
| **#747** | FinOps optimization (GCP L4 + RunPod) | Multi-run cost aggregation: query DuckDB analytics for cost trends across experiments. Add cost_per_epoch, cost_per_model_family summary tables. |
| **#717** | Dynamic GPU cost estimator | Extend `estimate_cost_from_first_epoch()` to produce total-job estimates logged to MLflow. Already partially implemented. |
| **#735** | MLflow + GCS stale run cleanup | Add `scripts/mlflow_cleanup.py` utility: archive runs older than N days, compute storage cost of artifacts, suggest deletions. |
| **#751** | GCP Docker pull optimization | Multi-stage Docker build with layer caching. Move pip install to earlier layer. GAR pull time from 15 min → <5 min target. |

**PR structure**:
```
src/minivess/observability/infrastructure_timing.py  — extend
src/minivess/observability/finops.py                 — NEW: cost aggregation
deployment/grafana/dashboards/infrastructure-cost.json — NEW dashboard
scripts/mlflow_cleanup.py                            — NEW utility
deployment/docker/Dockerfile.gpu                     — optimize layers
tests/v2/unit/test_finops.py                         — NEW
tests/v2/unit/test_mlflow_cleanup.py                 — NEW
```

---

### PR-2: Data Quality Pipeline (1 large issue)
**Issues**: #777
**Estimated tests**: 12-15
**Dependency**: None (standalone, but benefits from PR-1 cost visibility)
**Why second**: Data quality gates must be in place before collecting paper results.

Significant scaffolding already exists (`ge_runner.py`, `profiling.py`, `schemas.py`,
`deepchecks_vision.py`). The gap is **wiring these into the Prefect pipeline**.

| Component | Status | What's needed |
|-----------|--------|---------------|
| DeepChecks Vision | Stub exists | Wire 3D→2D adapter to `build_data_integrity_suite()`, add to data_flow |
| Great Expectations | Runner exists | Create batch validation suite for MiniVess + VesselNN, enforce in data_flow |
| whylogs profiling | **Done** (T-B2) | Already integrated via `WhylogsVolumeProfiler` |
| Pandera schemas | Schemas defined | Enforce in `extract_batch_features()` and `discover_nifti_pairs()` |

**PR structure**:
```
src/minivess/validation/deepchecks_vision.py         — wire to data_flow
src/minivess/validation/ge_runner.py                 — add MiniVess suite
src/minivess/orchestration/flows/data_flow.py        — add quality gates
configs/validation/                                  — GE expectation YAMLs
tests/v2/unit/test_data_quality_pipeline.py          — NEW
```

---

### PR-3: Knowledge Graph & Tooling Refresh (3 issues)
**Issues**: #736, #693, #691
**Estimated tests**: 5-8 (mostly structural/doc changes)
**Dependency**: None (standalone)
**Why third**: KG completeness is needed for manuscript coherence.

| # | Title | What's needed |
|---|-------|---------------|
| **#736** | KG scope blindness | Add planned-models section to `domains/models.yaml`. Map each model to paper section. Add VesselFM, MambaVesselNet++ decision nodes with R3 paper references. |
| **#693** | CLAUDE.md + Skills 2.0 | Modularize CLAUDE.md (already partly done via domain CLAUDE.md files). Update skill manifests to 2.0 format with `triggers`, `examples`, `evals` sections. |
| **#691** | Issue Creator Skill | Skill already exists at `.claude/skills/issue-creator/`. Needs eval harness + integration test to close. |

**PR structure**:
```
knowledge-graph/domains/models.yaml                  — add planned models
knowledge-graph/domains/manuscript.yaml              — fill stubs
.claude/skills/*/SKILL.md                            — upgrade to 2.0
CLAUDE.md                                            — trim redundancy
tests/v2/unit/test_kg_completeness.py                — NEW
```

---

### PR-4: GPU Experiment Runs (1 critical issue)
**Issues**: #734
**Estimated tests**: 0 (execution, not code)
**Dependency**: PR-1 (cost visibility), PR-2 (data quality gates)
**Why fourth**: This IS the paper results. Everything before is infrastructure.

| Model | GPU | Estimated cost | Status |
|-------|-----|----------------|--------|
| SAM3 Vanilla | RTX 4090 | ~$0.06 | Adapter ready, smoke-tested |
| SAM3 Hybrid | RTX 4090 | ~$0.09 | Adapter ready, finite val_loss confirmed |
| VesselFM | RTX 4090 | ~$0.12 | Adapter ready |
| MambaVesselNet++ | RTX 4090 | ~$0.10 | Adapter ready (T00-T08 complete) |

**Not a code PR** — this is `make smoke-test-all` + production training runs.
The issue tracks execution and result collection, not implementation.

**Execution plan**:
```
1. make smoke-test-preflight       # Validate prerequisites
2. make smoke-test-all             # All 4 models × 2 epochs (~$0.40)
3. make verify-smoke-test          # Check MLflow artifacts
4. Full training: 50 epochs × 3 folds × 4 models → RunPod batch job
```

---

### PR-5: Remaining Bug Fixes & DevEx (4 issues)
**Issues**: #778, #755, #754, #753
**Estimated tests**: 8-10
**Dependency**: PR-4 (bugs surface during GPU runs)
**Why last**: These are bugs and polish discovered during/after GPU runs.

| # | Title | What's needed |
|---|-------|---------------|
| **#778** | P2: RunPod DVC cache-skip test | Add "already cached" text to `dev_runpod.yaml` setup block. 1-line fix. |
| **#755** | Bug: MLflow Cloud Run multipart 500 | Investigate GCP Cloud Run artifact size limits. May need chunked upload or GCS artifact backend. |
| **#754** | Bug: RunPod provisioning stuck | Investigate SkyPilot volume mount timeout. Add health check to `dev_runpod.yaml` setup. |
| **#753** | feat: one-command dataset workflow | Wire `DatasetDownloader` (T-A1) + DVC versioning + RunPod upload into single `make dataset-setup DATASET=minivess` target. |

**PR structure**:
```
deployment/skypilot/dev_runpod.yaml                  — fix #778, #754
src/minivess/data/dataset_downloader.py              — extend for #753
Makefile                                             — add dataset-setup target
tests/v2/unit/test_runpod_dev_preflight.py           — fix #778
```

---

## Section 4: Execution Timeline

```
Week 1: PR-1 (FinOps + Timing)           — 5 issues → 1 PR
         Close 3 implemented issues
         Label + close 12 post-publication issues

Week 2: PR-2 (Data Quality Pipeline)     — 1 issue → 1 PR
         PR-3 (KG + Tooling)             — 3 issues → 1 PR (parallel)

Week 3: PR-4 (GPU Experiment Runs)       — 1 issue → execution
         PR-5 (Bug fixes)                — 4 issues → 1 PR (as bugs surface)

Post-submission:
         Revisit 12 post-publication issues
         Community-driven prioritization
```

---

## Section 5: Label Taxonomy

| Label | Color | Meaning |
|-------|-------|---------|
| `post-publication` | `#C5DEF5` (light blue) | Valuable work deferred until after R1 submission |
| `pre-publication` | `#0E8A16` (green) | Required for Nature Protocols submission gate |
| `already-implemented` | `#BFDADC` (light teal) | Code exists, issue can be closed |

---

## Appendix: Issue-to-PR Mapping

| Issue | PR | Cluster |
|-------|----|---------|
| #683 | PR-1 | FinOps |
| #747 | PR-1 | FinOps |
| #717 | PR-1 | FinOps |
| #735 | PR-1 | FinOps |
| #751 | PR-1 | FinOps |
| #777 | PR-2 | Data Quality |
| #736 | PR-3 | KG + Tooling |
| #693 | PR-3 | KG + Tooling |
| #691 | PR-3 | KG + Tooling |
| #734 | PR-4 | GPU Runs |
| #778 | PR-5 | Bug Fixes |
| #755 | PR-5 | Bug Fixes |
| #754 | PR-5 | Bug Fixes |
| #753 | PR-5 | Bug Fixes |
| #690 | — | Close (done) |
| #716 | — | Close (done) |
| #750 | — | Close (behavioral) |
| #709 | — | post-publication |
| #627 | — | post-publication |
| #682 | — | post-publication |
| #615 | — | post-publication |
| #612 | — | post-publication |
| #611 | — | post-publication |
| #652 | — | post-publication |
| #679 | — | post-publication |
| #653 | — | post-publication |
| #752 | — | post-publication |
| #713 | — | post-publication |
| #712 | — | post-publication |
