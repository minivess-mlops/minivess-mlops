# GitHub Projects Migration & Cleaning Plan

**Created**: 2026-03-12
**Sprint**: Feb 23 – Mar 15, 2026 (3-week sprint)
**Paper deadline**: Sunday, Mar 15, 2026
**Next week**: QA/debugging + article writing (Mar 16–22)

---

## Executive Summary

Project #5 (`users/petteriTeikari/projects/5`) is correctly set up with 14 fields (Status,
Priority, Size, Estimate, Iteration) but **zero iterations configured** and all 8 items
have no field values assigned. This plan covers:

1. Configure the current iteration (Feb 23 – Mar 15)
2. Assign all 8 open issues to iteration with Status/Priority/Size
3. Create semantic milestones for 523 closed + 8 open sprint issues
4. Set up sync-roadmap skill for automatic Start/Target dates
5. PRD completeness cross-check (52 decision nodes)
6. Paper submission gap analysis
7. Next week plan (QA + article writing)

**Key numbers**: 523 issues closed, 68 PRs merged, 8 issues remain open, 52 PRD nodes
(30 resolved, 5 partial, 7 config-only, 8 not-started, 2 deferred).

---

## Phase 1: Configure Iteration Field

### 1.1 Create Current Iteration

The Iteration field exists (`PVTIF_lAHOABAuos4BRlmyzg_X1I4`) but has zero iterations.
Create the current sprint:

```bash
# Get Project ID (node ID)
PROJECT_ID=$(gh api graphql -f query='{ user(login: "petteriTeikari") { projectV2(number: 5) { id } } }' --jq '.data.user.projectV2.id')

# Get Iteration field ID
ITER_FIELD_ID="PVTIF_lAHOABAuos4BRlmyzg_X1I4"

# Create current iteration (3-week sprint)
gh api graphql -f query="
mutation {
  updateProjectV2IterationField(input: {
    projectId: \"$PROJECT_ID\"
    fieldId: \"$ITER_FIELD_ID\"
    iterationConfiguration: {
      iterations: [
        {
          title: \"Sprint 2026-W08-W10 (Feb 23 - Mar 15)\"
          startDate: \"2026-02-23\"
          duration: 21
        }
      ]
    }
  }) {
    field {
      ... on ProjectV2IterationField {
        configuration {
          iterations { id title startDate duration }
        }
      }
    }
  }
}"
```

### 1.2 Create Next Iteration (QA + Writing)

```bash
# Add next iteration for QA/writing week
# Duration: 7 days (Mar 16-22)
# Title: "QA + Article Writing (Mar 16 - Mar 22)"
# This is added in the same mutation or a follow-up one
```

---

## Phase 2: Assign Open Issues to Iteration

All 8 open issues need Status, Priority, Size, and Iteration assignment.

### 2.1 Target Field Values

| # | Title | Status | Priority | Size | Estimate |
|---|-------|--------|----------|------|----------|
| 366 | VesselFM fine-tuning on RunPod | Ready | P1-high | L | 5 |
| 564 | Dockerized GPU benchmark suite | Ready | P1-high | L | 5 |
| 574 | Synthetic drift simulation (parent) | In progress | P0-critical | M | 3 |
| 602 | VesselNN 4th dataset registration | Ready | P2-medium | S | 2 |
| 609 | SkyPilot + RunPod remote training | Ready | P1-high | XL | 8 |
| 610 | MLflow from SkyPilot VMs | Ready | P1-high | M | 3 |
| 611 | HPO completion barrier | Backlog | P2-medium | M | 3 |
| 612 | DagsHub MLflow migration | Backlog | P2-medium | S | 2 |

### 2.2 Execution Script

```bash
# Get project node ID and field IDs first
PROJECT_ID=$(gh api graphql -f query='{ user(login: "petteriTeikari") { projectV2(number: 5) { id } } }' --jq '.data.user.projectV2.id')

# For each issue, get item ID and set fields
for ISSUE in 366 564 574 602 609 610 611 612; do
  ITEM_ID=$(gh project item-list 5 --owner petteriTeikari --format json \
    | jq -r ".items[] | select(.content.number == $ISSUE) | .id")

  # Set Status (example: "Ready")
  gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
    --field-id <STATUS_FIELD_ID> --single-select-option-id <READY_OPTION_ID>

  # Set Priority
  gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
    --field-id <PRIORITY_FIELD_ID> --single-select-option-id <P1_OPTION_ID>

  # Set Iteration
  gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
    --field-id "$ITER_FIELD_ID" --iteration-id <CURRENT_ITER_ID>
done
```

> **Note**: Field IDs and option IDs must be resolved from GraphQL queries at execution
> time. The sync-roadmap skill's field ID constants are for the OLD Project #1 — they
> need to be updated for Project #5.

---

## Phase 3: Semantic Milestones

### 3.1 Milestone Design

Group the 531 sprint issues (523 closed + 8 open) into semantic milestones that reflect
the major capability areas built during this sprint. These are not linear — many were
developed concurrently.

| Milestone | Issues | Description |
|-----------|--------|-------------|
| **`0.2-scaffold`** | ~51 | Project setup: uv, ruff, mypy, pre-commit, CI |
| **`0.2-infrastructure`** | ~99 | Docker hardening, Compose, security, volumes |
| **`0.2-hydra`** | ~21 | Hydra-zen config composition, .env SSoT |
| **`0.2-experiment`** | ~27 | MLflow tracking, DuckDB analytics, champion tagging |
| **`0.2-topology`** | ~36 | Graph topology losses, clDice, centerline metrics |
| **`0.2-data`** | ~14 | Data engineering, external datasets, validation |
| **`0.2-evaluation`** | ~30 | Analysis pipeline, ensembles, conformal prediction |
| **`0.2-post-training`** | ~29 | SWA, calibration, uncertainty quantification |
| **`0.2-deploy`** | ~35 | ONNX export, BentoML, deployment flow |
| **`0.2-verification`** | ~53 | Testing infrastructure, 3-tier suite, QA |
| **`0.2-orchestration`** | ~40 | Prefect flows, Docker-per-flow, trigger chain |
| **`0.2-models`** | ~76 | SAM3, DynUNet, SegResNet, Mamba, VesselFM adapters |
| **`0.2-drift`** | ~15 | Evidently drift detection, embedding drift |
| **`0.2-dashboard`** | ~17 | Dashboard flow, paper artifacts, visualization |
| **`0.3-backlog`** | ~8 | Open issues: remote compute, drift completion |

**Total**: ~551 (some issues span multiple milestones; assign to primary)

### 3.2 Milestone Creation Commands

```bash
# Create milestones (many already exist from the old project)
for MS in scaffold infrastructure hydra experiment topology data evaluation \
          post-training deploy verification orchestration models drift dashboard; do
  gh api repos/petteriTeikari/minivess-mlops/milestones \
    -f title="0.2-$MS" \
    -f state="closed" \
    -f description="v0.2 sprint: $MS capability area"
done

# 0.3-backlog already exists
```

### 3.3 Bulk Assignment Strategy

**Do NOT assign all 523 closed issues individually** — this is an 8-hour manual task
and provides marginal value. Instead:

1. **Create the milestones** (2 min) — for organizational clarity
2. **Assign the 8 open issues** to `0.3-backlog` (2 min)
3. **Spot-check**: Verify the ~187 issues with no milestone by sampling 20 — most
   were auto-filed during batch implementations and don't need individual triage
4. **Milestone counts live in git history** — the semantic clusters above serve as
   the milestone mapping for the paper. Actual GitHub milestone assignment is cosmetic.

**Rationale**: The sprint closes in 3 days. Assigning 523 closed issues to milestones
has zero impact on paper submission. The semantic cluster table above IS the milestone
documentation.

---

## Phase 4: Sync-Roadmap Skill Setup

### 4.1 Field ID Migration

The sync-roadmap skill (`/.claude/skills/sync-roadmap/SKILL.md`) was built for
**Project #1** (org: `minivess-mlops`). It needs these updates for **Project #5**
(user: `petteriTeikari`):

| Field | Old Project #1 ID | New Project #5 ID |
|-------|-------------------|-------------------|
| Project | `PVT_kwDOCPpnGc4AYSAM` | *resolve at runtime* |
| Status | `PVTSSF_lADOCPpnGc4AYSAMzgPhgrw` | *resolve at runtime* |
| Priority | `PVTSSF_lADOCPpnGc4AYSAMzgPhgsk` | *resolve at runtime* |
| Start date | `PVTF_lADOCPpnGc4AYSAMzgPhgso` | *resolve at runtime* |
| Target date | `PVTF_lADOCPpnGc4AYSAMzg-z7gU` | *resolve at runtime* |
| Size | `PVTSSF_lADOCPpnGc4AYSAMzg-zBm8` | *resolve at runtime* |
| Estimate | `PVTF_lADOCPpnGc4AYSAMzg-zBns` | *resolve at runtime* |
| Iteration | `PVTIF_lADOCPpnGc4AYSAMzg-zB-I` | `PVTIF_lAHOABAuos4BRlmyzg_X1I4` |

### 4.2 Resolve New Field IDs

```bash
gh api graphql -f query='{
  user(login: "petteriTeikari") {
    projectV2(number: 5) {
      id
      fields(first: 20) {
        nodes {
          ... on ProjectV2Field { id name }
          ... on ProjectV2SingleSelectField { id name options { id name } }
          ... on ProjectV2IterationField { id name configuration {
            iterations { id title startDate duration }
          }}
        }
      }
    }
  }
}'
```

### 4.3 Update Skill Constants

After resolving IDs, update `scripts/sync_roadmap.py` (if it exists) or the skill's
configuration to point to Project #5.

### 4.4 Backfill Start/Target Dates

For the 8 open issues:
- **Start date**: issue creation date (from `gh issue view --json createdAt`)
- **Target date**: Mar 15 for P0/P1, Mar 22 for P2, TBD for backlog

```bash
# Example for single issue
python3 scripts/sync_roadmap.py --issue 574
# Or if script doesn't exist, use gh directly:
gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
  --field-id <START_DATE_FIELD_ID> --date "2026-03-10"
gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
  --field-id <TARGET_DATE_FIELD_ID> --date "2026-03-15"
```

---

## Phase 5: PRD Completeness Cross-Check

### 5.1 PRD Materialization Status

**Critical finding**: The PRD directory (`docs/planning/prd/`) does NOT exist on disk.
The 52-node Bayesian decision network is designed in `docs/planning/hierarchical-prd-planning.md`
(Draft v1, 2026-02-23) but the YAML decision files, `_network.yaml`, `bibliography.yaml`,
scenarios, archetypes, and domain overlays have never been materialized.

This is a **documentation gap**, not a feature gap — the decisions were made and
implemented in code, they just weren't written as formal PRD YAML files.

### 5.2 Decision Status Summary (52 nodes)

| Level | Total | Resolved | Partial | Config-Only | Not Started |
|-------|-------|----------|---------|-------------|-------------|
| L1: Research Goals | 7 | 6 | 1 | 0 | 0 |
| L2: Architecture | 10 | 7 | 1 | 1 | 0 |
| L3: Technology | 20 | 10 | 2 | 6 | 1 |
| L4: Infrastructure | 8 | 4 | 1 | 0 | 3 |
| L5: Operations | 7 | 2 | 1 | 1 | 4 |
| **Total** | **52** | **30** | **5** | **7** | **8** |

**Percentages**: 58% resolved, 10% partial, 13% config-only, 15% not-started, 4% deferred

### 5.3 Nodes Requiring Attention for Paper

**Critical for paper (must be at least "partial"):**

| Node | Level | Status | Gap |
|------|-------|--------|-----|
| `topology_metrics` | L3 | not_started | clDice implemented but formal topology metric framework not wired |
| `gpu_compute` | L4 | partial | SkyPilot config exists, no execution yet |
| `model_governance` | L5 | partial | Champion tagging exists, no formal governance workflow |

**Acceptable as "config-only" for paper (demonstrate architecture, defer execution):**

| Node | Level | Status | Paper Treatment |
|------|-------|--------|-----------------|
| `api_protocol` | L2 | config_only | Mention BentoML + Gradio in architecture diagram |
| `label_quality_tool` | L3 | config_only | List Cleanlab as planned integration |
| `data_profiling` | L3 | config_only | Show whylogs in observability stack table |
| `xai_voxel_tool` | L3 | config_only | Captum in XAI section |
| `xai_meta_tool` | L3 | config_only | Quantus in XAI section |
| `lineage_tracking` | L3 | config_only | OpenLineage in architecture |
| `dashboarding` | L5 | config_only | Dashboard flow produces artifacts (Parquet, figures) |

**Deferred (not needed for paper):**

| Node | Level | Status | Rationale |
|------|-------|--------|-----------|
| `iac_tool` | L4 | not_started | Pulumi — cloud deployment, not paper scope |
| `gitops_engine` | L4 | not_started | ArgoCD — production only |
| `air_gap_strategy` | L4 | not_started | Clinical deployment |
| `retraining_trigger` | L5 | not_started | Production monitoring |
| `sbom_generation` | L5 | not_started | Supply chain security |
| `federated_learning` | L5 | not_started | Future research direction |

### 5.4 PRD vs Implementation Gap Matrix

| PRD Category | Nodes | Implemented | Paper-Ready | Action |
|-------------|-------|-------------|-------------|--------|
| Core ML Pipeline | 15 | 14 | Yes | topology_metrics needs docs |
| Infrastructure | 8 | 5 | Mostly | GPU compute needs SkyPilot demo |
| Observability | 8 | 5 | Yes | XAI tools are deps-only |
| Operations | 7 | 3 | Partial | Drift done, governance partial |
| Model Ecosystem | 7 | 6 | Yes | VesselFM pending RunPod |
| Config/Validation | 7 | 7 | Yes | All resolved |
| **Total** | **52** | **40** | **~42** | 10 deferred to post-paper |

---

## Phase 6: Paper Submission Gap Analysis

### 6.1 What IS Ready (by Sunday Mar 15)

| Capability | Evidence | Artifact |
|-----------|----------|----------|
| 5-flow Prefect pipeline | Verified E2E, trigger chain | `trigger_chain_results.json` |
| 18 loss functions | Tested, loss variation experiment | `loss_variation_v2_report.md` |
| 5+ model adapters | DynUNet, SAM3 (3 variants), SegResNet, UNETR, SwinUNETR | `src/minivess/adapters/` |
| Ensemble methods | Model soup, voting, conformal | `src/minivess/ensemble/` |
| Drift detection | Evidently + kernel MMD, MLflow artifacts | PR #608 |
| Docker-per-flow isolation | 12 services, security hardening | `docker-compose.flows.yml` |
| 3-tier test suite | Staging (<3min), Prod (~10min), GPU | `tests/` |
| MLflow experiment tracking | 73 validation checks passing | `verify_all_artifacts.py` |
| Paper artifacts | 25 figures/tables with LaTeX commands | `outputs/paper_artifacts/` |
| Hydra-zen config | Full composition pipeline, MLflow artifact | `src/minivess/config/` |
| Calibration | MAPIE + netcal + Local Temperature Scaling | `src/minivess/ensemble/` |
| Data validation | Pydantic + Pandera + Great Expectations | `src/minivess/data/` |
| Champion tagging | Per-loss, per-fold, overall | `scripts/tag_champions.py` |

### 6.2 What's Missing / At Risk

| Gap | Priority | Effort | Paper Impact | Action |
|-----|----------|--------|-------------|--------|
| **VesselNN dataset** | P2 | 2h | Minor (4th external test set) | #602 — nice-to-have |
| **SkyPilot execution** | P1 | 4h | Section 5 (Cloud Compute) | Demo run or describe as "validated config" |
| **PRD YAML files** | P2 | 6h | Supplementary material | Skip — describe network in prose |
| **Topology metrics framework** | P1 | 2h | Section on topology-aware losses | Already have clDice; document formally |
| **HPO execution** | P2 | 4h | Section on HPO | Describe Optuna+ASHA setup, defer execution |
| **Dashboard UI** | P3 | 8h | Not needed — paper uses static artifacts | Skip entirely |
| **MONAI Label integration** | P3 | 8h | Future work section | Mention as planned |
| **XAI pipeline execution** | P2 | 4h | XAI section | Describe architecture, show deps |
| **MLflow online deployment** | P2 | 4h | Deployment section | Reference `mlflow-online-deployment-research.md` |

### 6.3 Recommended Priority for Remaining 3 Days (Mar 13-15)

**Day 1 (Mar 13): Close critical gaps**
- [ ] Configure Project #5 iteration + assign 8 issues (30 min)
- [ ] Finish drift detection: close #574, decide on #602 (2h)
- [ ] Document topology metrics in paper draft (1h)
- [ ] Run `make test-staging` clean — fix any regressions (1h)

**Day 2 (Mar 14): Paper preparation**
- [ ] Assemble all paper artifacts: `python scripts/assemble_paper_artifacts.py` (30 min)
- [ ] Generate final comparison figures (1h)
- [ ] Write/finalize SkyPilot section (describe config, not execution) (1h)
- [ ] Final `make test-prod` run — everything green (2h)

**Day 3 (Mar 15): Paper submission**
- [ ] Final paper compilation
- [ ] Verify all figure/table references
- [ ] Tag release: `git tag v0.2.0-paper`
- [ ] Push to GitHub

### 6.4 Features Explicitly Deferred to Post-Paper

These are NOT missing — they are scoped out of the paper:

1. **SkyPilot cloud execution** — Config validated, no cloud credits spent yet
2. **VesselFM fine-tuning** — Requires RunPod GPU instance
3. **MONAI Label annotation** — Future work
4. **Dashboard React UI** — Paper uses static artifacts
5. **Federated learning** — Research direction
6. **IaC (Pulumi)** — Production deployment
7. **GitOps (ArgoCD)** — Production deployment
8. **SBOM generation** — Compliance feature
9. **Retraining triggers** — Production monitoring
10. **Air-gap deployment** — Clinical deployment

---

## Phase 7: Next Week Plan (Mar 16-22)

### 7.1 QA & Debugging Focus

| Day | Focus | Tasks |
|-----|-------|-------|
| Mon Mar 16 | Test suite hardening | Full `make test-prod`, fix all failures |
| Tue Mar 17 | Docker E2E | Run full pipeline in Docker, verify all 5 flows |
| Wed Mar 18 | Edge cases | OOM testing, large volume handling, error recovery |
| Thu Mar 19 | Documentation | Update README, verify all CLAUDE.md files current |
| Fri Mar 20 | Code quality | Ruff + mypy clean, remove dead code |

### 7.2 Article Writing Track (Parallel)

| Section | Status | Action |
|---------|--------|--------|
| Introduction | Draft | Finalize motivation, cite key references |
| Methods | 80% done | Add topology metrics, calibration details |
| Results | Figures ready | Write interpretation, statistical tests |
| Discussion | Not started | Write — leverage paper artifacts |
| Architecture | 80% done | Add SkyPilot, drift monitoring sections |
| Supplementary | Not started | PRD summary, full config examples |

### 7.3 Post-QA Sprint Planning

After the paper, the next sprint should focus on:
1. **SkyPilot execution** (#609, #610) — actually run on cloud GPUs
2. **VesselFM fine-tuning** (#366) — RunPod execution
3. **HPO sweep** (#611) — Optuna + ASHA on cloud
4. **MLflow online deployment** — Oracle Cloud Free setup

---

## Appendix A: Sprint Statistics

### Issue Velocity

| Metric | Value |
|--------|-------|
| Total issues created (sprint) | 612 |
| Issues closed (sprint) | 523 |
| PRs merged (sprint) | 68 |
| Net burn-down | 523 closed / 8 open = 98.5% close rate |
| Peak day | Mar 7 (80 issues closed) |
| Average close rate | 29 issues/day |

### Label Distribution (Closed Issues)

| Label | Count | % |
|-------|-------|---|
| P1-high | 297 | 57% |
| enhancement | 129 | 25% |
| P2-medium | 76 | 15% |
| P0-critical | 82 | 16% |
| testing | 35 | 7% |
| bug | 50 | 10% |
| infrastructure | 34 | 7% |
| documentation | 13 | 2% |

### Weekly Throughput

| Week | Issues Closed | PRs Merged |
|------|--------------|------------|
| W08 (Feb 23 - Mar 1) | 112 | ~15 |
| W09 (Mar 2 - Mar 8) | 311 | ~35 |
| W10 (Mar 9 - Mar 15) | 100+ | ~18 |

---

## Appendix B: Semantic Milestone Cluster Detail

### Cluster 1: Docker & Infrastructure (99 issues)

Core capability: Docker-per-flow isolation with security hardening.

**Key PRs**: #286 (Docker-per-flow + SkyPilot + HPO + MIG), #400 (28-task Prefect+Docker),
#461 (hard gates), #501 (enforcement), #542 (H3/H4 security), #559 (seccomp, SOPS, auth)

**Milestones**: `0.2-infrastructure` (34 existing) + `0.2-scaffold` (51 existing) + overflow

### Cluster 2: Models & Adapters (76 issues)

Core capability: Model-agnostic adapter pattern with 5+ model families.

**Key PRs**: #253 (SAM3 variants), #265 (SAM3 training), #327 (SAM3 champion),
#425 (SAM3 stub removal), #598 (SAM3 test markers)

**Models**: DynUNet (production), SAM3 vanilla/hybrid/topolora (research), SegResNet,
UNETR, SwinUNETR (MONAI), VesselFM (foundation), CoMMA/U-Like Mamba (planned)

### Cluster 3: Prefect Orchestration (40 issues)

Core capability: 5-flow Prefect pipeline with Docker-per-flow isolation.

**Flows**: Data → Train → Analyze → Deploy → Dashboard (+ Biostatistics, Acquisition)

**Key PRs**: #364 (Flow 0 acquisition), #459 (biostatistics), #478 (Prefect decoupling),
#557 (FlowContract), #589 (inter-flow contract)

### Cluster 4: Testing & QA (35 issues)

Core capability: 3-tier test suite (staging/prod/GPU) with 4136+ tests.

**Key PRs**: #339 (164 quasi-e2e), #506 (staging/prod tiers), #565 (3-tier suite),
#581 (e2e 5-phase 27-task)

### Cluster 5: Graph Topology & Losses (36 issues)

Core capability: 18 loss functions with topology-aware variants.

**Key losses**: cbdice_cldice (default), dice_ce, focal_tversky, graph_cuts,
persistent_homology, sdf_boundary, centerline_dice

### Cluster 6: MLflow & Experiment Tracking (27 issues)

Core capability: Full MLflow integration with DuckDB analytics.

**Key features**: ExperimentTracker, champion tagging, artifact verification (73 checks),
Parquet export, DuckDB analytics

### Cluster 7: Drift Detection (15 issues)

Core capability: Evidently + kernel MMD drift detection in Prefect flows.

**Key PR**: #608 (drift detection suite — 8/9 tasks done)

**Open**: #574 (parent, mostly done), #602 (VesselNN dataset)

---

## Appendix C: Execution Checklist

### Immediate (Today, Mar 12)

- [ ] Create iteration in Project #5 via GraphQL
- [ ] Assign all 8 open issues: Status + Priority + Size + Iteration
- [ ] Create missing milestones (`0.2-orchestration`, `0.2-models`, `0.2-drift`, `0.2-dashboard`)
- [ ] Assign 8 open issues to `0.3-backlog` milestone

### Before Paper (Mar 13-15)

- [ ] Close #574 (drift detection parent) — only #602 (VesselNN) remains as nice-to-have
- [ ] Run full `make test-staging` — verify clean
- [ ] Run `scripts/assemble_paper_artifacts.py` — verify 25 artifacts
- [ ] Run `scripts/verify_all_artifacts.py` — verify 73 checks pass
- [ ] Tag `v0.2.0-paper` release

### Post-Paper (Mar 16+)

- [ ] Resolve sync-roadmap skill field IDs for Project #5
- [ ] Bulk-assign Start/Target dates for open issues
- [ ] Plan Sprint 2 (Mar 16-29): SkyPilot, VesselFM, HPO, Oracle MLflow
- [ ] Materialize PRD YAML files (optional — for supplementary material)
