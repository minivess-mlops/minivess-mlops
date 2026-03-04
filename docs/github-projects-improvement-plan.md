# GitHub Projects Improvement Plan

**Date:** 2026-03-04
**Branch:** `fix/remove-unneeded-models`
**Goal:** Transform the GitHub Project from a 15% skeleton into a comprehensive, LLM-discoverable,
timeline-accurate project management view optimized for both human developers and AI agents.

---

## Failure Analysis: Why the Initial Upgrade Was Incomplete

### What was requested
Upgrade the org project (minivess-mlops/projects/1) to match the user project
(petteriTeikari/projects/4), which has 6 custom views with grouping, filtering,
and priority-based card layouts.

### What was actually delivered
- Fields were correctly upgraded (Status, Priority, Size, Estimate, Iteration) -- DONE
- Repo was linked to project (was missing) -- DONE
- Draft notes cleaned up -- DONE
- **But only 48 of 317 issues were in the project** -- CRITICAL MISS
- **No custom views were created** -- CRITICAL MISS
- Work was declared "complete" prematurely

### Root Causes

1. **Scope blindness** -- Focused narrowly on field schema (the data model) and ignored
   the two things that actually matter: (a) populating 100% of issues and (b) matching
   the 6-view layout. This is the equivalent of creating database columns but never
   inserting the rows or building the UI.

2. **False completion signal** -- After updating fields, statuses, and priorities for the
   48 items already in the project, the summary said "48 total items" without flagging
   that the repo has 317 issues. The 269 missing issues were never investigated.

3. **View API limitation not discovered upfront** -- The GitHub GraphQL API does NOT
   support creating or updating project views. This should have been discovered during
   planning, not after claiming the work was done. The 6 target views (Current iteration,
   Next iteration, Prioritized backlog, Roadmap, In review, My items) **must be created
   manually in the GitHub UI**.

4. **No comparison validation** -- Never compared the target project's structure
   (6 views, 241 items, groupBy/sortBy/filter config) against what was delivered.

### API Limitation: Views Are READ-ONLY

**CONFIRMED:** The GitHub Projects v2 GraphQL API has NO mutations for views:
- No `createProjectV2View` -- views cannot be created via API
- No `updateProjectV2View` -- view layout/filter/groupBy cannot be set via API
- No `deleteProjectV2View` -- views cannot be removed via API

Views can only be read via `ProjectV2.views` connection. The only operations
available via API are: field CRUD, item CRUD, item field value updates.

**What this means for Phase 8:** Views MUST be configured manually in the GitHub
web UI. The plan below provides exact specifications for each view so the user
(or an automated browser tool) can set them up quickly.

### Target Views (from petteriTeikari/projects/4)

| # | Name | Layout | Filter | GroupBy | VerticalGroupBy | SortBy |
|---|------|--------|--------|---------|-----------------|--------|
| 1 | Current iteration | Board | `iteration:@current` | -- | Status | -- |
| 2 | Next iteration | Board | `iteration:@next` | Priority | Status | -- |
| 3 | Prioritized backlog | Board | -- | Priority | Status | Priority ASC |
| 4 | Roadmap | Roadmap | -- | -- | -- | -- |
| 5 | In review | Table | `status:"In review"` | -- | -- | -- |
| 6 | My items | Table | `assignee:@me` | -- | -- | -- |

**Current views on our project (need manual upgrade):**

| # | Name | Layout | GroupBy | Notes |
|---|------|--------|---------|-------|
| 1 | Board | Board | -- | Needs: rename, add Priority groupBy |
| 2 | Table | Table | Priority | OK as base, needs filter |
| 3 | Roadmap | Roadmap | -- | OK |
| -- | (missing) | -- | -- | Need 3 new views |

### Manual Steps Required (UI only, ~10 min)

1. **Rename "Board" → "Prioritized backlog"**, set GroupBy=Priority, SortBy=Priority ASC
2. **Create "Current iteration" view** (Board), set filter=`iteration:@current`
3. **Create "Next iteration" view** (Board), set filter=`iteration:@next`, GroupBy=Priority
4. **Rename "Table" → "In review"**, set filter=`status:"In review"`
5. **Create "My items" view** (Table), set filter=`assignee:@me`
6. Reorder tabs to match: Current iteration | Next iteration | Prioritized backlog | Roadmap | In review | My items

---

## Current State Assessment

| Metric | Value | Problem |
|--------|-------|---------|
| Total repo issues | 317 (11 open, 306 closed) |  |
| Issues in Project board | 48 | **Only 15% coverage** |
| Issues missing from Project | 269 | 85% invisible in project view |
| Issues without priority label | 187 | 59% have no P0/P1/P2/P3 label |
| Issues without ANY label | 3 | #110, #111, #254 |
| Redundant/overlapping labels | ~8 pairs | deploy vs deployment, etc. |
| v0.1 era issues | 0 | 21 commits, zero audit trail |
| Milestones | 0 | No release/phase grouping |
| Batch-created issues (same day) | 9 batches, 286 issues | All show as instant open-close |

### Project Field State (already upgraded)

| Field | Options |
|-------|---------|
| Status | Backlog, Ready, In progress, In review, Done |
| Priority | P0, P1, P2 |
| Size | XS, S, M, L, XL |
| Estimate | Number |
| Iteration | Sprint 1-4 (2-week, Mar 4 - Apr 28) |

---

## Phase 1: Add All 269 Missing Issues to Project (automated)

**Why:** A project board that shows 15% of work is useless for planning or context.

**Script approach:**
```bash
# For each issue not in project, add it
gh project item-add 1 --owner minivess-mlops \
  --url "https://github.com/minivess-mlops/minivess-mlops/issues/$NUM"
```

**Status assignment logic:**
- Issue state = CLOSED → Status = Done
- Issue state = OPEN + has P0/P1 label → Status = Ready
- Issue state = OPEN + has P2/P3 label → Status = Backlog

**Estimated API calls:** 269 item-add + 269 status-set + 269 priority-set = ~807 calls.
GitHub API rate limit: 5000/hour. Doable in one run with small delays.

---

## Phase 2: Backfill Priority Labels from Title Text (187 issues)

**Problem:** 187 issues have priority embedded in their title (e.g., `[P1]`, `P0:`, `Phase 2:`)
but the actual GitHub label was never applied.

**Heuristic mapping:**

| Title pattern | Priority label | Count (est.) |
|---------------|---------------|--------------|
| Title starts with `P0:` or contains `[P0]` | P0-critical | ~5 |
| Title starts with `P1:` or contains `[P1]` | P1-high | ~10 |
| Title starts with `P2:` or contains `[P2]` | P2-medium | ~8 |
| Phase 0-3 retrospective issues | P0-critical | ~10 |
| R6-remediation issues (#52-#59) | P1-high | ~8 |
| Graph-topology issues (#112-#136) | varies by title | ~25 |
| SAM3 tasks (SAM-01 to SAM-18) | P1-high | ~18 |
| Multi-task T1-T20 | P1-high | ~20 |
| Deploy flow tasks (#162-#172) | P1-high | ~11 |
| Post-training plugins (#314-#324) | P1-high | ~11 |
| Quasi-E2E phases (#332-#338) | P1-high | ~7 |
| Remaining unlabeled closed issues | P2-medium (default) | ~64 |

**Algorithm:**
1. Parse title for explicit priority prefixes (`P0:`, `P1:`, `P2:`, `P3:`)
2. For phase/task issues, infer from parent PR/issue group
3. For issues with `phase-complete` label, set P0-critical (these were foundation)
4. Default to P2-medium for remaining unlabeled closed issues
5. Apply both GitHub label AND Project Priority field

**Also fix the 3 completely unlabeled issues:**
- #110: Add labels `config`, `mlflow`, `P1-high`
- #111: Add labels `config`, `infrastructure`, `P1-high`
- #254: Add labels `serving`, `bug`, `P2-medium`

---

## Phase 3: Consolidate Labels (reduce 62 → ~45)

**Redundant/overlapping labels to merge:**

| Keep | Delete (merge into Keep) | Rationale |
|------|--------------------------|-----------|
| `deployment` | `deploy`, `deploy-flow` | 3 labels for same concept |
| `infrastructure` | `decoupling` | Decoupling is infra work |
| `testing` | `quasi-e2e` | Quasi-E2E is a type of testing |
| `data` | `data-quality`, `data-acquisition` | Consolidate data-* to `data` |
| `models` | `sam3-variants`, `multi-task` | Model-specific labels → use `models` + title |
| `metrics` | `evaluation` | Evaluation IS metrics |
| `research` | `topology-aware` | Research subcategory |
| `compliance` | `sdd-template` | SDD is compliance work |

**Labels to ADD:**
- `v0.1-legacy` — for retroactive v0.1 audit trail issues
- `v2-foundation` — for the Phase 0-9 scaffold issues
- `automated-backfill` — tag for issues created retroactively by automation

**Labels to KEEP as-is** (well-used, distinct):
`bug`, `enhancement`, `documentation`, `models`, `training`, `metrics`, `uncertainty`,
`annotation`, `compliance`, `observability`, `ci-cd`, `P0-critical`, `P1-high`,
`P2-medium`, `P3-low`, `phase-complete`, `tech-debt`, `infrastructure`, `serving`,
`agents`, `data`, `refactor`, `testing`, `analysis`, `mlflow`, `ensemble`, `dashboard`,
`graph-topology`, `pipeline-verification`, `real-data`, `paper-artifacts`, `prefect`,
`trigger`, `hydra`, `config`, `integration`, `visualization`

**Approach:** Use `gh label delete` + `gh issue edit` to re-label affected issues before
deleting the redundant label. This preserves the audit trail.

---

## Phase 4: Create Retroactive v0.1 Issues (21 issues)

**Why:** The v0.1 era (2023-04-24 to 2024-02-29) has 21 significant commits but ZERO
issues. For LLM discoverability and audit trail, each major v0.1 feature should have a
closed issue with:

- Title matching the commit message
- Body containing: commit hash, date, description of what was done
- Labels: `v0.1-legacy`, `phase-complete`, + domain label
- Created → immediately closed with "Completed retroactively" comment

**v0.1 issues to create (grouped by phase):**

| # | Title | Commit | Date | Labels |
|---|-------|--------|------|--------|
| 1 | v0.1: Initial MONAI 3D segmentation pipeline | 7fb3072 | 2023-04-24 | v0.1-legacy, models, P0-critical |
| 2 | v0.1: Model ensembling with per-sample stats | b6ea8bb | 2023-09-01 | v0.1-legacy, ensemble |
| 3 | v0.1: TensorBoard logging + cross-validation | 9f64855 | 2023-09-02 | v0.1-legacy, observability |
| 4 | v0.1: Weights & Biases integration | 2b2ec3f | 2023-09-05 | v0.1-legacy, observability |
| 5 | v0.1: OmegaConf → Hydra config migration | a1ad19d+5f4632f | 2023-10-05 | v0.1-legacy, config |
| 6 | v0.1: ML testing infrastructure | 8cbc9f1 | 2023-10-13 | v0.1-legacy, testing |
| 7 | v0.1: Run modes (debug, test_data, etc.) | f919d08 | 2023-10-20 | v0.1-legacy, config |
| 8 | v0.1: Docker + GitHub Actions CI | 53d83e1+48bc71c | 2023-10 | v0.1-legacy, ci-cd, infrastructure |
| 9 | v0.1: DVC data versioning with S3 | 7d1286c | 2023-11-02 | v0.1-legacy, data |
| 10 | v0.1: MLflow + Dagshub tracking | 7f9bf90 | 2023-11-08 | v0.1-legacy, mlflow |
| 11 | v0.1: BentoML serving + Docker | 6c78fb4+2947230 | 2023-11-08 | v0.1-legacy, serving |
| 12 | v0.1: Pre-commit hooks (black, isort, flake8) | 5de5de4 | 2023-11-09 | v0.1-legacy, ci-cd |
| 13 | v0.1: MetricsReloaded evaluation | 8dca5e8 | 2024-02-29 | v0.1-legacy, metrics |

**Also create ~8 retroactive issues for v2 commits that lack issue references:**
- Code review rounds R1-R5 (5 commits → 1 issue)
- Self-learning TDD skill setup (1 issue)
- Pre-commit/mypy fixes (1 issue)
- SegResNet/SwinUNETR/VISTA3D removal (1 issue)
- 368 mypy errors fix (1 issue)

---

## Phase 5: Timeline Beautification for Batch-Created Issues

**Problem:** 9 batches of issues were created and closed on the same day. In a Roadmap/
Gantt view, they all appear as zero-width bars. For human readability and LLM timeline
understanding, we need to spread them out.

**Approach:** Use the Project's **Start date** field (already exists) and create an
**End date** field. For each batch:

1. Calculate `batch_duration = closedAt - createdAt` (typically 0-2 days)
2. If same-day batch with N phases: spread phases evenly across the batch duration
3. If batch_duration = 0 (same day): use 1-hour intervals per phase as symbolic spacing
4. Set Start date = createdAt + (phase_index / N) * batch_duration
5. Set Estimate = effort weight (XS=1, S=2, M=3, L=5, XL=8)

**Batch timeline strategy:**

| Batch Date | Issues | Strategy |
|------------|--------|----------|
| 2026-02-23 | #3-#33 (31) | Phase 0-9 retroactive: 1 day each, sequenced |
| 2026-02-24 | #34-#51 (18) | Action items: all start same day (parallel work) |
| 2026-02-25 | #52-#75 (22) | R6 + experiment eval: 2 groups, sequential |
| 2026-02-26 | #76-#98 (23) | Evaluation/ensemble/conformal: 3 groups |
| 2026-02-28 | #104-#136 (32) | Graph topology: 3 priority tiers |
| 2026-03-01 | #138-#172 (35) | Deploy+dashboard+viz: 3 flows |
| 2026-03-02 | #176-#268 (86) | **Largest batch**: SAM3+multitask+data+infra |
| 2026-03-03 | #269-#312 (41) | QA+Hydra+VesselFM+advanced |
| 2026-03-04 | #314-#338 (18) | Post-training+Quasi-E2E |

**Size estimation heuristic:**
- Issues with "Phase" or "Phase N:" in title → M (unless it's a sub-task → S)
- Issues that are part of a larger PR group → S
- Standalone feature issues (#3-#22) → L
- Bug fixes (#104-#108) → XS
- Research/exploration issues → M

---

## Phase 6: Create Milestones for Release/Phase Grouping

**Milestones to create:**

| Milestone | Issues | Description |
|-----------|--------|-------------|
| v0.1-alpha | retroactive #1-#13 | Legacy codebase (2023-2024) |
| v2.0-scaffold | #23-#32 | AI-generated foundation (Phase 0-9) |
| v2.0-experiment | #62-#75 | Experiment evaluation pipeline |
| v2.0-evaluation | #76-#98 | Evaluation, ensemble, conformal |
| v2.0-topology | #112-#136 | Graph-constrained models |
| v2.0-deploy | #138-#172 | Deploy, dashboard, visualization |
| v2.0-data | #176-#192 | Data engineering pipeline |
| v2.0-verification | #194-#268 | Real-data verification + SAM3 |
| v2.0-infrastructure | #248-#285 | Docker, SkyPilot, HPO, MIG |
| v2.0-hydra | #287-#302 | Hydra config migration |
| v2.0-post-training | #305-#338 | Post-training plugins + Quasi-E2E |
| v2.1-backlog | open issues | Current open backlog |

---

## Phase 7: LLM Discoverability Optimization

**Goal:** Make every issue self-contained enough that an LLM agent scanning issues
can understand what was done, why, and how it connects to the broader architecture.

### 7a. Issue Body Template (for retroactive issues)

```markdown
## Context
[1-2 sentences: why this work was needed]

## What Was Done
- [Bullet list of concrete changes]
- Commit: `abc1234`
- PR: #NNN (if applicable)

## Key Files
- `src/minivess/path/to/file.py` — [what this file does]

## Dependencies
- Blocked by: #NNN
- Enables: #NNN

## Architecture Impact
[How this changes the system — for LLM context understanding]
```

### 7b. Issue Title Conventions (normalize existing)

Current titles are inconsistent:
- Some use `P0:`, `P1:` prefixes (redundant with labels)
- Some use `feat:`, `fix:` prefixes (redundant with issue type)
- Some use task IDs like `SAM-01:`, `T1:`

**Normalization rules:**
1. Remove `P0:`, `P1:`, `P2:` prefixes from titles (use labels instead)
2. Remove `feat:`, `fix:` prefixes (use labels: enhancement, bug)
3. Keep task IDs (`SAM-01:`, `T1:`, `Deploy Task 1:`) for discoverability
4. Ensure titles are descriptive enough to understand without opening the issue
5. Max 80 characters (GitHub truncates longer titles in list views)

### 7c. Cross-Reference Network

For each issue, add a comment linking to:
- Parent PR that closed it
- Sibling issues from the same batch
- Successor issues that built on this work

**Example comment for #112 (ccDice metric):**
```
## Cross-References
- **PR:** #137 (feat: Graph-constrained topology models)
- **Batch:** Graph topology P0 (#112-#118, #134-#136)
- **Successor:** #125 (graph-topology experiment config)
- **Used by:** Champion selection flow, analysis pipeline
```

---

## Phase 8: Project Views Configuration (MANUAL — API does not support view mutations)

**GitHub API limitation:** Views CANNOT be created or updated via GraphQL API.
All view configuration below must be done in the GitHub web UI at:
https://github.com/orgs/minivess-mlops/projects/1

### View 1: Current iteration (Board)
- Layout: Board
- Filter: `iteration:@current`
- Columns (verticalGroupBy): Status
- Visible fields: Title, Assignees, Status, Linked PRs, Sub-issues progress, Priority, Size, Estimate

### View 2: Next iteration (Board)
- Layout: Board
- Filter: `iteration:@next`
- GroupBy: Priority (P0/P1/P2)
- Columns (verticalGroupBy): Status
- Visible fields: Title, Assignees, Status, Linked PRs, Sub-issues progress

### View 3: Prioritized backlog (Board)
- Layout: Board
- Filter: (none — show all)
- GroupBy: Priority (P0/P1/P2)
- Columns (verticalGroupBy): Status
- SortBy: Priority ASC
- Visible fields: Title, Assignees, Status, Linked PRs, Sub-issues progress, Priority, Size, Estimate, Iteration

### View 4: Roadmap (Roadmap)
- Layout: Roadmap
- Date field: Iteration (or Start date)
- Visible fields: Title, Assignees, Status, Size

### View 5: In review (Table)
- Layout: Table
- Filter: `status:"In review"`
- Visible fields: Title, Assignees, Linked PRs, Repository, Reviewers, Sub-issues progress

### View 6: My items (Table)
- Layout: Table
- Filter: `assignee:@me`
- Visible fields: Title, Linked PRs, Sub-issues progress, Priority, Size, Estimate, Iteration

---

## Phase 9: Ongoing Maintenance Skill

Create a Claude Code skill at `.claude/skills/github-project-sync/SKILL.md` that:

1. **On new issue creation:** Automatically adds to project, sets priority/size from labels
2. **On issue close:** Sets Status=Done, fills End date
3. **Periodic audit:** Checks for issues missing from project, label inconsistencies
4. **Context generation:** Exports project state as markdown for LLM context

This is NOT a Claude Code skill (those are interactive). Instead, implement as:
- A GitHub Action that runs on issue events (`.github/workflows/project-sync.yml`)
- A script `scripts/sync_project.py` for manual backfill runs

---

## Execution Order and Dependencies

```
Phase 1 (add 269 issues)           ←─ NO DEPS, do first
    ↓
Phase 2 (backfill priorities)      ←─ needs issues in project
    ↓
Phase 3 (consolidate labels)       ←─ needs priority audit done
    ↓
Phase 4 (v0.1 retroactive issues)  ←─ needs label cleanup done
    ↓
Phase 5 (timeline beautification)  ←─ needs all issues in project
    ↓
Phase 6 (milestones)               ←─ needs all issues present
    ↓
Phase 7 (LLM discoverability)      ←─ needs milestones for grouping
    ↓
Phase 8 (project views)            ←─ needs fields populated
    ↓
Phase 9 (maintenance automation)   ←─ final, builds on all above
```

**Phase 1-3 can be fully automated** (script-driven, no human judgment needed).
**Phase 4 requires light human review** (v0.1 issue descriptions).
**Phase 5-6 are automated** with heuristic rules.
**Phase 7 is the most labor-intensive** — needs per-issue context generation.
**Phase 8-9 are quick configuration/scripting.**

---

## Effort Estimate

| Phase | Effort | Automation |
|-------|--------|------------|
| Phase 1: Add missing issues | 30 min | Fully automated script |
| Phase 2: Backfill priorities | 30 min | Automated with title parsing |
| Phase 3: Consolidate labels | 20 min | Semi-automated (merge script + manual review) |
| Phase 4: v0.1 retroactive | 45 min | Automated creation, light review |
| Phase 5: Timeline beautification | 30 min | Automated heuristic |
| Phase 6: Milestones | 15 min | Automated |
| Phase 7: LLM discoverability | 2-3 hrs | Semi-automated (template + AI fill) |
| Phase 8: Project views | 10 min | **MANUAL UI ONLY** (API limitation) |
| Phase 9: Maintenance | 45 min | Script + GitHub Action |
| **Total** | **~5-6 hrs** | |

---

## Success Criteria

1. **100% issue coverage** — All 317+ issues in the Project board
2. **100% priority coverage** — Every issue has a P0/P1/P2/P3 label AND project field
3. **Zero redundant labels** — Consolidated from 62 to ~45
4. **v0.1 audit trail** — All 21 v0.1 commits have corresponding closed issues
5. **Timeline accuracy** — Roadmap view shows realistic phased timelines, not instant bars
6. **Milestone grouping** — 12 milestones covering all work from v0.1 through current
7. **LLM-ready context** — Any issue body has enough context for an AI agent to understand
   what was done, why, and how it connects to the architecture
8. **Automated sync** — New issues auto-added to project with correct fields

---

## Design Principles

### For Human Developers
- **Roadmap view** shows project arc at a glance
- **Priority + Size** enables sprint planning
- **Milestones** group work into logical releases
- **Board view** shows current state without noise from 300+ Done items

### For LLM Agents
- **Self-contained issue bodies** — no need to read 5 other issues to understand one
- **Cross-reference comments** — dependency graph is navigable
- **Consistent label taxonomy** — predictable categorization
- **Title conventions** — searchable, unambiguous
- **Retroactive v0.1 issues** — complete project history from day 1
- Issues serve as a **semantic index** into the codebase: each issue links to specific
  files, commits, and architectural decisions. An LLM scanning issues gets a structured
  map of the entire system.
