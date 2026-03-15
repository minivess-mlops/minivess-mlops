# Metalearning: Stale "5-flow" Count in Cover Letter — 2026-03-16

## What Failed

When asked about the NEUROVEX flow architecture, I stated "5-flow pipeline" — a number that
appeared throughout the cover letter (`docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md`)
multiple times. The actual architecture has **10 flows**.

## Root Cause

The cover letter was written when only 5 `type: core` flows existed. It was never updated when
extension flows were added. The `flows.yaml` header even says "5 core flows + 1 best-effort" —
which also undercounts by omitting the 4 extension flows listed immediately below it.

**Lazy reading pattern**: I read the cover letter prose and trusted its numbers without
cross-checking against the authoritative source (`knowledge-graph/code-structure/flows.yaml`).

## Correct Architecture (as of 2026-03-16)

**Core flows (5, type: core — pipeline stops on failure):**
1. `data_flow` — Data Engineering — persona: Data Engineer
2. `train_flow` — Model Training — persona: ML Engineer
3. `post_training_flow` — Post-Training — persona: ML Engineer
4. `analysis_flow` — Model Analysis — persona: Statistician
5. `deploy_flow` — Deployment — persona: MLOps Engineer

**Best-effort flows (1 — failure does NOT block pipeline):**
6. `dashboard_flow` — Dashboard & Reporting — persona: Researcher/PI

**Extension flows (4, type: extension — research extensions):**
7. `acquisition_flow` — Data acquisition from lab instruments
8. `annotation_flow` — Label Studio annotation workflow
9. `biostatistics_flow` — Statistical analysis beyond model comparison
10. ~~`hpo_flow`~~ — **VERIFY WITH USER: this was in flows.yaml but user did not list it**

⚠️ `hpo_flow` appears in `flows.yaml` as a `type: extension` flow but the user did NOT
mention it in their 10-flow list. This may be a stale entry. **Ask user before including
in manuscript.**

## Key Concept Missed: Division of Labour

The user explicitly stated the flow architecture enables:
- Each flow can have its own developer (or parallel agent)
- Fewer merge conflicts because flows are decoupled
- This is a core DevEx/agentic development selling point for the paper

This concept was absent from the cover letter and must be added to the M3 description.

## Fix Applied

All 5 occurrences of "5-flow" in the cover letter updated to "10-flow Prefect-orchestrated"
with the division-of-labour concept added to M3.

## Prevention Rule

**Before stating any count about flows, models, nodes, or tests: READ flows.yaml (or the
relevant KG file) directly. Never trust prose summaries in cover letters or README files —
they go stale. The KG YAML files are the authoritative source.**
