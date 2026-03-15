# 2026-03-16 — Knowledge Graph Exists. Read It First. Always.

## What Happened

When starting the UpCloud archival + RunPod dev environment task, I immediately asked
the user questions like:
- "What replaces UpCloud for DVC storage?"
- "Is GCP still in play?"
- "What MLflow backend do you want?"

**Every single one of these is answerable from the knowledge graph and existing docs.**

I asked anyway. The user had to tell me — again — to read the repo's own knowledge
management system before asking them anything.

## What I Should Have Done

```
1. Read knowledge-graph/navigator.yaml           (domain routing)
2. Read knowledge-graph/domains/infrastructure.yaml  (cloud/compute domain)
3. Read knowledge-graph/decisions/L4-infrastructure/gpu_compute.yaml
4. Read CLAUDE.md Cloud GPU Strategy table       (RunPod/GCP/Lambda/UpCloud roles)
5. Read docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md
6. Read .env.example for existing cloud vars
7. Grep for existing SkyPilot YAMLs to see real current patterns
```

Only AFTER doing all of this should I surface a question to the user — and ONLY for
information that is genuinely ambiguous and not documented anywhere.

## What Was Knowable from Existing Docs (No Question Needed)

| Question I Asked | Where the Answer Was |
|-----------------|----------------------|
| "Is GCP still in play?" | CLAUDE.md line 168: "GCP \| Staging/prod \| L4/A100" — clearly active |
| "What's Lambda status?" | CLAUDE.md line 168: "Lambda Labs \| Rejected \| No EU availability" |
| "What is UpCloud's role?" | CLAUDE.md line 169: "UpCloud \| MLflow server \| CPU VPS \| Fixed €5/mo" |
| "What replaces UpCloud S3 for DVC?" | smoke_test_gcp.yaml comment: "until migrated to GCS P2.3" — GCS is the target |
| "Where does MLflow live for dev?" | dev_runpod.yaml + #716 issue: RunPod Network Volume /vol/mlruns pattern |

## The Pattern Failure

The knowledge graph system was built precisely to prevent this. The cover letter doc
at `docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md` exists as
a zero-context onboarding brief. The navigator.yaml routes every topic to the right
domain file. The invariants in navigator.yaml state the highest priorities explicitly.

When I start ANY task that touches:
- Cloud providers / GPU strategy → read infrastructure domain KG FIRST
- Data storage / DVC → read data domain KG FIRST
- MLflow / experiment tracking → read observability domain KG FIRST
- Model architecture → read models domain KG FIRST
- Testing / CI → read testing domain KG FIRST

**Not after failing to find the answer. FIRST.**

## Action Plan: Prevent Recurrence

### Immediate fix (this session):
Read KG infrastructure domain, gpu_compute decision, and existing SkyPilot YAMLs
BEFORE making any architectural claims or surfacing any questions.

### Structural fix (create this task):
Add to ACTIVATION-CHECKLIST.md in self-learning-iterative-coder:
> "Step 0 (before reading the plan): Read knowledge-graph/navigator.yaml and the
> domain files relevant to the task. If task touches cloud/infra, read infrastructure
> domain. This step is non-skippable."

### Tracking fix:
When the KG is STALE (e.g., gpu_compute.yaml still lists Lambda Labs as a live option
when CLAUDE.md marks it Rejected), UPDATE THE KG immediately as part of the task.
Staleness in the KG is the root cause of "I had to look elsewhere" which is the root
cause of asking redundant questions.

## Key KG Gap Found Today

`knowledge-graph/decisions/L4-infrastructure/gpu_compute.yaml` is STALE:
- Still has `lambda_labs` as an active option (should be archived/rejected)
- No `upcloud` entry at all (it handled MLflow — a real infrastructure decision)
- No `runpod` entry as the primary dev provider (it's the actual primary)
- No `gcs_dvc` migration tracked

**This YAML needs updating as part of the UpCloud archival work.**

## Recurring Pattern

This is the third metalearning doc in two days about the same failure class:
- 2026-03-14: Wall-of-text questions instead of using AskUserQuestion
- 2026-03-16: Asking what tools can detect (Network Volume existence)
- 2026-03-16 (this doc): Asking what KG + docs already answer

The common root: defaulting to asking instead of defaulting to reading. **Always read
the repo's own documentation before surfacing ANY question to the user.**

## Related
- `.claude/metalearning/2026-03-16-asking-humans-what-tools-should-detect.md`
- `.claude/metalearning/2026-03-14-wall-of-text-bad-ux.md`
- `knowledge-graph/navigator.yaml` — use this as the first read for any task
- `docs/planning/cover-letter-to-sci-llm-writer-for-knowledge-graph.md` — zero-context onboarding brief
