---
name: ralph-loop
version: 1.1.0
description: >
  Autonomous cloud GPU job monitoring, diagnosis, and recovery loop for SkyPilot jobs.
  Use when launching SkyPilot training jobs, diagnosing cloud failures (OOM, preemption,
  Docker pull, DVC), or monitoring cloud training autonomously.
  Do NOT use for: local Docker training, unit tests, factorial multi-job monitoring
  (use factorial-monitor), or infrastructure provisioning (use Pulumi).
last_updated: 2026-03-19
activation: manual
invocation: /ralph-loop
metadata:
  category: operations
  tags: [skypilot, monitoring, cloud, gpu, diagnosis, recovery]
  relations:
    compose_with:
      - issue-creator
      - self-learning-iterative-coder
    depend_on: []
    similar_to: []
    belong_to:
      - overnight-runner
---

# Skill: ralph-loop

> Autonomous cloud GPU job monitoring, diagnosis, and recovery loop.
> Named after Ralph — the monitor that watches your cloud training while you sleep.

## When to Use This Skill

Use `ralph-loop` whenever you need to:
- Launch a SkyPilot training job and monitor it to completion
- Diagnose failures automatically (setup, runtime, OOM, preemption, artifact upload)
- Re-launch after applying fixes (auto or manual)
- Track spending and time across retry attempts
- Run cloud training autonomously while the user is away

**Activation triggers** (use this skill when the user says):
- "Launch training on Lambda/GCP/cloud"
- "Run the smoke test and monitor it"
- "Fix the cloud training"
- "Run autonomously" or "I'm going offline"
- Any SkyPilot launch + monitoring scenario

**Do NOT use for:**
- Local Docker training (`docker compose run train`)
- Unit tests or linting
- Infrastructure provisioning (use Pulumi directly)

---

## Invocation

### Launch + Monitor

```
/ralph-loop --model sam3_vanilla --cloud lambda
/ralph-loop --model sam3_vanilla --cloud gcp --spot
```

### Monitor existing job

```
/ralph-loop --monitor minivess-smoke-test
```

### Diagnose last failure

```
/ralph-loop --diagnose-last
```

---

## What This Skill Does (The Ralph Loop)

```
┌─────────────────────────────────────────────────────────┐
│                    RALPH LOOP                            │
│                                                          │
│  1. PRE-FLIGHT CHECK                                     │
│     ├── Check cloud credentials (sky check)              │
│     ├── Check GPU availability (Lambda API / sky gpus)   │
│     ├── Check MLflow server health                       │
│     ├── Check Docker image exists in registry            │
│     └── Check DVC data is pushed                         │
│                                                          │
│  2. LAUNCH                                               │
│     ├── Select best region (availability-aware)          │
│     ├── Launch via SkyPilot Python API                   │
│     ├── Record launch time + cost rate                   │
│     └── Write launch event to JSONL                      │
│                                                          │
│  3. MONITOR (poll every 30s)                             │
│     ├── Check cluster status (sky status)                │
│     ├── If INIT → report provisioning progress           │
│     ├── If UP → tail logs, check for warnings            │
│     ├── If SUCCEEDED → collect results, report, exit     │
│     └── If FAILED → go to step 4                        │
│                                                          │
│  4. DIAGNOSE (on failure)                                │
│     ├── Fetch setup + run logs                           │
│     ├── Match against known failure patterns             │
│     ├── Classify: auto-fixable vs needs-user             │
│     ├── Write structured diagnosis to JSONL              │
│     └── Report diagnosis to user                        │
│                                                          │
│  5. FIX + RELAUNCH (if auto-fixable)                     │
│     ├── Apply fix (config change, env var, etc.)         │
│     ├── Rebuild Docker image if needed                   │
│     ├── Push to registry if needed                       │
│     └── Go to step 2 (max 3 retries)                    │
│                                                          │
│  6. REPORT (always)                                      │
│     ├── Total time, cost, retries                        │
│     ├── MLflow experiment URL + run ID                   │
│     ├── Training metrics (loss, dice, etc.)              │
│     └── Write to outputs/ralph_diagnoses.jsonl           │
└─────────────────────────────────────────────────────────┘
```

---

## Failure Categories (Known Patterns)

14 failure categories with symptoms, auto-fix logic, and resolution steps.
For the full pattern table and detailed resolution steps, see [instructions/failure-patterns.md](instructions/failure-patterns.md).

---

## Key Files

| File | Purpose |
|------|---------|
| `.claude/skills/ralph-loop/SKILL.md` | This file — skill definition |
| `scripts/launch_smoke_test.py` | Multi-region launcher with retry |
| `scripts/ralph_monitor.py` | (TBD) Standalone monitoring script |
| `outputs/ralph_diagnoses.jsonl` | Structured failure diagnosis log |
| `deployment/skypilot/smoke_test_lambda.yaml` | Lambda SkyPilot YAML |
| `deployment/skypilot/smoke_test_gpu.yaml` | RunPod SkyPilot YAML |

---

## Pre-Flight Checklist

Before launching, verify ALL of these:

```
[ ] Cloud credentials: sky check <cloud> → enabled
[ ] GPU availability: Lambda API or sky gpus list → capacity exists
[ ] MLflow health: curl -u admin:$PASS http://<host>:5000/health → OK
[ ] Docker image: docker manifest inspect <registry>/<image>:<tag> → exists
[ ] DVC data: dvc status -r <remote> → up to date
[ ] .env populated: MLFLOW_CLOUD_*, DVC_S3_*, GITHUB_TOKEN set
[ ] Disk space: df -h → sufficient for image pull + training
```

---

## Cost Tracking

JSONL event format, hourly rates, and cost budget rules.
For the full cost tracking specification, see [instructions/cost-tracking.md](instructions/cost-tracking.md).

---

## Integration with Other Skills

| Skill | Integration |
|-------|-------------|
| `self-learning-iterative-coder` | On failure: RED (write test) → GREEN (fix) → VERIFY → CHECKPOINT → relaunch |
| `overnight-runner` | Ralph loops can be children in overnight batch configs |
| `planning-backlog` | Failed categories auto-create GitHub issues |

---

## Design Decisions

1. **Skill, not script**: Ralph Loop is a Claude Code skill (autonomous decision-making)
   backed by `scripts/launch_smoke_test.py` (mechanical execution). The skill decides
   WHAT to do; the script does the launching.

2. **JSONL output**: Every event (launch, status, diagnosis, fix) is appended to
   `outputs/ralph_diagnoses.jsonl`. This is the audit trail.

3. **No regex for log parsing**: Per CLAUDE.md Rule #16, all pattern matching uses
   `str.split()`, `in`, and `str.partition()`.

4. **Max 3 retries per failure category**: Prevents infinite loops. After 3 retries
   of the same category, escalate to user.

5. **Cost budget**: Each loop has a configurable cost budget (default $5). If cumulative
   cost exceeds budget, stop and report.

6. **Multi-region rotation**: For GPU_SOLD_OUT, rotate through 17 Lambda regions
   starting from unpopular EU/Asia regions (implemented in `launch_smoke_test.py`).

---

## Quality Evaluation

For quality checks and pass/fail criteria, see [eval/checklist.md](eval/checklist.md).
