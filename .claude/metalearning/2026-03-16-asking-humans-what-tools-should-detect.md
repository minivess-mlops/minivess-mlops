# 2026-03-16 — Asking Humans What Tools Should Detect

## What Happened

When planning the RunPod MambaVesselNet smoke test, I asked the user:

> "Do you already have a RunPod Network Volume set up for this project, or does the plan
> need to provision one first?"

This is a **fundamental DevEx violation**. The answer is trivially detectable by running:

```bash
sky status                          # shows running clusters + volumes
sky storage ls                      # lists SkyPilot-managed storage
# or via RunPod API:
curl -s "https://api.runpod.io/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query":"{ myself { networkVolumes { id name size datacenterName } } }"}'
```

I asked a human a question that a tool could answer in 2 seconds. This is the **inverse of
excellent DevEx** — it creates manual work and friction instead of eliminating it.

## Why This Is a Pattern Failure

CLAUDE.md TOP-2 says:
> "Automate everything. Nobody should ever manually launch pods, VMs, or instances."

Asking "do you have a Network Volume?" is asking a human to do infrastructure introspection
manually. Every infrastructure state question follows the same principle:

| BAD: Ask human | GOOD: Run the command |
|----------------|----------------------|
| "Do you have a Network Volume?" | `sky storage ls` or RunPod GraphQL API |
| "Is MLflow running?" | `curl $MLFLOW_URI/health` |
| "Do you have GPU availability?" | `sky gpus --cloud runpod` |
| "Is DVC data pushed?" | `dvc status -r remote` |
| "Is GHCR image up to date?" | `docker manifest inspect ghcr.io/...` |
| "Do you have a GCS bucket?" | `gsutil ls gs://bucket-name` |

## The Correct Planning Pattern

When a plan depends on infrastructure state, the task that uses it must:

1. **Detect first** — run the check command as part of task T0 (pre-flight)
2. **Auto-provision if missing** — make provisioning a sub-task with a clear command
3. **Never surface to user** unless:
   - The check fails AND the fix requires a cost/access decision
   - OR the check involves billing that has a non-trivial cost

**CORRECT plan structure:**
```
T0.1: Check for existing RunPod Network Volume
  cmd: sky storage ls | grep minivess-vol  OR RunPod GraphQL API
  if found: use volume ID from output
  if not found: run T0.1b (provision)

T0.1b: Provision Network Volume (only if T0.1 found nothing)
  cmd: sky storage mount minivess-vol --store s3 ...
  OR: call RunPod API to create volume
  cost: ~$0.05/GB/month (inform user in output, not as a question)
```

**NEVER:** "Do you already have a Network Volume?"

## Root Cause

The failure pattern is defaulting to asking because it feels "safer." It isn't. It
makes the user context-switch into infrastructure management instead of staying in
their scientific workflow. A plan that detects and provisions is always better than
one that asks.

## Fix Applied

Removed the "do you have a Network Volume?" question from the questionnaire.
Replaced it with a pre-flight check task that runs `sky storage ls` and the RunPod
GraphQL API to auto-detect, then conditionally provisions.

## Related

- CLAUDE.md TOP-2: Zero Manual Work + Reproducibility
- CLAUDE.md Design Goal #1: Excellent DevEx for PhD Researchers
- `.claude/metalearning/2026-03-14-wall-of-text-bad-ux.md` — asking too many questions
- `.claude/metalearning/2026-03-14-docker-resistance-anti-pattern.md` — "just do it"
