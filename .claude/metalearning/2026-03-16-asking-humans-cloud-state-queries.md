# Metalearning: Asking Humans What CLI Tools Can Answer (2026-03-16)

## Severity: P0 — RECURRING failure (at least 3rd occurrence)

## What Happened

Claude asked the user "Is the RunPod Network Volume created and populated?" when:
1. The RunPod API key is in `.env`
2. `sky storage ls` can query volumes programmatically
3. `sky status` can check running clusters
4. The user's RunPod dashboard clearly shows `minivess-dev` volume (50 GB, EU-RO-1)
5. The user has explicitly complained about this MULTIPLE TIMES before

This violates CLAUDE.md Design Goal #1 ("EXCELLENT DevEx") and the metalearning doc
`.claude/metalearning/2026-03-14-asking-instead-of-checking.md` which already documented
this exact failure pattern.

## Root Cause

Claude uses `AskUserQuestion` as a shortcut instead of running CLI commands to
determine cloud state. This is lazy — the information is deterministically queryable.

## What Should Have Happened

```bash
# Instead of asking "Is the Network Volume created?"
sky storage ls 2>&1 | grep minivess-dev
# → minivess-dev-aec1627c-5d00db  50 GB  EU-RO-1

# Instead of asking "Is data uploaded?"
sky exec minivess-dev -- "ls /opt/vol/data/raw/minivess/imagesTr/ | wc -l" 2>/dev/null
# → N files (or error if empty)

# Instead of asking "Is Docker image pushed?"
docker manifest inspect ghcr.io/petteriteikari/minivess-base:mamba-latest 2>/dev/null
# → manifest JSON (or error if not pushed)
```

## Rule (MUST internalize)

**NEVER ask the user about cloud resource state when CLI tools can answer.**

Queryable state (use CLI, not AskUserQuestion):
- RunPod volumes: `sky storage ls`
- RunPod GPUs: `sky gpus --cloud runpod`
- RunPod pods: `sky status`
- GCP resources: `pulumi stack output` (from deployment/pulumi/gcp/)
- Docker images: `docker manifest inspect`
- DVC data: `dvc status -r <remote>`
- MLflow experiments: `mlflow.search_experiments()`
- SkyPilot jobs: `sky jobs status`

AskUserQuestion is for DECISIONS, not STATE QUERIES.

## Cross-References

- `.claude/metalearning/2026-03-14-asking-instead-of-checking.md` — SAME failure, 2 days ago
- `.claude/metalearning/2026-03-16-asking-humans-what-tools-should-detect.md` — related
- `.claude/rules/no-unauthorized-infra.md` — ask for DECISIONS, not STATE
- CLAUDE.md Design Goal #1: "EXCELLENT DevEx — zero manual work"
