---
paths:
  - "**"
---

NEVER ask the user about cloud resource state when CLI tools can answer.
Query programmatically FIRST, then act on the result.

| State | CLI Command |
|-------|------------|
| RunPod volumes | `sky storage ls` |
| RunPod GPUs | `sky gpus --cloud runpod` |
| Running pods | `sky status` |
| SkyPilot jobs | `sky jobs status` |
| Docker images | `docker manifest inspect <image>` |
| DVC data | `dvc status -r <remote>` |
| GCP resources | `pulumi stack output` |
| MLflow experiments | `python -c "import mlflow; ..."` |

AskUserQuestion is for DECISIONS (preferences, budget, approach), not STATE QUERIES.
See: .claude/metalearning/2026-03-16-asking-humans-cloud-state-queries.md
