# 2026-03-14 — Claude's Systematic Resistance to Docker-Native Development

## Severity: CRITICAL (Recurring Systemic Failure)

## What Happened

Despite this repo having explicit Docker-only mandates (Rules #17-19, STOP Protocol #19),
Claude repeatedly defaults to bare-metal/host execution patterns:
- Suggesting `uv run python scripts/*.py` instead of `docker compose run`
- Writing SkyPilot YAMLs with bare VM setup scripts (apt-get, uv sync, git clone)
- Defaulting to host-side execution for "quick tests"
- Treating Docker as optional/overhead rather than the fundamental execution model

This is not a one-time mistake. It is a **systematic anti-pattern** rooted in Claude's
training data, where the vast majority of GitHub repos use pip + requirements.txt + bare
Python execution. Docker-native development is a minority pattern in training data, so
Claude's priors strongly favor bare-metal execution.

## Root Cause Analysis

### Training Data Bias
Claude's training corpus is dominated by repos that:
- Use pip/conda/poetry + requirements.txt
- Run Python scripts directly on the host
- Treat Docker as an optional deployment concern, not the development environment
- View Docker as "overhead" or "over-engineering" for dev workflows

This creates a strong prior that Docker is an optimization to add later, not the
foundation to build on. When under pressure (debugging, time constraints), Claude
regresses to these priors — suggesting "simpler" host-side execution.

### The "Over-Engineering" Misclassification
Claude classifies Docker-per-flow isolation as "over-engineering" because most repos
don't need it. But this repo's architecture is INTENTIONALLY Docker-native because:

1. **Reproducibility** — The #1 mission. "Works on my machine" is unacceptable in
   scientific computing. Docker eliminates environment-dependent failures.
2. **DevEx** — Paradoxically, Docker IMPROVES DevEx by guaranteeing consistent
   environments. No more "did you install the right CUDA version?" troubleshooting.
3. **Cloud parity** — The same Docker image runs locally, on RunPod, on AWS, on
   intranet servers. Zero adaptation needed.
4. **Isolation** — Each flow has exactly the deps it needs. No import leakage, no
   version conflicts, no "this flow broke because another flow updated a dep."

### The "Quick Fix" Trap
When something fails in Docker, Claude's instinct is to bypass Docker for a "quick fix"
on the host. This is exactly wrong — the fix should ALWAYS be inside Docker. A host-side
fix that doesn't work in Docker is not a fix at all.

## Three-Environment Model (Non-Negotiable)

| Environment | Docker? | Compute | Purpose |
|-------------|---------|---------|---------|
| **dev** | Docker-free | Local GPU | `uv run pytest` ONLY — fast iteration |
| **staging** | Docker Compose | Local GPU in container | Integration testing |
| **prod** | Docker + SkyPilot | Cloud spot / on-prem K8s | Full pipeline |

"dev" is Docker-free ONLY for `uv run pytest`. ALL training, ALL pipeline execution,
ALL deployment goes through Docker. There is no "just run it on host" option.

## What Must Change in Claude's Behavior

1. **Default to Docker** — When suggesting how to run anything beyond unit tests,
   the first thought should be `docker compose run`, not `uv run python`.
2. **SkyPilot = Docker image** — Cloud execution is ALWAYS via `image_id: docker:<image>`.
   No bare VM scripts. Ever.
3. **Fix Docker issues IN Docker** — When a Docker run fails, debug inside the
   container. Never suggest "try running on host to see if it works."
4. **Docker is not overhead** — Docker is the execution model. It's like saying
   "typing is overhead for programming." No. It's the medium.

## Connection to Repo Vision

The user's highest priorities:
1. **Excellent DevEx** — automate everything, zero manual work
2. **Reproducibility** — combat the reproducibility crisis in ML/science
3. **Zero manual infrastructure** — Pulumi for IaC, SkyPilot for compute, Docker for execution

Docker is not "an extra step." Docker IS the reproducibility guarantee. Bypassing Docker
is bypassing the repo's core mission.

## Resolution

- [x] Metalearning doc created (this file)
- [x] CLAUDE.md updated with stronger Docker-native language
- [x] SkyPilot analogy corrected (intercloud broker, not IaC)
- [x] Memory updated with DevEx/reproducibility vision
- [ ] Add pre-commit check: no `uv run python` in scripts/ (only in tests/)
