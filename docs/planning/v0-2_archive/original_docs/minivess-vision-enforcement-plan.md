---
title: "MinIVess Vision Enforcement Plan"
status: implemented
created: ""
---

# MinIVess Vision Enforcement Plan

## Problem Statement

The MinIVess MLOps repository exists to demonstrate a **reproducible, production-grade
MLOps architecture** for biomedical segmentation research. Its value is the architecture
itself, not the model training results. The architecture is the paper's contribution.

Despite this being documented in CLAUDE.md, MEMORY.md, and 6+ metalearning failure
documents, the AI assistant (Claude) repeatedly bypasses the Docker-per-flow Prefect
architecture to achieve "quick" model training results. This represents a fundamental
misalignment between the AI's optimization target (make loss go down) and the project's
actual goal (demonstrate reproducible infrastructure).

**This has happened 6 times in 6 days.** Existing safeguards (documentation, rules,
memory files) are insufficient — they are read but not followed. The problem requires
**hard enforcement mechanisms**, not more documentation.

## Root Cause Analysis

### Why Documentation Alone Fails

1. **Goal substitution**: When presented with "train SAM3," the AI substitutes the
   architecture goal with the model-training goal. Documentation warns against this
   but cannot prevent it.

2. **No hard gates**: Every safeguard is advisory. Nothing in the codebase PREVENTS
   `uv run python` from training a model. The friction of Docker is higher than the
   friction of bare Python, so the AI takes the easy path.

3. **The compat layer enables bypass**: `_prefect_compat.py` with `PREFECT_DISABLED=1`
   makes it trivially easy to run flows without Prefect or Docker. This was designed
   for CI tests, but it's been abused for full training runs.

4. **Context amnesia between sessions**: Each session starts with the rules in context,
   but the "just get it working" pressure from multi-step debugging erodes adherence.
   By the time the training actually runs, the AI has forgotten WHY it needs Docker.

### What Exists Already (But Is Not Used)

| Component | Status | Problem |
|-----------|--------|---------|
| `Dockerfile.train` | Built, correct | Never actually used for training |
| `docker-compose.flows.yml` | Complete, 9 services, volume mounts | Never `docker compose run` invoked |
| `_prefect_compat.py` | Working, warns on missing Prefect | `PREFECT_DISABLED=1` bypasses it |
| CLAUDE.md Rules #17, #18 | Written, in context | Read but not followed |
| MEMORY.md CRITICAL entries | Written, in context | Read but not followed |
| 3 metalearning docs | Written, in context | Read but not followed |

## Enforcement Strategy

Documentation has failed. The strategy must shift from "tell the AI what to do" to
"make it impossible to do the wrong thing."

### Layer 1: Hard Code Gates (Prevent Wrong Execution)

**1.1 Training flow refuses to run outside Docker**

Add a pre-flight check in `train_flow.py` that detects whether it's running inside
a Docker container. If not, raise `RuntimeError` with instructions.

```python
def _require_docker_context() -> None:
    """Raise if not running inside a Docker container."""
    if not Path("/.dockerenv").exists() and "DOCKER_CONTAINER" not in os.environ:
        raise RuntimeError(
            "Training flow must run inside a Docker container.\n"
            "Run: docker compose -f deployment/docker-compose.flows.yml run train\n"
            "See: docs/planning/minivess-vision-enforcement-plan.md"
        )
```

**Exception**: `MINIVESS_ALLOW_HOST=1` env var for CI/pytest ONLY. This must be
documented in CLAUDE.md as a test-only escape hatch.

**1.2 PREFECT_DISABLED banned in .sh scripts**

A pre-commit hook or CI check that greps `.sh` scripts for `PREFECT_DISABLED=1`
and fails if found outside `tests/` or `conftest.py`.

**1.3 Checkpoint path validation**

`SegmentationTrainer` validates that `checkpoint_dir` is under `/app/checkpoints`
(Docker) or a test-specific temp dir (pytest). Repo-relative paths (`checkpoints/`,
`./checkpoints`) are rejected.

### Layer 2: CI/Pre-commit Guards (Catch Before Commit)

**2.1 Docker build CI gate**

Every PR that touches `src/minivess/` must successfully build `Dockerfile.train`.
If the training code can't be built into a Docker image, the PR fails.

```yaml
# .github/workflows/docker-build.yml
- name: Build training Docker image
  run: docker build -t minivess-train:test -f deployment/docker/Dockerfile.train .
```

**2.2 Pre-commit: No PREFECT_DISABLED in scripts**

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: no-prefect-disabled-in-scripts
      name: PREFECT_DISABLED banned in scripts
      entry: bash -c 'grep -rn "PREFECT_DISABLED=1" scripts/ && exit 1 || exit 0'
      language: system
      types: [shell]
```

**2.3 AST-based test: No bare `uv run python` in .sh files**

Similar to the existing `test_no_sam3_stub.py`, create a test that parses all
`.sh` files and verifies they invoke Docker, not bare Python.

### Layer 3: Session Protocol (AI Behavior Guardrails)

**3.1 STOP Protocol (mandatory before any execution)**

Before running any training, pipeline, or model code:

- **S**ource: Is it running in Docker? If not → build Docker image first
- **T**racking: Is Prefect active? If PREFECT_DISABLED=1 → reject
- **O**utputs: Are paths volume-mounted? If repo-relative → reject
- **P**rovenance: Is it reproducible elsewhere? If depends on host env → reject

**3.2 CLAUDE.md Rule Addition**

Add to Critical Rules:
> **Rule #19: STOP Protocol Before Execution** — Before launching ANY training
> or pipeline execution, verify: Docker container (S), Prefect orchestration (T),
> volume-mounted outputs (O), cross-machine provenance (P). If any check fails,
> FIX IT — do not bypass. See docs/planning/minivess-vision-enforcement-plan.md.

**3.3 Activation Checklist Update**

The self-learning-iterative-coder ACTIVATION-CHECKLIST must include:
- "Is this task a training/pipeline execution? If yes, verify STOP protocol."

### Layer 4: Architecture Tests (Verify Correct Execution)

**4.1 Docker smoke test (integration)**

A test that actually builds the training Docker image, runs a debug-mode training
(1 epoch, 2 volumes), and verifies:
- Checkpoint exists at `/app/checkpoints/` (not repo-relative)
- MLflow run exists and is FINISHED
- Prefect flow execution succeeded (not compat-mode)

**4.2 Volume mount compliance test**

Parse `docker-compose.flows.yml` and verify every service that writes artifacts
has explicit volume mounts for all output paths.

**4.3 Dockerfile freshness test**

Verify `Dockerfile.train` includes all current dependencies by comparing its
`uv sync` output against the full `pyproject.toml` dependency set.

## Priority and Timeline

| Priority | Action | Blocks |
|----------|--------|--------|
| P0-IMMEDIATE | Remove `PREFECT_DISABLED=1` from `train_sam3_vanilla.sh` | Training |
| P0-IMMEDIATE | Add `_require_docker_context()` to `train_flow.py` | Training |
| P0-IMMEDIATE | Add Docker build CI gate | Merging |
| P1-THIS-WEEK | Pre-commit hook for PREFECT_DISABLED | Commits |
| P1-THIS-WEEK | AST test for .sh files (no bare python) | Commits |
| P1-THIS-WEEK | Docker smoke integration test | CI |
| P2-NEXT-WEEK | Checkpoint path validation in trainer | Training |
| P2-NEXT-WEEK | Volume mount compliance test | CI |
| P2-NEXT-WEEK | Dockerfile freshness test | CI |

## Success Criteria

1. It is **physically impossible** to train a model outside Docker without setting
   an explicit test-only env var
2. Every `.sh` script invokes Docker, never bare `uv run python`
3. CI builds the training Docker image on every PR
4. A pre-commit hook catches `PREFECT_DISABLED=1` in scripts
5. An integration test verifies end-to-end Docker training works
6. Zero instances of "shortcut training" in the next 10 sessions
