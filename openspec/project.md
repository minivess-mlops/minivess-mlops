# MinIVess MLOps v2 — OpenSpec Project Context

## Purpose

Model-agnostic biomedical segmentation MLOps platform, built as an extension to the
MONAI ecosystem. Designed for multiphoton vascular imaging research with PhD-researcher
DevEx as the primary design goal.

## Architecture

- **5 Prefect flows** running in Docker-per-flow isolation
- **MLflow** as the inter-flow communication contract (artifacts only)
- **Hydra-zen** for experiment configuration, **Dynaconf** for deployment
- **ModelAdapter ABC** for model-agnostic integration

## Non-Negotiable Constraints

1. MONAI-first: if MONAI has it, use it directly
2. Docker-per-flow: no shared filesystem between flows
3. TDD mandatory: tests before implementation
4. No standalone scripts as run paths — Prefect flows only
5. Config-driven: new tasks/models/datasets via YAML, not code

## Current State

- 52 PRD decisions tracked in `knowledge-graph/decisions/`
- 33 resolved, 19 open
- 4,200+ passing tests across staging/prod/GPU tiers
- 612 issues, 68 PRs in 3 weeks of development
