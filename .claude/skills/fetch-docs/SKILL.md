---
name: fetch-docs
version: 1.1.0
description: >
  Fetch upstream library documentation via Context Hub (chub CLI) before implementing
  with beyond-cutoff libraries. Use when unsure about function signatures, API behavior,
  or parameter names for libraries near or beyond training knowledge cutoff.
  Do NOT use for: project-specific patterns in domain CLAUDE.md files or well-known stdlib.
last_updated: 2026-03-19
activation: demand
invocation: /fetch-docs
metadata:
  category: research
  tags: [documentation, libraries, web-search, context-hub]
  relations:
    compose_with: []
    depend_on: []
    similar_to: []
    belong_to: []
---

# Fetch Docs Skill

> Fetch upstream library documentation via Context Hub before implementing.

## When to Use

- Before implementing with a library near or beyond your training knowledge cutoff
- When unsure about a function signature, parameter name, or API behavior
- When domain CLAUDE.md files do not cover the specific API surface needed
- NEVER as a substitute for web-search on beyond-cutoff models (Rule #10 still applies)

## When NOT to Use

- For project-specific patterns already in domain CLAUDE.md files
- For libraries well within training knowledge (standard Python, basic PyTorch)
- For quick one-liners where training knowledge is sufficient

## Workflow

### Step 1: Check local custom registry first

```bash
chub search monai --source minivess
chub get minivess/monai-losses --lang py
```

The custom registry at `docs/chub-registry/` has project-specific docs for:
- `minivess/monai-losses` — MONAI losses + our custom CbDiceClDiceLoss
- `minivess/skypilot-patterns` — SkyPilot YAML patterns with Docker image_id
- `minivess/model-adapters` — ModelAdapter ABC, SAM3 VRAM tables

### Step 2: Check community registry

```bash
chub search torch
chub get torch --lang py
```

### Step 3: For beyond-cutoff models, ALSO web-search (Rule #10)

Context Hub docs may be stale. For models like SAM3, VesselFM, or any library
released after Aug 2025, ALWAYS perform a web search in addition to chub fetch.

## Configuration

Local registry configured at `~/.chub/config.yaml`:

```yaml
sources:
  - name: community
    url: https://cdn.aichub.org/v1
  - name: minivess
    path: /path/to/minivess-mlops/docs/chub-registry/dist
```

## Constraints

- Max 3 chub calls per question (token budget)
- Demand-invoked only — never auto-trigger
- Custom registry content is human-authored, never LLM-generated
- Annotations (`chub annotate`) persist at `~/.chub/annotations/` (machine-local)
