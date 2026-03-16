---
title: "Context7 vs Context Hub — Evaluation for MinIVess MLOps"
date: 2026-03-16
status: decided
decision: Context Hub (A)
branch: fix/claude-harness
---

# Context7 vs Context Hub

## Problem

Claude Code hallucates API signatures for libraries beyond its training cutoff.
Rule #10 says "verify models beyond knowledge cutoff" but web search is slow and noisy.
Need structured, version-specific library docs available at coding time.

## Candidates

| | Context Hub (Andrew Ng) | Context7 CLI (Upstash) |
|-|------------------------|----------------------|
| **What** | Pre-curated markdown docs, local registries | Cloud API, auto-indexed from GitHub |
| **CLI** | `chub get <id>` | `ctx7 library` + `ctx7 docs` |
| **Skill** | `.claude/skills/get-api-docs/SKILL.md` | `.claude/skills/find-docs/SKILL.md` |
| **Stars** | 6.2k | 49k |
| **Version** | v0.1.2 | v0.3.5 |
| **Offline** | Yes (local registries) | No (cloud-only, no cache) |
| **Custom docs** | `chub build` from local markdown | Submit to Upstash, wait for indexing |
| **Annotations** | `~/.chub/annotations/` persists across sessions | None |
| **Node.js dep** | Yes | Yes |

## Evaluation

| Criterion | Hub | Ctx7 | Winner |
|-----------|-----|------|--------|
| Offline / Docker containers | Local registries work offline | Cloud-only, zero offline | **Hub** |
| Reproducibility (TOP-2) | Markdown in git, deterministic | API results vary over time | **Hub** |
| Custom docs (MONAI, adapters) | `chub build` custom registry | No custom doc mechanism | **Hub** |
| KG integration | Annotations map to metalearning | Black box, no local state | **Hub** |
| Coverage breadth | 68 curated APIs | Thousands auto-indexed | **Ctx7** |
| Token cost | Curated = smaller (~500 lines) | Auto-parsed = larger (uncontrolled) | **Hub** |
| Freshness | Community-contributed, manual | Auto-crawled but stale (MONAI: 4mo) | Tie |
| Our key libraries | Missing MONAI, SkyPilot, hydra-zen | Has MONAI (510 snippets), SkyPilot (2.3k) | **Ctx7** |

## Decision: Context Hub

Context7 fails three non-negotiable criteria:
1. **Offline** — Docker containers and air-gapped labs have no internet (TOP-2)
2. **Reproducibility** — Cloud API results change over time, non-deterministic
3. **Custom docs** — Cannot serve our ModelAdapter docs alongside upstream MONAI

Context Hub's coverage gap (missing MONAI/SkyPilot) is fixable by writing custom
DOC.md files in a local registry. Context7's architectural gaps are not fixable.

## Integration Plan

```
docs/chub-registry/              # Custom docs (chub build source)
  monai/losses/DOC.md            # MONAI losses + our custom losses
  monai/transforms/DOC.md        # MONAI transforms we use
  skypilot/yaml/DOC.md           # SkyPilot YAML patterns
  minivess/adapters/DOC.md       # ModelAdapter ABC, SAM3, VRAM tables

.claude/skills/fetch-docs/       # Skill for Claude Code
  SKILL.md                       # When to fetch vs rely on training knowledge
```

## References

- [Context Hub](https://github.com/andrewyng/context-hub)
- [Context7](https://github.com/upstash/context7)
- [Gloaguen et al. (2026). "Evaluating AGENTS.md." *arXiv:2602.11988*](https://arxiv.org/abs/2602.11988)
