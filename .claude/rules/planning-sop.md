---
paths:
  - docs/planning/**
  - .claude/skills/**
  - knowledge-graph/**
---

# Planning SOP (Non-Negotiable)

Before creating ANY plan, follow these steps IN ORDER. Skipping steps = context amnesia.

## Pre-Planning Context Load (MANDATORY)

1. **Navigator**: Read `knowledge-graph/navigator.yaml` → route to relevant domain(s)
2. **Domains**: Read relevant domain YAML(s) → load resolved decisions
3. **Metalearning**: Search `.claude/metalearning/` for task keywords → load top-5 failure patterns
4. **Decision registry**: Check `knowledge-graph/decisions/registry.yaml` → identify ALREADY DECIDED questions
5. **Memory**: Read `MEMORY.md` → check for prior session decisions
6. **Source of truth**: Read the authoritative document (e.g., `pre-gcp-master-plan.xml` for factorial)

## Interactive Questionnaire (When User Invokes)

- **State what I think FIRST** — present current understanding from context load
- **Highlight uncertainties ONLY** — ask about things NOT in decision registry
- **Max 4 questions per round** — use AskUserQuestion tool
- **NEVER re-ask decided questions** — "DO_NOT_RE_ASK" in registry = absolute ban
- **Show provenance** — cite which doc informed each claim

## Post-Planning Validation

- **Contradiction check**: cross-reference against CLAUDE.md, metalearning, KG
- **Decision capture**: record NEW decisions in registry
- **Metalearning update**: update if plan changes prior understanding
- **NEVER write metalearning in panic** — wait for user confirmation

## Absolute Bans

- NEVER ask "should debug include X?" — Rule 27: debug = production
- NEVER say "full factorial is 24 cells" — it's 4 layers, 720+ conditions
- NEVER write metalearning that contradicts the flow merger plan
- NEVER re-ask questions marked DO_NOT_RE_ASK in registry

See: `.claude/context-management-upgrade-plan.md` for full architecture.
See: `docs/planning/context-compounding-and-learning-repo-plan.md` for prevention plan.
