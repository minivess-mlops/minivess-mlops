# CHANGE_ME Project Name

CHANGE_ME — one-line project description.

## Design Goal #1: CHANGE_ME

> **CHANGE_ME — your primary design goal in one sentence.**
> CHANGE_ME — elaboration of the goal.

### Core Principles
1. **CHANGE_ME** — First principle
2. **CHANGE_ME** — Second principle
3. **CHANGE_ME** — Third principle

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Project Type** | CHANGE_ME |
| **Python Version** | 3.12+ |
| **Package Manager** | CHANGE_ME (e.g., uv, pip, poetry) |
| **Linter/Formatter** | CHANGE_ME |
| **Type Checker** | CHANGE_ME |
| **Test Framework** | pytest |
| **Config** | CHANGE_ME |
| **ML Framework** | CHANGE_ME |

## Critical Rules

1. **Package Manager** — CHANGE_ME: specify your package manager policy
2. **TDD MANDATORY** — All implementation MUST follow test-driven development.
   Write failing tests FIRST, then implement. No exceptions.
3. **Library-First (Non-Negotiable)** — Before implementing ANY algorithm,
   ALWAYS search for existing implementations in established libraries.
   Only write custom code when no suitable library exists.
4. **Pre-commit Required** — All changes must pass pre-commit hooks before commit.
5. **Encoding** — Always specify `encoding='utf-8'` for file operations.
6. **Paths** — Always use `pathlib.Path()`, never string concatenation.
7. **Timezone** — Always use `datetime.now(timezone.utc)`, never `datetime.now()`.
8. **`from __future__ import annotations`** — At the top of every Python file.

## TDD Workflow (Non-Negotiable)

Every feature, bugfix, or refactor MUST use test-driven development:

```
1. RED:        Write failing tests first
2. GREEN:      Implement minimum code to pass
3. VERIFY:     Run tests + lint + typecheck
4. FIX:        If failing, targeted fix
5. CHECKPOINT: Git commit
6. CONVERGE:   All green? Move to next task
```

## Quick Commands

```bash
# Install dependencies
CHANGE_ME

# Run tests
CHANGE_ME

# Lint and format
CHANGE_ME

# Type check
CHANGE_ME

# Full verify (all three gates)
CHANGE_ME
```

## Directory Structure

```
CHANGE_ME/
├── src/CHANGE_ME/             # Main package
│   ├── CHANGE_ME/             # Core modules
│   └── config/                # Configuration
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Service integration tests
│   └── e2e/                   # End-to-end tests
├── configs/                   # Experiment/deployment configs
├── docs/                      # Documentation
└── CHANGE_ME                  # Additional directories
```

## What AI Must NEVER Do

- Use unauthorized package managers
- Write implementation code before tests (violates TDD mandate)
- Claim tests pass without running them ("ghost completions")
- Write placeholder/stub implementations (`pass`, `TODO`, `NotImplementedError`)
- Skip pre-commit hooks
- Hardcode file paths as strings
- Use `datetime.now()` without timezone
- Commit secrets, credentials, or API keys
- Push untested changes

## PRD System

The project uses a **hierarchical probabilistic SDD** (Bayesian decision network)
to manage open-ended technology decisions.

- Decision schema: `_schema.yaml`
- Network topology: `decisions/_network.yaml`
- Bibliography: `bibliography.yaml` (central, append-only)
- Validator: `validate.py --sdd-root <path>`

### Citation Rules (NON-NEGOTIABLE)
1. **Author-year format only** — "Surname et al. (Year)", never numeric [1]
2. **Central bibliography** — All citations in `bibliography.yaml`, decision files reference by `citation_key`
3. **No citation loss** — References are append-only. Never remove citations.
4. **Sub-citations mandatory** — When ingesting a paper, also extract its relevant references
5. **Validation** — Run `python validate.py --sdd-root <path>` to check invariants

## SDD Operations

Available protocols for maintaining the probabilistic SDD:

| Operation | Protocol | When to Use |
|-----------|----------|-------------|
| Add Decision | `protocols/add-decision.md` | New technology/architecture decision |
| Update Priors | `protocols/update-priors.md` | New evidence shifts probabilities |
| Add Option | `protocols/add-option.md` | New option for existing decision |
| Create Scenario | `protocols/create-scenario.md` | New implementation path |
| Ingest Paper | `protocols/ingest-paper.md` | Process research paper into evidence |
| Validate | `protocols/validate.md` | Check all SDD invariants |
| Citation Guide | `protocols/citation-guide.md` | Academic citation formatting |

## See Also

- CHANGE_ME — link to project-specific documentation
- CHANGE_ME — link to deployment/infrastructure docs
