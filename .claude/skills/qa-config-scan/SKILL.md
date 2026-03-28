---
name: qa-config-scan
description: Detect configuration smells, hardcoded parameters, cross-file inconsistencies, and single-source-of-truth violations in the MinIVess MLOps codebase.
version: 1.0.0
author: Claude Code
triggers:
  - "/qa-config-scan"
  - "scan configs"
  - "check config smells"
  - "audit hardcoded params"
tags:
  - quality-assurance
  - configuration
  - static-analysis
  - testing
dependencies:
  tools: [Read, Grep, Bash, Write, Edit]
  skills: [self-learning-iterative-coder]
  python: [ast, pathlib, tomllib, yaml]
---

# QA Config Scanner Skill

Orchestrates a 4-phase pipeline to detect configuration quality issues in the
MinIVess MLOps codebase. Produces actionable violations with severity levels
and suggested pytest guard tests.

## When to Use

- Before any PR that touches config files, flow code, or pipeline modules
- After adding new Hydra config groups or Dynaconf settings
- When onboarding a new model/dataset/flow to verify config wiring
- Periodically as a codebase health check

## Architecture

```
Phase 1: COLLECT (deterministic)
    ├── AST scan: function defaults, keyword args, config class fields
    ├── YAML parse: Hydra groups, SkyPilot YAMLs, docker-compose
    ├── TOML parse: pyproject.toml, Dynaconf settings
    └── .env.example: version pins, feature flags

Phase 2: REVIEW (3 parallel LLM agents)
    ├── Agent A: Config Smell Detector (Sharma et al. MSR 2016 taxonomy)
    ├── Agent B: Cross-File Consistency Checker
    └── Agent C: Architecture Boundary Validator

Phase 3: VERIFY (filter false positives)
    ├── Allowlist check (known exceptions)
    ├── Context validation (is this actually config or infrastructure?)
    └── Severity assignment (critical / warning / info)

Phase 4: REPORT (output)
    ├── Violations list with severity + file:line + fix suggestion
    ├── Suggested new pytest guard tests
    └── Summary statistics
```

## Phase 1: COLLECT

**Goal**: Parse all configuration sources and run deterministic checks that
require zero LLM judgment.

**Protocol**: See `protocols/phase-1-collect.md`

### Checks

1. **Hardcoded params in flows/pipeline** (AST)
   - Function defaults with suspicious param names (seed, batch_size, etc.)
   - Keyword arguments in calls with literal values for config params
   - Guard test: `tests/v2/unit/config/test_no_hardcoded_params_in_flows.py`

2. **Version pin consistency** (.env.example vs Dockerfiles vs compose)
   - Every `*_VERSION` in .env.example must match all consumers
   - Guard test: `tests/v2/unit/config/test_version_pin_consistency.py`

3. **Config layer boundaries**
   - Hydra config groups reference only allowed keys
   - No Dynaconf settings leak into Hydra-zen space (and vice versa)
   - Guard test: `tests/v2/unit/config/test_config_layer_boundaries.py`

4. **Single-source .env.example compliance**
   - No `os.environ.get("VAR", "fallback")` in flow files
   - Guard test: `tests/v2/unit/test_env_single_source.py`

5. **YAML contract enforcement** (SkyPilot, docker-compose)
   - GPU allowlist, cloud provider list, schema drift
   - Guard test: `tests/v2/unit/deployment/test_yaml_contract_enforcement.py`

### Implementation

```python
# Phase 1 uses ONLY deterministic tools:
import ast          # Python source analysis (Rule 16: no regex)
import tomllib      # TOML parsing
import yaml         # YAML parsing (yaml.safe_load)
from pathlib import Path  # All paths via pathlib
```

## Phase 2: REVIEW

**Goal**: Three parallel LLM agents examine the COLLECT output and source
files for patterns that require judgment.

**Protocol**: See `protocols/phase-2-review.md`

### Agent A: Config Smell Detector

Applies the Sharma et al. (MSR 2016) taxonomy adapted for this project.
See `instructions/config-smell-patterns.md` for the full pattern catalog.

Smell categories:
- **Stale value**: Config value that no code path reads
- **Missing override**: Hardcoded value that should be in config
- **Type mismatch**: Config value parsed as wrong type
- **Implicit dependency**: Config A requires config B but relationship is undocumented
- **Scattered constant**: Same magic number in multiple files

### Agent B: Cross-File Consistency Checker

Verifies that values shared across files are consistent:
- GCS bucket names match across SkyPilot YAML, Pulumi config, .env.example
- Docker image tags match across compose files and SkyPilot YAML
- MLflow experiment names match across flow code and config
- Port numbers match across Dynaconf settings and docker-compose

### Agent C: Architecture Boundary Validator

Verifies config architectural rules from CLAUDE.md:
- Config definitions stay in `src/minivess/config/` (not scattered)
- Flows read config, never define it
- Pipeline modules receive config as parameters, never import .env directly
- No `os.environ.get()` with fallback defaults in flow/pipeline code

## Phase 3: VERIFY

**Goal**: Filter false positives from Phase 2 and assign severity levels.

### Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **CRITICAL** | Breaks reproducibility or violates CLAUDE.md Non-Negotiable | Must fix before merge |
| **WARNING** | Config smell that may cause issues | Should fix, create issue if deferred |
| **INFO** | Style improvement, minor inconsistency | Nice to have |

### False Positive Filters

1. **Config definition files** (`src/minivess/config/`) are allowed to define defaults
2. **Test files** are allowed to use literals (but should prefer config references)
3. **Private helpers** (`_helper()`) that always receive values from callers
4. **Display/layout values** (figsize, dpi, fontsize) are not research parameters
5. **Infrastructure constants** (ports, timeouts) that are NOT researcher-configurable

## Phase 4: REPORT

**Output format**:

```
## QA Config Scan Report

### CRITICAL (N violations)
- [file:line] Description — Fix: suggestion

### WARNING (N violations)
- [file:line] Description — Fix: suggestion

### INFO (N violations)
- [file:line] Description — Fix: suggestion

### Suggested Guard Tests
- `test_no_hardcoded_X_in_Y.py` — Description

### Summary
- Files scanned: N
- Violations: N critical, N warning, N info
- False positives filtered: N
```

## Usage

```
/qa-config-scan                    # Full scan, all phases
/qa-config-scan --phase 1          # Phase 1 only (deterministic, fast)
/qa-config-scan --focus flows      # Scan only orchestration/flows/
/qa-config-scan --focus pipeline   # Scan only pipeline/
/qa-config-scan --severity critical # Report only critical violations
```

## Related

- CLAUDE.md Rule 22: Single-Source Config via `.env.example`
- CLAUDE.md Rule 29: ZERO Hardcoded Parameters
- CLAUDE.md Rule 31: Zero Improvisation on Declarative Configs
- `tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py` (Issue #881)
- `.claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md`
