# QA Skill: Config Single-Source-of-Truth Scanner — Design Plan

**Date**: 2026-03-28
**Status**: Design approved, ready for implementation
**Skill name**: `qa-config-scan`
**Invocation**: `/qa-config-scan` (manual full scan) + pre-commit hook (fast targeted checks)

## Design Philosophy

> **The scanner is NOT a parallel enforcement system. It is a gap-finder that generates
> new pytest test cases.** The output of `/qa-config-scan` should be "here are N new test
> functions to add to existing test files," not a second PASS/FAIL system competing with pytest.

The codebase already has strong config enforcement:
- `tests/v2/unit/test_env_single_source.py` (336 lines, .env.example enforcement)
- `tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py` (95 lines, AST scanning)
- `scripts/validate_yaml_contract.py` (GPU type + cloud provider enforcement)
- `tests/v2/unit/finops/test_cost_governance.py` (10 tests, region + cost enforcement)

This skill fills the GAPS between these existing checks.

## Reviewer-Validated Architecture: 3 Modules (Not 4)

The original proposal had 4 modules. Reviewer critique identified fatal flaws in the
generic cross-file comparison approach and recommended 3 targeted modules instead.

### Module A: Version Pin Consistency (Highest ROI)

**What it catches**: The MLFLOW_SERVER_VERSION class of incident — a version pinned
in `.env.example` is hardcoded in 5 other files that can drift independently.

**Why highest ROI**: The `mlflow-version-mismatch-fuckup.md` metalearning doc documents
8+ hours of debugging caused by exactly this pattern. The `mlflow-413-10-passes` incident
was partially caused by the training Docker image having different deps than the server.

**Live violation RIGHT NOW** (reviewer found this):
- `.env.example`: `MLFLOW_SERVER_VERSION=3.10.0`
- `Dockerfile.mlflow` line 9: `FROM ghcr.io/mlflow/mlflow:v3.10.0` (hardcoded)
- `Dockerfile.mlflow-gcp` line 4: `FROM ghcr.io/mlflow/mlflow:v3.10.0` (hardcoded)
- `docker-compose.yml` line 102: `image: minivess-mlflow:v3.10.0` (hardcoded)
- `pulumi/gcp/__main__.py` line 36: `MLFLOW_SERVER_VERSION = "3.10.0"` (hardcoded)

**Implementation**: 50 lines of Python. Parse `.env.example` for version-pinned
values (keys ending in `_VERSION`). Grep Dockerfiles, docker-compose, Pulumi for
hardcoded version strings. FAIL if any disagree.

**Scope of version pins to track**:
- `MLFLOW_SERVER_VERSION` → Dockerfiles, docker-compose, Pulumi
- `PYTHON_VERSION` → Dockerfiles (if pinned)
- `CUDA_VERSION` → Dockerfile.base FROM line
- Any future `*_VERSION` in .env.example

### Module B: Config Layer Boundary Enforcement

**What it catches**: Cross-contamination between the three config systems:
- **Hydra** (training): seed, max_epochs, model, losses, patch_size, batch_size
- **Dynaconf** (deployment): agent_provider, langfuse_enabled, braintrust_enabled
- **.env.example** (infrastructure): ports, hostnames, credentials, paths

**Why needed**: These three systems serve different layers. If a Hydra YAML defines
`langfuse_enabled`, or a Dynaconf TOML defines `seed`, the boundary is violated and
maintenance becomes a nightmare.

**Implementation**: Define the boundary:
```python
HYDRA_ONLY_KEYS = {"seed", "max_epochs", "batch_size", "learning_rate", "loss_name",
                    "model", "patch_size", "in_channels", "out_channels", "optimizer"}
DYNACONF_ONLY_KEYS = {"agent_provider", "langfuse_enabled", "braintrust_enabled",
                       "environment", "drift_detection_threshold"}
```
Scan Dynaconf TOML for Hydra keys → violation.
Scan Hydra YAML for Dynaconf keys → violation.

### Module C: Hardcoded Parameter Detection in Pipeline Code

**What it catches**: `def train(alpha=0.05, seed=42)` in consumer code that should
read from config instead.

**Scope (NARROW — reviewer recommendation)**: Only scan:
- `src/minivess/orchestration/flows/` (flow files consume config)
- `src/minivess/pipeline/` (pipeline code consumes config)
- NOT `src/minivess/config/` (config DEFINITION is legitimate)
- NOT `tests/` (test fixtures may legitimately hardcode for testing)

**Implementation**: Extend `test_no_hardcoded_alpha.py` pattern:
```python
SUSPICIOUS_PARAMS = {"alpha", "seed", "learning_rate", "batch_size", "max_epochs",
                      "n_bootstrap", "n_permutations", "port", "timeout"}
```
AST scan for `FunctionDef.args.defaults` matching these names with numeric literals.
Allowlist for intentional cases.

**Does NOT attempt**: Generic cross-file parameter mapping. The reviewer correctly
identified this as unsolvable without an explicit parameter registry.

---

## What This Skill Does NOT Do (Reviewer-Validated Exclusions)

1. **NOT a generic cross-file parameter mapper**. "Same logical parameter" requires
   semantic understanding that AST/YAML parsing cannot provide. The `yaml_contract.yaml`
   pattern (explicit registry) is the right approach for cross-file consistency.

2. **NOT a Hydra completeness checker**. Hydra already fails loudly on MISSING values
   via `OmegaConf`. Reimplementing this in a scanner is duplication.

3. **NOT a 5-pass bidirectional reader**. AST parsers read the entire file into memory.
   "Top-down vs bottom-up reading" is a human strategy, not a scanner strategy. The
   skill uses 2 phases: (1) Collect into registry, (2) Compare against rules.

4. **NOT an auto-fixer** for config files. Fixing config violations requires knowing
   which value is canonical. The scanner produces a diff-format report; the human decides.

---

## Skill Structure (Following NEW-SKILL-GUIDE.md)

```
.claude/skills/qa-config-scan/
├── SKILL.md                           # Orchestrator + frontmatter
├── instructions/
│   ├── module-a-version-pins.md       # Version pin consistency rules
│   ├── module-b-layer-boundaries.md   # Config layer boundary rules
│   └── module-c-hardcoded-params.md   # Hardcoded parameter detection rules
├── protocols/
│   ├── phase-1-collect.md             # Parse all config sources into registry
│   └── phase-2-compare.md             # Run all rules against registry
├── eval/
│   └── checklist.md                   # 6 binary YES/NO criteria
└── templates/
    └── violation-report.md            # Output format template
```

### SKILL.md Frontmatter

```yaml
---
name: qa-config-scan
version: 1.0.0
description: >
  Config single-source-of-truth scanner. Detects version pin drift, config layer
  boundary violations, and hardcoded parameters in pipeline code. Use when reviewing
  config changes, before experiment launches, or as a periodic code quality audit.
  Do NOT use for: Hydra config completeness (Hydra validates at runtime),
  generic cross-file comparison (use yaml_contract.yaml pattern instead),
  or auto-fixing config violations (produces reports, not fixes).
last_updated: 2026-03-28
activation: manual
invocation: /qa-config-scan
metadata:
  category: development
  tags: [config, quality, yaml, single-source-of-truth, DRY, scanning]
  relations:
    compose_with:
      - self-learning-iterative-coder   # Fix violations found by scanner
    depend_on: []
    similar_to:
      - knowledge-reviewer              # Both are multi-agent review skills
    belong_to: []
---
```

### Eval Checklist (6 criteria)

```markdown
# eval/checklist.md

## Structural (machine-parseable)
1. [ ] Version pin scan found zero new disagreements (Module A)
2. [ ] Config layer boundary scan found zero violations (Module B)
3. [ ] Hardcoded parameter scan found zero new violations (Module C)

## Behavioral (require judgment)
4. [ ] Report includes specific file:line for every violation
5. [ ] No false positives in the allowlisted exceptions
6. [ ] Scan completed in < 30 seconds
```

---

## Implementation Plan

### Phase 1: pytest Tests (extend existing files)

| # | Test | Extends | Module |
|---|------|---------|--------|
| 1 | `test_version_pin_consistency()` | `test_env_single_source.py` | A |
| 2 | `test_dockerfile_mlflow_version_from_env()` | `test_env_single_source.py` | A |
| 3 | `test_pulumi_mlflow_version_from_env()` | `test_env_single_source.py` | A |
| 4 | `test_docker_compose_mlflow_version_from_env()` | `test_env_single_source.py` | A |
| 5 | `test_dynaconf_no_hydra_keys()` | NEW `test_config_layer_boundaries.py` | B |
| 6 | `test_hydra_no_dynaconf_keys()` | NEW `test_config_layer_boundaries.py` | B |
| 7 | `test_no_hardcoded_seed_in_flows()` | `test_no_hardcoded_alpha.py` | C |
| 8 | `test_no_hardcoded_batch_size_in_flows()` | `test_no_hardcoded_alpha.py` | C |

### Phase 2: Pre-commit Hook (fast, <5s)

Add to `.pre-commit-config.yaml`:
```yaml
- id: version-pin-consistency
  name: Version pin consistency (MLflow, CUDA)
  entry: uv run python scripts/check_version_pins.py
  language: system
  files: '(\.env\.example|Dockerfile|docker-compose|__main__\.py)$'
  pass_filenames: false
```

### Phase 3: Skill Definition

Write the SKILL.md, instructions/, protocols/, and eval/ files.
The skill orchestrates: (1) run Module A-C tests, (2) aggregate results,
(3) produce violation report, (4) suggest fixes.

---

## Live Violations to Fix BEFORE Implementing the Scanner

The reviewer found these violations that exist RIGHT NOW:

1. **MLFLOW_SERVER_VERSION hardcoded in 5 files** — Fix: make Dockerfiles use
   `ARG MLFLOW_VERSION` and read from `.env.example` at build time. Pulumi should
   read from env var, not hardcode.

2. **`minio_bucket = "mlflow-artifacts"` in settings.toml** — Fix: this should
   reference the `.env.example` value or be removed (MinIO is local-only).

These violations should be fixed as part of the scanner implementation (TDD: write
the test that catches the violation, then fix the code).

---

## References

### Academic
- Sharma, Fragkoulis & Spinellis, "Does Your Configuration Code Smell?" MSR 2016
  — 13 implementation smells + 11 design smells across 4,621 IaC repos
- Horton & Parnin, "V2: Fast Detection of Configuration Drift in Python" ASE 2019
  — feedback-directed drift detection in Python configurations

### Tools Evaluated
- **Conftest/OPA**: Powerful but adds Rego dependency — rejected for pure-Python repo
- **Yamale**: YAML schema validation — useful but we already have check-yaml pre-commit
- **yamllint**: Style linting — already handled by ruff for Python, check-yaml for YAML
- **Pydantic BaseSettings**: Good for runtime validation — we use Hydra+Dynaconf instead

### Project-Internal
- `.claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md`
- `.claude/metalearning/2026-03-27-mlflow-413-10-passes-never-fixed-self-reflection.md`
- `.claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md`
- `docs/planning/v0-2_archive/critical-failure-fixing-and-silent-failure-fix.md`
- `tests/v2/unit/test_env_single_source.py` (existing enforcement)
- `tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py` (existing AST scanner)
- `configs/cloud/yaml_contract.yaml` (existing parameter registry pattern)
