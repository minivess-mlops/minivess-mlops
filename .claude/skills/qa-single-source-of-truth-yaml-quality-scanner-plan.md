# QA Skill: Config Single-Source-of-Truth Scanner — Design Plan v2

**Date**: 2026-03-28 (v2 — dual-mode with web research insights)
**Status**: Design approved, ready for implementation
**Skill name**: `qa-config-scan`
**Invocation**: `/qa-config-scan` (full scan) + pre-commit hooks (deterministic gates)

## Architecture: 4-Layer Defense-in-Depth

Inspired by the DeepSource/Semgrep pattern: **"Deterministic First, LLM Second."**
The deterministic engine detects mechanical violations with zero false positives.
The LLM reviewer catches judgment-call issues that require semantic understanding.

```
Layer 1 (pre-commit, <5s):     Deterministic gates — version pins, env var consistency
Layer 2 (pytest, <30s):        Deterministic tests — AST scanning, cross-file checks
Layer 3 (local LLM, <3min):    /qa-config-scan skill — Claude reviews config with context
Layer 4 (remote, async):       Claude Code Review + CodeRabbit on PR creation
```

### Why 4 layers?

| Layer | Speed | False Positives | Catches |
|-------|-------|-----------------|---------|
| L1 Pre-commit | <5s | Zero | Mechanical violations (version drift, banned patterns) |
| L2 Pytest | <30s | Zero | Structural violations (AST scan, cross-file disagreement) |
| L3 Local LLM | <3min | Low (with verification) | Judgment calls (is this hardcoded value config-worthy?) |
| L4 Remote review | Async | Low (multi-agent) | Architectural issues (cross-layer coupling, design smells) |

**L1-L2 catch ~80% of violations deterministically.** L3-L4 catch the remaining ~20%
that require understanding intent, not just pattern matching.

---

## Layer 1: Pre-Commit Deterministic Gates (Already Implemented)

From the health harness (implemented this session):
- `ruff-strict-gate`: Zero lint errors across entire codebase
- `health-regression-gate`: Test count + ruff count monotonicity
- `yaml-contract-enforcement`: GPU types, cloud providers (existing)

**New additions (Module A):**
- `version-pin-consistency`: Parse `.env.example` for `*_VERSION` keys, verify
  matching values in Dockerfiles, docker-compose, Pulumi

**Implementation**: `scripts/check_version_pins.py` (50 lines), wired into
`.pre-commit-config.yaml` with file trigger on `(\.env\.example|Dockerfile|docker-compose|__main__\.py)$`

---

## Layer 2: Pytest Deterministic Tests (Extend Existing)

Existing enforcement (already works):
- `test_env_single_source.py` (336 lines) — .env.example as SSOT
- `test_no_hardcoded_alpha.py` (95 lines) — AST scan for alpha=0.05
- `test_cost_governance.py` (10 tests) — FinOps governance
- `test_health_harness.py` (15 tests) — harness integrity

**New tests to add:**

### Module A: Version Pin Consistency
```python
class TestVersionPinConsistency:
    def test_mlflow_version_matches_env() -> None:
        """MLFLOW_SERVER_VERSION in .env.example must match all Dockerfiles + Pulumi."""
    def test_dockerfile_mlflow_uses_build_arg() -> None:
        """Dockerfile.mlflow must use ARG, not hardcoded FROM tag."""
    def test_docker_compose_mlflow_version_from_env() -> None:
        """docker-compose.yml mlflow image tag must use ${MLFLOW_SERVER_VERSION}."""
```

### Module B: Config Layer Boundary Enforcement
```python
HYDRA_KEYS = {"seed", "max_epochs", "batch_size", "learning_rate", "model", ...}
DYNACONF_KEYS = {"agent_provider", "langfuse_enabled", "environment", ...}

class TestConfigLayerBoundaries:
    def test_dynaconf_no_hydra_keys() -> None:
        """Dynaconf TOML must not define training/model parameters."""
    def test_hydra_no_dynaconf_keys() -> None:
        """Hydra YAML must not define deployment-layer parameters."""
```

### Module C: Hardcoded Parameters in Pipeline Code (Narrow Scope)
```python
SUSPICIOUS_PARAMS = {"alpha", "seed", "learning_rate", "batch_size", "max_epochs",
                      "n_bootstrap", "n_permutations", "port", "timeout"}

class TestNoHardcodedParamsInFlows:
    def test_no_hardcoded_seed_in_flows() -> None:
        """flow files must not have seed=42 as function defaults."""
    def test_no_os_environ_get_with_fallback_in_flows() -> None:
        """flow files must not use os.environ.get('VAR', 'fallback')."""
```

**Scope**: Only `src/minivess/orchestration/flows/` and `src/minivess/pipeline/`
(consumer code). NOT config definitions, NOT test files.

---

## Layer 3: Local LLM Reviewer — `/qa-config-scan` Skill

This is where the LLM catches what deterministic tests cannot.

### What deterministic tests CANNOT catch:

1. **"Is this hardcoded value config-worthy?"**
   - `timeout=30` — is this a researcher-tunable parameter or an infrastructure constant?
   - `n_workers=8` — should this come from config or is it a reasonable default?
   - The answer depends on **context** (who calls this function, how the value is used)

2. **"Are these two values the same conceptual parameter?"**
   - `drift_detection_threshold = 0.05` in Dynaconf vs `alpha = 0.05` in Hydra
   - Same number, different concepts. Only an LLM with context can distinguish

3. **"Is this intentional duplication or a DRY violation?"**
   - Same loss function list in `debug.yaml` and `paper_full.yaml`
   - Intentional (different experiment configs). Not a violation.

4. **"Does this new config break the single-source-of-truth principle?"**
   - Adding a new env var to a SkyPilot YAML without adding it to `.env.example`
   - The deterministic test can check existing vars but not new/unknown ones

### Skill Architecture (inspired by awesome-skills/code-review-skill + claude-code-skills)

```
/qa-config-scan execution flow:

Phase 1: DETERMINISTIC COLLECT (5s)
  ├── Parse .env.example → key-value registry
  ├── Parse all YAML configs → flattened parameters
  ├── Parse Dynaconf TOML → deployment parameters
  ├── AST parse flow/pipeline Python files → function defaults, env lookups
  └── Run Module A/B/C checks → deterministic violations list

Phase 2: LLM REVIEW (2-3min, 3 specialized agents in parallel)
  ├── Agent 1: "Config Smell Detector"
  │   Read each Python file in flows/ + pipeline/
  │   Flag: hardcoded values that MIGHT be config-worthy
  │   Output: {file, line, value, confidence, reasoning}
  │
  ├── Agent 2: "Cross-File Consistency Auditor"
  │   Read the parameter registry from Phase 1
  │   Flag: values that appear suspiciously similar across files
  │   Output: {param_a, file_a, param_b, file_b, similarity, reasoning}
  │
  └── Agent 3: "Config Architecture Reviewer"
      Read the config directory structure + CLAUDE.md config rules
      Flag: new files that violate layering (Hydra/Dynaconf/env boundary)
      Output: {file, violation_type, reasoning}

Phase 3: VERIFICATION + DEDUP (30s)
  ├── Each agent's findings verified against allowlists
  ├── Cross-agent dedup (same issue found by multiple agents)
  ├── Severity assignment: ERROR (blocking) / WARN (investigate) / INFO (cosmetic)
  └── Report generation

Phase 4: REPORT
  ├── Deterministic violations (from Phase 1) — must fix before commit
  ├── LLM findings (from Phase 2) — review and decide
  └── Suggested new pytest tests to codify confirmed findings
```

### Multi-Pass LLM Review Protocol

The user correctly identified that single-pass LLM review is stochastic and misses
issues. The skill uses **3 specialized agents in parallel** (not 5 sequential passes):

1. Each agent focuses on ONE concern (smell detection, cross-file, architecture)
2. Agents run in parallel (not sequential) — faster, independent
3. A verification step filters false positives before reporting
4. If confidence < 70%, the finding is downgraded from ERROR to WARN

This follows the **multi-agent verification pattern** from:
- Claude Code Review (fleet of specialized agents + verification step)
- claude-code-skills (multi-model debate with AGREE/DISAGREE verdicts)

### Key Difference from Existing Review Tools

The existing tools (CodeRabbit, PR-Agent, Greptile) review **diffs** (what changed).
This skill reviews **the entire config system** (the current state, not just changes).
It's a periodic audit, not a per-commit gate.

---

## Layer 4: Remote Review Tools (External, Async)

For PR-level review, leverage managed services:

### Claude Code Review (Anthropic)
- **Setup**: GitHub App, `REVIEW.md` in repo root with config-specific rules
- **Cost**: ~$15-25 per review (Enterprise/Teams)
- **Config rules for REVIEW.md**:
  ```
  # Config single-source-of-truth rules
  - Flag any os.environ.get() with a fallback value in src/minivess/orchestration/
  - Flag any hardcoded version string that matches a *_VERSION key in .env.example
  - Flag new env vars in SkyPilot YAML that don't exist in .env.example
  ```
- **Docs**: https://code.claude.com/docs/en/code-review

### CodeRabbit (Free tier)
- **Setup**: GitHub App, `.coderabbit.yaml` at repo root
- **Cost**: Free for unlimited reviews on public + private repos
- **Config**: Custom review instructions for config validation
- **Docs**: https://coderabbit.ai

### Local CLI Alternative (no CI wait)
Both tools have CLI modes for local pre-push review:
- **Claude Code**: `claude review` (built into Claude Code CLI)
- **CodeRabbit**: `coderabbit review` (via npm)
- **PR-Agent** (open source): `docker run ... pr_agent review` (fully self-hosted)

---

## Reviewer-Validated Exclusions (What NOT to Build)

1. **NOT a generic cross-file parameter mapper** — semantic mapping is unsolvable
   generically. Use explicit parameter registry (`yaml_contract.yaml` pattern).

2. **NOT a Hydra completeness checker** — Hydra already fails loudly on MISSING.

3. **NOT an auto-fixer for config files** — produces reports, human decides
   which value is canonical. (Exception: deterministic Module A can auto-fix
   version pin updates via `scripts/update_version_pins.py`.)

4. **NOT a replacement for pytest** — the LLM skill GENERATES new pytest tests
   as its output. Confirmed findings become deterministic checks.

---

## The Virtuous Cycle: LLM Findings → Deterministic Tests

The key insight from DeepSource/Semgrep: **LLM findings that are confirmed should be
codified as deterministic tests.** This creates a ratchet:

```
Session 1: /qa-config-scan finds 5 issues
  → Developer confirms 3, dismisses 2 (false positives)
  → 3 new pytest tests added to test_env_single_source.py

Session 2: /qa-config-scan finds 3 NEW issues (old 3 caught by pytest now)
  → Developer confirms 2
  → 2 new pytest tests added

Session N: Most common patterns are now deterministic tests
  → LLM review only finds novel/rare violations
  → Scan is faster, more focused
```

This is the "Semgrep pattern" — the AI helps you write better deterministic rules,
not replace them.

---

## Implementation Roadmap

### Phase 1: Deterministic Tests (this session or next)
| # | Task | Module | Effort |
|---|------|--------|--------|
| 1 | `scripts/check_version_pins.py` + pre-commit hook | A | 30 min |
| 2 | Fix MLFLOW_SERVER_VERSION live violation | A | 15 min |
| 3 | `test_config_layer_boundaries.py` | B | 30 min |
| 4 | Extend `test_no_hardcoded_alpha.py` → seed, batch_size | C | 20 min |

### Phase 2: LLM Skill Definition
| # | Task | Effort |
|---|------|--------|
| 5 | Create `.claude/skills/qa-config-scan/SKILL.md` | 30 min |
| 6 | Write `instructions/config-smell-patterns.md` | 20 min |
| 7 | Write `protocols/phase-1-collect.md` and `phase-2-review.md` | 20 min |
| 8 | Write `eval/checklist.md` | 10 min |

### Phase 3: Remote Review Integration
| # | Task | Effort |
|---|------|--------|
| 9 | Create `REVIEW.md` with config rules for Claude Code Review | 15 min |
| 10 | Create `.coderabbit.yaml` with config review instructions | 15 min |

---

## References

### Tools (Reviewed and Evaluated)
- [Claude Code Review](https://code.claude.com/docs/en/code-review) — multi-agent fleet + verification
- [Claude Code Security Review](https://github.com/anthropics/claude-code-security-review) — open source GitHub Action
- [CodeRabbit](https://coderabbit.ai) — free tier, hybrid 40-linter + LLM
- [PR-Agent (Qodo)](https://github.com/qodo-ai/pr-agent) — open source, self-hosted
- [awesome-skills/code-review-skill](https://github.com/awesome-skills/code-review-skill) — best open-source Claude skill
- [claude-code-skills](https://github.com/levnikolaevich/claude-code-skills) — multi-model debate protocol
- [DeepSource](https://deepsource.com) — deterministic SAST + LLM autofix
- [Semgrep Assistant](https://semgrep.dev) — deterministic rules + LLM triage
- [Sourcery](https://sourcery.ai) — pre-commit Python review
- [Continue Dev](https://github.com/continuedev/continue) — privacy-first, self-hosted
- [Greptile](https://greptile.com) — full codebase graph indexing
- [Bito/CodeReviewAgent](https://github.com/gitbito/CodeReviewAgent) — AST knowledge graph

### Academic
- Sharma et al., "Does Your Configuration Code Smell?" (MSR 2016) — 13 implementation smells
- Horton & Parnin, "V2: Configuration Drift Detection" (ASE 2019)
- "Combining LLMs with Static Analyzers for Code Review" (arXiv 2502.06633)
- "Augmenting LLMs with Static Analysis" (arXiv 2506.10330) — 100% bug resolution
- Ciri: LLM Configuration Validation (arXiv 2310.09690)

### Key Architectural Patterns
1. **Deterministic First, LLM Second** (DeepSource, Semgrep) — lowest false positive rate
2. **LLM with Codebase Graph** (Greptile, Bito) — best for cross-file issues
3. **Multi-Agent Verification** (Claude Code Review) — highest finding quality
4. **Rule-Driven LLM** (Continue, REVIEW.md) — most customizable
5. **Virtuous Cycle** (Semgrep) — LLM findings become deterministic rules over time

### Project-Internal
- `.claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md`
- `.claude/metalearning/2026-03-27-mlflow-413-10-passes-never-fixed-self-reflection.md`
- `docs/planning/v0-2_archive/critical-failure-fixing-and-silent-failure-fix.md`
- `tests/v2/unit/test_env_single_source.py`
- `tests/v2/unit/biostatistics/test_no_hardcoded_alpha.py`
