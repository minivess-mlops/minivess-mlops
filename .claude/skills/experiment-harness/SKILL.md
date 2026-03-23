---
name: experiment-harness
version: 0.2.0
description: >
  Meta-skill that standardizes experiment run XML creation, enforces execution
  protocol on top of /factorial-monitor, and gates on live report writing.
  The "reproducible harness" — ensures every experiment pass compounds learning.
  ACTIVATE when: creating a new factorial experiment pass (debug or production),
  resuming a multi-pass experiment series, or preparing a production launch.
  DO NOT ACTIVATE when: monitoring a single job (use ralph-loop), running local
  tests (use self-learning-iterative-coder), continuing an existing pass mid-flight
  (use factorial-monitor directly), or doing non-experiment development work.
last_updated: 2026-03-23
activation: manual
invocation: /experiment-harness
metadata:
  category: operations
  tags: [experiment, harness, xml, factorial, compound-learning, reproducible]
  relations:
    compose_with:
      - factorial-monitor
      - issue-creator
      - self-learning-iterative-coder
      - plan-context-load
    depend_on:
      - factorial-monitor
      - ralph-loop
    similar_to: []
    belong_to:
      - overnight-runner
  # Formal skill tuple S = (C, π, T, R) per Bi et al. (2026)
  applicability_conditions:
    - "New experiment pass needed (debug or production)"
    - "Prior pass exists with report to reference"
    - "SkyPilot + GCP credentials available"
  termination_criteria:
    - "All jobs terminal AND report complete AND ≥1 test per observation"
    - "OR: max 2 relaunch cycles exhausted → escalate to user"
    - "OR: cost budget exceeded → emergency stop"
  interface:
    input: "Factorial config YAML + prior pass report"
    output: "XML plan + live report + new tests + issues + watchlist"
---

# Experiment Harness

> Meta-skill for reproducible experiment execution with compound learning.
> Creates XML plans, enforces execution protocol, and ensures every pass
> leaves the codebase measurably better than before.
>
> *"Things should compound."* — Guiding principle

## Why This Exists

Across 5 debug factorial passes, Claude Code repeatedly:
1. Created ad-hoc XML plans with inconsistent structure
2. Loaded `/factorial-monitor` but bypassed the XML protocol
3. Polled jobs ad-hoc without writing the live report
4. Lost compound observations to conversation context
5. Launched jobs with known-failing code (wasting cloud credits)

See: [references/failure-history.md](references/failure-history.md)

## Decision Tree: Which Skill?

```
Need to run experiments on cloud GPU?
  ├── New pass (first time or new fixes applied)?
  │     → /experiment-harness (THIS SKILL)
  ├── Continuing an existing pass mid-flight?
  │     → /factorial-monitor (monitoring only)
  ├── Single job diagnosis?
  │     → /ralph-loop
  └── Local code changes needed?
        → /self-learning-iterative-coder
```

## Architecture: Two-Layer Execution Model

```
Layer 1 (Deterministic): scripts/run_factorial.sh
  - Pure sky jobs launch calls in a loop
  - ANY researcher can run this without Claude Code
  - NO LLM involvement — reproducible, auditable

Layer 2 (Monitoring Harness): /experiment-harness → /factorial-monitor
  - Creates experiment XML (standardized template)
  - VALIDATES pre-launch gates (9 checks, 0 skips)
  - Creates report file BEFORE launching
  - Drives /factorial-monitor for monitoring + diagnosis
  - Updates report after EVERY state change
  - Captures compound learning (new tests, issues, observations)
```

**CRITICAL SEPARATION**: The .sh script is the PRODUCT. The harness is the
DEVELOPER TOOL. Production runs use ONLY the .sh script.

## Workflow (5 Phases)

```
Phase 1: GENERATE   → Create experiment XML from template     [protocols/generate.md]
Phase 2: VALIDATE   → Pre-launch gates (all must pass)        [protocols/validate.md]
Phase 3: EXECUTE    → Launch .sh + drive /factorial-monitor    [protocols/execute.md]
Phase 4: COMPOUND   → Tests, issues, observations, watchlist  [protocols/compound.md]
Phase 5: REFLECT    → Self-assess harness effectiveness       [protocols/reflect.md]
```

### Phase 1: GENERATE

Create the experiment XML from [templates/experiment-run.xml](templates/experiment-run.xml).

1. **Reference prior pass**: Read the most recent XML and its report.
   Extract failures, observations, and watchlist items.
2. **Load context**: `/plan-context-load` → navigator → domains → metalearning → registry
3. **Fill template**: All 8 required sections
4. **Validate XML**: `yaml.safe_load()` (NEVER regex) to verify structure
5. **Security check** (Bi et al. G1-G2): XML contains no hardcoded credentials,
   paths match actual config files, resource requests match declared limits

**Cognitive engagement checkpoint** (Shen et al. 2026): Before proceeding,
articulate in the report: *"WHY do I expect these specific watchlist items?
WHAT might go wrong that prior passes haven't revealed?"*

Output: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-{N}th-pass.xml`

### Phase 2: VALIDATE

Pre-launch gates that MUST ALL pass:

| Gate | Command | Criteria |
|------|---------|----------|
| Staging tests | `make test-staging` | 0 skipped, 0 failed |
| Prod tests | `make test-prod` | 0 skipped, 0 failed |
| Preflight GCP | `python scripts/preflight_gcp.py` | 9/9 passed |
| Docker image fresh | GAR timestamp > latest code commit | Image newer |
| Docker image verified | `docker run ... python3 -c "import ..."` | All imports OK |
| Factorial dry-run | `run_factorial.sh --dry-run` | Correct conditions |
| Report file created | `<report-output>` path from XML | File exists with headers |
| Prior pass referenced | `<session-context>` non-empty | Has root causes |
| Security scan | No credentials in XML, paths valid | Clean |

**If ANY gate fails → DO NOT LAUNCH. Fix first.**

### Phase 3: EXECUTE

1. Report file MUST exist at `<report-output>` with headers + empty tables (H1)
2. Launch: `bash scripts/run_factorial.sh <config.yaml>`
3. Enter `/factorial-monitor` Phase 2 (MONITOR)
4. After EACH poll: Update report status matrix (H2)
5. After EACH terminal job: Update cost table, check watchlist items (H2)
6. After ALL terminal: Write observations section

### Phase 4: COMPOUND

After all jobs are terminal, produce ALL of these (H3):

1. **Updated report** with timeline, cost, observations, test opportunities
2. **New tests** — at least 1 per cloud observation (not template-filling; each
   test must target a specific bug or edge case discovered during THIS pass)
3. **GitHub issues** for each unresolved problem
4. **Updated watchlist** for the NEXT pass (carry forward + new items)
5. **Metalearning doc** if any process failure occurred
6. **Cumulative state update** → append to `outputs/harness-state.jsonl`

**Compounding gate**: Session is NOT complete until all 6 artifacts exist.

**Cognitive engagement checkpoint**: Before writing tests, articulate:
*"WHAT did I observe that was NOT predictable from prior passes?
WHY does this observation matter for production reliability?"*

### Phase 5: REFLECT (Memento-Skills Read-Execute-Reflect-Write)

Self-assess after every pass:

1. **Compare metrics**: This pass vs. prior pass (success rate, cost, observation quality)
2. **Evaluate rules**: Did H1-H5 prevent failures, or did they add friction without value?
3. **Propose modifications**: Write `harness-reflection-{N}.md` with suggested rule changes
4. **Update harness-state.jsonl** with pass metrics for trend analysis

This phase prevents the harness from calcifying around rules that no longer apply.
Skills are living artifacts that self-improve from deployment (Zhou et al. 2026).

## XML Template

See [templates/experiment-run.xml](templates/experiment-run.xml) — 8 required sections.
Template is loaded on-demand (L3 progressive disclosure per Anthropic guide).

## Non-Negotiable Rules

Inherits F1-F5 from `/factorial-monitor`, plus:

| Rule | Name | Prevents | Detection |
|------|------|----------|-----------|
| H1 | REPORT-BEFORE-LAUNCH | Lost observations | Assert file exists before sky jobs launch |
| H2 | UPDATE-EVERY-POLL | Stale report | Timestamp check on report file after each poll |
| H3 | COMPOUND-OR-FAIL | Empty passes | Count artifacts ≥ 6 at session end |
| H4 | REFERENCE-PRIOR-PASS | Context amnesia | Assert `<session-context>` non-empty in XML |
| H5 | NO-AD-HOC-POLLING | Protocol bypass | All sky jobs queue goes through /factorial-monitor |
| H6 | REASON-BEFORE-TEMPLATE | Template-filling without thinking | Cognitive checkpoints in Phase 1 + 4 |

## Anti-Patterns

| Anti-Pattern | How It Manifests | Detection |
|-------------|-----------------|-----------|
| **Template zombie** | Fill all 8 XML sections with boilerplate, no genuine reasoning | Watchlist items identical to prior pass, no new hypotheses |
| **Launch-then-think** | Skip VALIDATE, launch immediately, fix on cloud | Gate failures discovered after job submission |
| **Report-at-end** | Write report only after all jobs terminal | Report file empty during execution |
| **Observation amnesia** | Cloud observations noted in chat, never written to report | `git diff report.md` shows no updates between polls |
| **Cost blindness** | Launching without checking prior pass cost or setting budget | No cost table in report during execution |

## Competency Questions (BDI validation)

After execution, these must all be answerable YES from artifacts:

1. Was the report file created BEFORE the first `sky jobs launch`?
2. Does every XML section contain non-placeholder content?
3. Did the cost table update after every terminal job?
4. Were at least N new tests written (N = number of cloud observations)?
5. Does the watchlist for the next pass carry forward ALL unresolved items?
6. Did the reflection phase identify at least 1 rule to keep and 1 to reconsider?

## Cumulative State (cross-pass trend tracking)

Append to `outputs/harness-state.jsonl` after each pass:

```json
{
  "pass": 6,
  "date": "2026-03-24",
  "branch": "test/run-debug-gcp-6th-pass",
  "jobs_total": 34,
  "jobs_succeeded": 34,
  "jobs_failed": 0,
  "cost_usd": 8.50,
  "observations": 8,
  "new_tests_written": 5,
  "issues_filed": 2,
  "watchlist_carried": 3,
  "watchlist_new": 2,
  "harness_version": "0.2.0",
  "normalized_gain": 0.15
}
```

This enables: "Is the harness getting better over time?" via `duckdb-skills:query`.

## Measurement Protocol (SkillsBench normalized gain)

Track before/after per pass:

```
g = (pass_success_rate - prior_pass_success_rate) / (1 - prior_pass_success_rate)
```

A positive g means the harness + fixes improved outcomes. A negative g means
the harness caused harm (bloated context, wrong rules, etc.) — trigger Phase 5
REFLECT with extra scrutiny.

## Integration Points

| Skill | Role | When |
|-------|------|------|
| `/plan-context-load` | Load context before XML generation | Phase 1 |
| `/factorial-monitor` | Monitoring + diagnosis | Phase 3 |
| `/ralph-loop` | Per-job log analysis | Phase 3 (via factorial-monitor) |
| `/self-learning-iterative-coder` | Writing new tests | Phase 4 |
| `/issue-creator` | Filing issues | Phase 4 |
| `/search-metalearning` | Check prior failure patterns | Phase 1 |

## References

- Bi et al. (2026). "Automating Skill Acquisition." arXiv:2603.11808 — S=(C,π,T,R) tuple, security gates
- Li et al. (2026). "SkillsBench." arXiv:2602.12670 — normalized gain, focused > comprehensive
- Zhou et al. (2026). "Memento-Skills." arXiv:2603.18743 — Read-Execute-Reflect-Write loop
- Shen & Tamkin (2026). "How AI Impacts Skill Formation." arXiv:2601.20245 — cognitive engagement
- Ye et al. (2026). "Meta Context Engineering via Agentic Skill Evolution." arXiv:2601.21557
- Anthropic (2026). "Guide to Building Skills for Claude." — progressive disclosure, patterns
- [Compound Engineering Plugin](https://github.com/EveryInc/compound-engineering-plugin) — compounding principle
