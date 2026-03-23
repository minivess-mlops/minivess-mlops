---
name: experiment-harness
version: 0.1.0
description: >
  Meta-skill that standardizes experiment run XML creation, enforces execution
  protocol on top of /factorial-monitor, and gates on live report writing.
  The "reproducible harness" — ensures every experiment pass compounds learning.
  Use when creating a new factorial experiment pass (debug or production).
  Do NOT use for: single-job monitoring (use ralph-loop), local tests (use
  self-learning-iterative-coder), or existing pass continuation (use factorial-monitor directly).
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
    depend_on:
      - factorial-monitor
      - ralph-loop
    similar_to: []
    belong_to:
      - overnight-runner
---

# Experiment Harness

> Meta-skill that creates reproducible experiment run XMLs and enforces their
> execution protocol. Wraps `/factorial-monitor` with pre-launch gates, live
> reporting requirements, and compound learning enforcement.

## Why This Exists

Across 5 debug factorial passes (2026-03-19 to 2026-03-23), Claude Code repeatedly:
1. Created ad-hoc XML plans with inconsistent structure
2. Loaded `/factorial-monitor` but bypassed the XML protocol
3. Polled jobs ad-hoc without writing the live report
4. Lost compound observations (timing, failure modes, cost) to conversation context
5. Launched jobs with known-failing code (wasting cloud credits)

This skill prevents ALL of these by making the harness the entry point, not the
monitoring skill.

## Architecture: Two-Layer Execution Model

```
Layer 1 (Deterministic): scripts/run_factorial.sh
  - Pure sky jobs launch calls in a loop
  - ANY researcher can run this without Claude Code
  - NO LLM involvement — reproducible, auditable

Layer 2 (Monitoring Harness): /experiment-harness → /factorial-monitor
  - Creates the experiment XML (standardized template)
  - VALIDATES pre-launch gates (9 checks, 0 skips)
  - Creates the report file BEFORE launching
  - Drives /factorial-monitor for monitoring + diagnosis
  - Updates report after EVERY state change
  - Captures compound learning (new tests, issues, observations)
```

**CRITICAL SEPARATION**: The .sh script is the PRODUCT. The harness is the
DEVELOPER TOOL. Production runs use ONLY the .sh script.

## Workflow (4 Phases)

```
Phase 1: GENERATE   → Create experiment XML from template
Phase 2: VALIDATE   → Pre-launch gates (preflight, tests, Docker image)
Phase 3: EXECUTE    → Launch .sh + drive /factorial-monitor
Phase 4: COMPOUND   → Write final report, create tests, file issues
```

### Phase 1: GENERATE

Create the experiment XML using the standardized template. MUST:

1. **Reference prior pass**: Read the most recent `run-debug-factorial-experiment-*-pass.xml`
   and its report. Extract failures, observations, and watchlist items.
2. **Load context**: navigator.yaml → domain files → metalearning search → decision registry
3. **Fill template**: All 8 required sections (see Template below)
4. **Validate XML**: Parse with yaml.safe_load (not regex!) to verify structure

Output: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-{N}th-pass.xml`

### Phase 2: VALIDATE

Pre-launch gates that MUST ALL pass before `run_factorial.sh` is invoked:

| Gate | Command | Must Be |
|------|---------|---------|
| Staging tests | `make test-staging` | 0 skipped, 0 failed |
| Prod tests | `make test-prod` | 0 skipped, 0 failed |
| Preflight GCP | `python scripts/preflight_gcp.py` | 9/9 passed |
| Docker image fresh | Compare GAR image timestamp vs latest commit | Image newer than code |
| Docker image verified | `docker run minivess-base:latest python3 -c "import ..."` | All imports OK |
| Factorial dry-run | `run_factorial.sh --dry-run ...` | Correct condition count |
| Report file created | `report-output` path from XML exists | File created with headers |
| Prior pass referenced | XML `<session-context>` references prior failures | Non-empty |

**If ANY gate fails → DO NOT LAUNCH. Fix first.**

### Phase 3: EXECUTE

1. Create report file at `<report-output>` with headers + empty tables
2. Launch: `bash scripts/run_factorial.sh configs/experiment/debug_factorial.yaml`
3. Enter `/factorial-monitor` Phase 2 (MONITOR)
4. **After EACH poll**: Update report file with current status matrix
5. **After EACH terminal job**: Update cost table, check watchlist items
6. **After ALL terminal**: Write final observations section

### Phase 4: COMPOUND

After all jobs are terminal, the harness MUST produce:

1. **Updated report** at `<report-output>` with all sections filled
2. **New tests** for each cloud observation (at least 1 per observation)
3. **GitHub issues** for each unresolved problem
4. **Updated watchlist** for the NEXT pass (carry forward + new items)
5. **Metalearning doc** if any process failure occurred

**Compounding gate**: The session is NOT complete until all 5 artifacts exist.

## XML Template (Required Sections)

See [templates/experiment-run.xml](templates/experiment-run.xml) for the full template.

| Section | Purpose | Validated? |
|---------|---------|-----------|
| `<metadata>` | Branch, date, status, priority, cost, report-output | YES |
| `<session-context>` | Prior pass results + fixes applied | YES — must be non-empty |
| `<pre-launch-qa>` | Fixes applied, tests added, preflight result | YES |
| `<factorial-design>` | Factors, levels, conditions | YES — must match config |
| `<monitoring>` | Polling interval, phases, kill-switch | YES |
| `<watchlist>` | Items from prior failures + new concerns | YES — must carry forward |
| `<dual-mandate>` | Platform validation + improvement opportunities | YES |
| `<cloud-test-upgrade>` | Specific new tests to write from observations | YES |

## Non-Negotiable Rules

Inherits all 5 rules from `/factorial-monitor` (F1-F5), plus:

| Rule | Name | Prevents |
|------|------|----------|
| H1 | REPORT-BEFORE-LAUNCH | Report file must exist before first sky jobs launch |
| H2 | UPDATE-EVERY-POLL | Report updated after every status change, not just terminal |
| H3 | COMPOUND-OR-FAIL | Session incomplete without new tests + issues + observations |
| H4 | REFERENCE-PRIOR-PASS | Every XML must reference the prior pass's failures |
| H5 | NO-AD-HOC-POLLING | All monitoring goes through /factorial-monitor, never raw sky jobs queue |

## Integration Points

| Skill | Role |
|-------|------|
| `/factorial-monitor` | Monitoring + diagnosis (Phase 3 inner loop) |
| `/ralph-loop` | Per-job log analysis (via factorial-monitor) |
| `/self-learning-iterative-coder` | Writing new tests from observations (Phase 4) |
| `/issue-creator` | Filing issues for unresolved problems (Phase 4) |
| `/plan-context-load` | Loading context before XML generation (Phase 1) |

## Quality Gate

The harness is complete when ALL of these are true:
- [ ] XML plan exists with all 8 sections filled
- [ ] Report file exists with timeline, cost, observations, test opportunities
- [ ] At least 1 new test written per cloud observation
- [ ] At least 1 GitHub issue filed per unresolved problem
- [ ] Watchlist for next pass written (carried forward + new items)
