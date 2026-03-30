# 2026-03-30 — Five Passes to Wire Docker Observability: Systemic Execution Gap

## Failure Pattern

Docker observability was PLANNED and CODE-WRITTEN in 5 separate passes across 3 weeks.
Each pass produced a council-reviewed XML plan (8-10/10 scores), wrote production code,
passed AST tests, and was merged to main. Yet after all 5 passes, when actually running
a Docker container, Claude Code had NO IDEA what was happening inside the container.

### Timeline of Passes

| Pass | Date | Plan Score | Code Written | Code Called at Runtime | Behavioral Test |
|------|------|-----------|-------------|----------------------|----------------|
| 1st  | ~Mar 14 | 9/10 | Modules created | Never | None |
| 2nd  | ~Mar 20 | 10/10 | Imports added to flows | Never | None |
| 3rd  | ~Mar 25 | 9.5/10 | Context managers wired | Partially | None |
| 4th  | ~Mar 28 | Proposed | Thread wiring | Partially | None |
| 5th  | Mar 30 | 6/10 (DevOps review) | Phase 2 stale | Already done | **STILL NONE** |

**Result: 5 planning documents, 5 code commits, 0 behavioral runtime tests.**

The council reviewer on pass 5 found that Phase 2 tasks were already implemented —
the plan author (Claude Code) didn't even read the current codebase before writing
remediation tasks. This is the meta-failure: **even the fix for "not verifying" was
itself not verified.**

## Root Cause Analysis

### 1. Planning ≠ Executing ≠ Verifying

Claude Code excels at planning (8-10/10 council scores) and code writing (tests pass).
But there is a systemic gap between "code that exists" and "code that runs in production."

The verification hierarchy has 5 layers, but only layers 1-3 are tested:

```
Layer 1: Code exists (AST scan)           ✅ Tested
Layer 2: Module imported (import check)   ✅ Tested
Layer 3: Function called in flow body     ✅ Tested (AST enforcement)
Layer 4: Function EXECUTES at runtime     ❌ NOT TESTED
Layer 5: Output is OBSERVABLE externally  ❌ NOT TESTED
```

### 2. Session Boundaries Reset Verification State

Each planning session starts fresh. Session N assumes Session N-1 completed everything.
But Session N-1 may have written code without running it in Docker. The state file
tracks "task DONE" but "DONE" means "code committed" not "verified in Docker."

### 3. No "Progress Tracking" Skill for Execution Verification

The `/self-learning-iterative-coder` skill tracks RED→GREEN→VERIFY for unit tests.
But there is no equivalent skill for Docker execution verification. The TDD loop
verifies code with `uv run pytest`, but never verifies with `docker compose run`.

### 4. Council Reviews Score Plans, Not Execution

The iterated-llm-council scores the plan document quality (completeness, consistency,
statistical validity). It does NOT verify that the plan was EXECUTED. A 10/10 plan
that is never executed produces zero value.

## What We Should Do Better

### A. Add a "Docker Execution Verification" Skill

Create `.claude/skills/docker-execution-verifier/SKILL.md` with:

1. **Pre-flight checklist** before ANY docker compose run:
   - Docker daemon running? (`docker info`)
   - Images exist? (`docker images | grep minivess`)
   - Output dirs writable? (`test -w outputs/`)
   - .env has secrets? (`grep MLFLOW .env`)
   - GPU accessible? (`nvidia-smi` for GPU flows)

2. **During execution monitoring protocol**:
   - `docker logs -f CONTAINER` in background
   - Poll `heartbeat.json` for GPU flows (every 30s)
   - Check `docker inspect --format '{{.State.Health.Status}}'`
   - If container exits: IMMEDIATELY read `docker logs` and diagnose

3. **Post-execution verification**:
   - Run `verify_artifact_chain()` from biostatistics_verification.py
   - Check events.jsonl for flow_start/flow_end
   - If ANY check fails: DO NOT mark task as DONE

### B. Behavioral Runtime Tests (Layer 4+5)

Add `tests/v2/behavioral/` test tier:
- `test_observability_smoke.py`: Run flow_observability_context(), verify events.jsonl
- `test_heartbeat_smoke.py`: Start GpuHeartbeatMonitor, verify heartbeat.json
- `test_healthcheck_smoke.py`: Create fresh/stale heartbeat, verify exit codes

These tests run in `make test-prod` (not staging — they need real file I/O).

### C. Modify TDD State to Track Execution Verification

Current TDD state: `{"status": "DONE", "note": "6 tests passing"}`
Proposed: `{"status": "DONE", "verified_in_docker": false, "note": "..."}`

A task is only fully DONE when both:
1. Unit tests pass (`uv run pytest`)
2. Docker execution verified (for tasks that produce Docker artifacts)

### D. Add "Execution Gate" to the TDD Skill

After CHECKPOINT in the TDD loop, add an EXECUTION-VERIFY step:
```
Step 5/7: CHECKPOINT - Git commit + state     [DONE]
Step 6/7: EXEC-VERIFY - Docker execution test [CURRENT]  ← NEW
Step 7/7: CONVERGE   - Quality gate check     [PENDING]
```

This step:
1. If the task modifies Docker-executed code → rebuild + run mini-test in Docker
2. Verify output files exist and contain expected events
3. If verification fails → iterate (like any other test failure)

## How to Apply

1. BEFORE any `docker compose run`: read `.claude/rules/docker-monitoring.md`
2. DURING execution: use the monitoring commands, NOT ad-hoc tail/sleep
3. AFTER execution: run verification module before marking task DONE
4. NEVER mark a Docker-related task as DONE without Docker execution proof
5. Council reviews should include "was this executed in Docker?" as a gate
