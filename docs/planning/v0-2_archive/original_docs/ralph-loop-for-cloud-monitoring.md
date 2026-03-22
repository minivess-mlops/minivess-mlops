# Ralph Loop for Cloud GPU Monitoring

## Problem Statement

Cloud GPU smoke tests on RunPod via SkyPilot fail silently unless actively monitored.
In the first session, 3 consecutive failures went unnoticed:

| Job | Status | Root Cause | Time Wasted |
|-----|--------|-----------|-------------|
| 1 | FAILED_SETUP | DVC env vars not resolved (literal `${VAR}`) | ~4 min |
| 2 | FAILED_SETUP | Same as Job 1 (blind retry) | ~4 min |
| 3 | FAILED | Old bare-VM approach: fragile filesystem + torch.save I/O error after spot preemption | ~18 min |

**Total wasted**: ~26 min of RunPod GPU time, ~$0.15. More importantly, the failures
accumulated without any automated diagnosis or corrective action.

## Design: Ralph Monitor Loop

A **monitor → diagnose → fix → relaunch** loop that:
1. Polls SkyPilot job status at configurable intervals
2. When a job fails, automatically fetches and analyzes logs
3. Categorizes the failure (setup vs runtime, known vs unknown)
4. Outputs a structured diagnosis that can drive code fixes
5. Optionally relaunches after fixes are applied

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                  ralph_monitor.py                      │
│                                                        │
│  while job_status not in {SUCCEEDED, CANCELLED}:       │
│    1. Poll: sky jobs queue → parse status               │
│    2. If RUNNING: tail logs, check for warnings         │
│    3. If FAILED_SETUP: fetch setup logs → diagnose      │
│    4. If FAILED: fetch run logs → diagnose              │
│    5. Write diagnosis to JSONL file                     │
│    6. If auto_relaunch and fixable: apply fix, relaunch │
│    7. Sleep(poll_interval)                              │
│                                                        │
│  Output: structured JSONL diagnosis file                │
└──────────────────────────────────────────────────────┘
```

### Failure Categories (Known Patterns)

| Category | Pattern in Logs | Severity | Auto-fixable? |
|----------|----------------|----------|---------------|
| `ENV_VAR_LITERAL` | `Invalid endpoint: ${VAR}` | SETUP | Yes (inline vars) |
| `UV_NOT_FOUND` | `uv: command not found` | SETUP | Yes (use python directly) |
| `DVC_NO_GIT` | `not a git repository` | SETUP | Yes (dvc init --no-scm) |
| `TORCH_SAVE_IO` | `inline_container.cc` | RUNTIME | Yes (atomic save) |
| `OOM` | `CUDA out of memory` | RUNTIME | Maybe (reduce batch/patch) |
| `SPOT_PREEMPTION` | `preempted` in status | RUNTIME | Auto (SkyPilot recovery) |
| `MLFLOW_AUTH` | `401 Unauthorized` | RUNTIME | No (check credentials) |
| `DATA_MISSING` | `No training data` | SETUP | No (DVC push first) |
| `DISK_FULL` | `No space left on device` | RUNTIME | Yes (increase disk_size) |

### Script Interface

```bash
# Monitor a specific job with 30s polling
uv run python scripts/ralph_monitor.py --job-id 4 --poll-interval 30

# Monitor the latest job
uv run python scripts/ralph_monitor.py --latest --poll-interval 30

# Launch + monitor in one command
uv run python scripts/ralph_monitor.py --launch sam3_vanilla --poll-interval 30

# Dry run: diagnose last failure without launching
uv run python scripts/ralph_monitor.py --diagnose-last
```

### Output: Diagnosis JSONL

Each failure produces a structured record:

```json
{
  "timestamp": "2026-03-14T06:00:00Z",
  "job_id": 4,
  "status": "FAILED_SETUP",
  "category": "ENV_VAR_LITERAL",
  "error_line": "Invalid endpoint: ${DVC_S3_ENDPOINT_URL}",
  "root_cause": "SkyPilot envs: section doesn't resolve ${} in DVC config files",
  "affected_files": ["deployment/skypilot/smoke_test_gpu.yaml"],
  "fix_suggestion": "Inline DVC remote config using shell variable expansion",
  "auto_fixable": true
}
```

### Integration with TDD Skill

The Ralph monitor feeds into the self-learning-iterative-coder loop:

```
OUTER: Plan execution (tasks S1.1.1 → S1.1.4)
  INNER: Ralph monitor loop
    1. Launch smoke test
    2. Poll status (30s interval)
    3. On failure: diagnose → update tdd-state.json
    4. RED: Write test for the failure pattern
    5. GREEN: Fix the code
    6. VERIFY: Run staging tests
    7. CHECKPOINT: Commit + rebuild Docker image + push GHCR
    8. Relaunch smoke test → back to step 2
```

### Key Design Decisions

1. **Script, not flow**: The monitor is a utility script (`scripts/ralph_monitor.py`),
   not a Prefect flow. It runs on the dev machine, not in Docker. It monitors
   remote SkyPilot jobs — it doesn't DO training.

2. **JSONL output**: Structured diagnosis enables automated analysis across sessions.
   File: `outputs/ralph_diagnoses.jsonl`

3. **No auto-fix without user confirmation**: The script diagnoses and suggests fixes.
   Automated code changes require the TDD skill (test first, then fix).

4. **Polling, not webhooks**: SkyPilot doesn't support webhooks. Polling via
   `sky jobs queue` is the only option. 30s default is a good balance.

5. **Known-pattern matching**: Uses exact string matching on log lines, not regex
   (per CLAUDE.md regex ban). Categories are defined as `(pattern, category)` tuples.

## Implementation Tasks

| Task | Description | Test |
|------|-------------|------|
| T1 | `scripts/ralph_monitor.py` — polling loop + status parsing | `test_ralph_monitor.py` |
| T2 | Log fetching + failure categorization | `test_ralph_diagnosis.py` |
| T3 | JSONL diagnosis output | `test_ralph_output.py` |
| T4 | Makefile integration (`make monitor-smoke-test`) | Manual |
| T5 | Atomic `torch.save()` fix (prevents `TORCH_SAVE_IO` category) | `test_atomic_checkpoint.py` |

## Cost Budget

Each Ralph loop iteration costs ~$0.06-0.12 per model per attempt on RunPod spot.
With proper diagnosis, most failures should be caught in 1-2 iterations, not 3+.

## Files

- `scripts/ralph_monitor.py` — Main monitoring script
- `outputs/ralph_diagnoses.jsonl` — Structured failure diagnoses
- `tests/v2/unit/test_ralph_monitor.py` — Unit tests for monitoring logic
- `src/minivess/pipeline/trainer.py` — Atomic checkpoint save (T5)
