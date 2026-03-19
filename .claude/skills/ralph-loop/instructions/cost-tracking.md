# Ralph Loop: Cost Tracking

## JSONL Event Format

Each Ralph Loop iteration appends a structured event to `outputs/ralph_diagnoses.jsonl`:

```json
{
  "attempt": 1,
  "cloud": "lambda",
  "region": "us-east-1",
  "gpu": "A100",
  "hourly_rate": 1.48,
  "start_time": "2026-03-14T16:00:00Z",
  "end_time": "2026-03-14T16:11:00Z",
  "duration_minutes": 11,
  "estimated_cost": 0.27,
  "status": "PARTIAL_SUCCESS",
  "training_ok": true,
  "artifact_upload_ok": false,
  "diagnosis": "MLFLOW_ARTIFACT_500"
}
```

## Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `attempt` | int | 1-indexed retry number within this Ralph session |
| `cloud` | str | Cloud provider: `"lambda"`, `"gcp"`, `"runpod"` |
| `region` | str | Cloud region where the job ran |
| `gpu` | str | GPU type: `"A100"`, `"A10"`, `"RTX4090"`, `"L4"`, etc. |
| `hourly_rate` | float | Cost per hour in USD for this GPU/cloud combo |
| `start_time` | str | ISO 8601 UTC timestamp of job launch |
| `end_time` | str | ISO 8601 UTC timestamp of job completion/failure |
| `duration_minutes` | int | Wall-clock duration in minutes |
| `estimated_cost` | float | `hourly_rate * duration_minutes / 60` |
| `status` | str | One of: `"SUCCESS"`, `"PARTIAL_SUCCESS"`, `"FAILED"`, `"PREEMPTED"` |
| `training_ok` | bool | Whether the training loop completed without error |
| `artifact_upload_ok` | bool | Whether MLflow artifacts were uploaded successfully |
| `diagnosis` | str | Failure category from the pattern table (null if SUCCESS) |

## Cost Tracking Rules

1. **Every attempt gets an event**: Even if the job fails in 30 seconds, log the event. The audit trail must be complete.

2. **Cost budget**: Each Ralph Loop session has a configurable cost budget (default $5). If cumulative cost across all attempts exceeds the budget, stop and report to user.

3. **Hourly rates**: Use the rates from SkyPilot's catalog (`sky show-gpus`). Do not hardcode rates -- they change frequently.

4. **Timestamps are UTC**: All `start_time` and `end_time` values use `datetime.now(timezone.utc)` per CLAUDE.md Rule #7.

5. **JSONL append-only**: Events are appended to `outputs/ralph_diagnoses.jsonl`, never overwritten. This file is the permanent audit trail across all Ralph sessions.

6. **Session report**: At the end of each Ralph session (success or budget exhausted), generate a summary report with:
   - Total attempts
   - Total cost
   - Total wall-clock time
   - Final status
   - MLflow experiment URL + run ID (if successful)
   - Training metrics summary (loss, dice, etc.) (if successful)
