# Ralph Loop: Quality Evaluation Checklist

Binary pass/fail criteria for every Ralph Loop invocation.

## Criteria

| # | Criterion | Pass | Fail |
|---|-----------|------|------|
| 1 | **Max 3 retries per failure category enforced** | YES: Same failure category never retried more than 3 times; escalated to user after limit | NO: Infinite retry loop or >3 retries of same category without escalation |
| 2 | **Cost event logged for every attempt** | YES: Every launch/failure/success has a corresponding JSONL entry in `outputs/ralph_diagnoses.jsonl` | NO: Missing events, gaps in the audit trail |
| 3 | **Diagnosis category assigned to every failure** | YES: Every failed attempt has a `diagnosis` field matching one of the 14 known categories or a documented new category | NO: Failure without classification, or generic "unknown error" without investigation |
| 4 | **Pre-flight checks all executed before launch** | YES: All 7 pre-flight checks ran (credentials, GPU, MLflow, Docker, DVC, .env, disk) | NO: Skipped checks, launched without verifying prerequisites |
| 5 | **No regex used for log parsing** | YES: All pattern matching uses `str.split()`, `in`, `str.partition()` per CLAUDE.md Rule #16 | NO: `import re` or regex patterns used for log analysis |
| 6 | **Report generated at session end** | YES: Summary report with total attempts, cost, time, status, and MLflow URL produced | NO: Session ended without summary, or partial report missing key fields |
