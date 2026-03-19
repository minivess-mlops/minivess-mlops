# Eval Checklist: overnight-runner

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **Heartbeat fires**: A heartbeat message is emitted every 60s during child process execution (verified via log timestamps). (YES/NO)
2. **Stall detection triggers**: When a child process produces 0 bytes of output for 10 minutes, stall detection fires and takes corrective action. (YES/NO)
3. **SKIP_TO resumption**: Setting SKIP_TO=N correctly skips the first N-1 children and resumes execution from child N. (YES/NO)
4. **Cost/time logged**: Each child session has its wall-clock time and estimated cost recorded in the run log. (YES/NO)

## Behavioral Criteria (require judgment)

5. **Screen-based execution**: Each child process runs inside a `screen` session for robustness -- `nohup` is never used. (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "run these 3 plans overnight"
- "execute batch plans unattended"
- "launch overnight batch with crash recovery"

### Should NOT trigger (2 prompts)

- "monitor a single SkyPilot job"
- "create a literature report"
