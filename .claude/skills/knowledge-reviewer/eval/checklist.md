# Eval Checklist: knowledge-reviewer

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **ERROR-severity fix or issue**: Every ERROR-severity failure results in either an immediate fix or a `gh issue create` call. (YES/NO)
2. **WARN-severity non-blocking**: WARN-severity failures are reported in output but do not block the workflow (exit code remains 0). (YES/NO)
3. **Quick mode skips bibliography**: In quick/pre-commit mode, the bibliography validation step is skipped for speed. (YES/NO)
4. **Exit codes correct**: Exit code is 0 for pass (warnings OK), 1 for any ERROR-severity failure. (YES/NO)

## Behavioral Criteria (require judgment)

5. **All 4 agents run in full mode**: Full mode executes all 4 review agents: link checker, PRD auditor, legacy detector, and staleness scanner. (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "validate the knowledge graph"
- "run knowledge reviewer"
- "check KG integrity"

### Should NOT trigger (2 prompts)

- "implement this feature"
- "launch training on cloud"
