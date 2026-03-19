# Eval Checklist: kg-sync

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **pdflatex compiles**: All generated `.tex` files compile with `pdflatex` without errors. (YES/NO)
2. **Idempotency**: Running the sync twice in succession produces byte-identical output files. (YES/NO)
3. **Orphan detection**: A planted orphan `run_id` (not present in MLflow) raises an error or warning during sync. (YES/NO)
4. **Schema validation**: All generated YAML files pass their respective JSON schema checks. (YES/NO)

## Behavioral Criteria (require judgment)

5. **No timestamps in generated .tex**: Generated `.tex` files contain no timestamps, build dates, or other non-deterministic content that would break idempotency. (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "sync the knowledge graph"
- "propagate belief changes to KG"
- "export KG snapshot"

### Should NOT trigger (2 prompts)

- "create a GitHub issue"
- "run tests with TDD"
