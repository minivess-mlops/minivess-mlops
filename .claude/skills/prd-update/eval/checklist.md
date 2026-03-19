# Eval Checklist: prd-update

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **Probability sum**: All `prior_probability` arrays in modified decision nodes sum to 1.0 (tolerance +/-0.01). (YES/NO)
2. **No citation loss**: Every `citation_key` present in the file before the update is still present after the update (append-only). (YES/NO)
3. **DAG acyclic**: `review_prd_integrity.py` reports zero cycle errors on the updated knowledge graph. (YES/NO)
4. **Bibliography entry exists**: Every `citation_key` referenced in decision nodes resolves to a valid entry in `bibliography.yaml`. (YES/NO)

## Behavioral Criteria (require judgment)

5. **Author-year format**: All in-text citations in `rationale` and `evidence` fields use "Surname et al. (Year)" format, never numeric [1] style. (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "update priors for DynUNet decision"
- "add new decision node for loss function"
- "ingest this paper into the PRD"

### Should NOT trigger (2 prompts)

- "run the training pipeline"
- "create a GitHub issue"
