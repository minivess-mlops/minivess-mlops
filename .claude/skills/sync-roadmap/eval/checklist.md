# Eval Checklist: sync-roadmap

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **Start date set**: All synced items on the project board have a non-empty start date. (YES/NO)
2. **Target date correct**: Closed issues use their `closedAt` date as the target date. (YES/NO)
3. **Size assigned**: Each synced item has a size field set using the correct heuristic (bug=XS, legacy=S, feature=M/L, epic=XL). (YES/NO)
4. **Estimate matches size mapping**: The numeric estimate field matches the size-to-points mapping (XS=1, S=2, M=3, L=5, XL=8). (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "sync roadmap timeline fields"
- "backfill project timeline dates"
- "sync recently closed issues"

### Should NOT trigger (2 prompts)

- "create a new issue"
- "run knowledge reviewer"
