# Eval Checklist: create-literature-report

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **Zero hallucinated citations**: Every `[Author (Year)]` in-text citation has a matching reference entry with a valid URL in the References section. (YES/NO)
2. **URL validity**: Every reference URL returns HTTP 200, 301, or 302 when fetched. No dead links. (YES/NO)
3. **No annotated bibliography**: The report does not contain sections with >5 consecutive single-sentence paper summaries (annotated bibliography anti-pattern = FAIL). (YES/NO)

## Behavioral Criteria (require judgment)

4. **Cross-domain synthesis**: At least 3 paragraphs cite papers from 2+ distinct research domains (e.g., topology + deep learning, or imaging + loss functions). (YES/NO)
5. **Markov novelty**: No two consecutive sentences start with the same syntactic pattern (e.g., "Smith et al. ... Smith et al. ..." or "This method ... This method ..."). (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "create literature report on vessel segmentation losses"
- "research report on topology-aware training"
- "literature survey for multiphoton imaging"

### Should NOT trigger (2 prompts)

- "create a GitHub issue"
- "sync the knowledge graph"
