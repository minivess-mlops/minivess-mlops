# Eval Checklist: issue-creator

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **YAML metadata block present**: Issue body contains a YAML metadata block with all 7 required fields (title, priority, domain, labels, assignees, milestone, linked_decisions). (YES/NO)
2. **Domain value valid**: The `domain` value in metadata exists as a key in `knowledge-graph/navigator.yaml`. (YES/NO)
3. **Priority label match**: The GitHub priority label (P0/P1/P2/P3) on the created issue matches the `priority` field in the YAML metadata block. (YES/NO)

## Behavioral Criteria (require judgment)

4. **All citations hyperlinked**: Every reference to a paper, tool, or external resource uses `[text](URL)` format -- no bare text references without hyperlinks. (YES/NO)
5. **No duplicate**: The issue title does not closely match any existing open issue in the repository (checked via `gh issue list`). (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "create issue for implementing CbDice loss"
- "file issue from test failure"
- "open issues from this plan"

### Should NOT trigger (2 prompts)

- "monitor the SkyPilot job"
- "write a literature report"
