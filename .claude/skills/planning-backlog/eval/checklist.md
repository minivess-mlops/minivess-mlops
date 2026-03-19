# Eval Checklist: planning-backlog

Tier: B
Max criteria: 8 (binary YES/NO)

---

## Structural Criteria (machine-parseable)

1. **Priority assigned**: Each created or triaged issue has exactly one priority label (P0, P1, P2, or P3) -- never zero, never multiple. (YES/NO)
2. **PRD linked**: Issue body contains a `## PRD Context` section with a valid `decision_id` that exists in the knowledge graph. (YES/NO)
3. **Project board updated**: The issue is added to the GitHub Project board with the `priority` field set correctly. (YES/NO)

## Behavioral Criteria (require judgment)

4. **Labels valid**: All labels applied to the issue exist in the repository's label set (no auto-creation of unknown labels). (YES/NO)

---

## Trigger Tests

### Should trigger (3 prompts)

- "plan the next sprint"
- "triage this paper into backlog"
- "review project board progress"

### Should NOT trigger (2 prompts)

- "monitor SkyPilot training"
- "create a literature report"
