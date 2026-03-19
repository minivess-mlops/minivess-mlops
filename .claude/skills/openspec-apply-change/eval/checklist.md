# Eval Checklist — OpenSpec Apply Change

Tier C skill. Binary pass/fail criteria.

## Structural Criteria

1. **Context files read before implementation**
   - All files listed in `contextFiles` from apply instructions were read before writing code.
   - Pass: All context files read first.
   - Fail: Implementation started without reading context.

2. **Task checkboxes updated**
   - After completing a task, the corresponding `- [ ]` was changed to `- [x]` in tasks.md.
   - Pass: Checkboxes reflect completion state.
   - Fail: Tasks completed but checkboxes not updated.

3. **Progress displayed**
   - Output includes "N/M tasks complete" summary.
   - Pass: Progress visible.
   - Fail: No progress indication.

## Trigger Tests

**Should trigger:**
- "implement the tasks from this change"
- "continue implementing the auth-middleware change"
- "work through the remaining tasks"

**Should NOT trigger:**
- "propose a new change"
- "archive this change"
