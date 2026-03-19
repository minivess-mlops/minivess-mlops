# Eval Checklist — OpenSpec Archive Change

Tier C skill. Binary pass/fail criteria.

## Structural Criteria

1. **Archive directory created**
   - `openspec/changes/archive/YYYY-MM-DD-<name>/` exists after execution.
   - Pass: Archive directory with timestamped name created.
   - Fail: No archive directory or wrong naming format.

2. **Original directory removed**
   - `openspec/changes/<name>/` no longer exists after archiving.
   - Pass: Original cleaned up.
   - Fail: Original still present (duplicated, not moved).

3. **User prompted on incomplete tasks**
   - If tasks were incomplete, the user was asked for confirmation before archiving.
   - Pass: Confirmation requested when needed.
   - Fail: Archived without confirming incomplete work.

## Trigger Tests

**Should trigger:**
- "archive the auth-middleware change"
- "finalize and archive this completed change"
- "clean up the finished change"

**Should NOT trigger:**
- "propose a new change"
- "implement the remaining tasks"
