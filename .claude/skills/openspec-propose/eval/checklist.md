# Eval Checklist — OpenSpec Propose

Tier C skill. Binary pass/fail criteria.

## Structural Criteria

1. **Change directory created**
   - `openspec/changes/<name>/` exists after execution.
   - Pass: Directory created with kebab-case name.
   - Fail: No directory created.

2. **All required artifacts generated**
   - `openspec status --json` shows all `applyRequires` artifacts as `done`.
   - Pass: All required artifacts present.
   - Fail: Missing artifacts.

3. **No context/rules leakage**
   - Generated artifact files do NOT contain raw `<context>` or `<rules>` XML blocks.
   - Pass: Clean artifacts.
   - Fail: Metadata leaked into content.

4. **Tasks file has checkboxes**
   - `tasks.md` contains at least one `- [ ]` item.
   - Pass: Actionable task list present.
   - Fail: No tasks or no checkboxes.

## Trigger Tests

**Should trigger:**
- "propose a new change for implementing the dashboard"
- "I want to build a new feature, create a proposal"
- "create a change proposal for refactoring the data pipeline"

**Should NOT trigger:**
- "implement the tasks from this change"
- "archive this change"
