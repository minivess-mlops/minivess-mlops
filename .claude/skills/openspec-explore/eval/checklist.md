# Eval Checklist — OpenSpec Explore

Tier C skill. Binary pass/fail criteria.

## Behavioral Criteria

1. **No code written**
   - Explore mode NEVER writes implementation code (src/ or tests/).
   - Pass: Only read operations and OpenSpec artifacts created.
   - Fail: Any Write/Edit to src/ or test files.

2. **Codebase grounded**
   - At least one Read/Grep/Glob call to actual project files during exploration.
   - Pass: Explored real code, not just theorized.
   - Fail: Entire session was abstract without reading any files.

3. **Visual aids used**
   - ASCII diagrams or structured comparisons used when clarifying complex ideas.
   - Pass: At least one visualization offered.
   - Fail: Complex topic discussed without visual aids.

## Trigger Tests

**Should trigger:**
- "let's explore how the loss function architecture should work"
- "I want to think through the deployment strategy"
- "help me investigate this problem before I propose a change"

**Should NOT trigger:**
- "implement this feature"
- "create issues from this plan"
