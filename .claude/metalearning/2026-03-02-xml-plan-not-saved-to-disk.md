# Metalearning Report: XML Plan Not Saved to Disk

## Date: 2026-03-02

## Severity: HIGH — User had to explicitly re-ask for a file they already requested

---

## What Happened

The user's original prompt (from a previous session) explicitly asked Claude to save
an XML implementation plan to `docs/planning/sam3-implementation-plan.xml`. The exact
instruction was:

> "Create an implementation plan in XML format at `docs/planning/sam3-implementation-plan.xml`"

Instead of writing the plan to disk using the Write tool, Claude wrote it to the
conversation plan file (`/home/petteri/.claude/plans/soft-doodling-wolf.md`) — a
Markdown file in a temporary location that is NOT the path the user requested.

The user had to ask again in a later session: "Did you save this .xml plan to disk?"
— at which point it was discovered that no XML file existed at the requested path.

### Timeline

1. **User's original prompt**: Explicitly specified `docs/planning/sam3-implementation-plan.xml`
2. **Previous session**: Generated the plan but wrote it to `.claude/plans/` as Markdown,
   not to the user-specified path as XML
3. **This session**: Continued implementing from the wrong plan (which also had the
   SAM2/SAM3 confusion), never checked whether the XML file existed
4. **User asked**: "Did you save this .xml plan to disk?" — discovered it didn't exist
5. **Fix**: Created the correct SAM3 XML plan at `docs/planning/sam3-implementation-plan.xml`

### Root Cause Analysis

| # | Root Cause | Should Have Done |
|---|-----------|------------------|
| 1 | **Ignored explicit file path**: User said "save to `docs/planning/sam3-implementation-plan.xml`", Claude wrote to `.claude/plans/` instead | Write the file to the exact path the user specified using the Write tool |
| 2 | **Plan mode artifacts ≠ user artifacts**: Claude's plan mode saves to `.claude/plans/`, but the user asked for a permanent file in `docs/planning/`. These are different things. | Distinguish between internal planning artifacts and user-requested deliverables |
| 3 | **No verification**: After plan mode, Claude never verified whether the requested file existed at the requested path | Always verify that requested artifacts were actually created |
| 4 | **Cross-session context loss**: The explicit file path instruction was in a previous session. This session received only the plan content, not the user's instruction about where to save it. | When resuming work, re-read the user's original prompt to check for specific deliverables |

## Corrective Actions

1. **When the user specifies an output path, use the Write tool to create that file**
   — plan mode's internal files are NOT the same as user-requested deliverables
2. **After completing a plan, verify** that all requested artifacts exist at their
   specified paths
3. **Added Rule 14 to CLAUDE.md**: "Write Requested Artifacts to Disk — When the user
   explicitly asks for a file at a specific path, ALWAYS write it using the Write tool"

## Connection to Broader Session Failures

This failure is part of a pattern observed throughout this session:

1. **SAM2/SAM3 confusion**: Plan generated with wrong model backbone
2. **XML plan not saved**: File written to wrong location
3. **Confabulation instead of verification**: Made up explanations instead of checking
4. **Self-reflection printed but not persisted**: Terminal output without file writes

All four failures share a common root: **prioritizing speed over accuracy, and
assumptions over verification.** Claude operated in "execution mode" (follow the plan,
produce output fast) instead of "verification mode" (check the plan matches intent,
verify outputs match requests, admit uncertainty).

## Lesson

> **The Write tool exists for a reason.** When a user says "save X to path Y",
> that is a direct instruction to use `Write(file_path=Y, content=X)`. Plan mode
> internal files, conversation context, and terminal output are NOT durable artifacts
> that fulfill the user's request. Only files on disk at the specified path count.
