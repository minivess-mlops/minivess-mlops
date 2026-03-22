# Superficial Audit: file-exists != file-works

**Date**: 2026-03-22
**Category**: audit-quality
**Severity**: high

## What Happened

When generating health scores for the plan archive navigator, the initial
implementation only checked `Path.exists()` to determine if a planning doc
was healthy. This is the "file-exists-not-works" anti-pattern:

1. A file can exist but be empty (0 bytes)
2. A file can exist but contain only a header with no content
3. A file can exist but reference other files that don't exist
4. A file can exist but describe a plan that was never implemented

## Why This Is Wrong

A health score based on file existence is meaningless. It tells you that
`git` tracked a file, not that the plan was useful, implemented, or even
complete. This is the same category of error as checking `import foo`
without checking `foo.bar()` actually works.

## Correct Approach

Code-verified health scores must check:
1. File exists AND is readable (not binary/corrupt)
2. File has substantive content (> 50 chars after stripping)
3. Implementation status from KG domain YAML (`status: implemented` vs `planned`)
4. Cross-references resolve (referenced files exist)

## Prevention Rule

**NEVER report a health score based only on file existence.**
Always verify content quality programmatically. The `_check_file_actually_works()`
function in `generate_audit_report.py` implements the correct check.

## See Also

- `.claude/metalearning/2026-03-22-repeated-superficial-audit-in-same-session.md`
- `scripts/generate_audit_report.py`
