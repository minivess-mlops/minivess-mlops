# Protocol: Create Issue from Observed Failure

## Purpose

When a test failure, import error, build error, or runtime warning is observed
during a session, create a structured GitHub issue to track it. This enforces
CLAUDE.md Rule #20 (Zero Tolerance for Observed Failures).

## When to Use

- Test failure that can't be fixed in <5 minutes
- Import error from a missing or incompatible dependency
- Build failure (Docker, pre-commit, mypy)
- Runtime warning that should be suppressed or fixed
- FORCE_STOP / STUCK outcome from the TDD skill

## Inputs

- Error output (test failure traceback, build log, warning text)
- File(s) where the failure occurs
- What was being done when the failure was observed
- Whether this is related to current work or pre-existing

## Steps

### 1. Capture Failure Details

Record from the terminal output or tool result:
- **Error type**: pytest failure, ImportError, mypy error, ruff error, Docker build, runtime
- **Error message**: Exact message (first 5 lines of traceback)
- **File + line**: Where the failure occurs
- **Test name**: If a test failure, the full test ID

### 2. Determine Root Cause Category

| Category | Signal | Priority |
|----------|--------|----------|
| Missing dependency | ImportError, ModuleNotFoundError | P1 |
| Type error | mypy error, TypeError at runtime | P1 |
| Logic bug | AssertionError in test, wrong output | P1 |
| Config error | KeyError, missing env var, bad YAML | P1 |
| Deprecation | DeprecationWarning, FutureWarning | P2 |
| Flaky test | Passes sometimes, fails sometimes | P2 |
| Performance | Timeout, OOM, slow test | P2 |

### 3. Check for Existing Issue

```bash
gh issue list --search "{error_type} {module_name}" --state open --json number,title
```

If an existing issue covers this failure, add a comment instead of creating a new one:
```bash
gh issue comment {number} --body "Also observed on $(date +%Y-%m-%d) during {context}. Traceback: ..."
```

### 4. Find Related Context

```bash
# Recent commits to the failing file
git log --oneline -5 -- {failing_file}

# Who last touched this code
git log --oneline -1 -- {failing_file}

# Any plan that covers this area
# Use Glob: docs/planning/*{module_keyword}*
```

### 5. Compose Issue Body

Use the bugfix template with failure-specific additions:

```markdown
<!-- METADATA
priority: {P1|P2}
domain: {from navigator}
type: bugfix
plan:
prd_decisions: []
relates_to: []
blocked_by: []
status: open
-->

## Summary

{Error type} in `{file}:{line}` — {one sentence description of what's broken}.
Observed during {what was being done}.

## Reproduction

```bash
{exact command to reproduce — e.g., uv run pytest tests/v2/unit/test_foo.py::test_bar -x}
```

## Error Output

```
{First 20 lines of traceback or error output}
```

## Context

- **Failing file**: [`{file}`]({permalink})
- **Last modified**: `{commit_sha}` by {author}
- **Observed during**: {what triggered it — e.g., "running make test-staging"}
- **Commits**: `{related_shas}`

## Root Cause Hypothesis

{Best guess at why this is failing. If unknown, say "Needs investigation."}

## Acceptance Criteria

- [ ] Error no longer reproduces with `{reproduction_command}`
- [ ] No regressions in `make test-staging`
- [ ] Pre-commit hooks pass

## References

{Any relevant docs, issues, or error documentation}
```

### 6. Create Issue

```bash
gh issue create --title "fix({scope}): {short description of failure}" \
  --label "bug,{priority_label},{domain_label}" \
  --body "$(cat <<'EOF'
{composed body}
EOF
)"
```

### 7. Report

```
Filed: {issue_url}
  Failure: {error_type} in {file}:{line}
  Priority: {priority}
  Rule #20 compliance: Issue created for observed failure
```

## FORCE_STOP Integration

When the TDD skill hits FORCE_STOP on a task:

1. Read the FORCE_STOP report from convergence.md output
2. Use the `Root cause hypothesis` and `Suggested next steps` as issue content
3. Title: `fix({scope}): task {id} STUCK — {root cause hypothesis}`
4. Add label `stuck` in addition to standard labels
5. Include iteration count and what was attempted in Implementation Notes
