# Template: Bugfix Issue

Use for bug fixes, error corrections, and failure resolution.

## Title Pattern

```
fix({scope}): {what was broken}
```

Examples:
- `fix(train): val_interval sentinel bypassed on epoch 0`
- `fix(data): mv02 outlier OOM with Spacingd at 4.97 um`
- `fix(deploy): stale flow images after base rebuild`

## Labels

```
bug, {priority_label}, {domain_label}
```

## Body

```markdown
<!-- METADATA
priority: {P0|P1|P2}
domain: {domain}
type: bugfix
plan:
prd_decisions: []
relates_to: [{#issues}]
blocked_by: []
status: open
-->

## Summary

{What is broken and what impact it has. One paragraph. Include: what fails,
when it fails, and what the user experiences.}

## Reproduction

```bash
{Exact command to reproduce the failure}
```

**Expected**: {what should happen}
**Actual**: {what happens instead}

## Error Output

```
{First 20 lines of traceback, error message, or unexpected output.
Truncate long outputs with "..." — full output in linked artifacts.}
```

## Context

- **Failing file**: [`{file}:{line}`]({permalink})
- **Last modified**: `{commit_sha}` — {date}
- **Observed during**: {what triggered it}
- **Commits**: `{related_shas}`
- **Environment**: {local/Docker/CI/cloud — if relevant}

## Root Cause Hypothesis

{Best guess at why this is failing. If investigated, describe findings.
If unknown, write "Needs investigation." Do not fabricate.}

## Acceptance Criteria

- [ ] Error no longer reproduces: `{reproduction_command}`
- [ ] Regression test added
- [ ] No regressions in `make test-staging`
- [ ] Pre-commit hooks pass

## References

- {Link to related error docs, upstream issues, or MONAI/PyTorch bugs if applicable}
```
