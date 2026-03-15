# Template: Tech Debt Issue

Use for refactoring, cleanup, deprecated code removal, and architectural
improvements that don't add user-visible features.

## Title Pattern

```
chore({scope}): {what to clean up}
```

Examples:
- `chore(config): consolidate duplicate Dynaconf TOML entries`
- `chore(tests): fix 12 silently-skipping tests from missing extras`
- `chore(docker): rebuild stale flow images after base change`

## Labels

```
enhancement, {priority_label}, {domain_label}
```

(Note: GitHub doesn't have a "debt" label by default. Use `enhancement` +
domain label. The `type: debt` in METADATA distinguishes it from features.)

## Body

```markdown
<!-- METADATA
priority: {P1|P2|P3}
domain: {domain}
type: debt
plan:
prd_decisions: []
relates_to: [{#issues}]
blocked_by: []
status: open
-->

## Summary

{What technical debt exists and why it should be addressed. Focus on the
concrete risk or cost of NOT fixing it: "X will cause Y if left unfixed."}

## Current State

{Describe what's wrong today:
- Where the debt lives (file paths, line numbers)
- How it manifests (warnings, slowness, fragility, confusion)
- How long it's been this way}

## Desired State

{What "fixed" looks like:
- Specific files/patterns to change
- What behavior changes
- What stays the same}

## Context

- **Origin**: {How this debt was introduced — commit, PR, or "always been this way"}
- **Discovered**: {When/how it was found}
- **Impact**: {Who/what is affected — tests, builds, devex, performance}
- **Files**: [`{file1}`]({path}), [`{file2}`]({path})

## Acceptance Criteria

- [ ] {Debt is resolved: specific measurable state}
- [ ] No regressions in `make test-staging`
- [ ] Pre-commit hooks pass
- [ ] No new warnings introduced

## References

- {Link to CLAUDE.md rule if this violates one}
- {Link to metalearning doc if this was a known antipattern}
```
