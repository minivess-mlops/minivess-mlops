# Protocol: Retroactive Issue Creation

## Purpose

Create GitHub issues for past work that was done without issue tracking.
This recovers the audit trail by linking completed work (commits, PRs,
plan executions) to structured issues marked `status: completed`.

## When to Use

- User says "create issues for past work on X"
- User says "we forgot to file issues for that PR"
- Audit reveals plan files with no corresponding issues
- Sprint review finds untracked completed work

## Inputs

- Topic or scope of past work (branch name, PR number, plan file, date range)
- Optional: specific commits to link

## Steps

### 1. Discover Past Work

Based on user input, identify the body of work:

**From a branch:**
```bash
git log main..{branch} --oneline --no-merges
```

**From a PR:**
```bash
gh pr view {number} --json title,body,commits,files,mergedAt
```

**From a plan file:**
```bash
# Read the plan, extract task IDs
# Check git log for commits referencing the plan
git log --oneline --all --grep="{plan_name}" -20
```

**From a date range:**
```bash
git log --oneline --after="{start_date}" --before="{end_date}" --no-merges
```

### 2. Cluster Work into Issue-Sized Units

Group related commits into logical issue units:
- Commits sharing a conventional commit scope → one issue
- Commits referencing the same plan task → one issue
- Commits on the same file/module → candidate for grouping

Present the clustering to the user:
```
Found {N} commits, clustered into {M} potential issues:

  Issue 1: "{scope}: {description}" ({K} commits)
    - {sha1} {message}
    - {sha2} {message}

  Issue 2: "{scope}: {description}" ({K} commits)
    ...

Adjust clustering? (yes/no/merge 1+2/split 3)
```

### 3. For Each Issue Unit

#### 3a. Find Cross-References

```bash
# Plan file that drove this work
# Use Glob: docs/planning/*{scope}*

# PR that merged this work
gh pr list --search "{scope}" --state merged --json number,title,url

# Related open issues
gh issue list --search "{scope}" --state all --json number,title,state
```

#### 3b. Check for Existing Issues

If an issue already exists for this work (open or closed), skip or
offer to enrich it with missing cross-references instead.

#### 3c. Compose Retroactive Issue Body

Key difference from standard issues: `status: completed` in metadata,
and the body documents what WAS done rather than what SHOULD be done.

```markdown
<!-- METADATA
priority: {P1|P2}
domain: {domain}
type: {feature|bugfix|refactor}
plan: {plan_file if found}
prd_decisions: []
relates_to: [{#related_issues}]
blocked_by: []
status: completed
-->

## Summary

{What was implemented and why. Past tense. One paragraph.}

## Context

- **Plan**: [`{plan_file}`]({path}) — if exists
- **PR**: #{pr_number} — if merged via PR
- **Branch**: `{branch_name}`
- **Commits**: `{sha1}`, `{sha2}`, ...
- **Date**: {completion_date}

## What Was Done

- {Bullet 1: concrete deliverable}
- {Bullet 2: concrete deliverable}
- {Bullet 3: concrete deliverable}

## Test Coverage

- {N} tests added/modified
- Key test files: `{test_file_1}`, `{test_file_2}`

## References

- [{citation if applicable}]({url})
```

### 4. Create Issues

Create with appropriate labels. For retroactive issues, always:
- Add label matching the work type
- Set priority based on the importance of the completed work
- Close immediately after creation (work is already done)

```bash
# Create
ISSUE_URL=$(gh issue create --title "{title}" \
  --label "{labels}" \
  --body "$(cat <<'EOF'
{body}
EOF
)" | tail -1)

# Close immediately (work is done)
gh issue close $(echo $ISSUE_URL | grep -oP '\d+$') \
  --comment "Retroactive issue — work completed in commits: {sha_list}"
```

### 5. Report Summary

```
Retroactive Issue Creation Complete
  Scope: {description of past work}
  Period: {date range}
  Issues created: {N} (all closed)

| # | Title | Issue | Commits | PR |
|---|-------|-------|---------|-----|
| 1 | {title} | #{number} | {shas} | #{pr} |
| 2 | {title} | #{number} | {shas} | — |
```

## Notes

- Retroactive issues are always created AND closed in the same operation
- They exist purely for audit trail and project history
- The `status: completed` metadata distinguishes them from regular issues
- Future agents analyzing the issue history can filter by this field
