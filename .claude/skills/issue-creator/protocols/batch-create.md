# Protocol: Batch Create Issues from Plan

## Purpose

Create multiple GitHub issues at once from a plan file's `<issues>` or
`<planned-issues>` block. This is the high-throughput path for plans that
define many tasks as separate trackable issues.

## Inputs

- Path to plan file (XML or MD)
- Optional: filter by task IDs, phase, or group
- Optional: dry-run mode (list without creating)

## Steps

### 1. Parse All Issue Directives

Read the plan file and extract ALL issue items. Build a creation manifest:

```
Manifest for: {plan_file}
  Total items: {N}
  Create: {list of issue stubs}
  Close:  {list of close directives}
  Skip:   {list of items referencing existing issues}
```

### 2. Deduplicate Against Open Issues

```bash
gh issue list --state open --limit 100 --json number,title
```

For each item in the manifest, check if a similar issue already exists.
Mark duplicates as SKIP with reason.

### 3. Present Manifest for Approval

Show the user the full manifest before creating anything:

```
Batch Issue Creation — {plan_file}
  Will CREATE {N} issues:
    1. [{P1}] {title_1}
    2. [{P2}] {title_2}
    ...
  Will CLOSE {M} issues:
    1. #{number} — {reason}
  Will SKIP {K} items (already exist):
    1. {title} → matches #{existing_number}

Proceed? (yes / dry-run only / select specific items)
```

Wait for user confirmation before creating.

### 4. Create Issues (Sequential)

Create issues one at a time (GitHub API rate limits). For each:

1. Compose body using SKILL.md template
2. Set `plan:` metadata to the source plan file
3. Set `relates_to:` to include the plan's parent issue if any
4. Create via `gh issue create`
5. Add to project board with correct priority
6. Record URL

Pause briefly between creations to avoid rate limiting:
```bash
sleep 1  # GitHub API courtesy
```

### 5. Handle Groups

If the plan uses `<github-issue-group>`:

1. Collect all tasks sharing the same group value
2. Create ONE issue per group
3. Title: `{plan_name}: {group_name}`
4. Body lists all grouped tasks with individual acceptance criteria

### 6. Execute Close Directives

For each `<close>` element:
```bash
gh issue close {number} --comment "$(cat <<'EOF'
Closed per plan: `{plan_file}`
Reason: {reason}
Note: {note}
EOF
)"
```

### 7. Report Summary Table

```
Batch Complete: {plan_file}
  Created: {N} issues
  Closed:  {M} issues
  Skipped: {K} items

| # | Action | Title | Issue | Priority |
|---|--------|-------|-------|----------|
| 1 | CREATED | {title} | #{number} | P1 |
| 2 | CREATED | {title} | #{number} | P2 |
| 3 | CLOSED  | #{old} | — | — |
| 4 | SKIPPED | {title} | #{existing} | — |
```

### 8. Update Plan File (Optional)

If the plan XML has `<planned-issues>` stubs without issue numbers,
offer to annotate them with the created issue numbers. This creates
a bidirectional link: plan references issues, issues reference plan.

**Only do this if the user agrees** — it modifies the plan file.

## Rate Limiting

- GitHub API: max ~30 requests/minute for authenticated users
- Space creations 2 seconds apart for batches >10
- If rate-limited (HTTP 429), wait 60 seconds and retry
