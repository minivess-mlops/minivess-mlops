# Protocol: Create Issue from Plan File

## Purpose

Extract issue directives from an XML or MD plan file and create structured
GitHub issues with full cross-references back to the plan.

## Inputs

- Path to plan file (XML or MD in `docs/planning/`)
- Optional: specific task IDs to create issues for (default: all)
- Optional: priority override

## Steps

### 1. Read and Parse Plan

Read the plan file. Determine format:

**XML plans** — look for these tag patterns:
```xml
<!-- Direct issue list -->
<issues>
  <issue task="1.1" title="..." labels="..." />
</issues>

<!-- Issue template (plan-specific) -->
<github-issues>
  <issue-template>
    <title>{pattern with {placeholders}}</title>
    <labels>{comma-separated}</labels>
    <body-template>{CDATA markdown}</body-template>
  </issue-template>
  <planned-issues>
    <issue task="0.1" title="..." />
  </planned-issues>
</github-issues>

<!-- Task-level grouping -->
<task id="T0.1" name="...">
  <github-issue-group>fix-interflow-seam</github-issue-group>
</task>

<!-- Plan metadata linking to existing issue -->
<metadata>
  <issue>305</issue>
</metadata>

<!-- Issue close directives -->
<close number="100" reason="not-planned" note="..." />
```

**MD plans** — look for these section patterns:
```markdown
## Issues
- [ ] Title (P1, domain: cloud)

## GitHub Issues
| Task | Title | Priority |
```

### 2. Extract Issue Items

For each issue directive found:

| Field | XML Source | Default |
|-------|-----------|---------|
| title | `title` attribute or `<title>` child | `{plan_name}: Task {id} — {name}` |
| labels | `labels` attribute or `<labels>` child | `enhancement,{priority}` |
| task_ids | `task` attribute | — |
| group | `<github-issue-group>` value | — |
| close | `<close number="X">` | — |

### 3. Handle Issue Groups

When multiple tasks share the same `<github-issue-group>` value:
- Create ONE issue for the group
- Title: `{plan_name}: {group_name}`
- List all grouped tasks in the issue body under `## Grouped Tasks`
- Acceptance criteria: one checkbox per task in the group

### 4. Handle Plan-Specific Templates

If the plan has `<issue-template>` with `<body-template>`:
- Use the plan's template instead of the default
- Replace `{placeholders}` with actual values from the task
- Still prepend the `<!-- METADATA -->` block (the plan template doesn't have it)

### 5. Resolve Cross-References

For each issue to create:

```bash
# Find commits related to this plan
git log --oneline --all --grep="{plan_name}" -10

# Check if plan references a parent issue
# (from <metadata><issue>X</issue></metadata>)
```

Populate the Context section:
- **Plan**: Always link to the source plan file
- **Parent Issue**: If plan has `<metadata><issue>X</issue></metadata>`
- **Commits**: Any commits mentioning the plan name
- **Branch**: From plan XML `branch` attribute if present

### 6. Check for Already-Created Issues

Before creating, check if issues from this plan already exist:

```bash
gh issue list --search "{plan_name}" --state all --json number,title,state
```

Skip any issue whose title closely matches an existing one. Report skipped items.

### 7. Create Issues (Sequential)

For each issue item (not skipped, not a close directive):

1. Compose body using SKILL.md template + plan-specific overrides
2. Create via `gh issue create`
3. Add to project board
4. Collect URL

### 8. Handle Close Directives

For `<close>` elements:

```bash
gh issue close {number} --comment "Closed via plan: {plan_file}. Reason: {reason}. {note}"
```

### 9. Report Summary

Print table of all actions taken:

```
Plan: {plan_file}
Issues created: {N}
Issues skipped (already exist): {M}
Issues closed: {K}

| # | Action | Title | URL |
|---|--------|-------|-----|
| 1 | CREATED | {title} | {url} |
| 2 | SKIPPED | {title} | {existing_url} |
| 3 | CLOSED  | #{number} | {reason} |
```

## Edge Cases

- **Plan has no issue directives**: Report "No issue directives found in {plan_file}"
  and ask user if they want to create issues from the plan's task list instead.
- **Plan references issues that don't exist**: Create them (they may have been planned
  but never filed).
- **Dry run**: If user says "show me what issues would be created", list them without
  creating. Wait for confirmation.
