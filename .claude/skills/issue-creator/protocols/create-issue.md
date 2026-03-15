# Protocol: Create Issue (Standard)

## Purpose

Create a single structured GitHub issue from a user description or observed need.

## Inputs

- User description of what to track (title idea, context, priority)
- Optional: related files, commits, plan references

## Steps

### 1. Gather Context

Read the current repo state to populate cross-references:

```bash
# Recent commits for cross-referencing
git log --oneline -15

# Open issues for deduplication and relates_to
gh issue list --state open --limit 20 --json number,title,labels

# Current branch
git branch --show-current
```

### 2. Determine Domain

Read `knowledge-graph/navigator.yaml` and use the `keywords:` section to match
the issue's topic to a domain. If ambiguous, prefer the domain whose `covers:`
list best matches.

### 3. Determine Type and Priority

| Type | Signal |
|------|--------|
| feature | "add", "implement", "create", "new" |
| bugfix | "fix", "broken", "error", "failing" |
| refactor | "clean up", "simplify", "reorganize" |
| research | "explore", "investigate", "spike", "evaluate" |
| debt | "tech debt", "deprecated", "workaround", "hack" |
| docs | "document", "README", "guide" |

Priority: Ask user if not specified. Use these heuristics:
- P0: Blocks other work, production broken, data loss risk
- P1: Should do soon, important but not blocking
- P2: Nice to have, improvement
- P3: Future consideration, exploratory

### 4. Check for Duplicates

```bash
gh issue list --search "{key terms from title}" --state open --json number,title
```

If a similar issue exists, report it to the user and ask whether to:
- Skip (it's a duplicate)
- Create anyway (different scope)
- Update the existing issue instead

### 5. Find Cross-References

Search for related artifacts:

```bash
# Related plan files
# Use Glob tool: docs/planning/*{keyword}*

# Related commits
git log --oneline --grep="{keyword}" -5

# Related code
# Use Grep tool for relevant symbols/modules
```

### 6. Compose Issue Body

Use the template from SKILL.md. Fill in ALL fields:

- `<!-- METADATA -->` block with all required fields
- `## Summary` — one clear paragraph
- `## Context` — cross-references found in step 5
- `## Acceptance Criteria` — measurable items, always include TDD
- `## References` — hyperlinked citations if applicable

### 7. Select Labels

Map priority + domain + type to GitHub labels:

```
Priority: P0-critical | P1-high | P2-medium | (no label for P3)
Type:     enhancement (feature/research) | bug (bugfix)
Domain:   models | training | monitoring | etc. (from navigator)
```

### 8. Create Issue

```bash
gh issue create --title "{title}" \
  --label "{label1},{label2},{label3}" \
  --body "$(cat <<'EOF'
{composed body}
EOF
)"
```

### 9. Add to Project Board

Follow post-creation steps from SKILL.md.

### 10. Report

Print:
```
Created: {issue_url}
  Title:    {title}
  Priority: {P0|P1|P2|P3}
  Domain:   {domain}
  Type:     {type}
  Labels:   {labels}
```

## Output

The created issue URL and a one-line summary.
