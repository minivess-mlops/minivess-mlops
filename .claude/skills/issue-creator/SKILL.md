---
name: issue-creator
description: >
  Create structured GitHub issues with progressive disclosure metadata.
  Triggers on: "create issue", "file issue", "open issues from plan",
  "create issues from this XML", "retroactive issues", "track this as issue".
  Also triggers when executing TDD plans that contain <issues>, <planned-issues>,
  or <github-issue-group> XML tags.
model: sonnet
allowed-tools: Read, Grep, Glob, Bash(gh:*), Bash(git:*)
argument-hint: [plan-file-or-description]
---

# Issue Creator Skill

Create structurally consistent GitHub issues with progressive disclosure —
compact YAML metadata for LLM consumption, human-readable summaries, and
cross-references to commits, plans, reports, and code permalinks.

> "Format syntax does NOT significantly affect accuracy. Information architecture
> — how content is partitioned and navigated — is the dominant variable."
> — [McMillan (2026). arXiv:2602.05447](https://arxiv.org/abs/2602.05447)

## When to Activate

This skill activates in THREE scenarios:

### Scenario 1: Explicit Plan Execution
User asks to "open issues from this plan" or "create issues from this XML/MD".
- Read the plan file (XML or MD) from `docs/planning/`
- Extract `<issues>`, `<planned-issues>`, `<github-issue-group>` XML tags
- Or extract `## Issues` / `## Acceptance Criteria` sections from MD plans
- Create one issue per extracted item using the appropriate template

### Scenario 2: TDD Skill Integration
During self-learning-iterative-coder execution, when the plan contains:
- `<issues>` block with `<issue task="..." title="..." labels="..." />`
- `<planned-issues>` with issue stubs
- `<github-issue-group>` on tasks (group multiple tasks into one issue)
- `<issue>` in `<metadata>` (link plan to existing issue)
- FORCE_STOP / STUCK outcomes that need issue tracking

When the TDD skill finishes a plan (or hits FORCE_STOP), check if the plan
had issue directives that haven't been created yet. If so, offer to batch-create.

### Scenario 3: Retroactive Issue Creation
User asks to "create issues for past work" or "retroactive issues".
- Scan `git log` for commits without associated issues
- Scan `docs/planning/` for plans that reference issues not yet created
- Scan closed PRs for work that should have tracking issues
- Create issues with `status: completed` metadata linking to the existing commits/PRs

## Repository Context

**GitHub Project**: `minivess-mlops/minivess-mlops` (Project #1)
- Project ID: `PVT_kwDOCPpnGc4AYSAM`
- Priority field ID: `PVTSSF_lADOCPpnGc4AYSAMzgPhgsk`
  - P0: `c128192a` | P1: `b419b06f` | P2: `5bb35602` | P3: `4b1f5dd3`

**Domain routing**: Read `knowledge-graph/navigator.yaml` to assign `domain:` field.
Use the `keywords:` section to map issue content to domains.

**Existing labels**: P0-critical, P1-high, P2-medium, P3-low, enhancement, models,
monitoring, training, metrics, uncertainty, compliance, annotation, data-quality,
validation, observability, research, ci-cd, documentation.

## Issue Body Template (MANDATORY)

Every issue MUST use this exact structure. The YAML metadata block inside an HTML
comment is invisible in GitHub's rendered view but parseable by agents.

```markdown
<!-- METADATA
priority: {P0|P1|P2|P3}
domain: {domain from navigator.yaml}
type: {feature|bugfix|refactor|research|debt|docs}
plan: {path/to/plan.xml or path/to/plan.md — omit if no plan}
prd_decisions: [{decision_node_ids} — omit if none]
relates_to: [{#issue_numbers}]
blocked_by: [{#issue_numbers}]
status: {open|completed — use completed for retroactive issues}
-->

## Summary

{One paragraph: what this issue delivers and why it matters. Written for a
developer who has never seen this repo. Lead with the user-visible outcome,
not the implementation detail.}

## Context

- **Plan**: [`{plan_filename}`]({relative_path_to_plan})
- **Report**: [`{report_filename}`]({relative_path}) — if a research report exists
- **Commits**: `{sha1}`, `{sha2}` — related commits (use short 7-char SHAs)
- **PRD**: `{decision_node.yaml}` — if this implements a PRD decision
- **Branch**: `{branch_name}` — if work is in progress

## Acceptance Criteria

- [ ] {Measurable criterion — "X works" not "implement X"}
- [ ] Unit tests (TDD mandatory per CLAUDE.md)
- [ ] Pre-commit hooks pass

## Implementation Notes

{Optional: key constraints, gotchas, architectural decisions the implementor
needs to know. Delete this section if empty.}

## References

- [{Author (Year). "Title." *Journal*.}]({URL})
```

## Protocols

Select the appropriate protocol based on the scenario:

| Scenario | Protocol | When |
|----------|----------|------|
| Single issue from description | [create-issue.md](protocols/create-issue.md) | User describes what to track |
| Issues from plan file | [create-from-plan.md](protocols/create-from-plan.md) | User points to XML/MD plan |
| Issue from failure | [create-from-failure.md](protocols/create-from-failure.md) | Test failure / build error observed |
| Batch from plan | [batch-create.md](protocols/batch-create.md) | Plan has `<issues>` block with multiple items |
| Retroactive issues | [retroactive-create.md](protocols/retroactive-create.md) | Past work needs tracking issues |

## Templates

Select template based on `type:` field:

| Type | Template | For |
|------|----------|-----|
| feature | [feature.md](templates/feature.md) | New capability or enhancement |
| bugfix | [bugfix.md](templates/bugfix.md) | Bug fix with reproduction steps |
| research | [research.md](templates/research.md) | Exploration spike, literature review |
| debt | [debt.md](templates/debt.md) | Tech debt, refactoring, cleanup |

## Validation Rules (Hard Gates)

Before creating ANY issue, verify:

1. **YAML metadata present** — `<!-- METADATA` block with all required fields
2. **Domain valid** — `domain:` value exists in `knowledge-graph/navigator.yaml`
3. **Priority label matches** — `priority:` in metadata matches the GitHub label
4. **Citations hyperlinked** — Every reference has a clickable URL (CLAUDE.md rule)
5. **No duplicate** — `gh issue list --search "{title keywords}"` returns no close match
6. **Plan cross-ref valid** — If `plan:` is set, the file exists at that path
7. **Commit SHAs valid** — If commits listed, `git log --oneline {sha}` resolves

## Post-Creation Steps

After `gh issue create` succeeds:

1. **Add to project board**:
   ```bash
   ITEM_ID=$(gh project item-add 1 --owner minivess-mlops --url {issue_url} \
     --format json | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
   gh project item-edit --project-id PVT_kwDOCPpnGc4AYSAM --id "$ITEM_ID" \
     --field-id PVTSSF_lADOCPpnGc4AYSAMzgPhgsk \
     --single-select-option-id {priority_option_id}
   ```

2. **Report to user**: Print issue URL and metadata summary.

3. **If batch**: Collect all created issue URLs and print summary table.

## XML Plan Tag Reference

When parsing XML plans, recognize these tag patterns:

| XML Tag | Meaning | Action |
|---------|---------|--------|
| `<issues>` | Block of issue directives | Parse all child `<issue>` elements |
| `<issue task="X" title="Y" labels="Z" />` | Create issue for task X | Use title and labels directly |
| `<issue>` in `<metadata>` | Plan links to existing issue | Cross-reference, don't create |
| `<planned-issues>` | Stubs for future issues | Create with plan cross-ref |
| `<github-issue-group>` on `<task>` | Group tasks into one issue | Merge tasks sharing same group value |
| `<issue-template>` | Custom template for this plan | Use instead of default template |
| `<close number="X" reason="Y" />` | Close existing issue | Run `gh issue close X` with comment |
| `<prerequisite-issues>` | Existing issues that block this plan | Add to `blocked_by:` in new issues |
