# Issue Creator Skill — Design Plan

**Status**: PLAN (not yet implemented)
**Date**: 2026-03-14
**Author**: Claude Code + Petteri

## 1. Problem Statement

GitHub issues in this repo are created ad-hoc with inconsistent structure. When Claude
(or a new developer) needs to reconstruct the audit trail — what was planned, what was
implemented, why decisions changed — the issues are unstructured free text that requires
expensive context loading to parse.

### Key insight from McMillan (Feb 2026)

[McMillan (2026). "Structured Context Engineering for File-Native Agentic Systems."
*arXiv:2602.05447*.](https://arxiv.org/abs/2602.05447)

> Format syntax (Markdown vs YAML vs JSON) does NOT significantly affect aggregate
> accuracy (chi-squared=2.45, p=0.484). **Information architecture** — how content is
> partitioned, navigated, and structured — is the dominant variable after model capability.

This means: the issue body can stay as Markdown (human-readable, GitHub-native rendering)
but must be **architecturally structured** — consistent sections, machine-parseable metadata
block, progressive disclosure via cross-references.

## 2. Design Principles

### 2.1 Progressive Disclosure (4 Levels)

| Level | What | Where | Loaded When |
|-------|------|-------|-------------|
| **L0** | Title + labels + priority | GitHub issue list view | Always (free) |
| **L1** | Structured issue body: YAML metadata + human summary | Issue description | On issue open (~200 tokens) |
| **L2** | Cross-referenced artifacts (commit SHAs, plan files, report docs, code permalinks) | Links in issue body | On demand (agent follows links) |
| **L3** | Full context (research reports, metalearning docs, plan XMLs) | Referenced files in repo | Only when deep-diving |

### 2.2 Issue Body Structure

```markdown
<!-- METADATA (machine-readable, human-skimmable) -->
```yaml
priority: P1
domain: cloud          # maps to knowledge-graph/navigator.yaml domains
type: feature          # feature | bugfix | refactor | research | debt | docs
prd_decisions:         # PRD decision nodes this implements
  - cloud_compute_broker
relates_to: [#680, #681]
blocked_by: []
```
<!-- /METADATA -->

## Summary
One paragraph: what and why. Written for a developer who has never seen this repo.

## Context
- **Plan**: [`docs/planning/cloud-architecture-decisions-2026-03-14.md`](permalink)
- **Research**: [`docs/research/skypilot-multi-cloud-report.md`](permalink)
- **Commits**: `6368290`, `3adeef5`
- **PRD**: `knowledge-graph/decisions/cloud_compute_broker.yaml`

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Unit tests (TDD mandatory per CLAUDE.md)

## Implementation Notes
Optional: key technical decisions, constraints, gotchas for the implementor.

## References
- [Author et al. (Year). "Title." *Journal*.](URL)
```

### 2.3 Why This Structure Works

1. **L0 (labels)**: `gh issue list --label P1-high,cloud` → instant filtering
2. **L1 (YAML block)**: Claude can `json.loads(yaml.safe_load(...))` the metadata without
   parsing the entire issue body. Domain field routes to knowledge-graph navigator.
3. **L2 (cross-refs)**: Permalinks to specific commits, files, and line ranges. Agent
   fetches only what it needs — no grep tax.
4. **L3 (deep context)**: Research reports and plans live in `docs/` — only loaded when
   the agent is actually implementing the issue.

### 2.4 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Metadata format | YAML in HTML comment | GitHub renders it invisibly; agents can parse it; humans see clean issue |
| Cross-ref format | Relative repo paths | Permalinks survive branch changes; `gh api` can resolve |
| Domain taxonomy | Reuse `knowledge-graph/navigator.yaml` | Single source of truth for domain routing |
| Issue types | 6 types (feature/bugfix/refactor/research/debt/docs) | Covers all work categories without over-taxonomizing |
| Priority | Reuse existing P0/P1/P2/P3 labels | Already in use via planning-backlog skill |

## 3. Skill Architecture (Skills 2.0)

### 3.1 Frontmatter

```yaml
---
name: issue-creator
description: >
  Create structured GitHub issues with progressive disclosure.
  Use when creating issues, filing bugs, or tracking implementation work.
  Triggers on: "create issue", "file issue", "open issue", "track this".
model: sonnet
context: fork
allowed-tools: Read, Grep, Glob, Bash(gh:*), Bash(git:*)
argument-hint: [title-or-description]
---
```

Key Skills 2.0 features used:
- **`context: fork`** — Issue creation is a side-effect workflow; isolate it so the
  main conversation context isn't polluted with issue template rendering.
- **`model: sonnet`** — Issue creation is structured output, not deep reasoning.
  Sonnet is sufficient and faster.
- **`allowed-tools`** — Restrict to read-only repo access + GitHub CLI. No file writes,
  no code execution. The skill creates issues, not code.

### 3.2 Directory Structure

```
.claude/skills/issue-creator/
├── SKILL.md                    # Main skill (frontmatter + instructions)
├── SKILL-PLAN.md               # This file (design rationale)
├── protocols/
│   ├── create-issue.md         # Standard issue creation protocol
│   ├── create-from-plan.md     # Issue from an existing plan/report
│   ├── create-from-failure.md  # Issue from an observed test/build failure
│   ├── batch-create.md         # Multiple issues from a plan file
│   └── link-artifacts.md       # Add cross-references to existing issue
├── templates/
│   ├── feature.md              # Feature issue template
│   ├── bugfix.md               # Bug report template
│   ├── research.md             # Research exploration template
│   └── debt.md                 # Tech debt template
└── scripts/
    └── validate-issue.sh       # Lint: check YAML metadata, required sections
```

### 3.3 Dynamic Context Injection

The SKILL.md will use backtick-prefixed shell commands to inject live repo state:

```markdown
## Current repo state
`!git log --oneline -10`
`!gh issue list --state open --limit 5 --json number,title,labels`
`!cat knowledge-graph/navigator.yaml | head -50`
```

This gives the skill awareness of recent commits (for cross-referencing) and open
issues (for deduplication and `relates_to` linking) without manual context gathering.

### 3.4 Validation Hook

```yaml
hooks:
  post-tool-use:
    - matcher: "Bash"
      command: sh
      script: |
        # Validate that gh issue create includes YAML metadata block
        if echo "$TOOL_INPUT" | grep -q "gh issue create"; then
          if ! echo "$TOOL_INPUT" | grep -q "priority:"; then
            echo "ERROR: Issue body missing YAML metadata block" >&2
            exit 1
          fi
        fi
```

### 3.5 Workflow

1. **User invokes**: `/create-issue <description>` or Claude auto-triggers on
   "create an issue for...", "file a bug about...", "track this as..."
2. **Context injection**: Recent commits, open issues, navigator domains loaded
3. **Domain routing**: Skill reads `knowledge-graph/navigator.yaml` to assign domain
4. **Template selection**: Based on issue type (feature/bugfix/research/debt)
5. **Cross-reference gathering**:
   - Find related commits via `git log --grep`
   - Find related plans/reports via `Glob` in `docs/planning/`
   - Find related code via `Grep` for relevant symbols
   - Find related issues via `gh issue list --search`
6. **Issue creation**: `gh issue create` with structured body
7. **Project board**: Add to GitHub project with priority field
8. **Validation**: Hook checks YAML metadata presence and required sections

## 4. Issue Taxonomy (Label System)

Extend existing labels from `planning-backlog/SKILL.md`:

| Label | Existing? | Purpose |
|-------|-----------|---------|
| `P0-critical` | Yes | Must do next |
| `P1-high` | Yes | Should do soon |
| `P2-medium` | Yes | Nice to have |
| `P3-low` | Yes | At some point maybe |
| `type:feature` | NEW | New capability |
| `type:bugfix` | NEW | Bug fix |
| `type:refactor` | NEW | Code improvement |
| `type:research` | NEW | Exploration / spike |
| `type:debt` | NEW | Technical debt |
| `type:docs` | NEW | Documentation |
| Domain labels | Existing | `cloud`, `models`, `training`, etc. |

## 5. Reviewer Agent Integration

### 5.1 Issue Quality Gate (Post-Creation)

After issue creation, a reviewer subagent validates:

1. **Structure completeness**: All required sections present (Summary, Context, Acceptance Criteria)
2. **YAML metadata validity**: All fields populated, domain exists in navigator
3. **Cross-reference integrity**: Linked commits exist, linked files exist, linked issues exist
4. **Citation format**: All references have clickable hyperlinks (CLAUDE.md rule)
5. **Deduplication check**: No open issue with >80% title similarity
6. **Priority consistency**: P0 issues have `blocked_by: []` (nothing blocking critical work)

### 5.2 Audit Trail Agent (Periodic)

A separate agent (or protocol) that:
- Scans all open issues for structural compliance
- Reports issues missing YAML metadata or required sections
- Identifies stale issues (no activity in 14+ days)
- Generates a summary report for sprint planning

## 6. Implementation Order

| Phase | What | Depends On |
|-------|------|-----------|
| **Phase 1** | SKILL.md + create-issue protocol + feature template | Nothing |
| **Phase 2** | Remaining templates (bugfix, research, debt) | Phase 1 |
| **Phase 3** | Dynamic context injection (`!` commands) | Phase 1 |
| **Phase 4** | Validation hook + validate-issue.sh | Phase 1 |
| **Phase 5** | batch-create protocol (from plan XML/MD) | Phase 1 |
| **Phase 6** | Reviewer agent integration | Phase 4 |
| **Phase 7** | Audit trail periodic scan | Phase 6 |

## 7. Testing Strategy

Following McMillan's methodology — test information retrieval accuracy, not format:

1. **Template rendering test**: Given inputs, does the skill produce valid YAML metadata?
2. **Cross-reference resolution test**: Do all linked artifacts actually exist?
3. **Domain routing test**: Does the navigator correctly route issue domains?
4. **Deduplication test**: Does the skill detect near-duplicate issues?
5. **Round-trip test**: Create issue → parse issue → recreate issue → diff = empty

## 8. Key References

- [McMillan (2026). "Structured Context Engineering for File-Native Agentic Systems." *arXiv:2602.05447*.](https://arxiv.org/abs/2602.05447)
- [Anthropic (2026). "The Complete Guide to Building Skills for Claude." *Anthropic Resources*.](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf)
- [Hightower (2026). "Claude Code Agent Skills 2.0." *Towards AI*.](https://pub.towardsai.net/claude-code-agent-skills-2-0-from-custom-instructions-to-programmable-agents-ab6e4563c176)
- [Anthropic (2026). "Improving Skill Creator: Test, Measure, and Refine Agent Skills." *Claude Blog*.](https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills)
- [GitHub: anthropics/skills](https://github.com/anthropics/skills) — Official skill repository and skill-creator meta-skill
