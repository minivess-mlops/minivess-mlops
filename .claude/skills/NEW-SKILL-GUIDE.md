# Creating a New Skill

Reference exemplar: `factorial-monitor/` (10 files, born-upgraded March 2026).

## Step 1: Scaffold

```bash
cp -r .claude/skills/factorial-monitor/ .claude/skills/my-new-skill/
```

Rename files and clear content — keep the structure.

## Step 2: Fill Frontmatter

Every SKILL.md starts with this YAML block:

```yaml
---
name: my-new-skill
version: 1.0.0
description: >
  What it does. Use when [trigger phrases].
  Do NOT use for: [negative triggers pointing to other skills].
last_updated: YYYY-MM-DD
activation: manual
invocation: /my-new-skill
metadata:
  category: development | research | knowledge | orchestration | operations | planning
  tags: [relevant, tags, from, existing, skills]
  relations:
    compose_with: []      # Skills whose input this skill produces
    depend_on: []          # Skills that must exist for this to work
    similar_to: []         # Functionally equivalent alternatives
    belong_to: []          # Parent skill this is a sub-component of
---
```

**Required:** `name`, `description` (with triggers AND negative triggers), `metadata.relations`.

## Step 3: Write Protocols

One file per workflow phase in `protocols/`. Name them after phases:

```
protocols/
  phase-1-gather.md
  phase-2-process.md
  phase-3-validate.md
```

Cross-cutting rules that apply to ALL phases stay in SKILL.md.
Phase-specific rules go in `instructions/`.

## Step 4: Write Eval Criteria

`eval/checklist.md` with 3-6 binary YES/NO criteria. Split into:

- **Structural** (machine-parseable): file exists, YAML valid, API returns expected
- **Behavioral** (require judgment): output quality, no silent dismissal

Include 3 should-trigger and 2 should-not-trigger prompts.

## Step 5: Verify Relations

Check that every skill referenced in `compose_with` / `depend_on` exists:

```bash
for skill in $(grep -A5 'compose_with:' .claude/skills/my-new-skill/SKILL.md | grep '^ *-' | sed 's/.*- //'); do
  [ -d ".claude/skills/$skill" ] && echo "✓ $skill" || echo "✗ $skill MISSING"
done
```

## Step 6: Validate

After creation, verify:

- [ ] `head -1 SKILL.md` is `---` (YAML frontmatter present)
- [ ] Description includes "Use when" AND "Do NOT use for"
- [ ] `eval/checklist.md` exists with 3-6 criteria
- [ ] `metadata.relations` has all 4 edge types (even if empty lists)
- [ ] Skill appears in system-reminder after saving

## When NOT to Create a New Skill

- If an existing skill covers the use case — extend it instead
- If the task is a one-off — just do it, don't build infrastructure
- If the task has no repeatable workflow — skills are for repeatable patterns
- If the skill would have <80 lines total — consider adding to an existing skill

## Anti-Pattern: Generated Rules

Never auto-generate domain rules or anti-patterns. The best rules in this repo
(TDD Rule #9, factorial-monitor Rule F1-F5) all trace to real failures documented
in `.claude/metalearning/`. Rules come from experience, not templates.
