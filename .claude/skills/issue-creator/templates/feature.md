# Template: Feature Issue

Use for new capabilities, enhancements, and additions.

## Title Pattern

```
{scope}: {what it adds}
```

Examples:
- `cloud: multi-cloud launcher with Lambda Labs multi-region rotation`
- `training: add boundary loss functions (HausdorffDT + LogHausdorffDT)`
- `data: VesselNN dataset adapter for external validation`

## Labels

```
enhancement, {priority_label}, {domain_label}
```

## Body

```markdown
<!-- METADATA
priority: {P0|P1|P2|P3}
domain: {domain}
type: feature
plan: {path/to/plan — omit if none}
prd_decisions: [{decision_ids}]
relates_to: [{#issues}]
blocked_by: [{#issues}]
status: open
-->

## Summary

{What this feature delivers and why it matters. One paragraph. Lead with the
user-visible outcome: "Researchers can now X" or "The pipeline gains the ability
to Y." Avoid implementation-first language.}

## Context

- **Plan**: [`{plan_file}`]({path})
- **Research**: [`{report_file}`]({path})
- **PRD**: `{decision_node.yaml}`
- **Commits**: `{sha1}`, `{sha2}`

## Acceptance Criteria

- [ ] {User-visible behavior works: "X produces Y"}
- [ ] {Integration point validated: "Works with existing Z"}
- [ ] Unit tests (TDD mandatory per CLAUDE.md)
- [ ] Pre-commit hooks pass
- [ ] Config-driven (no hardcoded values per CLAUDE.md rules)

## Implementation Notes

{Key architectural constraints:
- Which adapter/registry to extend
- Which config file to add/modify
- Docker volume mount considerations
- MONAI-first: check if MONAI already has this}

## References

- [{Author (Year). "Title." *Journal*.}]({URL})
```
