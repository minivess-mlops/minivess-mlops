# Template: Research Issue

Use for exploration spikes, literature reviews, model evaluations, and
technology assessments.

## Title Pattern

```
research({scope}): {what to explore}
```

Examples:
- `research(models): evaluate CoMMA Mamba for 3D vessel segmentation`
- `research(calibration): compare MAPIE vs netcal for conformal prediction`
- `research(data): assess VesselFM data leakage risk on MiniVess`

## Labels

```
research, {priority_label}, {domain_label}
```

## Body

```markdown
<!-- METADATA
priority: {P1|P2|P3}
domain: {domain}
type: research
plan:
prd_decisions: [{decision_ids — research often feeds PRD updates}]
relates_to: [{#issues}]
blocked_by: []
status: open
-->

## Summary

{What question this research answers and why it matters for the project.
Frame as a decision to be made, not just "look into X."}

## Research Questions

1. {Specific question with measurable answer — e.g., "Does model X achieve
   >0.85 clDice on MiniVess with <8 GB VRAM?"}
2. {Second question}
3. {Third question — keep to 3-5 max}

## Context

- **PRD Decision**: `{decision_node.yaml}` — if this feeds a PRD choice
- **Literature**: [`{report_file}`]({path}) — if a literature report exists
- **Related work**: [{Author (Year). "Title."}]({URL})
- **Prior art in repo**: {existing code/config that relates}

## Methodology

{How to answer the research questions:
- Literature review scope (which databases, keywords)
- Experimental setup (dataset, metrics, hardware constraints)
- Comparison baselines
- Success criteria}

## Deliverables

- [ ] Literature summary in `docs/research/{topic}.md`
- [ ] PRD decision update (if evidence changes priors)
- [ ] Prototype implementation (if promising) — filed as separate feature issue
- [ ] VRAM measurements in `configs/model_profiles/{model}.yaml` (if model eval)

## Time Box

{Estimated effort: e.g., "1 day literature + 1 day prototype" or
"2 hours web search + comparison table"}

## References

- [{Author (Year). "Title." *Journal*.}]({URL})
```
