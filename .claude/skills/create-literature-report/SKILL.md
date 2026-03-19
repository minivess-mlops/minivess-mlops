---
name: create-literature-report
version: 2.0.0
description: >
  Reproducible scientific literature research report with citation-verified,
  hallucination-free synthesis. 6-phase pipeline composing Markov novelty
  writing, parallel web research, and zero-tolerance citation verification.
last_updated: 2026-03-19
activation: See ACTIVATION-CHECKLIST.md
invocation:
  - "create literature report on [topic]"
  - "research report on [topic]"
  - "literature survey for [topic]"
revision_notes: >
  v2.0.0: Decomposed monolith into protocols + prompts + state schema.
  Added ACTIVATION-CHECKLIST, agent prompt templates, WebFetch failure
  handling, deduplication algorithm, FORCE_STOP protocol, workspace
  structure. Addresses 19 architectural gaps from 3-reviewer audit.
metadata:
  category: research
  tags: [literature, citations, web-search, synthesis, hallucination-free]
  relations:
    compose_with:
      - prd-update
      - kg-sync
    depend_on:
      - fetch-docs
    similar_to: []
    belong_to: []
---

# create-literature-report v2.0.0

> Zero-hallucination scientific literature synthesis in 6 phases.

## Quick Start

1. Run `ACTIVATION-CHECKLIST.md` pre-flight
2. Execute phases 0→5 sequentially
3. State file enables crash recovery at any sub-step

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `topic` | Yes | — | Research topic or question |
| `seed_papers` | Yes | — | Markdown list: `- [Author (Year). "Title."](URL)` per line |
| `output_path` | Yes | — | Path for the .md report file |
| `repo_context` | No | Current repo | Which repo/manuscript this informs |
| `target_paper_count` | No | 50 | Target total papers (seeds + discovered) |
| `quality_target` | No | MINOR_REVISION | Convergence: MAJOR_REVISION / MINOR_REVISION / ACCEPT |
| `create_issues` | No | true | Create GitHub issues from findings |
| `update_kg` | No | false | Propose KG decision node updates |
| `max_iterations` | No | 3 | Max council review iterations |
| `biblio_search_paths` | No | [] | Additional local bibliography directories to search |
| `report_length_words` | No | 6000 | Target report body length in words |

## Architecture: 6-Phase Pipeline

```
Phase 0: CAPTURE & ALIGN         (~5 min)
    ↓
Phase 1: GATHER (parallel)       (~15 min)
    ↓
Phase 2: SYNTHESIZE              (~15 min)
    ↓
Phase 3: VERIFY (parallel)       (~10 min)
    ↓
Phase 4: REVIEW (iterative)      (~10 min, optional)
    ↓
Phase 5: DELIVER                 (~5 min)
```

Total time budget: **<60 minutes** for 50-paper report.

## Phase Protocols

Each phase has a dedicated protocol file with exact steps:

| Phase | Protocol | Purpose |
|-------|----------|---------|
| 0 | `protocols/phase-0-capture.md` | Save prompt, alignment questions, state init |
| 1 | `protocols/phase-1-gather.md` | Parallel research agents, deduplication |
| 2 | `protocols/phase-2-synthesize.md` | Markov novelty writing rules |
| 3 | `protocols/phase-3-verify.md` | Citation URL/title/author verification |
| 4 | `protocols/phase-4-review.md` | Iterated council convergence loop |
| 5 | `protocols/phase-5-deliver.md` | Issues, KG updates, final commit |

## Agent Prompt Templates

| Agent | Template | Used in |
|-------|----------|---------|
| Local inventory | `prompts/gather-local-inventory.md` | Phase 1 |
| Web research | `prompts/gather-web-research.md` | Phase 1 |
| Citation verifier | `prompts/verify-citation.md` | Phase 3 |
| Review: domain | `prompts/review-domain-expert.md` | Phase 4 |
| Review: novelty | `prompts/review-novelty-assessor.md` | Phase 4 |

## State Management

- Schema: `state/literature-report-state.schema.json`
- Example: `state/example-state.json`
- Location: `state/literature-report-{topic_slug}-state.json`
- Sub-step tracking enables precise crash recovery

## Workspace Structure

```
state/
  literature-report-{topic_slug}-state.json
workspace/literature-report-{topic_slug}/
  phase-1-gather/
    local-inventory.json       # Structured inventory from local agent
    web-research-a.json        # Papers from research agent A
    web-research-b.json        # Papers from research agent B
    deduplicated-papers.json   # Final merged, deduplicated list
  phase-3-verify/
    verification-batch-1.json  # Results from verifier agent 1
    verification-batch-2.json  # Results from verifier agent 2
    verification-batch-3.json  # Results from verifier agent 3
    corrections-log.json       # All corrections applied
  phase-4-review/
    iter-1/
      review-domain.md         # Domain expert feedback
      review-novelty.md        # Novelty assessor feedback
      synthesis.md             # L2 synthesis
      verdict.md               # L1 verdict
```

## Convergence Criteria

| Target | Criteria | Max Iterations |
|--------|----------|---------------|
| MAJOR_REVISION | 0 hallucinated citations, body text coherent | 1 |
| MINOR_REVISION | Above + Markov novelty score ≥2.0/section | 2 |
| ACCEPT | Above + all "So What?" criteria met, ≥3 cross-domain insights | 3 |

## FORCE_STOP Protocol

If any phase exceeds its time budget or hits 3 consecutive failures:

1. Save current state with `"force_stopped": true, "reason": "..."`
2. Git commit all work so far
3. Report to user: phase, substep, failure reason, recommended action
4. Do NOT continue to next phase
5. On resume: read state, fix the blocking issue, restart current phase

## Anti-Patterns (BANNED)

| Pattern | Detection | Prevention |
|---------|-----------|------------|
| Memory-based citations | Phase 3 verifies ALL URLs | Never write a citation without a URL |
| Annotated bibliography | Phase 4 novelty reviewer | "So What?" test per section |
| Serial verification | Phase 3 runs ALL in parallel | GATHER→CATEGORIZE→FIX pattern |
| Skip verification | Phase 3 is MANDATORY | State blocks Phase 4 without Phase 3 |
| Unattributed numbers | Phase 4 domain reviewer | Every number needs `[Author (Year)]` |
| Generic gaps | Phase 4 novelty reviewer | "More research needed" is BANNED |
| Citation spam | Phase 2 rules | Max 3 inline citations per sentence |

## Eval Criteria

### Quantitative (automated)

| Metric | Target | How to measure |
|--------|--------|----------------|
| Hallucination rate | 0% | `verified / total` from Phase 3 |
| URL validity | 100% | All links resolve (Phase 3) |
| Title accuracy | 100% | Verified titles match (Phase 3) |
| Paper count | ≥ target_paper_count | `state.verified_count` |

### Qualitative (reviewer agents)

| Criterion | Target | How to measure |
|-----------|--------|----------------|
| No annotated bibliography | Pass | Phase 4 novelty reviewer |
| Cross-domain synthesis | ≥3 domains connected | Phase 4 novelty reviewer |
| Specific research gaps | ≥3 gaps (not generic) | Phase 4 novelty reviewer |
| Repo-contextualized | All findings mapped to repo | Phase 4 domain reviewer |

### Regression test

Run on biomedical-agentic-ai topic with original 27 seeds.
Expected: 0 hallucinations, ≥55 verified papers, ≥3 novel insights.

## Sub-Skill Dependencies

| Skill | Location | Used for |
|-------|----------|----------|
| engaging-review-writer | sci-llm-writer | Markov novelty rules (Phase 2) |
| citation-content-verifier | sci-llm-writer | Verification protocol (Phase 3) |
| iterated-llm-council | sci-llm-writer | Review loop (Phase 4) |

**Note**: These skills provide *principles* that are inlined into this skill's
protocol files. This skill is self-contained — it does NOT require the
sci-llm-writer repo to be present at runtime.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-03-18 | Decomposed into protocols + prompts + state schema. 19 gaps fixed. |
| 1.0.0 | 2026-03-18 | Initial release (monolithic). |
