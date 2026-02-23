# PRD-Update Skill

## Purpose
Maintain the MinIVess MLOps v2 hierarchical probabilistic PRD at `docs/planning/prd/`.
This skill provides structured operations for adding decisions, updating probabilities,
ingesting research papers, and validating the Bayesian decision network.

**This is an academic software project.** All PRD content must maintain peer-review-ready
citation standards. The PRD serves as the evidence base for a future peer-reviewed article.

## Activation
Use this skill when:
- A new paper or tool is discovered that affects technology decisions
- Prior probabilities need updating based on new evidence
- A new decision node needs to be added to the network
- A new scenario or archetype needs to be created
- The PRD needs validation after changes

## Available Operations

| Operation | Protocol | When to Use |
|-----------|----------|-------------|
| `add-decision` | `protocols/add-decision.md` | Add new decision node to the network |
| `update-priors` | `protocols/update-priors.md` | Update probabilities from evidence |
| `add-option` | `protocols/add-option.md` | Add option to existing decision |
| `create-scenario` | `protocols/create-scenario.md` | Create new composed scenario |
| `ingest-paper` | `protocols/ingest-paper.md` | Read paper and extract decisions |
| `validate` | `protocols/validate.md` | Validate DAG integrity and probabilities |
| `citation-guide` | `protocols/citation-guide.md` | Academic citation format reference |

## Key Files
- `docs/planning/prd/decisions/_network.yaml` — DAG topology (source of truth)
- `docs/planning/prd/decisions/_schema.yaml` — Decision node JSON Schema
- `docs/planning/prd/bibliography.yaml` — Central bibliography (ALL cited works)
- `docs/planning/prd/schema.yaml` — PRD document frontmatter schema
- `docs/planning/prd/llm-context.md` — AI assistant context

## Invariants (MUST hold after every operation)
1. **DAG acyclic** — No cycles in the decision network
2. **Probabilities sum to 1.0** — All prior_probability arrays, conditional tables, archetype overrides
3. **Cross-references resolve** — All `parent_decision_id` references exist in `_network.yaml`
4. **Node-file consistency** — Every node in `_network.yaml` has a `.decision.yaml` file and vice versa
5. **Option IDs match** — All option references in conditional tables match actual option_ids
6. **Citation integrity** — Every `citation_key` in any `.decision.yaml` MUST resolve to an entry in `bibliography.yaml`
7. **No citation loss** — When updating a decision file, ALL existing references MUST be preserved. References may only be ADDED, never removed without explicit user approval. This is non-negotiable for peer review readiness.
8. **Author-year format** — All in-text citations in `rationale`, `description`, and `research_notes` fields MUST use author-year format: "Surname et al. (Year)" — never numeric references

## CRITICAL: Citation Preservation Rules
- **NEVER delete a reference** from any `.decision.yaml` or from `bibliography.yaml` unless the user explicitly requests it
- **NEVER rewrite rationale text** in a way that drops existing author-year citations
- **When updating a decision file**, read the existing references array FIRST, then APPEND new ones
- **When rewriting rationale**, preserve all existing `(Author et al., Year)` citations and add new ones
- **Sub-citations are mandatory**: when ingesting a paper, extract all relevant references that paper cites
- **Validate after every change**: run the `validate` protocol which checks citation integrity

## Templates
- `templates/decision-node.yaml` — Template for new decision nodes
- `templates/scenario.yaml` — Template for new scenarios

## Workflow
1. Run the operation's protocol
2. Run `validate` protocol to check ALL invariants (including citation integrity)
3. Update `_network.yaml` if nodes/edges changed
4. Update `bibliography.yaml` if new references were cited
5. Commit changes with descriptive message

## Reviewer Agent Checklist
When reviewing PRD changes (manually or via CI), verify:
- [ ] No references were removed from any `.decision.yaml` file
- [ ] All new `citation_key` values exist in `bibliography.yaml`
- [ ] All `rationale` fields contain at least one `(Author, Year)` citation
- [ ] All `bibliography.yaml` entries have `doi` or `url`
- [ ] The `bibliography.yaml` `topics` array includes all decision files that cite each entry
- [ ] Sub-citations from ingested papers have been extracted and added
