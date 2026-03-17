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

## GitHub Project Integration

PRD updates often generate implementation work. The project backlog lives in:
- **Repository**: `minivess-mlops/minivess-mlops`
- **Project**: `Minivess MLOps` (ID: `PVT_kwDOCPpnGc4AYSAM`, Number: 1)
- **Priority field**: `PVTSSF_lADOCPpnGc4AYSAMzgPhgsk`
  - High: `9ac26196` (P0 — must do next)
  - Medium: `9d51ccf2` (P1 — should do soon)
  - Low: `17b6c978` (P2 — nice to have)

### Creating Issues from PRD Changes
When a PRD update reveals new implementation work:
1. Create a GitHub issue with PRD context (decision ID, option, bibliography key)
2. Add to project: `gh project item-add 1 --owner minivess-mlops --url <issue_url>`
3. Set priority based on evidence strength and implementation readiness
4. Use labels: `enhancement`, `models`, `monitoring`, `training`, `metrics`,
   `uncertainty`, `compliance`, `annotation`, `data-quality`, `validation`,
   `observability`, `research`, `ci-cd`, `documentation`
5. Priority labels: `P0-critical`, `P1-high`, `P2-medium`

### Issue Body Template
```markdown
## Summary
[What and why, with key evidence]

## PRD Context
- **Decision**: `decision_id` (level)
- **Option**: `option_id` (prior: X.XX)
- **Bibliography**: `citation_key`

## Acceptance Criteria
- [ ] [Specific, testable items]
- [ ] Unit tests (TDD mandatory)

## References
- Author et al. (Year). "Title." DOI/arXiv
```

## PRD -> KG Materialization Protocol

When a PRD decision node is resolved (posterior >= 0.80 for the winner), it must
be materialized into the Knowledge Graph domain files. This is the connection
between probabilistic reasoning (PRD) and deterministic implementation (KG).

See: `docs/planning/prd-kg-openspec-architecture.md` for the full architecture.

### Materialization Steps

1. **Check resolution threshold**: `status: resolved` AND winner's `posterior_probability >= 0.80`
2. **Update KG domain file**: Add/update the decision entry in the appropriate
   `knowledge-graph/domains/*.yaml` with `winner:`, `rationale:`, `implementation:`, `prd_node:`
3. **Update _network.yaml**: Ensure node and edges exist in `knowledge-graph/_network.yaml`
4. **Update navigator.yaml**: Ensure the decision ID appears in the domain's `decisions:` list
5. **Create OpenSpec spec** (optional): If the decision is implementation-ready,
   create a GIVEN/WHEN/THEN spec in `openspec/specs/` via `/opsx:propose`
6. **Propagation check**: Check `_network.yaml propagation:` section for downstream
   nodes that need `requires_review: true` flags

### Status -> KG Mapping

| PRD Status | KG Representation | OpenSpec? |
|-----------|-------------------|-----------|
| `resolved` (posterior >= 0.80) | `winner:` field in domain YAML | Yes |
| `partial` (no clear winner) | `candidates:` list in domain YAML | No |
| `config_only` (tool selected, not integrated) | `winner:` with note | Optional |
| `not_started` (unexplored) | `candidates:` or omitted | No |

### Invariant: KG-PRD Consistency

After materialization, these MUST hold:
- Every `prd_node:` in domain YAML points to an existing decision file
- Every resolved decision in `_network.yaml` has a corresponding `winner:` in its domain
- The `decisions:` list in `navigator.yaml` includes all decision IDs for each domain

## Workflow
1. Run the operation's protocol
2. Run `validate` protocol to check ALL invariants (including citation integrity)
3. Update `_network.yaml` if nodes/edges changed
4. Update `bibliography.yaml` if new references were cited
5. **Materialize to KG** if decision status changed to resolved (see protocol above)
6. Create GitHub issues for implementation work revealed by the update
7. Commit changes with descriptive message

## Reviewer Agent Checklist
When reviewing PRD changes (manually or via CI), verify:
- [ ] No references were removed from any `.decision.yaml` file
- [ ] All new `citation_key` values exist in `bibliography.yaml`
- [ ] All `rationale` fields contain at least one `(Author, Year)` citation
- [ ] All `bibliography.yaml` entries have `doi` or `url`
- [ ] The `bibliography.yaml` `topics` array includes all decision files that cite each entry
- [ ] Sub-citations from ingested papers have been extracted and added
- [ ] Implementation issues created for new options/significant evidence updates
