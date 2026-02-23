# PRD-Update Skill

## Purpose
Maintain the MinIVess MLOps v2 hierarchical probabilistic PRD at `docs/planning/prd/`.
This skill provides structured operations for adding decisions, updating probabilities,
ingesting research papers, and validating the Bayesian decision network.

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

## Key Files
- `docs/planning/prd/decisions/_network.yaml` — DAG topology (source of truth)
- `docs/planning/prd/decisions/_schema.yaml` — Decision node JSON Schema
- `docs/planning/prd/schema.yaml` — PRD document frontmatter schema
- `docs/planning/prd/llm-context.md` — AI assistant context

## Invariants (MUST hold after every operation)
1. **DAG acyclic** — No cycles in the decision network
2. **Probabilities sum to 1.0** — All prior_probability arrays, conditional tables, archetype overrides
3. **Cross-references resolve** — All `parent_decision_id` references exist in `_network.yaml`
4. **Node-file consistency** — Every node in `_network.yaml` has a `.decision.yaml` file and vice versa
5. **Option IDs match** — All option references in conditional tables match actual option_ids

## Templates
- `templates/decision-node.yaml` — Template for new decision nodes
- `templates/scenario.yaml` — Template for new scenarios

## Workflow
1. Run the operation's protocol
2. Run `validate` protocol to check invariants
3. Update `_network.yaml` if nodes/edges changed
4. Commit changes with descriptive message
