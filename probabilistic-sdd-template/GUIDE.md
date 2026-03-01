# Probabilistic SDD — User Guide

A **Probabilistic Software Design Document (SDD)** encodes open-ended technology
and architecture decisions as a Bayesian decision network. Instead of resolving
each decision upfront, you assign prior probabilities to options and let evidence
(benchmarks, papers, experiments) update those probabilities over time.

This template provides the reusable framework. You supply the decisions.

---

## What is a Probabilistic SDD?

Traditional design documents pick *one* answer per question ("We use PostgreSQL").
A probabilistic SDD keeps *all viable options open* with calibrated probabilities:

```yaml
options:
  - option_id: postgresql
    prior_probability: 0.45
    status: recommended
  - option_id: sqlite
    prior_probability: 0.35
    status: viable
  - option_id: duckdb
    prior_probability: 0.20
    status: experimental
```

As evidence accumulates (benchmarks, team experience, paper findings), you run
Bayesian updates to shift probabilities. Decisions can also *condition* on each
other — choosing a cloud deployment shifts the database probabilities differently
than choosing local-first.

### Why Use This?

1. **Explicit uncertainty** — Acknowledge that early-stage projects have open questions
2. **Evidence-driven** — Every probability shift is backed by citations
3. **Composable** — Archetypes (team profiles) and domain overlays modulate probabilities
4. **Auditable** — Append-only citation trail suitable for peer review
5. **Machine-readable** — YAML + JSON Schema enables automated validation

---

## Quick Start

Get a working SDD in 3 steps (~5 minutes):

### Step 1: Copy the Template

```bash
cp -r probabilistic-sdd-template/ my-project-sdd/
```

### Step 2: Edit the Network Skeleton

Open `_network-template.yaml` and replace `CHANGE_ME` with your project name.
Copy it to `decisions/_network.yaml`:

```bash
mkdir -p my-project-sdd/decisions
cp my-project-sdd/_network-template.yaml my-project-sdd/decisions/_network.yaml
```

### Step 3: Create Your First Decision

```bash
cp my-project-sdd/templates/decision-node.yaml \
   my-project-sdd/decisions/L1-goals/my_first_decision.decision.yaml
```

Edit the file — replace all `CHANGE_ME` markers. Add the node to your
`_network.yaml`. Run the validator:

```bash
python my-project-sdd/validate.py --sdd-root my-project-sdd/
```

---

## Step-by-Step Population Guide

Populating a full SDD takes 1-2 days for a medium-sized project. Follow these
9 steps in order.

### Step 1: Define Your Decision Hierarchy (~30 min)

Sketch your project's key decisions across 5 levels:

| Level | Focus | Examples |
|-------|-------|---------|
| L1 | Research Goals | Project purpose, impact target, compliance depth |
| L2 | Architecture | Model strategy, data management, serving approach |
| L3 | Technology | Specific frameworks, tools, libraries |
| L4 | Infrastructure | Compute targets, CI/CD, containers |
| L5 | Operations | Monitoring, governance, retraining triggers |

Aim for 5-15 decisions per level. Don't overthink — you can always add more later.

### Step 2: Create the Network DAG (~15 min)

Edit `decisions/_network.yaml`:
- Add each decision as a node (id, level, file path, title)
- Add edges for conditional dependencies (which decisions influence others?)
- Mark influence strength: strong, moderate, or weak

### Step 3: Create Decision Files (~2-4 hours)

For each node in the network, create a `.decision.yaml` file using
`templates/decision-node.yaml`. For each decision:

1. Define 2-5 options with descriptions
2. Assign prior probabilities (must sum to 1.0)
3. Write conditional tables for parent dependencies
4. Cite at least one academic reference per decision
5. Write a rationale with author-year in-text citations

### Step 4: Set Up the Bibliography (~30 min)

Create `bibliography.yaml` with entries for every cited work:

```yaml
bibliography:
  - citation_key: surname2024
    inline_citation: "Surname et al. (2024)"
    authors: "Surname, A. B., Coauthor, C. D."
    year: 2024
    title: "Paper Title"
    venue: "Conference/Journal"
    doi: "10.xxxx/xxxxx"
    evidence_type: benchmark
    topics:
      - decision_id_1
```

### Step 5: Define Archetypes (~1 hour)

Create 2-3 team archetypes using `templates/archetype.yaml`. Each archetype
represents a different team profile (solo researcher, large lab, clinical team)
with different probability overrides per decision.

### Step 6: Create Domain Overlays (~1 hour)

If your project spans multiple application domains, create overlays using
`templates/domain-overlay.yaml`. Each overlay adjusts metric weights, loss
preferences, and decision priors for a specific domain.

### Step 7: Compose Scenarios (~30 min)

Create scenarios using `templates/scenario.yaml`. A scenario resolves every
decision to exactly one option — it represents a concrete implementation path.

### Step 8: Validate (~5 min)

Run the validator to check all invariants:

```bash
python validate.py --sdd-root <your-sdd>/
```

The validator checks:
- DAG acyclicity (no circular dependencies)
- Probability sums (all distributions sum to 1.0)
- Cross-reference integrity (all IDs resolve)
- File existence (every node has a file)
- Schema compliance (required fields present)

### Step 9: Iterate with Evidence

As your project evolves, update probabilities using the protocols in `protocols/`:
- `update-priors.md` — Shift probabilities based on new evidence
- `add-option.md` — Add new technology options to existing decisions
- `add-decision.md` — Add entirely new decisions to the network
- `ingest-paper.md` — Process a research paper into evidence updates

---

## Concepts

### Decision Node

A node in the Bayesian network representing a single technology or architecture
decision. Contains 2+ options with prior probabilities, conditional dependencies,
archetype weights, and academic references.

**File**: `*.decision.yaml` — validated against `_schema.yaml`

### Conditional Probability Table (CPT)

When a child decision depends on a parent, the CPT specifies how parent choices
shift child probabilities. Each row (given a parent option) must sum to 1.0.

### Archetype

A team persona that modulates decision probabilities. Different team profiles
(budget, expertise, hardware) naturally prefer different options. Archetype
weights override default priors when computing scenario probabilities.

### Scenario

A fully resolved set of decisions — exactly one option per node. Represents
a concrete, implementable configuration. Its joint probability indicates how
likely this combination is given the current evidence.

### Domain Overlay

A configuration layer that adjusts backbone defaults (metrics, losses,
augmentations) for a specific application domain. Applied on top of the
domain-agnostic backbone defaults.

### Bibliography

Central `bibliography.yaml` file containing all academic references in
author-year format. Decision files link to entries by `citation_key`.
References are **append-only** — never remove citations, as they form the
evidence trail for peer review.

---

## Validation

### Running the Validator

```bash
# Validate your SDD
python validate.py --sdd-root path/to/your/sdd

# Validate the included example
python validate.py --sdd-root examples/minivess
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks pass |
| 1 | FAIL — structural violations found |

### What Gets Checked

1. **DAG Acyclicity** — No circular dependencies in the decision network
2. **Probability Sums** — All prior, conditional, and archetype distributions sum to 1.0 (±0.01)
3. **Cross-References** — All edge endpoints exist as nodes
4. **File Existence** — Every node has a corresponding `.decision.yaml` file
5. **Schema Compliance** — Required fields present in every decision file

---

## FAQ

### How many decisions should my SDD have?

Start small (10-20 decisions) and grow organically. A mature project might have
50-100 decisions. The five-level hierarchy keeps things organized regardless of
scale.

### Can I use this without the academic citation system?

Yes, but it's not recommended. The citation system ensures every probability
assignment has a justification. For non-academic projects, you can use informal
evidence types (tool_release, experience) instead of formal papers.

### How do I handle decisions that are already resolved?

Set their status to `resolved` and `implementation_status` to `implemented`.
Keep the probability distribution — it documents *why* that option was chosen
and what alternatives were considered.

### What if probabilities don't sum to 1.0?

The validator enforces this with ±0.01 tolerance. Adjust the smallest option(s)
to compensate. This is a fundamental invariant of the Bayesian framework.

### Can I extend the 5-level hierarchy?

The L1-L5 structure works for most projects. If you need more granularity,
use tags or sub-IDs within a level rather than adding new levels.

### How does this relate to Architecture Decision Records (ADRs)?

ADRs capture *resolved* decisions with context. A probabilistic SDD captures
*open* decisions with calibrated uncertainty. They're complementary — ADRs
are snapshots, SDDs are living probability distributions. You can generate
ADRs from resolved SDD decisions.

### What tools can process this format?

The SDD uses standard YAML validated against JSON Schema. Any tool that reads
YAML can process it. The `validate.py` script provides structural validation.
For visualization, export the network as a DOT graph or use any DAG visualization
library.

### How do I integrate this with CI/CD?

Add `python validate.py --sdd-root <path>` to your CI pipeline. The validator
returns exit code 1 on failures, making it compatible with any CI system.
See `protocols/validate.md` for the full validation protocol.

---

## Worked Examples

The `examples/minivess/` directory contains a curated instantiation from the
MinIVess biomedical segmentation project, demonstrating:

- **L1 decision**: Project Purpose (4 options, 3 archetypes)
- **L3 decision**: Loss Functions (6 options, conditional on model strategy)
- **Archetype**: Solo Researcher profile with hardware constraints
- **Domain overlay**: Vascular segmentation with topology-aware metrics
- **Bibliography**: 6 entries in author-year format
- **Network**: 2-node DAG with one conditional edge

Run the validator on it:

```bash
python validate.py --sdd-root examples/minivess
```

---

## File Reference

| File | Purpose |
|------|---------|
| `_schema.yaml` | JSON Schema for decision node validation |
| `_network-template.yaml` | Empty DAG skeleton to copy and populate |
| `backbone-defaults.yaml` | Domain-agnostic default configuration |
| `validate.py` | Standalone structural validator (CLI) |
| `templates/decision-node.yaml` | Template for new decision nodes |
| `templates/scenario.yaml` | Template for resolved scenarios |
| `templates/archetype.yaml` | Template for team archetypes |
| `templates/domain-overlay.yaml` | Template for domain overlays |
| `protocols/*.md` | Step-by-step operational protocols |
| `examples/minivess/` | Complete example instantiation |
| `CLAUDE-TEMPLATE.md` | Template CLAUDE.md for AI-assisted development |
