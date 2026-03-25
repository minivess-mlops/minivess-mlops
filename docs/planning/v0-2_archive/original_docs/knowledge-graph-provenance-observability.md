# Knowledge Graph Provenance and Observability — Research Report

**Date**: 2026-03-24
**Context**: VASCADIA/NEUROVEX knowledge graph (52-node Bayesian DAG, YAML-in-git, pre-commit validated)
**Purpose**: Identify practical tools and patterns for KG provenance, observability, and traceability to inform implementation decisions and produce one infographic for the agentic-development slide deck.

---

## 1. The Problem We Actually Have

The VASCADIA knowledge graph is a production-grade specification system: 52 decision nodes across 5 levels (L1 research goals → L5 operations), connected by ~100 edges with hard/soft/signal propagation types. It is YAML-based, git-versioned, and validated by pre-commit hooks (`review_prd_integrity.py`, `validate_yaml_contract.py`, `validate_prd_citations.py`). Every decision links to evidence, implementation files, tests, and bibliography entries.

What we lack:

- **Provenance at the decision level**: git blame tells us *who* changed a line and *when*, but not *why* a posterior probability shifted from 0.30 to 0.85, which experiment motivated it, or which downstream decisions were affected.
- **Observability of KG health over time**: no dashboard showing schema drift, orphaned nodes, stale decisions, or propagation debt (decisions flagged for review but not yet reviewed).
- **Semantic diff**: git diff shows YAML line changes; we need "decision X changed from option A to option B, affecting decisions Y and Z."
- **Reproducibility of KG states**: if a downstream system breaks, can we reconstruct the KG state at any prior commit and understand the chain of decisions that led there?

This report surveys the practical tools and patterns available to address these gaps, grounded in the academic literature and the reality that we are a small team with a repo-local KG (not an enterprise triplestore deployment).

---

## 2. The W3C PROV Standard — Foundation for All Provenance

The W3C PROV family defines three core concepts that map directly to our KG operations:

- **Entity**: Any thing with provenance — a decision node YAML file, a domain materialization, a bibliography entry, a KG state at a given commit
- **Activity**: Something that changes entities — updating a posterior probability, resolving a decision, running an experiment, triggering propagation
- **Agent**: Who or what is responsible — a human researcher, a pre-commit hook, an agentic session (Claude Code), an experiment script

Key relationships: `wasGeneratedBy`, `wasDerivedFrom`, `wasAttributedTo`, `used`, `wasAssociatedWith`. These are sufficient to answer the questions "where did this fact come from?" and "what was affected when it changed?"

### Practical Implementation

**`prov` Python library** (v2.1.1, MIT, 829 commits): The canonical implementation. Creates in-memory PROV documents, serialises to PROV-O/RDF, PROV-JSON, PROV-N, PROV-XML. Converts to NetworkX for graph analysis. Powers ProvStore (openprovenance.org).

**Application to VASCADIA**: After each decision update, generate a PROV record:

```python
from prov.model import ProvDocument

doc = ProvDocument()
doc.set_default_namespace('https://vascadia.dev/prov/')

# The decision node before and after
old_entity = doc.entity('decision:loss_function_v3', {'prov:value': 'posterior=0.30'})
new_entity = doc.entity('decision:loss_function_v4', {'prov:value': 'posterior=0.85'})

# The activity that changed it
activity = doc.activity('update:loss_resolution', '2026-03-15T14:30:00')

# The evidence that motivated it
evidence = doc.entity('experiment:dynunet_loss_variation_v2')

# Relationships
new_entity.wasGeneratedBy(activity)
new_entity.wasDerivedFrom(old_entity)
activity.used(evidence)
activity.wasAssociatedWith(doc.agent('agent:petteri'))
```

This creates a machine-queryable audit trail that goes beyond git blame: it records not just *what* changed but *why* (the experiment) and *what was used* (the evidence).

---

## 3. Git as Provenance Infrastructure

For a YAML-in-git KG, git already provides coarse-grained provenance. Two tools extract W3C PROV from git history:

**Git2PROV** (IDLab Research, Node.js): Maps commits → Activities, files → Entities, authors → Agents. Generates W3C PROV documents from any git repository.

**GitLab2PROV** (DLR German Aerospace Center, Python): Extracts PROV from GitLab API including issues and merge requests. Presented at USENIX TaPP 2021.

**What this gives us for free**: Every commit to a KG file is already a tracked Activity with Agent (committer) and Entity (file version). The gap is *semantic* provenance — git knows a line changed, but not that "the loss function decision resolved in favour of cbdice_cldice based on experiment results."

**Recommendation**: Use git as the coarse layer. Add a thin PROV wrapper (via the `prov` library) in pre-commit or post-commit hooks to generate semantic provenance documents that augment git's structural provenance with decision-level semantics.

---

## 4. Triple-Level Provenance — The Bosch Approach

Henrik Dibowski's FOIS 2024 paper "Full Traceability and Provenance for Knowledge Graphs" (Bosch Research) introduces **PROV-STAR**: an RDF-star extension of W3C PROV-O that enables provenance annotation at the individual triple level.

Architecture:
1. **Provenance Engine** intercepts every SPARQL/Update query
2. **PROV-STAR Ontology** uses RDF-star's embedded triple syntax: `<< subject predicate object >> prov:wasGeneratedBy activity`
3. **Query Transformation** rewrites incoming queries to simultaneously record changes in a separate provenance KG

Key innovation: stores only change deltas, not complete versions. Any historical KG state can be reconstructed from a single query.

**Relevance to VASCADIA**: Our KG is YAML, not RDF triples. But the *principle* applies: instead of versioning entire files, track individual decision changes with provenance metadata. This could be implemented as a YAML-level diff + PROV annotation layer, stored alongside the KG files.

---

## 5. Schema Validation as Tests for Knowledge Graphs

The harness pattern for KGs: formal constraints that reject invalid states before they enter the repository. Three practical approaches:

### 5.1 SHACL (Shapes Constraint Language)

W3C standard for RDF graph validation. **pySHACL** provides CLI validation with exit codes (0 = conformant, 1 = violation), directly usable in pre-commit hooks and CI.

Enterprise adoption: eccenca Corporate Memory v25.1 offers one-click SHACL shape creation; Stardog stores SHACL in named graphs; GraphDB validates against materialised inference.

### 5.2 LinkML — The Strongest Candidate for YAML/JSON KGs

**LinkML** (Linked Data Modeling Language) is purpose-built for our pattern. A single YAML schema definition generates:
- SHACL shapes (for RDF validation)
- JSON Schema (for YAML/JSON validation)
- Pydantic models (for Python type checking)
- Markdown documentation

Originally developed for the Biolink Model (biological knowledge graphs), it is the "CLAUDE.md for KGs" pattern: one specification file generates all validation artifacts.

```yaml
# LinkML schema — single source of truth
classes:
  DecisionNode:
    attributes:
      decision_id:
        identifier: true
        pattern: "^[a-z_]+$"
      level:
        range: DecisionLevel
        required: true
      status:
        range: DecisionStatus
        required: true
      options:
        range: Option
        multivalued: true
```

From this, `linkml generate` produces validators for every format our KG uses.

**Recommendation**: Evaluate LinkML as a replacement for `_schema.yaml` + custom `review_prd_integrity.py`. It would unify schema definition with validation code generation, reducing the maintenance burden of keeping schema docs and validation scripts in sync.

### 5.3 Pydantic for Immediate Value

For the existing YAML files, Pydantic v2 with `pydantic-yaml` provides immediate schema enforcement without adopting a new framework. The `_schema.yaml` contract can be translated directly to Pydantic models and run in pre-commit hooks.

---

## 6. KG Versioning Beyond Git

### 6.1 What Git Already Gives Us

For a KG under ~1000 entities in YAML, plain git is the right default: full diff, blame, bisect, and branch/merge semantics. No new tooling needed for structural versioning.

### 6.2 When to Consider Alternatives

| Situation | Tool | What It Adds |
|-----------|------|-------------|
| KG files exceed git comfort (>100MB) | **DVC** | Git pointers + remote storage; pipeline DAGs |
| Need SPARQL over git-backed RDF | **QuitStore** | "Quads in Git" — SPARQL 1.1 endpoint backed by git, with branch/merge for RDF |
| Need version materialisation | **OSTRICH** | Snapshot+delta RDF storage; reconstruct any historical state via single query |
| Need immutable audit trail | **Fluree** | Ledger-based RDF database with time-travel queries and cryptographic integrity |

**For VASCADIA currently**: Plain git is sufficient. The KG is ~52 nodes with ~100 edges across ~50 YAML files — well within git's comfortable range. If the KG grows to thousands of entities or requires SPARQL access, QuitStore (for RDF) or Oxigraph (embedded triplestore with Python bindings) become relevant.

---

## 7. KG Observability — The Missing Layer

The core finding from this research: **there is no "Datadog for knowledge graphs."** No off-the-shelf platform monitors KG health the way application performance monitoring tracks services. This is the gap between "works in demo" and "works in production" — as WildPinesAI noted, "teams spend 90% of effort on the model and 10% on knowing when it's falling silently."

### What Production KG Monitoring Requires

| Check | What It Catches | Implementation |
|-------|----------------|---------------|
| **Schema conformance** | Type violations, missing required fields | pySHACL or Pydantic in pre-commit |
| **DAG integrity** | Cycles, broken edges, orphaned nodes | `review_prd_integrity.py` (already exists) |
| **Propagation debt** | Decisions flagged for review but not reviewed | Custom query on `requires_review` flags |
| **Freshness** | Stale decisions (not updated for N months) | Git log timestamps per file vs threshold |
| **Completeness** | Decisions with `status: not_started` or missing evidence | Custom YAML scan |
| **Bibliography integrity** | Citation keys that don't resolve | `validate_prd_citations.py` (already exists) |
| **Cross-reference integrity** | Implementation files that reference deleted decisions | `review_knowledge_links.py` (exists but incomplete) |
| **Probability consistency** | Option probabilities that don't sum to 1.0 | Schema validation (exists in pre-commit) |

### Recommended Dashboard Metrics

For a periodic health check (CI cron job or a Prefect flow):

1. **Resolved ratio**: N resolved / N total decisions (currently ~54%)
2. **Propagation debt**: count of `requires_review` flags not yet addressed
3. **Evidence coverage**: decisions with ≥1 resolution_evidence entry vs those without
4. **Freshness distribution**: histogram of days since last update per decision
5. **Orphan count**: nodes in `_network.yaml` with no implementation files listed
6. **Bibliography utilisation**: citation keys in bibliography.yaml that are never referenced by any decision

---

## 8. Pipeline Orchestration with Provenance

### Dagster: The Best Fit for KG Pipelines

Dagster's Software-Defined Assets map naturally to KG files: each YAML/JSON file is an asset with explicit inputs, outputs, and automatic lineage tracking. The asset dependency graph is the KG dependency graph.

**How this maps to VASCADIA**:
- Each domain YAML (`domains/training.yaml`) is a Software-Defined Asset
- Its upstream dependencies are the decision nodes it materialises
- Dagster tracks when each asset was last materialised and whether its inputs have changed
- The Dagster UI provides the observability dashboard we lack

### Prefect (Already in VASCADIA)

Prefect's new "assets" feature (2025) provides lineage within flows. Since VASCADIA already uses Prefect for the 10-flow pipeline, adding KG validation as a Prefect flow with asset tracking would be the lowest-friction path to KG observability.

---

## 9. The Spec-Driven Connection

The structural parallel between spec-driven development and KG maintenance is exact:

| Software (Harness Pattern) | KG (Provenance Pattern) |
|---------------------------|------------------------|
| CLAUDE.md | `_schema.yaml` + `navigator.yaml` |
| Test suite | SHACL shapes / Pydantic validators |
| CI/CD pipeline | Pre-commit hooks + `review_prd_integrity.py` |
| `git blame` | W3C PROV provenance chain |
| Linting | Schema completeness checks |
| Type checking | OWL reasoning + SHACL validation |
| Code review | KG diff review (semantic diff) |
| Spec-driven development | Decision-driven development (PRD → KG → OpenSpec → code) |

The repository-as-source-of-truth principle (from the SGLang "how-to-sglang" case study) applies directly: all expert agents draw knowledge from YAML files inside the repo, with no dependency on external databases or verbal agreements. The navigator.yaml provides structured routing (what the harness community calls "mechanised constraints"), and the domain partition provides progressive disclosure.

**The post-PRD insight**: In a traditional workflow, the PRD is a document that goes stale. In the VASCADIA pattern, the PRD is a live Bayesian DAG that updates as evidence arrives, materialises into deterministic specifications, and generates testable OpenSpec scenarios. The KG *is* the PRD, and provenance tracking ensures every change is traceable to its evidence.

---

## 10. Recommended Implementation Path for VASCADIA

### Phase 1: Immediate (This Sprint)

1. **Add PROV generation to decision updates**: After each `_network.yaml` or decision YAML change, generate a PROV-JSON sidecar file recording the activity, evidence, and agent. Store in `knowledge-graph/provenance/`.
2. **Extend `review_prd_integrity.py`** with propagation debt and freshness checks.
3. **Create a `kg-health` Prefect flow** that runs the extended integrity checks and reports metrics.

### Phase 2: Near-Term (Next Month)

4. **Evaluate LinkML** as a replacement for `_schema.yaml`. If adopted, generate Pydantic models for runtime validation and SHACL shapes for comprehensive checks.
5. **Implement semantic diff**: A script that compares two KG states (git commits) and reports decision-level changes (not line-level), including affected downstream decisions.
6. **Add `prov` library** to generate W3C PROV documents in post-commit hooks.

### Phase 3: When Needed

7. **Dagster integration** if the KG grows beyond what git + Prefect can track.
8. **QuitStore or Oxigraph** if SPARQL access becomes necessary for querying.
9. **RDF-star provenance** if triple-level annotation becomes valuable (e.g., for the manuscript claims layer).

---

## 11. Key References

### Academic Papers

- **Dibowski (2024)** "Full Traceability and Provenance for Knowledge Graphs" — FOIS 2024 (Bosch Research). PROV-STAR ontology for triple-level provenance via RDF-star. [ebooks.iospress.nl/volumearticle/71408](https://ebooks.iospress.nl/volumearticle/71408)
- **Kleinsteuber et al. (2024)** "Managing Provenance Data in Knowledge Graph Management Platforms" — Datenbank-Spektrum 24:43-52. Framework for KG provenance in web portals using W3C PROV-O. [link.springer.com/article/10.1007/s13222-023-00463-0](https://link.springer.com/article/10.1007/s13222-023-00463-0)
- **Ding et al. (2026)** "Bridging Data and Discovery: A Survey on Knowledge Graphs in AI for Science" — National Science Review. Comprehensive survey of scientific KGs and LLM+KG synergy. [doi.org/10.1093/nsr/nwag140](https://doi.org/10.1093/nsr/nwag140)
- **Ahmetaj et al. (2025)** "Common Foundations for SHACL, ShEx, and PG-Schema" — WWW 2025. Formal comparison of shape languages.
- **Arndt et al. (2018)** "Decentralized Collaborative Knowledge Management Using Git" (QuitStore) — JWS. Git-backed SPARQL endpoint.
- **NPCS (2024)** "Native Provenance Computation for SPARQL" — ACM Web Conference. How-provenance via query rewriting; scales to billions of triples.
- **Perez et al.** Systematic review of 25 provenance systems (taxonomy of provenance approaches).

### Tools and Libraries

| Tool | URL | Role |
|------|-----|------|
| `prov` (Python) | pypi.org/project/prov | W3C PROV-DM implementation |
| `pySHACL` | github.com/RDFLib/pySHACL | SHACL validation with CI-friendly exit codes |
| LinkML | linkml.io | YAML schema → Pydantic + SHACL + JSON Schema + docs |
| `rdflib` | rdflib.readthedocs.io | RDF manipulation in Python |
| `kglab` | derwen.ai/docs/kgl | Abstraction over rdflib + pySHACL + NetworkX |
| Git2PROV | github.com/IDLabResearch/Git2PROV | Extract W3C PROV from git history |
| GitLab2PROV | github.com/DLR-SC/gitlab2prov | Extract PROV from GitLab API |
| QuitStore | github.com/AKSW/QuitStore | Git-backed SPARQL endpoint |
| Oxigraph | github.com/oxigraph/oxigraph | Embedded Rust triplestore with Python bindings |
| Fluree | flur.ee | Immutable ledger-based RDF with time-travel |
| Dagster | dagster.io | Pipeline orchestration with Software-Defined Asset lineage |
| Great Expectations | greatexpectations.io | Data quality expectations as code |
| `pydantic-yaml` | pypi.org/project/pydantic-yaml | YAML serialisation for Pydantic models |

---

## 12. Infographic Concept: "KG Observability Stack"

The accompanying infographic for the agentic-development slide deck should visualise the layered observability architecture for a repository-local knowledge graph:

**Layer 1 (bottom, TEAL)**: Git — structural versioning, line-level diff, blame, bisect
**Layer 2**: Schema Validation — SHACL/LinkML/Pydantic as "tests" for the KG (pre-commit)
**Layer 3**: Semantic Provenance — W3C PROV recording who/what/when/why at decision level
**Layer 4**: Health Monitoring — freshness, propagation debt, orphans, completeness (CI cron)
**Layer 5 (top, ROSE)**: Semantic Diff — decision-level change detection across KG states

The visual should show these as horizontal bands with the KG (represented as a small DAG) in the centre, and arrows flowing upward from raw git through increasingly semantic layers of understanding.

This maps to the harness principle: each layer adds constraint and verification, shifting left from "discover the problem in production" to "prevent the problem at commit time."
