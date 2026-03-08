# Metalearning: SQLite/In-Memory Fallback Bias

**Date:** 2026-03-08
**Severity:** HIGH — violates project architecture and reproducibility
**Pattern:** Trained-in "tutorial default" overrides explicit project context
**Research context:** amplifying.ai/research/claude-code-picks (2,430 prompts, 4 project types)

---

## What Happened

While planning HPO allocation strategies for Issue #504, I wrote:

```yaml
OPTUNA_STORAGE_URL=       # (empty = in-memory for sequential)
```

```markdown
- `sequential`: `optuna_storage` can be null (in-memory) or SQLite
- `parallel`: MUST be postgresql://...  (SQLite has file-locking failures)
```

**The project already has PostgreSQL running** — `POSTGRES_DOCKER_HOST=postgres` used by
MLflow, Prefect, and Langfuse. There is zero justification for SQLite or in-memory.
I invented a two-tier system that does not exist in this project's architecture.

---

## Root Cause Analysis

### 1. Optuna Tutorial Bias (Training Data)

Every Optuna tutorial starts with:
```python
study = optuna.create_study()  # in-memory, no storage arg
```

This is massively over-represented in training data. My first instinct reproduces
the tutorial default, not the project production pattern.

### 2. "Complexity Proportionality" Heuristic (False Economy)

Implicit rule fired: *"simple use case → simpler tool."*
One sequential trial feels light → matched to lightweight storage (in-memory/SQLite).
This is wrong because:
- PostgreSQL is already in the stack (zero marginal cost to use it)
- Adding SQLite creates a SECOND database system — MORE complexity, not less
- In-memory = ephemeral = not reproducible = wrong for any research run

### 3. Contextual Amnesia

I had read `.env.example` and saw `POSTGRES_DB_MLFLOW`, `POSTGRES_DB_PREFECT`,
`POSTGRES_DB_LANGFUSE`. I knew PostgreSQL was the standard. I still wrote
`OPTUNA_STORAGE_URL=` (empty). The trained-in default fired despite evidence.

Worse: in the SAME document I correctly wrote:
> "SQLite has file-locking failures under concurrent writers."
> "PostgreSQL is already in the compose stack."

Then immediately proposed SQLite for the "simple" case. Confident writing masked
the self-contradicting logic.

### 4. amplifying.ai Research Connection

From the study (2,430 prompts, 3 models, 4 project types):
- PostgreSQL: **58.4%** of database picks
- SQLite: **16%** of database picks

Even in open-ended prompts I prefer PostgreSQL 3.6x over SQLite. Yet with explicit
project context (PostgreSQL in stack), I defaulted to SQLite for the "simple" case.

Key finding from the research that describes this failure mode:
> "Models select tools **proportionate to stated requirements** rather than
> defaulting to enterprise solutions; lightweight, framework-native approaches
> consistently win over heavyweight dependencies."

The heuristic is correct in context-free settings. It is wrong when the project
has an explicit technology choice that applies regardless of scale.

---

## The Fix (Applied 2026-03-08)

**Rule — absolute, no exceptions:**
> **PostgreSQL is the ONLY database in this project. No SQLite. No in-memory.**
> If PostgreSQL is in the stack, ALL database needs use PostgreSQL: Optuna,
> MLflow, Prefect, Langfuse, application data, HPO studies.

`.env.example` default:
```
OPTUNA_STORAGE_URL=postgresql+psycopg2://minivess:minivess_secret@postgres:5432/optuna
```

`HPOEngine.from_config()` for ALL strategies (sequential, parallel, hybrid):
```python
if not storage_url.startswith(("postgresql://", "postgresql+psycopg2://")):
    raise ValueError(
        "PostgreSQL storage required for Optuna studies. "
        "Set OPTUNA_STORAGE_URL to a postgresql:// URL. "
        "SQLite and in-memory are not supported in this project."
    )
```

---

## Broader Principle

When a project has an explicit technology choice, that choice applies to ALL
instances of that category — not just the "complex" ones.

| Category | This Project's Standard | Never fallback to |
|----------|------------------------|-------------------|
| Database | PostgreSQL | SQLite, in-memory, DuckDB for writes |
| Package manager | uv | pip, conda, poetry |
| Orchestration | Prefect + Docker | standalone scripts |
| Tracking | MLflow | stdout logs only |
| Config source | `.env.example` | hardcoded literals |

**Detection check before any storage/database recommendation:**
```
1. grep .env.example for POSTGRES → found? → postgresql:// for EVERYTHING
2. grep docker-compose for postgres service → found? → same as above
3. Is this a "simple" case? → irrelevant if PostgreSQL is already in the stack
4. "In-memory is fine here"? → NO. In-memory = not reproducible.
```
