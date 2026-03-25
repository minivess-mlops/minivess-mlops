# Label Harmonization Plan

## Problem

55 labels with significant overlap. Many single-use or redundant labels. Priority labels are inconsistent (`P0` vs `P0-critical`, `P1` vs `P1-high`).

## Target: 20 Coherent Labels

### Priority (4) — standardized names
| Keep | Merge Into It | Count |
|------|---------------|-------|
| `P0-critical` | `P0` (3 issues) | 121 |
| `P1-high` | `P1` (17 issues) | 358 |
| `P2-medium` | — | 168 |
| `P3-low` | — | 18 |

### Type (4)
| Keep | Merge Into It | Count |
|------|---------------|-------|
| `bug` | — | 82 |
| `enhancement` | — | 263 |
| `refactor` | `tech-debt` (19) | 35 |
| `documentation` | — | 29 |

### Domain (8) — map to milestones
| Keep | Merge Into It | Count |
|------|---------------|-------|
| `infrastructure` | `docker` (33), `prefect` (30), `ci-cd` (8), `config` (15), `hydra` (2), `trigger` (1) | 191 |
| `training` | `tdd` (3) | 48 |
| `data` | `external-data` (4), `real-data` (8), `annotation` (4) | 36 |
| `models` | — | 72 |
| `observability` | `mlflow` (38), `monitoring` (19), `metrics` (13) | 85 |
| `compliance` | `security` (12) | 34 |
| `testing` | `validation` (12), `integration` (1), `pipeline-verification` (4), `reproducibility` (4) | 68 |
| `deployment` | `serving` (10) | 28 |

### Scope (4)
| Keep | Merge Into It | Count |
|------|---------------|-------|
| `science` | `research` (44), `uncertainty` (6), `graph-topology` (28), `ensemble` (9), `manuscript` (8), `paper-artifacts` (7), `visualization` (8) | 119 |
| `gpu-runs` | — | 1 |
| `blocked` | — | 2 |
| `dashboard` | — | 5 |

### Delete (no merge needed)
| Label | Reason | Count |
|-------|--------|-------|
| `phase-0` | Historical, no value | 10 |
| `phase-complete` | Historical, replaced by milestones | 32 |
| `automated-backfill` | Internal automation artifact | 18 |
| `R6-remediation` | One-time audit, complete | 8 |
| `v0.1-legacy` | Historical | 13 |
| `v2-foundation` | Everything is v2 | 5 |
| `post-publication` | Use P3-low instead | 12 |
| `knowledge-graph` | Niche, use `infrastructure` | 10 |
| `agents` | Niche, use `infrastructure` | 2 |
| `analysis` | Use `observability` or milestone | 28 |

## Execution Steps

1. **Merge priority labels**: `P0` → `P0-critical`, `P1` → `P1-high`
2. **Merge domain labels**: docker/prefect/ci-cd → infrastructure, mlflow/monitoring → observability, etc.
3. **Merge type labels**: tech-debt → refactor, security → compliance
4. **Merge science labels**: research/uncertainty/graph-topology/ensemble/manuscript/visualization → science
5. **Delete obsolete labels**: phase-0, phase-complete, automated-backfill, R6-remediation, v0.1-legacy, v2-foundation
6. **Rename for consistency**: All lowercase, hyphenated
