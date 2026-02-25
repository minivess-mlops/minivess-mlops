# R6 Remediation Plan — 2nd-Pass Code Review Execution

> **Created:** 2026-02-25
> **Baseline:** 868 tests passing (after R5 remediation)
> **Source:** `code-review-report-v0-2-2nd-pass.md` + 3 specialist agent reports
> **Issues:** #52–#59 (8 issue packages)

---

## Background

### Review History

| Phase | What | Tests | Rating |
|-------|------|-------|--------|
| v0.1 | 1st-pass code review (42 issues) | 662 | 7.5/10 |
| R1-R4 | 1st-pass remediation | 813 | 8.5/10 |
| R5 | 2nd-pass remediation (5 tasks) | 868 | 8.5/10 |
| **R6** | **Remaining 2nd-pass items** | **Target: ~940** | **Target: 9.0/10** |

### What R5 Already Fixed

| ID | Description | Tests |
|----|-------------|-------|
| R5.1 | Checkpoint/save/export → ModelAdapter base (~350 LOC removed) | +16 |
| R5.2 | ONNX export consolidated with R5.1 | (above) |
| R5.6 | Agent graph node function unit tests | +24 |
| R5.19 | CUDA determinism (`cudnn.deterministic=True`) | +4 |
| R5.20 | DataLoader worker seeding (`_worker_init_fn`) | +4 |
| R5.26 | Factory naming → `build_*` prefix | +3 |
| R5.27 | `__all__` exports in exceptions.py | +4 |

---

## Execution Order

Issues are ordered by **dependency chain** and **risk reduction**:

```
Sprint 1 (HIGH — correctness & safety)
├── #55 R6.4: Reproducibility Phase 2 (version pins, seeds, checksums)
├── #53 R6.2: Test coverage — loss, WeightWatcher, numerics
└── #54 R6.3: Test coverage — config, data, dataclasses

Sprint 2 (HIGH → MEDIUM — structure)
├── #52 R6.1: Adapter DRY Phase 2 (output, config, comma)
├── #57 R6.6: Validation gate refactoring
└── #56 R6.5: Dependency injection (Trainer, BentoML, loader)

Sprint 3 (MEDIUM — architecture)
├── #58 R6.7: Configuration architecture (adapter hypers, defaults)
└── #59 R6.8: Code hygiene (god modules, docstrings, dict ordering)
```

---

## Sprint 1: Correctness & Safety

**Goal:** Ensure reproducible results and cover critical test gaps.
**Est. new tests:** ~48

### #55 R6.4: Reproducibility Phase 2
| Task | Files | Tests | Priority |
|------|-------|-------|----------|
| R5.21: Dep version upper bounds | `pyproject.toml` | 0 | HIGH |
| R5.22: Domain randomization seed | `data/domain_randomization.py` | 2 | HIGH |
| R5.23: VesselFM weight checksums | `adapters/vesselfm.py` | 2 | HIGH |
| R5.25: MAPIE seed configurable | `ensemble/mapie_conformal.py` | 1 | LOW |
| Datetime style fixes | 3 compliance/observability files | 0 | LOW |

### #53 R6.2: Test Coverage — Loss & Quality Gates
| Task | Files | Tests | Priority |
|------|-------|-------|----------|
| R5.7: VesselCompoundLoss | `pipeline/loss_functions.py` | 8 | HIGH |
| R5.8: WeightWatcher | `ensemble/weightwatcher.py` | 6 | HIGH |
| R5.11: Numerical edge cases | metrics, calibration, drift, qc | 8 | MEDIUM |

### #54 R6.3: Test Coverage — Config & Data
| Task | Files | Tests | Priority |
|------|-------|-------|----------|
| R5.9: Config validation | `config/models.py` | 5 | MEDIUM |
| R5.10: Data loading edge cases | `data/loader.py` | 6 | MEDIUM |
| R5.12: Dataclass structure | 5 dataclass files | 10 | MEDIUM |

**Sprint 1 totals:** ~48 new tests, 868 → ~916

---

## Sprint 2: Structure & Deduplication

**Goal:** Reduce remaining code duplication and improve testability.
**Est. new tests:** ~18

### #52 R6.1: Adapter DRY Phase 2
| Task | Files | Tests | LOC saved |
|------|-------|-------|-----------|
| R5.3: `_build_output()` | 7 adapters + base | 2 | ~60 |
| R5.5: `_build_config()` | 6 adapters + base | 2 | ~78 |
| R5.28: comma.py naming | `adapters/comma.py` | 2 | 0 |

### #57 R6.6: Validation Gate Refactoring
| Task | Files | Tests | LOC saved |
|------|-------|-------|-----------|
| R5.4: `_validate_with_schema()` | `gates.py`, `ge_runner.py` | 4 | ~47 |

### #56 R6.5: Dependency Injection
| Task | Files | Tests |
|------|-------|-------|
| R5.14: Trainer DI | `pipeline/trainer.py` | 4 |
| R5.13: BentoML DI | `serving/bento_service.py` | 2 |
| R5.17: Loader transforms | `data/loader.py` | 2 |

**Sprint 2 totals:** ~18 new tests, ~916 → ~934, ~185 LOC reduced

---

## Sprint 3: Architecture & Hygiene

**Goal:** Improve configurability and code consistency.
**Est. new tests:** ~6

### #58 R6.7: Configuration Architecture
| Task | Files | Tests |
|------|-------|-------|
| R5.15: Adapter hyperparameters | 3 adapter files + config | 4 |
| R5.16: Centralize defaults | New `config/defaults.py` | 2 |

### #59 R6.8: Code Hygiene
| Task | Files | Tests |
|------|-------|-------|
| R5.18: God module splits | 3 files (if warranted) | 0 |
| R5.24: Sorted dict logging | `observability/tracking.py` | 0 |
| R5.29: Docstring style | Multiple files | 0 |

**Sprint 3 totals:** ~6 new tests, ~934 → ~940

---

## Summary

| Sprint | Issues | New Tests | LOC Impact | Focus |
|--------|--------|-----------|------------|-------|
| 1 | #55, #53, #54 | ~48 | +tests only | Correctness & safety |
| 2 | #52, #57, #56 | ~18 | -185 LOC | Structure & DI |
| 3 | #58, #59 | ~6 | Neutral | Architecture & style |
| **Total** | **8 issues** | **~72** | **-185 LOC** | **868 → ~940 tests** |

---

## TDD Workflow (Mandatory)

Every task follows the self-learning-iterative-coder skill:

```
1. RED:    Write failing tests first
2. GREEN:  Implement minimum code to pass
3. VERIFY: pytest + ruff + mypy
4. COMMIT: One commit per issue (or per sub-task for large issues)
```

---

## Definition of Done

- [ ] All 8 issues (#52–#59) closed
- [ ] Test count ≥ 930
- [ ] `uv run pytest tests/ -x -q` — 0 failures
- [ ] `uv run ruff check src/ tests/` — 0 new errors
- [ ] Code review rating ≥ 9.0/10
- [ ] `docs/claude-code-patterns.md` updated with new patterns
- [ ] 3rd-pass review confirms no regressions
