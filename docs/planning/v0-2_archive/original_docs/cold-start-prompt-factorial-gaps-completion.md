# Cold-Start Prompt: Complete Factorial Pipeline Gaps

**Date**: 2026-03-21
**Branch**: `test/debug-factorial-run`
**Session goal**: Close all remaining gaps for the 6-factor factorial debug run

---

## How to Use This Prompt

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-factorial-gaps-completion.md
Branch: test/debug-factorial-run." --dangerously-skip-permissions
```

---

## 1. Context: What Was Done (2026-03-21 Session)

### 1a. RAM Crash Fix (COMPLETE — all 3 phases)
- **Root cause**: Missing mocks in `test_biostatistics_flow.py` (3 unmocked tasks)
- **Fixes applied**:
  - Added 3 missing `@patch` decorators + real `BiostatisticsConfig` (no more MagicMock)
  - `resource.setrlimit(RLIMIT_AS, 32 GB)` in `tests/v2/unit/conftest.py`
  - AST-based mock count validation test (catches future regressions)
  - `MemoryMonitor` ported from foundation-PLR → `src/minivess/observability/memory_monitor.py`
  - Permutation test refactored: pre-allocated buffers, in-place shuffle, periodic GC
- **Report**: `docs/planning/ram-issue-mock-data-biostatistics-duckup-report.md`
- **Metalearning**: `.claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md`

### 1b. Post-Training Flow Double-Check (COMPLETE — Phases 0-4)
- **Critical fix**: `EXPERIMENT_POST_TRAINING` changed from `"minivess_post_training"` to `"minivess_training"` (synthesis Part 2.3 — same experiment for Analysis Flow discovery)
- **Files changed**: `constants.py`, `post_training_config.py`, `analysis_flow.py`, `configs/post_training/default.yaml`
- **Tests added**: 20 new tests (tag schema, checkpoint fallback chain, factorial YAML factors)
- **Constant test updated**: `test_no_duplicate_values` → `test_no_unexpected_duplicate_values` (allows intentional sharing)

### 1c. Analysis Flow Double-Check (PARTIAL — Priorities 1-2 done)
- **Implemented**: `_resolve_ensemble_strategies()` in `analysis_flow.py` — reads from factorial YAML
- **Tests added**: 15 new tests (factorial YAML strategies, experiment names, tag schema, layered ANOVA factor groups)
- **Verified**: Existing ensemble builder (4 strategies), logit-level averaging, per-volume metrics all have coverage

### 1d. Test Suite State
```
5530 passed, 3 skipped, 0 failed (4m 12s)
```
**Skips**: 1 CTK config.toml (#884), 2 acceptable hardware-gated

---

## 2. Uncommitted Changes (MUST COMMIT FIRST)

```
Modified (12 files):
  configs/post_training/default.yaml
  src/minivess/config/biostatistics_config.py
  src/minivess/config/post_training_config.py
  src/minivess/orchestration/constants.py
  src/minivess/orchestration/flows/analysis_flow.py
  src/minivess/orchestration/flows/biostatistics_flow.py
  src/minivess/pipeline/biostatistics_specification_curve.py
  tests/unit/orchestration/test_constants.py
  tests/v2/unit/conftest.py
  tests/v2/unit/test_biostatistics_flow.py
  tests/v2/unit/test_post_training_config.py
  tests/v2/unit/test_post_training_mlflow.py

New (8 files):
  .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md
  docs/planning/ram-issue-mock-data-biostatistics-duckup-report.md
  src/minivess/observability/memory_monitor.py
  tests/v2/unit/test_memory_monitor.py
  tests/v2/unit/test_analysis_flow_factorial_tags.py
  tests/v2/unit/test_analysis_flow_factorial_yaml.py
  tests/v2/unit/test_post_training_checkpoint_resolution.py
  tests/v2/unit/test_post_training_factorial_yaml.py
```

**Action**: Commit all changes before starting new work. Suggested message:
```
fix: RAM crash + post-training/analysis factorial pipeline verification

- Fix 62 GB RAM crash: add missing mocks, MemoryMonitor, memory cap
- Fix EXPERIMENT_POST_TRAINING: minivess_post_training → minivess_training
- Add _resolve_ensemble_strategies() for factorial YAML-driven analysis
- 43 new tests, all 5530 passing
```

---

## 3. Open Issues — Execution Order

### P0-Critical (MUST do next)

| Issue | Title | Scope | Est. |
|-------|-------|-------|------|
| **#885** | Post-training: per-method MLflow runs for factorial discovery | Architectural change to `post_training_flow.py` — create SEPARATE MLflow runs per method variant (e.g., `dynunet__cbdice__fold0__swa`) instead of one aggregate run | 3-4 hr |

**Why P0**: Without per-method runs, Biostatistics cannot discover post-training variants as separate factorial conditions. The 5-way ANOVA (Layers A+B) is impossible.

**Implementation approach**:
1. `run_factorial_post_training()` currently returns metadata dicts → change to also create MLflow runs
2. Each run in `minivess_training` experiment with:
   - `run_name`: `{model}__{loss}__{calib}__{fold}__{method}`
   - Tags: `post_training_method`, `upstream_training_run_id`, inherited `model_family`/`loss_function`/`fold_id`/`with_aux_calib`
3. Update `_factorial_result()` to return `mlflow_run_id`
4. Tests: verify N runs created, correct naming, tag inheritance

### P1-High (should do soon — unblock full factorial)

| Issue | Title | Scope | Est. |
|-------|-------|-------|------|
| **#889** | EnsembleBuilder: include post-training variants | `builder.py`: query for `post_training_method` tag, include SWA/calibrated models as ensemble candidates | 2 hr |
| **#888** | Zero-shot baseline discovery and tagging | New `discover_zero_shot_baselines()` task, `is_zero_shot` tag, VesselFM DeepVess-only constraint | 2 hr |
| **#886** | UQ from deep ensembles | `predict_with_uncertainty()` — entropy/MI decomposition (Lakshminarayanan 2017) | 4 hr |
| **#887** | Debug vs production storage policy | Conditional: debug=summary stats, production=full 5D maps (~50MB/vol) | 2 hr |
| **#884** | CTK config.toml skip fix | Gate test with `@pytest.mark.gpu` or mock for non-GPU | 30 min |

### Execution Order (recommended):

```
1. Commit current changes
2. #885 — per-method MLflow runs (P0, unblocks everything)
3. #889 — post-training variants in EnsembleBuilder (depends on #885)
4. #888 — zero-shot baseline discovery
5. #886 — UQ from deep ensembles
6. #887 — storage policy
7. #884 — CTK skip fix (quick win)
8. Run integration test: debug-factorial-local-post-analysis-biostats.xml
```

---

## 4. XML Plans Status

| Plan | Status | Remaining |
|------|--------|-----------|
| `post-training-flow-debug-double-check.xml` | **80% done** | Phase 1 T1.1-T1.4 (per-method runs = #885), Phase 5 (needs cloud data) |
| `analysis-flow-debug-double-check.xml` | **40% done** | Phase 1 T1.6 (#889), Phase 3 (#886), Phase 4 T4.2 (6-factor tags after #885), Phase 6 (#887), Phase 7 (integration) |
| `debug-factorial-local-post-analysis-biostats.xml` | **Not started** | Depends on above two being complete + cloud artifacts (Issue #882) |

---

## 5. Mandatory Files to Read Before Any Action

```
1. CLAUDE.md                                          # Project rules (29 rules)
2. docs/planning/cold-start-prompt-factorial-gaps-completion.md  # THIS FILE
3. docs/planning/ram-issue-mock-data-biostatistics-duckup-report.md  # RAM fix context
4. docs/planning/post-training-flow-debug-double-check.xml  # Post-training plan
5. docs/planning/analysis-flow-debug-double-check.xml  # Analysis plan
6. docs/planning/intermedia-plan-synthesis-pre-debug-run.md  # Synthesis (authoritative)
7. src/minivess/orchestration/flows/post_training_flow.py  # 561 lines, current impl
8. src/minivess/orchestration/flows/analysis_flow.py  # 2078+ lines, current impl
9. src/minivess/config/factorial_config.py  # Composable factorial YAML parser
10. configs/factorial/debug.yaml  # Debug factorial definition (384 conditions)
```

---

## 6. Key Architecture Decisions (Already Resolved)

| Decision | Resolution | Source |
|----------|-----------|--------|
| Post-training experiment | SAME as training (`minivess_training`) | Synthesis Part 2.3 |
| Analysis experiment | SEPARATE (`minivess_evaluation`) | Synthesis Part 2.3 |
| Ensemble averaging | LOGIT-level, not metric-level | Synthesis Part 1.2 Layer C |
| Debug scope | Full 6 factors, fewer levels (384 conditions) | CLAUDE.md Rule #27 |
| Factorial config | Composable YAML (`configs/factorial/*.yaml`) | Already implemented |
| Factor name derivation | Auto-derived from YAML, zero hardcoded | Already implemented |
| Memory safety | MemoryMonitor + `resource.setrlimit` + periodic GC | Already implemented |

---

## 7. TDD Protocol

Use the self-learning-iterative-coder skill for all implementation:

```
/self-learning-iterative-coder <task description>
```

Constraints:
- `make test-staging` must pass (5530+ tests, <5 min)
- `from __future__ import annotations` in every Python file
- `pathlib.Path` for all paths
- `encoding='utf-8'` for all file operations
- No `import re` for structured data (Rule #16)
- No hardcoded parameters (Rule #29)
- Read ALL existing tests before writing new ones (Rule #11)

---

## 8. Verification Commands

```bash
# Quick: specific test files
MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/test_post_training_flow.py -v --timeout=60

# Staging gate (must pass before any commit)
make test-staging

# Lint
uv run ruff check src/ tests/

# Memory profile (verify no OOM)
/usr/bin/time -v make test-staging 2>&1 | grep "Maximum resident"
```

---

## 9. What NOT to Do

- Do NOT add cloud providers beyond RunPod + GCP
- Do NOT enable GitHub Actions CI triggers
- Do NOT hardcode experiment names — use `constants.py`
- Do NOT create per-method MLflow runs without inheriting ALL upstream tags
- Do NOT bypass Docker context checks (use `MINIVESS_ALLOW_HOST=1` only in tests)
- Do NOT assume the RAM crash is fully solved — always monitor memory in long-running tests
- Do NOT run `make test-staging` without first cleaning `file:*` MLflow pollution:
  ```bash
  rm -rf /home/petteri/Dropbox/github-personal/minivess-mlops/file:*
  ```
