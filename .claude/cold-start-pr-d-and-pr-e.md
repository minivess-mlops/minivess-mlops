# Cold-Start: PR-D + PR-E Sequential Execution (TDD)

**Created**: 2026-03-17
**Estimated context**: ~70-80% of 1M window for both PRs
**Checkpoint gate**: If PR-D exceeds 8 inner iterations, STOP and save PR-E for next session.

## Paste this to resume:

```
Execute two XML plans sequentially using the self-learning-iterative-coder skill.

## SESSION PLAN

### Phase 1: PR-D (Deploy Flow Factorial)
- Branch: feat/deploy-flow-factorial (CREATE from main)
- Plan: docs/planning/pr-d-deploy-flow-plan.xml
- Tasks: T1-T5 (5 tasks, issues #825-#829)
- CRITICAL: Read existing deploy_flow.py and champion_evaluator.py FIRST

### Phase 2: PR-E (Cost Reporting) — ONLY if PR-D completes in <8 iterations
- Branch: feat/cost-reporting (CREATE from main AFTER merging PR-D)
- Plan: docs/planning/pr-e-cost-reporting-plan.xml
- Tasks: T1-T3 (3 tasks, issues #830-#832)
- Closes parent issue #795

### Between PRs:
1. Create PR for PR-D, merge via squash
2. Pull main
3. Create new branch feat/cost-reporting
4. Start PR-E execution

## WHAT'S ALREADY DONE (do NOT redo):
- PR-C merged (hL1-ACE loss, factorial config, execution loop, Hydra, checkpoint tracking)
- PR-A merged (ANOVA, interaction plots, LaTeX tables, calibration metrics, Riley instability, TRIPOD)
- The cost appendix table generator exists in biostatistics_tables.py (_generate_cost_appendix_table)
- The checkpoint tracking exists in observability/checkpoint_tracking.py
- LineageEmitter is fully functional in observability/lineage.py
- All task issues created (#825-#832)
- PR-A T8 (#815) still open — integration test deferred, ignore for now

## KEY CONSTRAINTS:
- TDD: RED (tests first) → GREEN (implement) → VERIFY (make test-staging) → FIX → CHECKPOINT → CONVERGE
- Docker IS the execution model. NEVER propose "simpler" local alternatives.
- uv ONLY (--all-extras). No pip/conda. No import re.
- from __future__ import annotations at top of every Python file
- encoding='utf-8' for file ops, pathlib.Path() for paths
- Close GitHub issues (#825-#832) as tasks complete
- ONNX opset 17 for export (PR-D T2)
- No hardcoded cloud rates (PR-E T1) — read from SkyPilot env or config
- Slash-prefix convention for all metric keys (cost/)

## ISSUE TRACKING:
When completing each task:
1. git commit with "Closes #NNN" in message
2. gh issue close NNN --reason completed

PR-D tasks → issues:
  T1 (#825): Champion model selection from factorial results
  T2 (#826): BentoML import chain verification
  T3 (#827): Serving endpoint health check
  T4 (#828): Champion metadata logging for manuscript
  T5 (#829): OpenLineage emission for deploy flow

PR-E tasks → issues:
  T1 (#830): Instance type and cost param logging
  T2 (#831): Spot vs on-demand savings computation
  T3 (#832): Cost appendix integration into biostatistics flow

## KEY FILES TO READ FIRST:
PR-D:
  - src/minivess/orchestration/flows/deploy_flow.py (existing deploy flow)
  - src/minivess/serving/champion_evaluator.py (if exists)
  - src/minivess/serving/bento_model_import.py (existing BentoML import)
  - src/minivess/serving/mlflow_wrapper.py (existing pyfunc models)
  - src/minivess/config/deploy_config.py (ChampionCategory enum)

PR-E:
  - src/minivess/observability/cost_logging.py (may not exist — create)
  - src/minivess/observability/metric_keys.py (cost/ prefix already defined)
  - src/minivess/pipeline/biostatistics_tables.py (_generate_cost_appendix_table already exists from PR-A)
  - src/minivess/pipeline/biostatistics_lineage.py (add cost_summary field)
  - deployment/skypilot/*.yaml (SkyPilot config for cost env vars)

## VERIFICATION COMMANDS:
make test-staging          # Staging tier (<3 min, no model loading)
uv run ruff check src/ tests/   # Lint
uv run mypy src/           # Type check

## KNOWN PRE-EXISTING FAILURES (IGNORE):
- test_kg_completeness.py::test_valid_level_values (#800)
- test_benchmark_cache_integration.py::test_valid_cache_logged_to_mlflow (slash-prefix migration)

## START:
1. git checkout main && git pull origin main
2. git checkout -b feat/deploy-flow-factorial
3. Read docs/planning/pr-d-deploy-flow-plan.xml
4. Read src/minivess/orchestration/flows/deploy_flow.py
5. Read src/minivess/serving/ (all files)
6. Begin T1 RED phase: write test_champion_factorial_selection.py

## OPTIONAL: PR-A T8 Integration Test (#815) — if context remains after PR-E
If both PR-D and PR-E complete with budget remaining, close the deferred T8:
- Branch: feat/biostatistics-factorial-gaps-t8 (CREATE from main)
- Test file: tests/v2/integration/test_biostatistics_factorial_integration.py
- Issue #815 has full implementation spec (6 tests, synthetic data requirements)
- All component functions already exist — this wires them into one integration test
- Key: create synthetic per_volume_data with planted model/loss/interaction effects,
  run compute_factorial_anova(), generate figures and tables, verify outputs exist
- condition_key format: "model__loss" (double underscore, NO regex)
- Test must run in <60s with no GPU or Docker
```
