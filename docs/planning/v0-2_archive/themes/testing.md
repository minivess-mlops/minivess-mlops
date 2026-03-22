---
theme: testing
health_score: 70
doc_count: 13
created: "2026-03-22"
kg_domain: knowledge-graph/domains/testing.yaml
archive_dir: docs/planning/v0-2_archive/original_docs
status: archived-with-synthesis
---

# Theme: Testing

Test tiers, staging/prod splits, E2E verification, GPU instance tests, model testing
best practices, and the evolution from ad-hoc test runs to a principled three-tier
architecture with 5500+ tests.

---

## Key Scientific Insights

1. **Three-tier test architecture solves the GPU-gated-CI problem.** Academic ML repos
   face a unique challenge: model loading and GPU tests are expensive and slow, but
   skipping them entirely hides bugs that only manifest on cloud GPUs. The three-tier
   split (staging < 3 min / prod ~10 min / GPU = external only) balances PR velocity
   with correctness. The staging tier is pure code validation (no model instantiation),
   prod includes model loading but not GPU-heavy forward passes, and GPU tests run
   exclusively on external instances (RunPod RTX 4090, GCP L4).

2. **The "model construction gap" wasted cloud dollars.** The 2026-03-20 debug factorial
   run found 14/26 conditions failing due to bugs (LoRA Conv2d type error, missing
   mamba-ssm import, Pydantic max_epochs=0) that required zero GPU time to diagnose.
   All three bugs were catchable by CPU-only unit tests in < 1 second. This motivated
   the `@pytest.mark.model_construction` marker and a proposed `make test-models` tier
   that tests adapter instantiation, LoRA application, and config validation without
   training loops.

3. **MLflow as inter-flow contract is the critical E2E seam.** The 2026-03-09 debug
   run showed all 7 flows returning "OK" but producing zero real results. The root
   cause: train_flow never logged `checkpoint_dir` tags to MLflow, so post-training
   received 0 checkpoints and analysis evaluated 0 models. This "silent success"
   pattern -- where every component passes locally but the integration is broken --
   is the dominant failure mode in orchestrated ML pipelines.

4. **TDD with failure-triage protocol catches bugs at the right abstraction level.**
   Two recurring anti-patterns were identified: (a) silent dismissal of pre-existing
   failures ("not related to current changes") and (b) whac-a-mole serial fixing
   (fix one, re-run, fix next). The TDD skill upgrade (v2.1 to v3.0) introduced
   the GATHER-CATEGORIZE-PLAN-FIX-VERIFY protocol and a "tokens upfront" philosophy
   (30% reading / 70% implementing, not 5%/95%).

5. **Code review via multi-agent specialist panels scales quality assessment.** The
   two-pass code review (662 tests at 1st pass, 813 at 2nd pass) used 6 parallel
   Claude Code sub-agents (dead code, duplicate code, test coverage, decoupling,
   reproducibility, API consistency). This raised the overall quality score from
   7.5/10 to 8.5/10 and identified 31 remaining issues, primarily in adapter
   boilerplate duplication.

---

## Architectural Decisions Made

| Decision | Winner | Rationale | KG Node |
|----------|--------|-----------|---------|
| Test framework | pytest + Hypothesis | pytest for unit/integration, Hypothesis for property-based testing | `test_framework` |
| Test tier strategy | Three-tier (staging/prod/GPU) | Staging < 3 min (no models), Prod ~10 min (all except GPU), GPU (external instances only) | `test_tier_strategy` |
| SAM3 test location | `tests/gpu_instance/` only | SAM3 model loading banned from CI; tests validate new instance types only | CLAUDE.md Rule, metalearning |
| E2E verification | Triple: Prefect API + MLflow artifacts + container logs | Single-source verification insufficient for orchestrated pipelines | `e2e-testing-1st-pass.xml` |
| Validation framework | Pandera + Great Expectations + whylogs | Deepchecks rejected (2D only, incompatible with 3D MONAI CacheDataset) | `validation-pipeline-plan.md` |
| Pre-debug QA | Debug run = full production minus epochs/data/folds | Never reduce factors, flows, or baselines in debug scope | CLAUDE.md Rule 27 |
| Code review method | Multi-agent specialist panels (6 agents) | Independent reviewers avoid groupthink; 151 tests added through remediation | `code-review-report-v0-2.md` |

---

## Implementation Status

| Document | Status | Key Deliverable | Implementation Evidence |
|----------|--------|-----------------|----------------------|
| `code-review-report-v0-2.md` | **Implemented** | 1st-pass review: 42 issues, R1-R4 remediation | 662 to 813 tests (+151). All critical issues resolved. |
| `code-review-report-v0-2-2nd-pass.md` | **Reference** | 2nd-pass review: 31 remaining issues | Quality score 8.5/10. Adapter boilerplate duplication identified as largest gap. |
| `debug-training-testing-plan.xml` | **Partial** | Quasi-E2E mechanics validation for train-post_training-analyze chain | Phase -1 blockers (UPSTREAM_EXPERIMENT propagation, missing model configs) addressed. Full debug pipeline partially validated. |
| `e2e-testing-1st-pass.xml` | **Partial** | 5-phase E2E plan covering all 5 local-GPU models | Phase 0 (inter-flow seam fixes) implemented. Phases 1-4 (individual flows, MLflow contracts, BentoML/Grafana, pytest-docker) partially complete. |
| `e2e-testing-user-prompt.md` | **Reference** | Verbatim user prompt for E2E test plan alignment | 24 Q&A decisions captured. Serves as alignment record. |
| `final-verification-report.md` | **Implemented** | PRD gap analysis: +20 bibliography entries, DataLad assessment | DVC retained as winner (0.35 posterior). DataLad added as experimental option (0.10). |
| `pre-debug-qa-verification-plan.md` | **Partial** | Pre-debug QA: 24-condition factorial, config cleanup | GROUP A (config cleanup) and GROUP B (aux_calib wiring) in progress. SkyPilot debug YAML created. |
| `prod-staging-gcp-doublecheck-code-review.xml` | **Partial** | GCP infrastructure verification + spot recovery | GCS, Cloud SQL, GAR, Cloud Run deployed. MLflow multipart upload verified. GPU pricing analysis completed. |
| `pytorch-model-testing-best-practices-report.md` | **Reference** | Research report on PyTorch model testing patterns | Patterns documented: shape tests, determinism, NaN/Inf detection, gradient flow, LoRA. Informed `@pytest.mark.model_construction` proposal. |
| `staging-prod-remote-test-suite-splits.xml` | **Partial** | 5-tier test redesign (staging/prod/prod-remote) | Staging and prod tiers implemented in Makefile. `requires_services` marker proposed but not yet added. Prod-remote tier planned. |
| `tdd-skill-upgrade-plan.md` | **Implemented** | TDD skill v3.0: failure-triage, tokens-upfront, agentic evals | 8 work items. Phase 2 failure-triage protocol implemented. CLAUDE.md Rules 23-24 added. |
| `test-suite-improvement-report.md` | **Planned** | `make test-models` tier for model construction testing | `@pytest.mark.model_construction` marker proposed. Conditional triggering via Makefile or pre-commit hook designed. Not yet implemented. |
| `validation-pipeline-plan.md` | **Partial** | Pandera schemas, GE suites, drift detection, whylogs profiling | `src/minivess/validation/` exists with schemas.py, expectations.py, drift.py, gates.py, profiling.py. Tests partially written. |

---

## Cross-References

- **KG domain**: `knowledge-graph/domains/testing.yaml` -- 2 resolved decisions (test_framework, test_tier_strategy)
- **KG metalearning**: `.claude/metalearning/2026-03-07-silent-existing-failures.md`, `.claude/metalearning/2026-03-10-silent-test-skips-optional-deps.md`
- **Makefile targets**: `make test-staging`, `make test-prod`, `make test-gpu`, `make test-e2e`
- **pytest config**: `pytest-staging.ini`, `pyproject.toml` markers section
- **Test directories**: `tests/v2/unit/` (514 files), `tests/v2/integration/` (47 files), `tests/gpu_instance/` (15 files)
- **Architecture theme**: `prefect-flow-connectivity.md` (inter-flow contract is the E2E testing seam)
- **Deployment theme**: `pr-d-deploy-flow-plan.xml` (deploy flow verification is an E2E test concern)
- **CLAUDE.md rules**: Rule 20 (zero tolerance for failures), Rule 23 (never fix serially), Rule 24 (tokens upfront), Rule 27 (debug = full production), Rule 28 (zero silent skips)

---

## Constituent Documents

1. `code-review-report-v0-2.md` -- 1st-pass multi-agent deep review (2026-02-25). 6 specialist agents, 42 issues, R1-R4 remediation. Quality baseline 7.5/10.
2. `code-review-report-v0-2-2nd-pass.md` -- 2nd-pass multi-agent review (2026-02-25). 31 remaining issues. Adapter boilerplate and ONNX export duplication as primary gaps.
3. `debug-training-testing-plan.xml` -- Dual-mandate debug plan: (A) mechanics validation, (B) bug fixing with TDD. 3 pre-run blockers identified by reviewer agents.
4. `e2e-testing-1st-pass.xml` -- 5-phase E2E testing plan (2026-03-10). Covers all 5 local-GPU models, 2 epochs, 3 folds. Champion metric: val_compound_masd_cldice.
5. `e2e-testing-user-prompt.md` -- Verbatim user Q&A (24 decisions) aligning E2E scope. Key decisions: VesselNN excluded, Docker via Prefect, pytest-docker plugin.
6. `final-verification-report.md` -- PRD gap analysis (2026-02-24). +20 bibliography entries. DataLad assessed vs DVC. 5 genuine novelty gaps identified.
7. `pre-debug-qa-verification-plan.md` -- Pre-debug QA plan (2026-03-19). 24-condition factorial matrix. 9 reviewer corrections. GROUP A-H task groups.
8. `prod-staging-gcp-doublecheck-code-review.xml` -- GCP infrastructure verification (2026-03-15). MLflow multipart upload, spot recovery, GPU pricing analysis.
9. `pytorch-model-testing-best-practices-report.md` -- Research report (2026-03-20). Shape tests, determinism, NaN/Inf, gradient flow, LoRA, VRAM-tiered execution.
10. `staging-prod-remote-test-suite-splits.xml` -- 5-tier test redesign (2026-03-19). Staging/prod/prod-remote/gpu separation with clear infrastructure requirements.
11. `tdd-skill-upgrade-plan.md` -- TDD skill v3.0 upgrade (2026-03-18). 8 work items: failure-triage, tokens-upfront, agentic evals, Ralph Loop integration.
12. `test-suite-improvement-report.md` -- Model-specific test tier proposal (2026-03-20). `@pytest.mark.model_construction` marker. Motivated by $1-2 cloud waste on CPU-catchable bugs.
13. `validation-pipeline-plan.md` -- Validation pipeline plan (2026-02-24). Pandera, GE, whylogs, drift detection. Deepchecks rejected for 3D incompatibility.
