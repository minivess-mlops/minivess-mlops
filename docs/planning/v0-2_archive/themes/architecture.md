---
theme: architecture
health_score: 68
doc_count: 13
created: "2026-03-22"
kg_domain: knowledge-graph/domains/architecture.yaml
archive_dir: docs/planning/v0-2_archive/original_docs
status: archived-with-synthesis
---

# Theme: Architecture

Hydra-zen/Dynaconf dual config system, Prefect 5-flow topology, inter-flow connectivity
via MLflow contracts, adaptive compute profiles, script consolidation, and the 6-layer
knowledge architecture (PRD-KG-OpenSpec).

---

## Key Scientific Insights

1. **Dual config architecture resolves the experiment-vs-deployment tension.** Training
   experiments need composable hyperparameter sweeps (Hydra-zen), while deployment
   needs environment-specific settings with secrets (Dynaconf + .env). Attempting to
   unify these into a single system creates friction in both directions. The resolved
   architecture: Hydra-zen for experiment sweeps (`configs/base.yaml` + groups),
   Dynaconf for deployment (`configs/deployment/*.toml`), and `.env.example` as the
   single source of truth for secrets and URLs. The three systems are complementary,
   not competing.

2. **MLflow as the ONLY inter-flow communication channel eliminates state coupling.**
   Flows communicate exclusively through MLflow tags, metrics, and artifacts. No shared
   filesystem paths (beyond volume-mounted artifact directories that MLflow references),
   no direct function calls between flow containers, no Prefect artifacts as inter-flow
   data. This is the critical architectural insight that enables Docker-per-flow
   isolation: each flow container reads upstream state from MLflow and writes downstream
   state to MLflow. The FlowContract abstraction (`flow_contract.py`) codifies
   `find_upstream_run()`, `log_flow_completion()`, and `resolve_experiment_name()`.

3. **Config composability requires separating infrastructure from training behavior.**
   The 2026-03-15 `sam3_hybrid val_loss=NaN` incident across 4 GCP jobs exposed that
   `configs/cloud/` carried only infrastructure params (GPU type, Docker image, region)
   but not training-behavior params that differ by platform (val_interval,
   mixed_precision). This created three parallel config systems for the same concern:
   experiment YAMLs, bash shell overrides in SkyPilot YAML, and hardcoded heuristics
   in Python. The fix was to encode model-specific constraints (e.g., SAM3 requires
   `mixed_precision: false`) in model configs, not cloud configs.

4. **Five-persona Prefect flow topology maps to academic lab roles.** The flow division
   (acquisition, train, analysis, deploy, dashboard) mirrors how a small academic team
   divides labor: data engineer (acquisition + data quality), data scientist (training +
   post-training), analysis expert (evaluation + biostatistics), DevOps (deployment),
   and the PI or lab (dashboard for overview). Even a solo researcher benefits from this
   separation because each flow has independent retry, scheduling, and failure isolation.
   The dashboard flow is explicitly "best-effort" -- its failure never blocks the pipeline.

5. **The 6-layer knowledge architecture (L0-L5) formalizes the decision-to-code pipeline.**
   Information flows downward: PRD decisions (L3a, Bayesian) materialize into KG domain
   entries (L3b, deterministic) when posterior >= 0.80, which are operationalized as
   OpenSpec GIVEN/WHEN/THEN scenarios (L4), which are implemented as code + tests (L5).
   Evidence flows upward: experiments update KG posteriors, which propagate to PRD
   beliefs. This bidirectional flow ensures that the codebase reflects deliberate
   decisions, not accumulated accidents. The navigator.yaml (L2) routes queries to
   the correct domain, preventing knowledge fragmentation.

6. **Merging training and post-training into one SkyPilot job eliminates a broken seam.**
   The original architecture used two separate SkyPilot jobs (training, then
   post-training), but the handoff was never wired: post-training could not discover
   training checkpoints across job boundaries. The solution was a parent flow with
   two sub-flows running in a single GPU session: training sub-flow produces
   checkpoints, and post-training sub-flow (SWAG, calibration) consumes them in-memory
   without requiring MLflow artifact download. The "none" post-training condition is
   free -- just the training sub-flow output.

---

## Architectural Decisions Made

| Decision | Winner | Rationale | KG Node |
|----------|--------|-----------|---------|
| Config architecture | Dual: Hydra-zen + Dynaconf | Hydra-zen for experiment sweeps, Dynaconf for deployment. `.env.example` as single source for secrets. | `config_architecture` |
| Flow topology | 5-flow architecture | Acquisition, Train, Analysis, Deploy, Dashboard (best-effort). Each flow is a Docker container. | `flow_topology` |
| Model adapter pattern | ABC protocol | `ModelAdapter` ABC with train/predict/export contract. Subclasses override only when needed. | `model_adapter_pattern` |
| Ensemble strategy | Heterogeneous multi-model | Multi-loss x multi-fold heterogeneous ensembles. Per-model best fold + cross-model ensemble. | `ensemble_strategy` |
| Serving architecture | BentoML primary | BentoML + ONNX Runtime for inference serving. Gradio for demo UI. | `serving_architecture` |
| Inter-flow communication | MLflow-only contract | No shared filesystems, no direct function calls, no Prefect artifacts for data. MLflow tags/metrics/artifacts only. | `prefect-flow-connectivity.md` |
| Script consolidation | Hydra-zen native | Debug configs are standard Hydra experiment YAMLs, not a parallel system. CLI overrides flow through Hydra grammar. | `script-consolidation.xml` |
| Knowledge architecture | 6-layer (L0-L5) | Constitution - Hot Context - Navigator - Evidence - Specifications - Implementation. Bidirectional information flow. | `prd-kg-openspec-architecture.md` |
| Oracle Cloud | **Rejected** | ARM capacity exhausted in Frankfurt, no region change possible, garbage DevEx. Switched to GCP. | `oracle-config-planning.md` |
| Train+Post-training merge | Sub-flows under one parent | One SkyPilot job, two Prefect sub-flows. Eliminates broken handoff between separate GPU sessions. | `training-and-post-training-into-two-subflows-under-one-flow.md` |

---

## Implementation Status

| Document | Status | Key Deliverable | Implementation Evidence |
|----------|--------|-----------------|----------------------|
| `consolidated-devex-training-evaluation-plan.md` | **Implemented** | 10-phase plan: profiler, adaptive profiles, Prefect, experiment runner, loss sweep | All 10 phases complete. 103 new tests. 4 loss functions verified. Hardware budget for RTX 2070 Super (8 GB). |
| `double-check-all-wiring.xml` | **Partial** | P0 wiring audit: 6 blockers, 4 P1 issues, 35 wired components | 6 P0 blockers identified (external test datasets, metric key convention, biostatistics split). Phases 1-2 in progress. |
| `flow-data-acquisition-plan.md` | **Implemented** | Flow 0: dataset download, format conversion, provenance logging | `acquisition_flow.py` exists. Registry for 4 datasets. VesselNN git clone downloader. TIFF-to-NIfTI conversion. |
| `hydra-config-verification-report.md` | **Reference** | Audit of three-config system (Hydra-zen, Dynaconf, .env) | Training, deployment, and secrets configs verified as single-source. SkyPilot GPU config identified as the composability gap. |
| `hydra-double-check.md` | **Reference** | Root cause analysis: cloud configs carry only infrastructure, not training behavior | Three parallel config systems identified. Fix: model-specific constraints in model configs, not cloud configs. `compose_experiment_config()` verified as structurally sound. |
| `oracle-config-planning.md` | **Rejected** | Oracle Cloud Always Free for MLflow hosting | Rejected due to Frankfurt ARM capacity exhaustion, permanent home region lock, card verification blocks, garbage DevEx. GCP chosen instead. |
| `prd-kg-openspec-architecture.md` | **Implemented** | 6-layer knowledge architecture (L0-L5), materialization protocol | `knowledge-graph/navigator.yaml` (L2) exists with 11 domains. `knowledge-graph/decisions/` (L3) has 65+ Bayesian nodes. `openspec/` (L4) exists with specs directory. |
| `prefect-analysis-dashboard-flow-improvement.md` | **Partial** | Analysis flow completion + 5th dashboard flow | 5-flow architecture designed. Dashboard flow (`dashboard_flow.py`) exists. Reproducibility verification and external test evaluation tasks partially implemented. |
| `prefect-and-devex-profiling-optimizations.md` | **Implemented** | Adaptive compute profiles, hardware detection, dataset-aware patching | `HardwareBudget` detection, 6 compute profiles, dataset-constrained patch sizes. Prefect made obligatory. Design principles (P1-P7) codified in CLAUDE.md. |
| `prefect-flow-connectivity-execution-plan.xml` | **Implemented** | Fix train-post_training-analysis seam via MLflow tags | Phase 0 (6 tasks): FlowContract enhancements, resolve_experiment_name(), checkpoint_dir tags, FLOW_COMPLETE status. `flow_contract.py` and `constants.py` exist. |
| `prefect-flow-connectivity.md` | **Implemented** | MLflow as inter-flow contract design document | 7 root causes diagnosed. MLflow-only communication principle established. Config-driven flow dispatch pattern from foundation-PLR adopted. Structured run name encoding. |
| `script-consolidation.xml` | **Implemented** | Bridge Hydra-zen composition with Prefect flow execution | `compose_experiment_config()` called by `train_flow.py`. Resolved config logged to MLflow as artifact. Debug configs are standard Hydra experiment YAMLs. v3 revision eliminated parallel config systems. |
| `training-and-post-training-into-two-subflows-under-one-flow.md` | **Planned** | Merge train + post-training into parent flow with 2 sub-flows | Architecture designed. File changes specified. Hydra config structure for `post_training/none.yaml` and `post_training/swag.yaml` defined. Tests specified. Awaiting implementation. |

---

## Cross-References

- **KG domain**: `knowledge-graph/domains/architecture.yaml` -- 6 decisions (model_adapter_pattern, ensemble_strategy, config_architecture, serving_architecture, api_protocol, flow_topology)
- **KG infrastructure domain**: `knowledge-graph/domains/infrastructure.yaml` -- Docker, Prefect, CI/CD (complementary to architecture)
- **Source implementations**:
  - `src/minivess/config/compose.py` -- Hydra-zen composition
  - `src/minivess/orchestration/flow_contract.py` -- MLflow inter-flow contract
  - `src/minivess/orchestration/constants.py` -- Flow names, experiment name resolution
  - `src/minivess/orchestration/flows/` -- 18 flow files (5 core + extensions)
- **Testing theme**: E2E testing depends on flow connectivity; the inter-flow MLflow contract is the primary E2E testing seam
- **Deployment theme**: BentoML serving architecture, ONNX export, champion selection are downstream of the flow topology
- **OpenSpec**: `openspec/specs/` contains testable specifications derived from KG decisions
- **CLAUDE.md rules**: Rule 9 (task-agnostic architecture), Rule 17 (never standalone scripts), Rule 22 (single-source config), Rule 29 (zero hardcoded parameters)

---

## Constituent Documents

1. `consolidated-devex-training-evaluation-plan.md` -- 10-phase DevEx + training plan (2026-02-25). Dataset profiler, adaptive compute profiles, Prefect compatibility, experiment runner, DynUNet loss sweep. All phases complete with 103 tests.
2. `double-check-all-wiring.xml` -- P0 wiring audit (2026-03-19). Systematic audit of all 5 flows + trainer + evaluation_runner + biostatistics. Found 6 P0 blockers (external test datasets not wired, metric key convention, biostatistics split awareness).
3. `flow-data-acquisition-plan.md` -- Flow 0 implementation (2026-03-04). Dataset download registry, format conversion (TIFF to NIfTI), provenance logging. Covers MiniVess, DeepVess, TubeNet-2PM, VesselNN.
4. `hydra-config-verification-report.md` -- Config system audit (2026-03-14). Three-config system (Hydra-zen, Dynaconf, .env) verified as correctly divided. SkyPilot GPU config identified as the composability gap requiring Hydra cloud config groups.
5. `hydra-double-check.md` -- Root cause analysis (2026-03-15). Triggered by sam3_hybrid val_loss=NaN across 4 GCP jobs. Three parallel config paths identified. Model-specific constraints (mixed_precision) must live in model configs, not cloud configs.
6. `oracle-config-planning.md` -- Oracle Cloud plan (2026-03-13). REJECTED. ARM capacity issues, permanent home region lock, API key bootstrap via Cloud Shell, intermittent authentication errors. Decision: switch to GCP (later confirmed as staging+prod provider).
7. `prd-kg-openspec-architecture.md` -- 6-layer knowledge architecture design. L0 Constitution, L1 Hot Context, L2 Navigator, L3 Evidence (PRD decisions + KG materialization), L4 Specifications (OpenSpec), L5 Implementation. Bidirectional information flow.
8. `prefect-analysis-dashboard-flow-improvement.md` -- Analysis flow completion plan (2026-03-01). 5 phases: reproducibility verification, external test datasets, paper-quality figures, interactive DuckDB-WASM dashboard, 5th dashboard flow.
9. `prefect-and-devex-profiling-optimizations.md` -- Adaptive DevEx profiling plan (2026-02-25). Hardware detection (`HardwareBudget`), two-phase adaptive profiles, dataset-constrained patch sizes, Prefect 4-persona flow architecture. Verbatim user prompt preserved.
10. `prefect-flow-connectivity-execution-plan.xml` -- Execution plan for MLflow contract fixes (2026-03-09). Phase 0: FlowContract enhancements (6 tasks). Phase 1: Data Engineering to Train handoff (3 tasks). TDD-first with real local MLflow.
11. `prefect-flow-connectivity.md` -- Design document for inter-flow connectivity (2026-03-09). 7 root causes diagnosed. MLflow-only communication. Config-driven flow dispatch from foundation-PLR. Structured run name encoding via `str.split("__")`.
12. `script-consolidation.xml` -- Script consolidation plan v3 (2026-03-08). Bridges Hydra-zen composition with Prefect flow execution. `compose_experiment_config()` called from `train_flow.py`. Resolved config logged to MLflow. Reproducibility guarantee: download YAML artifact, re-run.
13. `training-and-post-training-into-two-subflows-under-one-flow.md` -- Sub-flow merger plan (2026-03-21). One SkyPilot job, two Prefect sub-flows. Eliminates broken handoff. Post-training method configurable via Hydra (`configs/post_training/none.yaml`, `swag.yaml`).
