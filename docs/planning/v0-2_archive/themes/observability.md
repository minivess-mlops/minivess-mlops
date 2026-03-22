---
title: "Theme: Observability — MLflow, Tracking, Monitoring, Agents"
theme_id: observability
doc_count: 18
archive_path: docs/planning/v0-2_archive/original_docs/
kg_domain: knowledge-graph/domains/observability.yaml
created: "2026-03-22"
status: archived
---

# Theme: Observability

MLflow experiment tracking, metric key conventions, cloud deployment research,
GPU efficiency logging, LLM agent orchestration, diagnostic frameworks, and
multi-backend logging. This theme covers everything from local filesystem tracking
through cloud MLflow deployment to agentic AI micro-orchestration.

---

## Key Scientific Insights

### 1. MLflow Is the Inter-Flow Contract -- Not Just a Logger

MLflow is not merely an experiment tracker; it is the sole inter-flow communication
mechanism in MinIVess. All 5 Prefect flows communicate exclusively through MLflow
artifacts and run metadata. `find_upstream_run()` discovers the best training run for
post-training; `FlowContract` defines what each flow produces and consumes. This
contract-first design means MLflow reliability is a hard dependency, not a nice-to-have.

### 2. Metric Naming Convention: Slash-Prefix for MLflow 2.11+ Grouping

Resolved decision: all metric keys use slash-prefix convention (`val/dice`, `sys/gpu_model`,
`fold/0/best_val_loss`). MLflow 2.11+ auto-groups metrics by slash prefix in the UI.
Migration from underscore to slash was completed via Issue #790. CI suffixes use
`/ci95_lo`, `/ci95_hi` for confidence intervals.

### 3. Neptune.ai Is Dead -- Never Adopt

Neptune was acquired by OpenAI in December 2025 (~$400M in stock). The platform shut
down March 5, 2026 with irreversible data deletion. This validates the MLflow-first
strategy: open-source self-hosted tracking survives vendor acquisitions. The meta-logger
plan documents DagsHub (hosted MLflow, drop-in), ClearML (strongest OSS alternative),
W&B (proprietary but popular), and Aim (visualization-focused) as the 2026 landscape.

### 4. GPU-Util% Is More Important Than VRAM

The `gpu-params-logging.md` document reveals a critical observability gap: the system
monitor logs VRAM allocation (static once model loads) but not volatile GPU utilization
percentage. The key diagnostic combination is `gpu_util < 60% AND cpu > 85%` indicating
DataLoader bottleneck, and `gpu_util ~100% AND sm_clock < base_clock * 0.92` indicating
thermal throttling. Both are invisible without proper GPU efficiency logging.

### 5. LangGraph Deprecated in Favor of Pydantic AI + PrefectAgent

Architectural decision (ADR-0007): Pydantic AI replaces LangGraph as the agent
framework. The original `langgraph-agents-plan.md` proposed LangGraph state graphs;
the advanced plan (`langgraph-agents-plan-advanced.md`) recognized that Pydantic AI
provides cleaner type-safe agents with first-class Prefect integration via PrefectAgent
durable execution. The two-layer architecture separates MACRO orchestration (Prefect
flows in Docker) from MICRO reasoning (Pydantic AI agents inside specific tasks).

### 6. Three Concrete Agents Defined

The advanced plan defines three production agents:

- **drift_triage**: Analyzes drift detection results, decides remediation strategy
- **experiment_summarizer**: Generates natural-language experiment comparison reports
- **figure_narrator**: Produces figure captions from statistical results

All three are implemented as Pydantic AI agents wrapped in Prefect tasks, with LiteLLM
for provider flexibility and Langfuse for OTEL tracing.

### 7. MLflow Cloud Deployment: Oracle Always Free Was the Choice

Extensive evaluation (7 hosting targets across 4 documents) concluded that Oracle Cloud
Always Free (A1.Flex ARM, 4 OCPUs, 24 GB RAM, 200 GB storage) provides sufficient
resources for MLflow + PostgreSQL + MinIO at zero cost. However, this decision was
superseded by the GCP consolidation -- MLflow tracking for production now targets
GCP Cloud Run (or local filesystem for dev). The Oracle research remains valuable as
a reference for budget-constrained deployments.

### 8. Four-Layer Composable MLflow Test Suite

The `mlflow-cloud-test-suite-initial-plan.md` defines a 4-layer test architecture:

- **L1**: Generic MLflow (filesystem + server, no creds, staging tier)
- **L2**: Cloud MLflow (live deployment, cloud creds required)
- **L3**: Pulumi IaC (string checks + YAML parse, no creds, staging tier)
- **L4**: SkyPilot-to-Cloud (tracking URI propagation, preemption recovery)

This addresses the gap that all 30+ existing MLflow tests use local filesystem; zero
tests verify remote connectivity, managed PostgreSQL, or S3-compatible artifact storage.

### 9. DiLLS-Style Agent Diagnostics for Session Introspection

The `dills-diagnostics-plan.md` proposes layered diagnostics: AgentInteraction (per-step),
SessionSummary (conversation-level), and AgentDiagnostics (cross-session aggregates).
This enables debugging agent decision quality by examining interaction chains, latency
distributions, and token consumption patterns.

---

## Architectural Decisions Made

| Decision | Winner | Evidence Doc | KG Node |
|----------|--------|-------------|---------|
| Experiment tracker | MLflow local filesystem | mlflow-tracking-plan.md | `observability.experiment_tracker` |
| Metric naming | Slash-prefix (`val/dice`) | mlflow-tracking-plan.md | `observability.metric_naming_convention` |
| LLM tracing | Langfuse self-hosted | langgraph-agents-plan-advanced.md | `observability.llm_tracing` |
| LLM evaluation | Braintrust autoevals | (referenced in agents plan) | `observability.llm_evaluation` |
| LLM provider | LiteLLM multi-provider | langgraph-agents-plan.md | `observability.llm_provider_strategy` |
| Agent architecture | Pydantic AI + PrefectAgent | langgraph-agents-plan-advanced.md | `observability.agent_architecture` |
| Observability depth | Full-stack OTEL | monitoring-research-report.md | `observability.observability_depth` |
| XAI strategy | Partial (Captum candidates) | N/A | `observability.xai_strategy` |
| Multi-backend logging | MLflow primary, W&B optional mirror | mlflow-and-weights-and-biases-meta-logger-plan.md | N/A |

---

## Implementation Status

| Document | Type | Status | Key Impl Files |
|----------|------|--------|----------------|
| cyclops-plan.md | plan | Implemented (structure) | `compliance/fairness.py` (227 lines) |
| dills-diagnostics-plan.md | plan | Not started | Agent diagnostics not yet built |
| experiment-run-to-mlflow-plan.md | plan | Implemented | `observability/tracking.py` (927 lines) |
| gpu-params-logging.md | plan | Partial | GPU-Util% not yet in console log; CSV written but not uploaded to MLflow |
| langgraph-agents-plan-advanced.md | plan | Partial | `agents/drift_triage.py`, `experiment_summarizer.py`, `figure_narrator.py` exist (329 lines) |
| langgraph-agents-plan.md | plan | Superseded | Superseded by ADR-0007 (Pydantic AI) |
| mlflow-and-weights-and-biases-meta-logger-plan.md | plan | Not started | Meta-logger abstraction not built |
| mlflow-cloud-test-suite-initial-plan.md | plan | Partial | L1 tests exist (30+), L2-L4 not built |
| mlflow-deployment-storage-analysis.md | reference | Reference only | 7 hosting targets evaluated |
| mlflow-online-deployment-research.md | reference | Superseded | Oracle choice superseded by GCP consolidation |
| mlflow-robustifying-plan-and-report-for-reproducibility.md | plan | Implemented | Robustified MLflow URI resolution, retry logic |
| mlflow-serving-and-evaluation-plan.xml | execution_plan | Partial | MLflow pyfunc serving wrapper designed |
| mlflow-tracking-plan.md | plan | Implemented | Full tracking with slash-prefix convention |
| mlflow-training-learnings.md | reference | Implemented | Learnings from dynunet_loss_variation_v2 |
| mlflow-vs-tensorboard-status-report-for-epoch-level-metrics.md | reference | Implemented | MLflow chosen over TensorBoard for epoch metrics |
| mlruns-evaluate-verifications.md | reference | Implemented | Post-training MLflow artifact verification |
| mlruns-verification-tdd-plan.xml | execution_plan | Implemented | TDD plan for MLflow artifact verification tests |
| monitoring-research-report.md | reference | Reference only | Full-stack monitoring landscape survey |

---

## Cross-References

- **KG Domain**: `knowledge-graph/domains/observability.yaml` -- 10 decision nodes, agent architecture resolved
- **Training Theme**: MLflow metric keys defined in training docs (`val/dice`, `val/cldice`)
- **Manuscript Theme**: MLflow artifacts are the data source for all results tables and figures
- **Operations Theme**: Audit trail (compliance) feeds OpenLineage events from MLflow
- **Infrastructure Theme**: Docker Compose orchestrates MLflow + PostgreSQL + Prometheus + Grafana
- **Data Theme**: DVC handles data versioning; MLflow handles experiment artifacts (zero overlap)
- **Key Source Files**:
  - `src/minivess/observability/tracking.py` (927 lines) -- ExperimentTracker, MLflow integration
  - `src/minivess/observability/analytics.py` (196 lines) -- DuckDB analytics queries
  - `src/minivess/agents/` (11 files, 885 lines total):
    - `drift_triage.py` (115 lines) -- drift analysis agent
    - `experiment_summarizer.py` (105 lines) -- experiment comparison agent
    - `figure_narrator.py` (109 lines) -- figure caption generation agent
    - `tracing.py` (108 lines) -- Langfuse OTEL tracing
    - `config.py` (65 lines), `evaluation.py` (66 lines), `factory.py` (50 lines), `models.py` (114 lines)

---

## Constituent Documents

1. `cyclops-plan.md` -- CyclOps healthcare ML auditing: subgroup fairness evaluation (Issue #12)
2. `dills-diagnostics-plan.md` -- DiLLS-style layered agent diagnostics: per-step, session, aggregate (Issue #16)
3. `experiment-run-to-mlflow-plan.md` -- DynUNet loss variation experiment pipeline: 4 losses x 3 folds x 100 epochs
4. `gpu-params-logging.md` -- GPU efficiency logging: volatile GPU-Util%, Mem-BW%, thermal throttle detection
5. `langgraph-agents-plan-advanced.md` -- Pydantic AI + PrefectAgent two-layer orchestration (supersedes LangGraph)
6. `langgraph-agents-plan.md` -- Original LangGraph agent plan (deprecated, superseded by ADR-0007)
7. `mlflow-and-weights-and-biases-meta-logger-plan.md` -- Meta-logger: MLflow primary + W&B/ClearML/DagsHub optional mirrors
8. `mlflow-cloud-test-suite-initial-plan.md` -- 4-layer composable MLflow test suite (L1 generic through L4 SkyPilot)
9. `mlflow-deployment-storage-analysis.md` -- Storage analysis across 7 MLflow hosting targets (Oracle, Hetzner, DagsHub, etc.)
10. `mlflow-online-deployment-research.md` -- MLflow online deployment: Oracle Always Free decision (superseded by GCP)
11. `mlflow-robustifying-plan-and-report-for-reproducibility.md` -- MLflow reliability improvements: URI resolution, retry logic
12. `mlflow-serving-and-evaluation-plan.xml` -- XML plan for MLflow pyfunc serving + model evaluation
13. `mlflow-tracking-plan.md` -- MLflow tracking plan: slash-prefix conventions, fold encoding, CI suffixes
14. `mlflow-training-learnings.md` -- Learnings from first training runs: artifact logging, metric granularity
15. `mlflow-vs-tensorboard-status-report-for-epoch-level-metrics.md` -- MLflow vs TensorBoard for epoch-level metrics (MLflow wins)
16. `mlruns-evaluate-verifications.md` -- Post-training MLflow artifact verification checklist
17. `mlruns-verification-tdd-plan.xml` -- TDD plan for MLflow artifact verification test suite
18. `monitoring-research-report.md` -- Comprehensive monitoring landscape: Prometheus, Grafana, AlertManager, OTEL
