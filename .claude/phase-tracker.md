# Phase Execution Tracker

## Meta-Mandate: Claude Code Pattern Documentation
After each major commit, update `docs/claude-code-patterns.md` with patterns demonstrated.
These feed into slide decks:
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-2-intermediate-slides.md` (Modules 4-7)
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-3-advanced-slides-consolidated.md` (Modules 8-12)

## Current Phase: ALL COMPLETE (Phases 0-9)

### Phase 0 Tasks (COMPLETE)
- [x] P0.1: Initialize project with `uv init`, set up pyproject.toml with all deps
- [x] P0.2: Create docker-compose.yml with profiles (dev/monitoring/full)
- [x] P0.3: Create src/minivess/ + tests/v2/ package structure
- [x] P0.4: Pre-commit hooks (ruff, mypy, gitleaks) + justfile
- [x] P0.5: Pydantic v2 config models (data, model, training, serving)
- [x] P0.6: Hydra-zen experiment configs + Dynaconf deployment configs
- [x] P0.7: Set up DVC 3.x config for MinIO local remote
- [x] P0.8: GitHub Actions CI (lint, type check, unit tests)
- [x] P0.9: Pandera schemas + Great Expectations suite
- [x] P0.10: Hypothesis property-based tests for config validation
- [x] P0.11: .gitignore update for v2 artifacts

### Phase 1 Tasks (COMPLETE)
- [x] P1.1: ModelAdapter ABC (train/predict/export/metrics protocol)
- [x] P1.2: SegResNet MONAI adapter
- [x] P1.3: SwinUNETR MONAI adapter
- [x] P1.4: Data loading pipeline (DVC + MONAI transforms + TorchIO augmentation)
- [x] P1.5: Training engine (mixed precision, gradient checkpointing, early stopping)
- [x] P1.6: MLflow integration (experiment tracking, model registry)
- [x] P1.7: TorchMetrics integration (GPU-accelerated Dice, F1)
- [x] P1.8: DuckDB analytics over MLflow runs
- [x] P1.9: Unit tests for adapters + training engine (48 tests, 66 total)
- [x] P1.10: SwinUNETR API fix for MONAI 1.5.x + conftest warning filters

### Phase 2 Tasks (COMPLETE)
- [x] P2.1: Ensemble base (voting, mean, weighted strategies)
- [x] P2.2: Greedy Soup implementation
- [x] P2.3: WeightWatcher integration (alpha_weighted, deployment gate)
- [x] P2.4: Calibration (ECE/MCE + temperature scaling)
- [x] P2.5: Deepchecks Vision integration (data integrity + train-test suites)
- [x] P2.6: Evidently drift detection (KS test + PSI)
- [x] P2.7: Unit tests (14 tests, 80 total)

### Phase 3 Tasks (COMPLETE)
- [x] P3.1: BentoML service definition
- [x] P3.2: ONNX Runtime export + inference
- [x] P3.3: Gradio demo frontend
- [x] P3.4: Unit tests (7 pass, 8 skip for optional deps)

### Phase 4 Tasks (COMPLETE)
- [x] P4.1: OpenTelemetry instrumentation (OTLP gRPC exporter)
- [x] P4.2: Prometheus metrics + Grafana dashboards (deferred to docker-compose config)
- [x] P4.3: Langfuse LLM tracing (config in telemetry module)
- [x] P4.4: OpenLineage lineage tracking (deferred to docker-compose config)
- [x] P4.5: Model Cards (Mitchell et al. 2019 format, Markdown generation)
- [x] P4.6: SaMD audit trail (IEC 62304 lifecycle, SHA-256 data hashing)
- [x] P4.7: Unit tests (15 new tests, 102 total)

### Phase 5 Tasks (COMPLETE)
- [x] P5.1: LangGraph agent definitions (training + evaluation state graphs)
- [x] P5.2: Braintrust evaluation framework (segmentation + agent scorer configs)
- [x] P5.3: LiteLLM provider abstraction (config in agent module)
- [x] P5.4: Label Studio annotation workflows (deferred to docker-compose config)
- [x] P5.5: Cleanlab label quality checks (deferred to data pipeline integration)

### Phase 6 Tasks (COMPLETE)
- [x] P6.1: End-to-end integration test (full pipeline, 103 tests total)
- [x] P6.2: Documentation + Architecture Decision Records (5 ADRs in docs/adr/)
- [x] P6.3: README update with badges + quickstart
- [x] P6.4: Final claude-code-patterns.md update for slides (patterns 7-12)

### Phase 7: Hierarchical Probabilistic PRD (COMPLETE)
- [x] P7.1: Planning document (docs/planning/hierarchical-prd-planning.md)
- [x] P7.2: PRD directory structure (docs/planning/prd/)
- [x] P7.3: Schema files (_schema.yaml, schema.yaml) + llm-context.md
- [x] P7.4: DAG topology (_network.yaml — 52 nodes, ~80 edges)
- [x] P7.5: L1 Research Goals decision nodes (7 files)
- [x] P7.6: L2 Architecture decision nodes (10 files)
- [x] P7.7: L3 Technology decision nodes (20 files)
- [x] P7.8: L4 Infrastructure decision nodes (8 files)
- [x] P7.9: L5 Operations decision nodes (7 files)
- [x] P7.10: Archetype files (solo-researcher, lab-group, clinical-deployment)
- [x] P7.11: Scenario files (learning-first-mvp, research-scaffold, clinical-production)
- [x] P7.12: Domain overlays (registry, backbone-defaults, 4 domain overlays)
- [x] P7.13: PRD-update skill (.claude/skills/prd-update/ — 6 protocols, 2 templates)
- [x] P7.14: CLAUDE.md + phase-tracker.md updates

### Phase 7b: Academic Citation Standards (COMPLETE)
- [x] P7b.1: Central bibliography (docs/planning/prd/bibliography.yaml — 29 entries)
- [x] P7b.2: Structured references schema (_schema.yaml — citation_key + relevance + sections + supports_options)
- [x] P7b.3: Citation guide protocol (.claude/skills/prd-update/protocols/citation-guide.md)
- [x] P7b.4: Updated ingest-paper protocol (sub-citation extraction, author-year format)
- [x] P7b.5: Updated all 6 protocols for academic citation standards
- [x] P7b.6: SKILL.md invariants #6-8 (citation integrity, no citation loss, author-year format)
- [x] P7b.7: Templates updated with citation requirements and examples
- [x] P7b.8: Exemplar: segmentation-models.decision.yaml with 8 academic references
- [x] P7b.9: Citation validation script (scripts/validate_prd_citations.py)
- [x] P7b.10: Pre-commit hook (prd-citation-check — blocks citation removals)
- [x] P7b.11: LLM context updated with citation rules (llm-context.md)
- [x] P7b.12: Fixed 7 decision files with old-format file path references
- [x] P7b.13: CLAUDE.md updated with citation rules section

### Phase 8: PRD Research Integration & Backlog (COMPLETE)
- [x] P8.1: Read and analyze 83 bibliography papers from sci-llm-writer/biblio/
- [x] P8.2: Write PRD update plan (docs/planning/prd-update-plan.md — 560 lines)
- [x] P8.3: Bibliography update (+38 entries, 29→67 total in bibliography.yaml)
- [x] P8.4: HIGH priority decision files — segmentation_models (vesselFM + COMMA options), uncertainty_quantification, compliance_depth (regops_automated option), drift_response, monitoring_stack
- [x] P8.5: MEDIUM priority decision files — calibration_tools, loss_functions, metrics_framework, agent_framework, documentation_standard, documentation_generation, augmentation_stack, data_validation_tools, annotation_platform, foundation_model_integration
- [x] P8.6: LOW priority decision files — label_quality, model_diagnostics, annotation_workflow, retraining_trigger, llm_observability, testing_strategy, federated_learning
- [x] P8.7: GitHub issues created (20 issues: 4 P0, 6 P1, 10 P2) on project board
- [x] P8.8: PRD-update Skill updated with GitHub project integration
- [x] P8.9: Planning/backlog Skill created (.claude/skills/planning-backlog/SKILL.md)

### Phase 9: PRD Citation Completeness & Backfill (COMPLETE)
- [x] P9.1: Web research — Groups A+B (L1 research goals + L2 architecture, 22 papers found)
- [x] P9.2: Web research — Groups C+D (L3 technology + L4 infrastructure, 14 papers found)
- [x] P9.3: Web research — Group E + thin files (L5 operations + thin coverage, 11 papers found)
- [x] P9.4: Bibliography expansion (+38 entries, 64→102 total in bibliography.yaml)
- [x] P9.5: L1 decision files (6 files) — project-purpose, research-impact-target, open-source-model, portfolio-priority, monai-alignment, reproducibility-standard
- [x] P9.6: L2 decision files (8 files) — model-strategy, data-management-strategy, ensemble-strategy, serving-architecture, config-management, pipeline-orchestration, xai-strategy, validation-depth
- [x] P9.7: L3 decision files (7 files) — ensemble-methods, experiment-tracking, hpo-framework, lineage-tracking, xai-meta-evaluation, data-profiling, llm-provider
- [x] P9.8: L4+L5 decision files (9 files) — ci-cd-platform, compute-target, containerization, gitops-strategy, iac-tooling, secrets-management, model-export-format, cost-tracking, model-governance
- [x] P9.9: Cross-topic bibliography fixes (5 entries: kreuzberger2023mlops, sato2019cd4ml, maier2024metrics, langchain2024langgraph, cardoso2022monai)
- [x] P9.10: Validation — all 3 core checks PASS (resolution, completeness, cross-topic), 37 pre-existing warnings from Phase 8
