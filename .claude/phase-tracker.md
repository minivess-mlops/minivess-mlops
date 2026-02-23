# Phase Execution Tracker

## Meta-Mandate: Claude Code Pattern Documentation
After each major commit, update `docs/claude-code-patterns.md` with patterns demonstrated.
These feed into slide decks:
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-2-intermediate-slides.md` (Modules 4-7)
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-3-advanced-slides-consolidated.md` (Modules 8-12)

## Current Phase: ALL COMPLETE (Phases 0-6)

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
