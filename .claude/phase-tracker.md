# Phase Execution Tracker

## Meta-Mandate: Claude Code Pattern Documentation
After each major commit, update `docs/claude-code-patterns.md` with patterns demonstrated.
These feed into slide decks:
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-2-intermediate-slides.md` (Modules 4-7)
- `sci-llm-writer/manuscripts/vibe-coding-slides/archived/tier-3-advanced-slides-consolidated.md` (Modules 8-12)

## Current Phase: 1 — Model-Agnostic Training Pipeline

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

### Phase 1 Tasks
- [ ] P1.1: ModelAdapter ABC (train/predict/export/metrics protocol)
- [ ] P1.2: SegResNet MONAI adapter
- [ ] P1.3: SwinUNETR MONAI adapter
- [ ] P1.4: Data loading pipeline (DVC + MONAI transforms + TorchIO augmentation)
- [ ] P1.5: Training engine (mixed precision, gradient checkpointing, early stopping)
- [ ] P1.6: MLflow integration (experiment tracking, model registry)
- [ ] P1.7: TorchMetrics integration (GPU-accelerated Dice, clDice, NSD)
- [ ] P1.8: DuckDB analytics over MLflow runs
- [ ] P1.9: Unit tests for adapters + training engine
- [ ] P1.10: Integration test: train SegResNet on MinIVess for 2 epochs

### Phase 2 Tasks (pending)
- [ ] P2.1: Ensemble base (voting, mean, weighted strategies)
- [ ] P2.2: Greedy Soup implementation
- [ ] P2.3: WeightWatcher integration (alpha_weighted, deployment gate)
- [ ] P2.4: MAPIE conformal prediction wrapper
- [ ] P2.5: netcal calibration (temperature scaling, ECE)
- [ ] P2.6: Deepchecks Vision integration
- [ ] P2.7: Evidently drift detection
- [ ] P2.8: Unit + integration tests for ensemble + UQ

### Phase 3 Tasks (pending)
- [ ] P3.1: BentoML service definition
- [ ] P3.2: ONNX Runtime export + inference
- [ ] P3.3: Gradio demo frontend
- [ ] P3.4: CML integration (PR comments with metrics)
- [ ] P3.5: ArgoCD deployment manifests (stretch)

### Phase 4 Tasks (pending)
- [ ] P4.1: OpenTelemetry instrumentation
- [ ] P4.2: Prometheus metrics + Grafana dashboards
- [ ] P4.3: Langfuse LLM tracing
- [ ] P4.4: OpenLineage lineage tracking
- [ ] P4.5: Model Cards + Data Cards
- [ ] P4.6: SaMD audit trail (IEC 62304 lifecycle)

### Phase 5 Tasks (pending)
- [ ] P5.1: LangGraph agent definitions
- [ ] P5.2: Braintrust evaluation framework
- [ ] P5.3: LiteLLM provider abstraction
- [ ] P5.4: Label Studio annotation workflows
- [ ] P5.5: Cleanlab label quality checks

### Phase 6 Tasks (pending)
- [ ] P6.1: End-to-end integration test (full pipeline)
- [ ] P6.2: Documentation + Architecture Decision Records
- [ ] P6.3: README update with badges + quickstart
- [ ] P6.4: Final claude-code-patterns.md update for slides
