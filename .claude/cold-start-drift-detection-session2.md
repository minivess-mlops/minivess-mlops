# Cold-Start Prompt: Drift Detection Implementation — Session 2

**Created**: 2026-03-16
**Branch**: `feat/observability-drift-detection` (on top of `test/mambavesselnet`)
**State file**: `state/tdd-state.json`
**Plan**: `docs/planning/drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md`

## Paste this to resume:

```
Continue implementing the drift detection monitoring plan on branch `feat/observability-drift-detection`.

## Context
- Plan: `docs/planning/drift-detection-grafana-evidently-vesselnn-synthetic-generator-deploy-monitoring-plan.md`
- TDD state: `state/tdd-state.json` — 9/19 tasks DONE, 10 remaining
- 20 GitHub issues created (#758-#777), P0 data quality issue #777
- 3 metalearning docs created this session (Level 4 mandate, infrastructure scaffold, RunPod dev-only)
- All work uses the self-learning-iterative-coder TDD skill (`/tdd-iterate`)

## What's DONE (9 tasks, 79 tests, all green + committed + pushed):
1. T-D1 (#768): SyntheticGeneratorAdapter ABC + registry + debug generator — `src/minivess/data/synthetic/`
2. T-F1 (#776): Active learning ABCs (UncertaintySampler, AnnotationRequest, MONAILabelAdapter) — `src/minivess/active_learning/`
3. T-C3 (#767): 3D volume vectorizer (statistical, histogram, FFT) — `src/minivess/data/volume_vectorizer.py`
4. T-C1 (#765): Multi-family champion registry — `src/minivess/serving/champion_registry.py`
5. T-B1 (#762): Evidently 0.7+ drift reporter + Prometheus export — `src/minivess/observability/evidently_service.py`
6. T-B3 (#764): Alerting (DriftAlert, AlertManager, JSONL+webhook) — `src/minivess/observability/alerting.py`
7. T-C2 (#766): Dual-mode champion evaluator (supervised+unsupervised) — `src/minivess/serving/champion_evaluator.py`
8. T-A3 (#760): Docker Compose — Evidently + Alertmanager services in monitoring profile
9. T-A4 (#761): Grafana drift monitoring timeline dashboard (9 panels, 5th dashboard)

## What's REMAINING (10 tasks, in priority order):
1. **T-B2 (#763)**: whylogs continuous profiling — profile every volume, mergeable profiles, Prometheus exporter
2. **T-A1 (#758)**: Universal dataset downloader — cloud-agnostic, VesselNN from GitHub, upload to any cloud
3. **T-A2 (#759)**: DVC drift simulation setup — partition 12 VesselNN vols into 6 batches of 2, git tags
4. **T-D2 (#769)**: vesselFM d_drand adapter — wrap github.com/bwittmann/vesselFM, MONAI-native, GPL-3.0
5. **T-D3 (#770)**: MONAI VQ-VAE adapter — `monai.networks.nets.VQVAE`, train on patches, Apache-2.0
6. **T-D4 (#771)**: VaMos procedural adapter — wrap gitlab.univ-nantes.fr/autrusseau-f/vamos/
7. **T-D5 (#772)**: VascuSynth C++ wrapper — subprocess around compiled binary, Apache-2.0
8. **T-D6 (#773)**: Synthetic generation Prefect flow (Flow 7) — `generate_stack(method='vesselFM_drand')`
9. **T-E1 (#774)**: Drift simulation Prefect flow (Flow 6) — single automated run, 6 VesselNN batches
10. **T-E3 (#775)**: E2E integration test — full pipeline verification

## Critical constraints (NEVER violate):
- Level 4 MLOps is NON-NEGOTIABLE — never offer lighter alternatives
- Implement ALL viable synthetic generators — this is an infrastructure scaffold, not one tool
- GCP = primary (staging/prod), RunPod = dev backup only
- Evidently 0.7+ API: `from evidently.presets import DataDriftPreset`, `from evidently import Report, Dataset`
- TDD mandatory: RED (tests first) → GREEN (implement) → VERIFY → FIX → CHECKPOINT → CONVERGE
- `uv` only, `--all-extras` required, `from __future__ import annotations` in every file

## To resume, invoke the TDD skill:
/tdd-iterate

Then read `state/tdd-state.json` and continue from T-B2 (whylogs continuous profiling).
```
