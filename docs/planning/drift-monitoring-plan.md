# Drift Detection & Monitoring Plan

**Issue**: #38 — Integrate drift detection into v2 pipeline (Evidently + Alibi-Detect)
**Date**: 2026-02-24
**Status**: Draft → Implementation

---

## 1. Problem Statement

PRD decisions `drift_detection_method` (L3) and `drift_response` (L5) are well-researched
with 12+ references but have zero implementation. The project has:

- Synthetic drift generators (`data/drift_synthetic.py`) — 4 drift types tested
- Evidently already in `pyproject.toml` as a core dependency (v0.7+)
- No actual drift *detection* code anywhere

**Key references motivating the design**:
- Perkonigg et al. (2021) — Dynamic Memory for continual learning across scanner
  acquisition shifts. Demonstrates that scanner vendor changes (Siemens → GE → Philips →
  Canon) cause catastrophic forgetting without drift-aware adaptation.
- Roschewitz et al. (2023) — Performance drift from acquisition shift in mammography.
  Shows that model calibration thresholds become invalid when scanner hardware changes,
  even when overall discriminative power (AUC) is preserved. Proposes UPA (unsupervised
  prediction alignment) — a label-free correction method.
- Rabanser et al. (2024, Nature Comms) — Monitoring model performance is NOT a good
  proxy for detecting data drift; input data monitoring is essential.

---

## 2. Decision: Both Evidently AND Alibi-Detect (complementary tiers)

**Do we need both?** Yes — they serve complementary detection tiers per the PRD:

| Criterion | Evidently | Alibi-Detect (MMD) |
|-----------|-----------|-------------------|
| Already installed | Yes (core dep) | No — add `alibi-detect[torch]` |
| Detection level | Feature-level (KS, PSI, Wasserstein) | Embedding-level (kernel MMD) |
| Catches | Scanner acquisition shifts (intensity, SNR) | Semantic distribution shifts |
| Interpretability | Per-feature drift scores + HTML reports | Single p-value for batch |
| Dependency | Light (scipy/sklearn) | PyTorch backend (`alibi-detect[torch]`) |
| Best for | Monitoring raw image statistics | Monitoring model embeddings |

**Architecture**: Two-tier detection following PRD `composite_tiered` option:
1. **Tier 1 — Evidently** on extracted image features (fast, interpretable)
2. **Tier 2 — Alibi-Detect MMD** on model embeddings (statistically rigorous)

Both tiers alert independently. Either alone can trigger drift response.

**PRD status**: The PRD `drift_detection_method.decision.yaml` correctly models both
as complementary (Alibi-Detect 0.40, Evidently 0.25). No PRD update needed — our
implementation matches the design.

---

## 3. Architecture

```
Volume batch
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  Tier 1: Features    │     │  Tier 2: Embeddings      │
│  FeatureExtractor    │     │  Model.forward() hook     │
│  (mean, std, SNR,    │     │  → penultimate layer      │
│   contrast, entropy) │     │  → Alibi-Detect MMD       │
└─────────┬────────────┘     └────────────┬─────────────┘
          │                               │
          ▼                               ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  Evidently Report    │     │  MMDDrift.predict()       │
│  DataDriftPreset     │     │  kernel MMD p-value       │
│  Per-feature KS/PSI  │     │  on embedding vectors     │
└─────────┬────────────┘     └────────────┬─────────────┘
          │                               │
          └───────────┬───────────────────┘
                      ▼
             ┌─────────────────┐
             │  DriftReport     │
             │  .drift_detected │
             │  .tier1_result   │
             │  .tier2_result   │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  Alert / Log     │
             │  (Dynaconf       │
             │   thresholds)    │
             └─────────────────┘
```

---

## 4. Implementation Plan

### Task T1: Install alibi-detect + image feature extraction module

**Deps**: `uv add --optional quality "alibi-detect[torch]>=0.12"`

Create `src/minivess/data/feature_extraction.py`:
- `extract_volume_features(volume: ndarray) -> dict[str, float]`
  - mean, std, min, max, p5/p95 percentiles
  - SNR (mean/std of foreground)
  - contrast (p95 - p5)
  - histogram entropy (scipy.stats.entropy on binned values)
- `extract_batch_features(volumes: list[ndarray]) -> pd.DataFrame`
  - Applies above to a batch, returns DataFrame (one row per volume)

**Tests** (`test_drift_detection.py`):
- `test_extract_volume_features_keys` — correct keys present
- `test_extract_volume_features_types` — all values are float
- `test_extract_batch_features_shape` — correct DataFrame shape
- `test_features_change_under_drift` — drifted volumes have different features

### Task T2: Evidently drift detector (Tier 1)

Create `src/minivess/observability/drift.py`:
- `FeatureDriftDetector` class (Tier 1 — Evidently):
  - `__init__(reference_features: pd.DataFrame, *, threshold: float = 0.05)`
  - `detect(current_features: pd.DataFrame) -> DriftResult`
  - `generate_html_report(output_path: Path) -> Path`
- `DriftResult` dataclass:
  - `drift_detected: bool`
  - `dataset_drift_score: float`
  - `feature_scores: dict[str, float]` (per-feature p-values)
  - `drifted_features: list[str]`
  - `n_features: int`
  - `n_drifted: int`
  - `timestamp: datetime`

Uses `evidently.report.Report` with `DataDriftPreset`.

**Tests**:
- `test_no_drift_same_distribution` — identical features → no drift
- `test_drift_detected_shifted_features` — shifted features → drift detected
- `test_drift_result_has_correct_fields` — dataclass fields
- `test_drift_result_feature_scores` — per-feature p-values present
- `test_drift_threshold_configurable` — custom threshold changes sensitivity
- `test_generate_html_report` — creates HTML file on disk

### Task T3: Alibi-Detect MMD drift detector (Tier 2)

Add to `src/minivess/observability/drift.py`:
- `EmbeddingDriftDetector` class (Tier 2 — Alibi-Detect):
  - `__init__(reference_embeddings: ndarray, *, p_val_threshold: float = 0.05)`
  - `detect(current_embeddings: ndarray) -> DriftResult`
- Uses `alibi_detect.cd.MMDDrift` with PyTorch backend

**Tests**:
- `test_embedding_drift_no_shift` — same distribution → no drift
- `test_embedding_drift_detected` — shifted embeddings → drift detected
- `test_mmd_detector_configurable_threshold` — custom p-value threshold

### Task T4: Alert configuration via Dynaconf

Add to `configs/deployment/settings.toml`:
```toml
drift_detection_threshold = 0.05
drift_embedding_threshold = 0.05
drift_alert_enabled = true
drift_report_dir = "reports/drift"
```

**Tests**:
- `test_settings_have_drift_config` — drift keys exist in settings
- `test_drift_detector_reads_threshold_from_config` — reads Dynaconf value

### Task T5: Integration test — synthetic drift → detection pipeline

End-to-end integration test:
1. Generate reference features from clean synthetic volumes
2. Apply `drift_synthetic.apply_drift()` with known severity
3. Tier 1: Extract features → `FeatureDriftDetector.detect()` → drift=True
4. Tier 2: Extract embeddings (SegResNet penultimate) → `EmbeddingDriftDetector.detect()` → drift=True
5. Verify clean data → drift_detected=False on both tiers

**Tests**:
- `test_synthetic_intensity_drift_tier1` — intensity shift → Evidently detects
- `test_synthetic_noise_drift_tier1` — noise injection → Evidently detects
- `test_clean_data_no_drift_tier1` — no drift on same distribution
- `test_embedding_drift_full_pipeline` — model embeddings → MMD detects shift
- `test_gradual_drift_sensitivity` — higher severity → lower p-values

### Task T6: Topology metrics (clDice drift as monitoring signal)

Add topology-aware feature:
- `compute_cl_dice_proxy(volume, mask)` — simplified clDice approximation
  using skeletonization (scipy.ndimage) for vessel connectivity
- Add as an extracted feature for drift detection

**Tests**:
- `test_cldice_proxy_returns_float` — returns a float in [0, 1]
- `test_cldice_proxy_detects_topology_change` — eroded vessels → lower clDice

---

## 5. Execution Order

```
T1 (feature extraction + alibi install) → T2 (Evidently tier 1) → T3 (Alibi-Detect tier 2) → T4 (Dynaconf) → T5 (integration) → T6 (topology)
```

---

## 6. Out of Scope

- WATCH sequential testing (no production library yet)
- Automated retraining trigger (issue #40+)
- Grafana dashboard integration
- Continual learning / dynamic memory (Perkonigg 2021 approach — future work)
- UPA prediction alignment (Roschewitz 2023 — future issue)
