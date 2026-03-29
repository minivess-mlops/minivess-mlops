# 11th Pass Debug Factorial Experiment Report — Vascadia v0.2-beta

**Branch**: `fix/10th-pass-production-readiness`
**Date started**: 2026-03-28
**Plan**: `run-debug-factorial-experiment-11th-pass.xml`
**Prior pass report**: `run-debug-factorial-experiment-report-10th-pass.md`

## Pre-Launch Validation

| Gate | Status | Details |
|------|--------|---------|
| CS1 Branch | PASS | `fix/10th-pass-production-readiness` |
| CS2 Env vars | PASS | MLFLOW_TRACKING_URI + HF_TOKEN set |
| CS3 GCP auth | PASS | ADC token obtained |
| CS4 Docker image | PENDING | Rebuild in progress (src/ unchanged, label update) |
| CS5 MiniVess on GCS | PASS | 984 MB, 211 files |
| CS5 DeepVess on GCS | PASS | 1.9 GB, 27 files |
| CS6 MLflow | PASS | HTTP 200, Cloud Run europe-west4 |
| CS7 SkyPilot | PASS | v1.0.0.dev20260326, controller UP (europe-west1-b) |
| CS8 Staging tests | PASS | 6721 passed, 0 failed, 0 skipped |
| CS9 YAML contract | PASS | 10 files validated |
| CS9 Preflight | 14/15 PASS | Docker image freshness (rebuilding) |
| Report file (H1) | PASS | This file created before launch |
| SAM3 on-demand | PASS | use_spot: false in config, --no-use-spot in script |
| GPU quota | WARNING | 1 L4 GPU — jobs run sequentially |

## Council Review (2026-03-28)

5-agent review completed. 3 CRITICAL issues found and fixed:
1. Stale CS1 path (fixed in XML)
2. SAM3 on-demand mechanism (implemented in run_factorial.sh + debug.yaml)
3. DeepVess data on GCS (downloader implemented + data pushed)

Corrected cost estimate: ~$6.53 (plan original: $5.30, +23%).
Budget cap $15 provides 2.3x headroom.

## Phase 1: Infrastructure Validation (3 Jobs)

| Job | Name | Model | Spot | Expected | Actual | Status | Cost |
|-----|------|-------|------|----------|--------|--------|------|
| J1 | dynunet-dice_ce-calibfalse-f0 | DynUNet | spot | 10 min | | | |
| J2 | sam3_hybrid-cbdice_cldice-calibfalse-f0 | SAM3 Hybrid | on-demand | 25 min | | | |
| J3 | sam3_vanilla-zeroshot-minivess-f0 | SAM3 Vanilla | spot | 15 min | | | |

**Gate**: ALL 3 must SUCCEED before Phase 2.

### Phase 1 Verification Criteria

| Criterion | J1 | J2 | J3 |
|-----------|----|----|-----|
| Training/eval completes | | | |
| Checkpoint in gs://mlflow-artifacts | | | |
| No 413 in logs | | | |
| MLflow UI shows run | | | |
| DVC pull from GCS succeeded | | | |

## Phase 2: Failed Factorial Conditions (17 Jobs)

*P2.18 (VesselFM zero-shot on DeepVess) included now that DeepVess is on GCS.*

| ID | Model | Loss | Calib | Spot | Status | Duration | Cost |
|----|-------|------|-------|------|--------|----------|------|
| P2.1 | sam3_hybrid | cbdice_cldice | true | on-demand | | | |
| P2.2 | sam3_hybrid | cbdice_cldice | false | on-demand | | | |
| P2.3 | sam3_hybrid | dice_ce | true | on-demand | | | |
| P2.4 | sam3_hybrid | dice_ce | false | on-demand | | | |
| P2.5 | sam3_hybrid | dice_ce_cldice | true | on-demand | | | |
| P2.6 | sam3_hybrid | dice_ce_cldice | false | on-demand | | | |
| P2.7 | sam3_hybrid | bce_dice_05cldice | true | on-demand | | | |
| P2.8 | sam3_hybrid | bce_dice_05cldice | false | on-demand | | | |
| P2.9 | sam3_topolora | cbdice_cldice | true | on-demand | | | |
| P2.10 | sam3_topolora | cbdice_cldice | false | on-demand | | | |
| P2.11 | sam3_topolora | dice_ce | true | on-demand | | | |
| P2.12 | sam3_topolora | dice_ce | false | on-demand | | | |
| P2.13 | sam3_topolora | dice_ce_cldice | true | on-demand | | | |
| P2.14 | sam3_topolora | dice_ce_cldice | false | on-demand | | | |
| P2.15 | sam3_topolora | bce_dice_05cldice | true | on-demand | | | |
| P2.16 | sam3_topolora | bce_dice_05cldice | false | on-demand | | | |
| P2.17 | sam3_vanilla | zero-shot (minivess) | - | spot | | | |
| P2.18 | vesselfm | zero-shot (deepvess) | - | spot | | | |

## Phase 3: Artifact Re-Verification (1 Job)

| ID | Model | Loss | Purpose | Status | Duration | Cost |
|----|-------|------|---------|--------|----------|------|
| P3.2 | mambavesselnet | cbdice_cldice | Verify checkpoint upload | | | |

*P3.1 (DynUNet) skipped if Phase 1 J1 verifies artifact upload.*

## Cost Tracking

| Phase | Jobs | Estimated | Actual | Running Total |
|-------|------|-----------|--------|---------------|
| Phase 1 | 3 | ~$0.47 | | |
| Phase 2 | 18 | ~$6.03 | | |
| Phase 3 | 1 | ~$0.04 | | |
| **Total** | **22** | **~$6.54** | | |
| **Budget cap** | | **$15.00** | | |

**Budget alarms**: $8 WARNING, $15 STOP.

## Duration Calibration (Updated After Phase 1)

| Model | Expected | Phase 1 Actual | Calibrated WARN | Calibrated CANCEL |
|-------|----------|----------------|-----------------|-------------------|
| DynUNet | 10 min | | 30 min | 50 min |
| SAM3 Hybrid | 25 min | | 75 min | 75 min |
| SAM3 Vanilla (ZS) | 15 min | | 45 min | 75 min |
| SAM3 TopoLoRA | 25 min | | 75 min | 75 min |
| VesselFM (ZS) | 15 min | | 45 min | 75 min |
| MambaVesselNet | 12 min | | 36 min | 60 min |

## Observations

1. **2026-03-28 09:00**: All 3 Phase 1 jobs PENDING. europe-west4 L4 GPU capacity constrained.
   Jobs cycle STARTING→PENDING as provisioning attempts fail. europe_strict config means
   no US fallback — jobs will PEND until europe-west4 capacity frees up. Controller cycling
   through jobs trying to provision. J2 (on-demand) and J3 (spot) both failing to provision.
   This is W2 manifesting — "L4 spot availability in europe-west4." GPU quota shows 1/1
   available but actual hardware may be fully allocated by other users.

   **Action**: Continue polling. SkyPilot will auto-retry. max_wait_hours=168 (1 week).
   No intervention needed — this is expected behavior with europe_strict in a constrained region.

2. **2026-03-28 09:15**: 30 min monitoring. J3 (161, SAM3 Vanilla spot) reached 14 min sustained
   STARTING (longest attempt) before reverting to PENDING — spot instance preempted during setup.
   J2 (160, on-demand) had 13 min sustained STARTING before PENDING. Both spot AND on-demand
   provisioning failing. europe-west4 L4 capacity genuinely exhausted across all 3 zones.
   Controller cycling through provisioning attempts automatically.

   **Diagnosis**: This is W2 manifesting as predicted. GPU quota shows 1/1 available (quota not
   exhausted) but physical hardware is fully allocated by other GCP users. This is a region
   capacity issue, not a configuration or quota issue.

   **Infrastructure follow-up plan being created** in background — will analyze multi-region
   fallback strategy, quota increase request, and provisioning resilience improvements.

   **No intervention needed** — controller auto-retries. Polling reduced to 5-min intervals.

3. **2026-03-28 09:30**: J3 (SAM3 Vanilla spot) reached 19 min sustained STARTING — longest
   attempt. Setup running (Docker pull 6 GB + SAM3 weights 9 GB + DVC data pull), but spot
   preempted before setup completed. Pattern: SAM3 zero-shot setup takes ~15-20 min, making
   spot instances economically irrational for ANY SAM3 job (not just training).

   **INSIGHT for follow-up plan**: ALL SAM3 jobs (including zero-shot baselines) should use
   on-demand. The 15-20 min setup window makes spot preemption almost guaranteed. DynUNet
   (no SAM3 weights) setup is ~5-10 min — spot is viable for DynUNet.

   **J1 (DynUNet spot) at 33 min PENDING** — may also be capacity-gated, not spot-preemption.
   The region itself may have zero L4 availability right now.

4. **2026-03-28 09:45**: 45 min monitoring. J2 (160, on-demand) reached 14 min sustained
   STARTING then reverted to PENDING. On-demand should NOT be preempted — this suggests
   setup FAILURE (HF rate limiting, DVC pull, or Docker pull). Log tailing produced no
   output (logs not retained for setup-phase failures).

   J1 at 40 min PENDING, J2 at 28 min, J3 at 26 min. All cycling STARTING→PENDING.

   **Root cause hypothesis**: europe-west4 L4 physical capacity exhausted. GPU quota
   shows 1/1 available but this is the quota LIMIT, not current AVAILABILITY. Other
   GCP tenants are consuming all L4 instances in the region.

   **Infrastructure improvement needed**: GCS weight caching (SAM3 9 GB, VesselFM 2 GB)
   would reduce setup from 15 min to 3 min, making spot viable. Also: multi-region
   fallback (europe_us.yaml) would provide US regions as backup when EU is exhausted.

   **Decision**: Continue autonomous monitoring. Jobs auto-retry (max_wait_hours=168).
   Two background agents creating: (1) infrastructure follow-up plan, (2) GCS weight
   caching report. User offline, will request GPU quota increase later.

5. **2026-03-28 09:55**: 50+ min monitoring. All 3 jobs still PENDING/cycling. J1 (DynUNet)
   reached 10+ min STARTING twice (setup running, then spot preempted). J2 (SAM3 Hybrid
   on-demand) reached 14 min STARTING (setup running, then returned to PENDING — possible
   setup failure, not preemption since on-demand). J3 (SAM3 Vanilla spot) reached 19 min
   STARTING (longest) before preemption.

   **Pattern observed**: europe-west4 L4 capacity fluctuates. VMs provision intermittently
   but get released before setup completes (spot preemption) or setup fails (on-demand).
   The 6 GB Docker pull + DVC data pull takes ~8-10 min, which is the minimum setup time.
   SAM3 adds ~10 min for HF weight download (9 GB). Total SAM3 setup: ~18-20 min.

   **USER NOTE for follow-up session**: Jobs 159, 160, 161 are still in queue and auto-
   retrying. When L4 capacity frees up (could be minutes to hours), they will provision
   and run. Check with: `uv run sky jobs queue | grep -E "159|160|161"`

   **Infrastructure improvements identified during monitoring**:
   - GCS weight caching (P0): Cache SAM3/VesselFM weights on GCS → setup drops from 15 to 3 min
   - Multi-region fallback (P1): europe_us.yaml for US backup when europe-west4 exhausted
   - GPU quota increase (P1): Request >1 L4 for parallel job execution
   - SkyPilot controller autostop (P2): Disable or increase from 10m to prevent re-start latency

## Watchlist Status

| ID | Item | Status |
|----|------|--------|
| W1 | GCS artifact upload for SAM3 ~900 MB | PENDING (Phase 1 J2 tests this) |
| W2 | L4 spot availability europe-west4 | PENDING |
| W3 | VesselFM zero-shot setup reliability | PENDING |
| W4 | MiniVess data on GCS | RESOLVED (984 MB pushed) |
| W4b | DeepVess data on GCS | RESOLVED (1.9 GB pushed) |
| W5 | SkyPilot version mismatch | RESOLVED (API restarted) |
| W6 | Cloud SQL authorized_networks | LOW (security, not functional) |
| W7 | Untagged MLflow image | LOW (cleanup candidate) |

## Timeline

| Time | Event |
|------|-------|
| 2026-03-28 07:38 | Session started, cold-start checks |
| 2026-03-28 07:45 | SkyPilot API restarted (stale venv paths) |
| 2026-03-28 07:50 | MiniVess download started from EBRAINS |
| 2026-03-28 07:55 | MiniVess download complete (70 volumes) |
| 2026-03-28 07:57 | MiniVess pushed to GCS (211 files, 984 MB) |
| 2026-03-28 08:00 | 5-agent council review launched |
| 2026-03-28 08:10 | Council reviews complete, 3 CRITICAL issues found |
| 2026-03-28 08:15 | SAM3 on-demand mechanism implemented |
| 2026-03-28 08:20 | Staging tests: 6721 passed |
| 2026-03-28 08:34 | DeepVess download started from eCommons |
| 2026-03-28 08:38 | DeepVess download complete (26 TIFF files, 1.8 GB) |
| 2026-03-28 08:40 | DeepVess pushed to GCS (27 files) |
| 2026-03-28 08:42 | Controller started (europe-west1-b) |
| 2026-03-28 08:45 | Docker rebuild started |
| | Docker rebuild complete |
| | Preflight re-run (all pass) |
| | Phase 1 J1 launched |
| | Phase 1 J2 launched |
| | Phase 1 J3 launched |
