# Cold-Start Prompt: Track B — RunPod + GCP Remaining QA
<!-- Generated: 2026-03-16 | Branch: test/mambavesselnet | Plan: v4.0 -->

## How to start a fresh Claude Code session for Track B

Paste the prompt below into a new Claude Code session. It is self-contained — no prior
context needed.

---

## THE PROMPT (copy everything below this line)

---

You are picking up Track B of the MinIVess MLOps QA plan. Track A (MambaVesselNet
smoke test) may or may not still be running in a separate session. This session
handles only Track B.

## What you must read first (do NOT skip)

Read these files in order before doing anything else:

1. `docs/planning/remaining-runpod-gcp-qa-4th-pass.xml` — the full v4 QA plan
2. `knowledge-graph/domains/cloud.yaml` — RunPod + GCP architecture (two-provider rule)
3. `.claude/metalearning/2026-03-16-unauthorized-aws-s3-architecture-migration.md` — why S3 is banned
4. `deployment/skypilot/smoke_test_gpu.yaml` — RunPod SkyPilot YAML
5. `deployment/skypilot/smoke_test_gcp.yaml` — GCP SkyPilot YAML

## Current state (as of 2026-03-16, commit df18293)

**Branch**: `test/mambavesselnet`
**Track B should branch from**: `qa/gcp-runpod-3rd-pass` (create if missing:
`git checkout -b qa/gcp-runpod-3rd-pass`)

**Already done (do NOT redo):**
- `smoke_test_gcp.yaml` fixed: GCS data source, MLFLOW_TRACKING_URI direct ref (commit ffad741)
- GCP Pulumi stack deployed: Cloud SQL, Cloud Run MLflow, GCS, GAR all UP
- `smoke_test_mamba.yaml` created on test/mambavesselnet
- FinOps plan written, Issue #747 created
- All S3 references removed from RunPod YAMLs (commits bdfb005, df18293)
- KG updated: correct per-flow compute flexibility documented

**Outstanding work (what this session executes):**
- P0: Pre-flight verification
- P2: RunPod sequential regression + inter-flow handoff test (T2.4 is NEW)
- P3: Concurrent RunPod (2 pods, 1 shared volume, separate mlruns subdirs) — redesigned from v3
- P4: GCP sequential (4 models) + quota exhaustion test
- P5: Cross-provider concurrent (GCP + RunPod simultaneously)
- P6: GCP artifact verification
- P7: TDD integration tests
- P8: FinOps Sprint 1 (Issue #747)
- P9: QA report + merge prep

## Budget cap (HARD)

Global: **$6.00**. Check `outputs/ralph_diagnoses.jsonl` cumulative cost before every
new job launch. If remaining < cost of next phase, skip that phase and document it.

## Architecture constraints (NON-NEGOTIABLE — violations are bugs)

1. **RunPod = local rsync only. NO S3, EVER.**
   Data path: researcher's local disk → `make dev-gpu-upload-data` → `/opt/vol/data/raw/`.
   If data is missing on the Network Volume, the setup script MUST exit 1 with upload
   instructions — NOT fall back to any S3 URL.
   Any `s3://` in a RunPod YAML is a bug. Check with:
   ```bash
   grep -r "s3://" deployment/skypilot/smoke_test_gpu.yaml deployment/skypilot/dev_runpod.yaml
   ```
   Expected output: empty (zero matches).

2. **Two cloud providers only: RunPod (dev) + GCP (staging/prod).**
   No AWS, no Lambda Labs, no UpCloud. If you find yourself adding a third provider,
   STOP and ask the user for explicit authorization.

3. **T4 GPU BANNED.**
   T4 = Turing architecture = no BF16 = NaN in SAM3 encoder during validation.
   GCP accelerator must be L4 (Ada Lovelace). Verify in smoke_test_gcp.yaml:
   `accelerators: {L4: 1}` (not T4).

4. **GCP quota = 1 L4 GPU.**
   Sequential training only. P4 tests intentional quota exhaustion (T4.2) — that test
   is diagnostic, the second job is cancelled after 2 minutes.

5. **MLflow sync-back is MANDATORY after every RunPod job.**
   Without `sky rsync down <pod>:<mlruns_path>/ mlruns/`, the run is invisible locally
   and the entire RunPod path has zero scientific value. Never skip sync.

6. **Per-flow compute is NEVER hardcoded.**
   Any flow can run anywhere — Train on RunPod, Analysis on GCP, Dashboard on Vercel.
   The platform does not prescribe which flows run where. Never write this assumption
   into any YAML, KG, or code.

## Track A coordination

Check Track A state before starting P2:
```bash
sky jobs status 2>&1 | grep -E "minivess|mamba"
sky status 2>&1 | grep -E "minivess|mamba"
```

- **Track A still running** → Track B uses `MLFLOW_TRACKING_URI=/opt/vol/mlruns/track-b`
  (same `minivess-dev` volume, isolated mlruns subdir). NO second volume needed.
- **Track A done** → Track B uses `/opt/vol/mlruns` (default path, full volume).

## Skills to use

- `/ralph-loop` — for all SkyPilot job launches + monitoring (polls sky jobs status,
  diagnoses failures, retries with fixes, writes to `outputs/ralph_diagnoses.jsonl`)
- `/self-learning-iterative-coder` — for P7 TDD integration tests (RED → GREEN → VERIFY)

## Prerequisites before spending money

Run P0 first — all tasks are cheap ($0) and verify infra health:

```bash
# 1. Check Track A state
sky jobs status 2>&1 | grep -E "minivess|mamba"

# 2. GCP MLflow health
set -a && source .env && set +a
curl -sf "${MLFLOW_TRACKING_URI}/health" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)"

# 3. GCS DVC bucket populated
gsutil ls gs://minivess-mlops-dvc-data/ | wc -l  # expect ≥350

# 4. SkyPilot cloud access
uv run sky check runpod gcp 2>&1 | grep -E "enabled|ENABLED|OK"

# 5. No banned env vars in GCP YAML
python3 -c "
import pathlib, sys
banned = ['DVC_S3_ENDPOINT_URL','DVC_S3_ACCESS_KEY','upcloud','storage.fi-hel']
content = pathlib.Path('deployment/skypilot/smoke_test_gcp.yaml').read_text()
hits = [b for b in banned if b in content]
if hits: print('BANNED VARS FOUND:', hits); sys.exit(1)
print('OK')
"

# 6. RunPod Network Volume exists and has data
sky volumes ls 2>&1 | grep minivess-dev
```

If any P0 check fails, fix it before proceeding. See `on-failure` handlers in the plan XML.

## How to execute the plan

Work through the plan phases in order: P0 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9.

P3 (concurrent RunPod) and P4 (GCP sequential) can overlap if you start P4 while P3
is running — GCP and RunPod use independent quotas.

For each task, execute the `<commands>` block verbatim and check against `<acceptance>`
criteria. If acceptance fails, follow the `<on-failure>` handler. If no handler exists
and the failure is blocking, use ralph-loop to diagnose.

**DO NOT skip T2.4** (the inter-flow handoff test). This is the key proof that the
laptop→RunPod→Analysis contract works. It runs AFTER T2.2 (sync) and BEFORE T2.3
(teardown).

## P3 concurrent test — key design (read before running T3.2)

Both concurrent pods share ONE volume (`minivess-dev`). Isolation is via
MLFLOW_TRACKING_URI subdirectories:

```bash
# Pod A — dynunet
sky jobs launch deployment/skypilot/smoke_test_gpu.yaml \
  --env MODEL_FAMILY=dynunet \
  --env MLFLOW_TRACKING_URI=/opt/vol/mlruns/concurrent-a \
  --name minivess-concurrent-a -y &

# Pod B — sam3_vanilla
sky jobs launch deployment/skypilot/smoke_test_gpu.yaml \
  --env MODEL_FAMILY=sam3_vanilla \
  --env MLFLOW_TRACKING_URI=/opt/vol/mlruns/concurrent-b \
  --name minivess-concurrent-b -y &
```

NO second volume. NO S3 fallback. Both pods read the same `/opt/vol/data/raw/minivess`.

## Primary GCP QA goal (DO NOT merge without this)

`sam3_hybrid` on GCP L4 must produce **finite val_loss** (not NaN, not inf).
SAM3 Hybrid uses BF16 + sliding_window_inference. MONAI #4243: AMP + sliding_window = NaN.
The `smoke_test_gcp.yaml` MUST have `++mixed_precision=false` in `CLOUD_OVERRIDES`.
If you get NaN val_loss from sam3_hybrid, that is a P0 blocker — do NOT merge.

## When done — final gate before P9 merge

All of these must be true:
- [ ] P6.T6.1: all 4 GCP models with finite val_loss — PASSED
- [ ] P3.T3.4: zero MLflow run ID collisions — PASSED
- [ ] P2.T2.4: find_upstream_run() succeeds after RunPod sync — PASSED
- [ ] P7.T7.3: all TDD tests GREEN
- [ ] `uv run ruff check src/ tests/` → 0 errors
- [ ] `uv run mypy src/` → 0 errors
- [ ] `make test-staging` → all pass

## Key files

| File | Purpose |
|------|---------|
| `docs/planning/remaining-runpod-gcp-qa-4th-pass.xml` | Full v4 QA plan |
| `deployment/skypilot/smoke_test_gpu.yaml` | RunPod SkyPilot YAML |
| `deployment/skypilot/smoke_test_gcp.yaml` | GCP SkyPilot YAML |
| `deployment/skypilot/minivess-dev-volume.yaml` | Network Volume definition |
| `outputs/ralph_diagnoses.jsonl` | Ralph loop cost + failure log |
| `knowledge-graph/domains/cloud.yaml` | Two-provider architecture reference |
| `.env` | All credentials (RUNPOD_API_KEY, HF_TOKEN, MLFLOW_TRACKING_URI, etc.) |

---
## END OF PROMPT
