# GCP Artifact Upload SUCCESS — SA Key in Cloud Run (2026-03-15)

## The Victory

After 16+ hours of debugging across RunPod → Lambda → GCP, the 910 MB checkpoint
artifact upload finally works. The full chain:

```
GCP T4 Spot (europe-west1) → Cloud Run MLflow (europe-north1) → GCS (europe-north1)
  ↑ multipart client             ↑ SA key for signing              ↑ artifacts stored
```

## What Was Required (All Three Together)

1. **Client**: `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` in SkyPilot envs
   → Makes MLflow client request multipart upload instead of single HTTP PUT

2. **Server**: `GOOGLE_APPLICATION_CREDENTIALS=/app/sa-key.json` on Cloud Run
   → SA key file (mounted via Secret Manager) gives MLflow the ability to
   generate **signed GCS URLs** for multipart upload chunks

3. **IAM**: `mlflow-server@` SA with `storage.objectAdmin` + `serviceAccountTokenCreator`
   → Permissions to write to GCS AND sign URLs

## Why Each Previous Attempt Failed

| Attempt | What Was Missing | Error |
|---------|-----------------|-------|
| UpCloud v2.20 | Wrong MLflow version | HTTP 500 (no multipart support) |
| UpCloud v3.10 | Still wrong (need server+client multipart) | HTTP 500 (single PUT too large) |
| Lambda → UpCloud | Cross-Atlantic latency | HTTP 500 (timeout) |
| GCP Cloud Run (rev 1) | No SA key for signing | `you need a private key to sign credentials` |
| GCP Cloud Run (rev 2) | SA without key file | Same error (Compute Engine creds can't sign) |
| **GCP Cloud Run (rev 3)** | **All three requirements met** | **SUCCESS** |

## Key Insight

Cloud Run service accounts use **Compute Engine credentials** by default — these
are token-based and **cannot generate signed URLs**. You need an explicit service
account **key file** (JSON) mounted via Secret Manager + `GOOGLE_APPLICATION_CREDENTIALS`.

This is NOT documented clearly in MLflow or GCP docs. The error message
("you need a private key to sign credentials") is the only hint.

## Files Changed

- `deployment/docker/Dockerfile.mlflow-gcp` — Custom MLflow image with psycopg2 + GCS
- `deployment/pulumi/gcp/__main__.py` — Full GCP Pulumi stack
- `deployment/skypilot/smoke_test_gcp.yaml` — GCP SkyPilot YAML with multipart env
- `.env.example` — GCP section with MLFLOW_SERVER_VERSION pinned
- Cloud Run deployed via `gcloud run deploy` with `--update-secrets` for SA key
