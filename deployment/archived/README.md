# deployment/archived/

Archived infrastructure code for providers that have been dropped from the active stack.
Code is preserved here for reference but is **never used in active workflows**.

## upcloud/

UpCloud VPS (Helsinki, fi-hel1) was the MLflow server + S3-compatible DVC storage.
**Dropped 2026-03-16.** Replaced by:
- MLflow: RunPod Network Volume file-based (`/opt/vol/mlruns`, `MLFLOW_TRACKING_URI=/opt/vol/mlruns`)
- DVC storage: AWS S3 (`s3://minivessdataset`) + RunPod Network Volume local cache

| File | What it was |
|------|------------|
| `pulumi/__main__.py` | Pulumi IaC for UpCloud Managed PostgreSQL + Object Storage + VPS |
| `scripts/upcloud-setup-script.sh` | Manual VPS bootstrap (pre-Pulumi) |
| `docs/pulumi-upcloud-managed-deployment.md` | Architecture doc |
| `docs/upcloud-mlflow-plan.md` | MLflow-on-UpCloud deployment plan |

## lambda/

Lambda Labs was evaluated as staging/prod GPU provider but rejected (no EU availability).
**Never deployed in production. Archived 2026-03-16.**

Active providers: RunPod (dev), GCP L4 (production).

| File | What it was |
|------|------------|
| `skypilot/smoke_test_lambda.yaml` | SkyPilot smoke test YAML for Lambda Labs |
| `configs/lambda.yaml` | Hydra cloud config for Lambda Labs |
| `docs/lambda-labs-staging-env-for-training-plan.md` | Architecture doc |
| `docs/lambda-mlflow-failure-analysis.md` | MLflow artifact upload failure analysis |
