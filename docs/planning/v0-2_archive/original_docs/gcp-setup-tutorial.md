# GCP Setup Tutorial for MinIVess MLOps

> **Target audience**: Researchers setting up GCP for the first time.
> **Time**: ~30 minutes (plus quota approval wait).
> **Cost**: ~$30-50/month for infrastructure + GPU usage (spot).

---

## Prerequisites

- Google account (gmail)
- Credit card (for billing — GCP may offer $300 free trial for new accounts)
- Docker installed locally with `minivess-base:latest` built
- Pulumi CLI installed (`curl -fsSL https://get.pulumi.com | sh`)

---

## Step 1: Install Google Cloud CLI

```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell, then initialize
gcloud init
# → Sign in via browser
# → Select project: minivess-mlops (or create new)
```

## Step 2: Create GCP Project (if not done)

```bash
gcloud projects create minivess-mlops --name="MinIVess MLOps"
gcloud config set project minivess-mlops
```

## Step 3: Enable Billing

```bash
# List billing accounts
gcloud billing accounts list

# Link billing to project (replace ACCOUNT_ID)
gcloud billing projects link minivess-mlops --billing-account=ACCOUNT_ID
```

## Step 4: Enable Required APIs

```bash
# All APIs needed for MLflow + GPU training + storage
for api in \
  compute.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  storage.googleapis.com \
  storage-component.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com; do
    echo "Enabling $api..."
    gcloud services enable $api --project=minivess-mlops
done
```

## Step 5: Application Default Credentials

```bash
# Required for SkyPilot + Pulumi to authenticate
gcloud auth application-default login
# → Browser opens, sign in, credentials saved to:
# ~/.config/gcloud/application_default_credentials.json
```

## Step 6: Request GPU Quota

New GCP projects have `GPUS_ALL_REGIONS = 0`. You must request an increase:

```bash
# Check current quota
gcloud compute project-info describe --project=minivess-mlops \
  --format="json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'GPU' in q.get('metric', '').upper() and 'ALL' in q.get('metric', ''):
        print(f'{q[\"metric\"]}: limit={q[\"limit\"]}')
"
```

If `GPUS_ALL_REGIONS: 0.0`, go to:
https://console.cloud.google.com/iam-admin/quotas?project=minivess-mlops

1. Filter: `gpus_all_regions`
2. Click "Edit Quotas"
3. Set new limit: `1`
4. Submit (approved in minutes to hours for small requests)

## Step 7: Verify SkyPilot GCP Integration

```bash
# Install GCP Python SDK (if not already)
uv pip install google-api-python-client google-auth google-cloud-storage

# Ensure httplib2 compatibility
uv pip install "httplib2>=0.22,<0.23"

# Make gcloud accessible to SkyPilot API server
mkdir -p ~/.local/bin
cat > ~/.local/bin/gcloud << 'EOF'
#!/bin/bash
exec "$HOME/google-cloud-sdk/bin/gcloud" "$@"
EOF
chmod +x ~/.local/bin/gcloud

# Restart SkyPilot API server
uv run sky api stop
uv run sky api start

# Check GCP
uv run sky check gcp
# Should show: GCP: enabled [compute, storage]
```

## Step 8: Check Available GPUs

```bash
uv run sky gpus list --cloud gcp
# T4: 1-4, L4: 1-8, A100: 1-8, H100: 1-8
```

## Step 9: Update .env

Add to your `.env` file:

```bash
# ─── GCP ─────────────────────────────────────────────────────────────
GCP_PROJECT=minivess-mlops
GCP_REGION=europe-north1
GCS_MLFLOW_BUCKET=minivess-mlflow-artifacts
GCS_DVC_BUCKET=minivess-dvc-data
GCS_CHECKPOINT_BUCKET=minivess-checkpoints
MLFLOW_SERVER_VERSION=3.10.0
SKYPILOT_DEFAULT_CLOUD=gcp
```

## Step 10: Deploy Infrastructure (Pulumi)

```bash
cd deployment/pulumi/gcp
pulumi up
# Deploys: GCS buckets + Cloud SQL + Cloud Run MLflow + Artifact Registry
```

## Step 11: Push Docker Image to Artifact Registry

```bash
# Authenticate Docker with GAR
gcloud auth configure-docker europe-north1-docker.pkg.dev

# Tag and push
docker tag minivess-base:latest \
  europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
docker push \
  europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

## Step 12: Launch Training

```bash
# Smoke test on GCP T4 spot ($0.14/hr)
uv run python scripts/launch_smoke_test.py --model sam3_vanilla --cloud gcp

# Monitor
uv run sky status
uv run sky logs minivess-gcp-test
```

---

## GPU Pricing (Spot)

| GPU | VRAM | Spot $/hr | On-Demand $/hr | Discount |
|-----|------|-----------|----------------|----------|
| T4 | 16 GB | $0.14 | $0.35 | 60% |
| L4 | 24 GB | $0.22 | $0.56 | 60% |
| A100 40GB | 40 GB | $1.15 | $2.93 | 61% |
| A100 80GB | 80 GB | $1.57 | $3.93 | 60% |
| H100 | 80 GB | $2.25 | $9.80 | 77% |

## Regions with GPUs

| Region | GPUs | Notes |
|--------|------|-------|
| europe-north1 (Finland) | T4, L4, A100 | Default — closest to Helsinki |
| europe-west1 (Belgium) | T4, L4, V100, A100 | Cheapest EU |
| us-central1 (Iowa) | T4, L4, A100, H100 | Cheapest globally |
| asia-northeast1 (Tokyo) | T4, L4, A100 | For Asia-Pacific users |

## Customization for Your Lab

Create `configs/user/your_name.yaml`:

```yaml
# @package _global_
cloud:
  provider: gcp
  region: asia-northeast1  # Tokyo for your lab
  accelerators: [L4, A100]
  use_spot: true
```

Then launch with: `HYDRA_OVERRIDES="user=your_name" uv run python scripts/launch_smoke_test.py`

---

## Troubleshooting

### `GPUS_ALL_REGIONS quota exceeded`

Request quota increase: https://console.cloud.google.com/iam-admin/quotas

### `LazyImport object is not callable`

Fix httplib2/pyparsing version conflict:
```bash
uv pip install "httplib2>=0.22,<0.23"
```

### `gcloud: command not found` in SkyPilot

Create wrapper script:
```bash
mkdir -p ~/.local/bin
echo '#!/bin/bash' > ~/.local/bin/gcloud
echo 'exec "$HOME/google-cloud-sdk/bin/gcloud" "$@"' >> ~/.local/bin/gcloud
chmod +x ~/.local/bin/gcloud
uv run sky api stop && uv run sky api start
```

### MLflow 403 `Invalid Host header`

Set `MLFLOW_SERVER_ALLOWED_HOSTS=*` in MLflow server environment.

### MLflow artifact upload 500 errors

Ensure server version matches client (both 3.10.0):
```bash
MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true
MLFLOW_SERVER_VERSION=3.10.0  # Pin in .env.example
```
