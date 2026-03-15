"""MinIVess MLOps — GCP Full Stack via Pulumi.

Deploys same-region infrastructure in europe-north1 (Finland):
  - GCS buckets (MLflow artifacts, DVC data, checkpoints)
  - Cloud SQL PostgreSQL (MLflow backend + Optuna)
  - Artifact Registry (Docker images for SkyPilot)
  - Cloud Run MLflow server (v3.10.0 with GCS artifact store)
  - IAM service account (SkyPilot VMs + Cloud Run)

Usage:
  cd deployment/pulumi/gcp
  pulumi stack init dev
  pulumi config set minivess-gcp:mlflow_admin_password --secret "YOUR_PASSWORD"
  pulumi config set minivess-gcp:db_password --secret "YOUR_DB_PASSWORD"
  pulumi up

All config from .env.example (CLAUDE.md Rule #22).
MLflow version pinned via MLFLOW_SERVER_VERSION (metalearning 2026-03-14).
"""

from __future__ import annotations

import pulumi
import pulumi_gcp as gcp

# ── Config ───────────────────────────────────────────────────────────────

config = pulumi.Config("minivess-gcp")
project = config.get("project") or "minivess-mlops"
region = config.get("region") or "europe-north1"
mlflow_password = config.require_secret("mlflow_admin_password")
db_password = config.require_secret("db_password")

# MLflow version — MUST match client (pyproject.toml → uv.lock → 3.10.0)
# See: .claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md
MLFLOW_SERVER_VERSION = "3.10.0"

# ── GCS Buckets ──────────────────────────────────────────────────────────

mlflow_artifacts_bucket = gcp.storage.Bucket(
    "mlflow-artifacts",
    name=f"{project}-mlflow-artifacts",
    location=region.upper(),
    # Use exact region for single-region bucket (same-region as training VMs)
    force_destroy=True,
    uniform_bucket_level_access=True,
    lifecycle_rules=[
        gcp.storage.BucketLifecycleRuleArgs(
            action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
            condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                age=7,
                with_state="ARCHIVED",
            ),
        ),
    ],
)

dvc_data_bucket = gcp.storage.Bucket(
    "dvc-data",
    name=f"{project}-dvc-data",
    location=region.upper(),
    force_destroy=True,
    uniform_bucket_level_access=True,
)

checkpoints_bucket = gcp.storage.Bucket(
    "checkpoints",
    name=f"{project}-checkpoints",
    location=region.upper(),
    force_destroy=True,
    uniform_bucket_level_access=True,
    lifecycle_rules=[
        gcp.storage.BucketLifecycleRuleArgs(
            action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
            condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                age=30,
                with_state="ARCHIVED",
            ),
        ),
    ],
)

# ── Cloud SQL PostgreSQL ─────────────────────────────────────────────────

db_instance = gcp.sql.DatabaseInstance(
    "mlflow-db",
    database_version="POSTGRES_15",
    region=region,
    deletion_protection=False,
    settings=gcp.sql.DatabaseInstanceSettingsArgs(
        tier="db-g1-small",  # 1.7 GB RAM (reviewer R9: db-f1-micro too small)
        disk_size=10,
        disk_type="PD_SSD",
        ip_configuration=gcp.sql.DatabaseInstanceSettingsIpConfigurationArgs(
            ipv4_enabled=True,
            # Public IP for initial setup; switch to private after VPC setup
            authorized_networks=[
                gcp.sql.DatabaseInstanceSettingsIpConfigurationAuthorizedNetworkArgs(
                    name="allow-all-initial",
                    value="0.0.0.0/0",
                ),
            ],
        ),
        backup_configuration=gcp.sql.DatabaseInstanceSettingsBackupConfigurationArgs(
            enabled=True,
            start_time="03:00",
        ),
    ),
)

# MLflow database
mlflow_db = gcp.sql.Database(
    "mlflow-database",
    instance=db_instance.name,
    name="mlflow",
)

# Optuna database
optuna_db = gcp.sql.Database(
    "optuna-database",
    instance=db_instance.name,
    name="optuna",
)

# Database user
db_user = gcp.sql.User(
    "mlflow-user",
    instance=db_instance.name,
    name="mlflow",
    password=db_password,
)

# ── Artifact Registry (Docker) ──────────────────────────────────────────

docker_repo = gcp.artifactregistry.Repository(
    "docker-repo",
    repository_id="minivess",
    location=region,
    format="DOCKER",
    description="MinIVess MLOps Docker images (same-region as training VMs)",
)

# ── IAM Service Account ─────────────────────────────────────────────────

skypilot_sa = gcp.serviceaccount.Account(
    "skypilot-sa",
    account_id="skypilot-training",
    display_name="SkyPilot Training Service Account",
    description="Used by SkyPilot VMs for GPU training + GCS access + GAR pulls",
)

# Grant roles to the service account
for role_name, role in [
    ("compute-admin", "roles/compute.admin"),
    ("storage-admin", "roles/storage.objectAdmin"),
    ("sql-client", "roles/cloudsql.client"),
    ("gar-reader", "roles/artifactregistry.reader"),  # R5: for Docker image pulls
    ("log-writer", "roles/logging.logWriter"),
    ("monitoring-writer", "roles/monitoring.metricWriter"),
]:
    gcp.projects.IAMMember(
        f"skypilot-{role_name}",
        project=project,
        role=role,
        member=skypilot_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

# ── Cloud Run MLflow Server ──────────────────────────────────────────────

# Build the connection string for Cloud SQL
db_connection_uri = pulumi.Output.all(db_instance.public_ip_address, db_password).apply(
    lambda args: f"postgresql://mlflow:{args[1]}@{args[0]}:5432/mlflow"
)

# NOTE: Cloud Run MLflow deployment is Phase 1.5 — requires custom Docker image
# built and pushed to GAR first (with psycopg2 + google-cloud-storage baked in).
# For now, deploy infrastructure (GCS + Cloud SQL + GAR + IAM) first.
# Cloud Run will be added after the base image is ready.
# TODO(P1.5): Uncomment and fix Cloud Run service after GAR image is pushed.
_CLOUD_RUN_DEFERRED = True

if not _CLOUD_RUN_DEFERRED:
    mlflow_service = gcp.cloudrunv2.Service(
        "mlflow-server",
        name="minivess-mlflow",
        location=region,
        ingress="INGRESS_TRAFFIC_ALL",
        template={
            "scaling": {
                "min_instance_count": 1,  # R13: avoid cold starts
                "max_instance_count": 2,
            },
            "containers": [
                {
                    "image": f"ghcr.io/mlflow/mlflow:v{MLFLOW_SERVER_VERSION}",
                    "ports": [{"container_port": 5000}],
                    "envs": [
                        {
                            "name": "MLFLOW_BACKEND_STORE_URI",
                            "value": db_connection_uri,
                        },
                        {
                            "name": "MLFLOW_ARTIFACTS_DESTINATION",
                            "value": mlflow_artifacts_bucket.name.apply(
                                lambda n: f"gs://{n}"
                            ),
                        },
                        {
                            "name": "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD",
                            "value": "true",
                        },
                        {"name": "MLFLOW_SERVER_ALLOWED_HOSTS", "value": "*"},
                        {
                            "name": "MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT",
                            "value": "1800",
                        },
                    ],
                    "resources": {"limits": {"cpu": "1", "memory": "2Gi"}},
                    "commands": ["sh"],
                    "args": [
                        "-c",
                        (
                            "pip install --no-cache-dir psycopg2-binary "
                            "google-cloud-storage boto3 flask-wtf && "
                            "mlflow server "
                            "--host 0.0.0.0 "
                            "--port 5000 "
                            "--backend-store-uri $MLFLOW_BACKEND_STORE_URI "
                            "--artifacts-destination $MLFLOW_ARTIFACTS_DESTINATION "
                            "--serve-artifacts"
                        ),
                    ],
                    "startup_probe": {
                        "http_get": {"path": "/health", "port": 5000},
                        "initial_delay_seconds": 30,
                        "period_seconds": 10,
                        "failure_threshold": 10,
                    },
                },
            ],
            # Cloud SQL connection via Cloud SQL Auth Proxy sidecar (R4)
            "volumes": [
                {
                    "name": "cloudsql",
                    "cloud_sql_instance": {
                        "instances": [db_instance.connection_name],
                    },
                },
            ],
        },
    )

    # Make Cloud Run service publicly accessible (with basic auth via MLflow)
    mlflow_iam = gcp.cloudrunv2.ServiceIamMember(
        "mlflow-public-access",
        name=mlflow_service.name,
        location=region,
        role="roles/run.invoker",
        member="allUsers",
    )

# ── Outputs ──────────────────────────────────────────────────────────────

if not _CLOUD_RUN_DEFERRED:
    pulumi.export("mlflow_url", mlflow_service.uri)
pulumi.export("mlflow_version", MLFLOW_SERVER_VERSION)
pulumi.export("db_connection_name", db_instance.connection_name)
pulumi.export("db_public_ip", db_instance.public_ip_address)
pulumi.export("mlflow_artifacts_bucket", mlflow_artifacts_bucket.name)
pulumi.export("dvc_data_bucket", dvc_data_bucket.name)
pulumi.export("checkpoints_bucket", checkpoints_bucket.name)
pulumi.export(
    "docker_registry",
    docker_repo.name.apply(lambda n: f"{region}-docker.pkg.dev/{project}/{n}"),
)
pulumi.export("skypilot_service_account", skypilot_sa.email)
pulumi.export("region", region)
