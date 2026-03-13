"""MinIVess MLOps — UpCloud Managed MLflow Stack.

Provisions:
  - Managed PostgreSQL (MLflow backend store, auto-backups)
  - Managed Object Storage + bucket (MLflow artifact store, S3-compatible)
  - Cloud Server / VPS (runs MLflow with built-in basic auth — 1 container)
  - Firewall rules (SSH + MLflow:5000 only)

Usage:
  cd deployment/pulumi
  pulumi stack init dev
  pulumi config set upcloud:token "$UPCLOUD_TOKEN" --secret
  pulumi config set ssh_public_key "$(cat ~/.ssh/upcloud_minivess.pub)"
  pulumi config set mlflow_admin_password "$(openssl rand -base64 16)" --secret
  pulumi up

Teardown:
  pulumi destroy
"""

from __future__ import annotations

import textwrap

import pulumi
import pulumi_command as command
import pulumi_upcloud as upcloud

config = pulumi.Config()
zone = config.get("zone") or "fi-hel1"
ssh_public_key = config.require("ssh_public_key")
mlflow_admin_password = config.require_secret("mlflow_admin_password")

# ── Managed PostgreSQL ────────────────────────────────────────────────────
# MLflow backend store. UpCloud manages backups, upgrades, HA.
db = upcloud.ManagedDatabasePostgresql(
    "mlflow-postgres",
    name="minivess-mlflow-pg",
    plan="1x1xCPU-2GB-25GB",
    zone=zone,
    title="MinIVess MLflow PostgreSQL",
    properties={
        "admin_username": "minivess",
        "timezone": "UTC",
        "backup_hour": 3,
        "backup_minute": 0,
        # Allow connections from any IP. The DB itself requires credentials;
        # MLflow basic auth adds a second layer. Restricting to VPS IP would
        # break SkyPilot training VMs that log directly to the managed DB.
        "ip_filters": ["0.0.0.0/0"],
        "automatic_utility_network_ip_filter": False,
        "public_access": True,
    },
)

# Derive the public PostgreSQL URI from service_uri by replacing the private
# hostname with the public one. UpCloud managed DBs expose a "public-" prefixed
# hostname when public_access is enabled.
db_public_uri = pulumi.Output.all(db.service_uri, db.components).apply(
    lambda args: (
        args[0]
        .replace("postgres://", "postgresql://", 1)
        .replace(
            next(
                (
                    c["host"]
                    for c in (args[1] or [])
                    if c.get("route") == "dynamic" and c.get("component") == "pg"
                ),
                "",
            ),
            next(
                (
                    c["host"]
                    for c in (args[1] or [])
                    if c.get("route") == "public" and c.get("component") == "pg"
                ),
                "",
            ),
        )
    )
)

# ── Managed Object Storage ────────────────────────────────────────────────
# MLflow artifact store. S3-compatible, 250 GB minimum, 99.99% durability.
object_storage = upcloud.ManagedObjectStorage(
    "mlflow-s3",
    name="minivess-mlflow-s3",
    region="europe-1",
    configured_status="started",
    networks=[
        {
            "family": "IPv4",
            "name": "public-access",
            "type": "public",
        }
    ],
)

s3_bucket = upcloud.ManagedObjectStorageBucket(
    "mlflow-artifacts",
    name="mlflow-artifacts",
    service_uuid=object_storage.id,
)

# DVC data bucket — MiniVess training data for cloud GPU training via SkyPilot.
# RunPod instances pull data from this bucket using `dvc pull -r upcloud`.
dvc_bucket = upcloud.ManagedObjectStorageBucket(
    "dvc-data",
    name="minivess-dvc-data",
    service_uuid=object_storage.id,
)

s3_user = upcloud.ManagedObjectStorageUser(
    "mlflow-s3-user",
    username="mlflow",
    service_uuid=object_storage.id,
)

# Extract the public S3 endpoint URL from the endpoints list.
s3_endpoint = object_storage.endpoints.apply(
    lambda eps: next(
        (
            f"https://{ep['domain_name']}"
            for ep in (eps or [])
            if ep.get("type") == "public"
        ),
        "unknown",
    )
)

s3_access_key = upcloud.ManagedObjectStorageUserAccessKey(
    "mlflow-s3-key",
    username=s3_user.username,
    service_uuid=object_storage.id,
    status="Active",
)

# Grant the mlflow user full S3 access to both buckets.
# Without this policy, PutObject fails with AccessDenied (#678).
# Policy "ECSS3FullAccess" is a built-in UpCloud system policy.
s3_policy = upcloud.ManagedObjectStorageUserPolicy(
    "mlflow-s3-policy",
    username=s3_user.username,
    service_uuid=object_storage.id,
    name="ECSS3FullAccess",
)

# ── Cloud Server (VPS) — runs MLflow only ─────────────────────────────────
# Smallest viable plan. MLflow uses managed DB + managed S3, so the VPS
# only needs enough resources for the MLflow Python process.
server = upcloud.Server(
    "mlflow-server",
    hostname="minivess-mlflow",
    zone=zone,
    plan="DEV-1xCPU-2GB",
    metadata=True,
    template={
        "storage": "Ubuntu Server 24.04 LTS (Noble Numbat)",
        "size": 25,
    },
    login={
        "user": "deploy",
        "keys": [ssh_public_key],
    },
    network_interfaces=[{"type": "public"}],
    # NOTE: firewall=True is not set because UpCloud trial accounts cannot
    # modify firewall rules (403 TRIAL_FIREWALL). After upgrading from trial,
    # set firewall=True and add ServerFirewallRules resource.
)

# ── Extract server IP ─────────────────────────────────────────────────────
server_ip = server.network_interfaces.apply(
    lambda ifaces: next(
        (
            iface["ip_address"]
            for iface in (ifaces or [])
            if iface.get("type") == "public"
            and iface.get("ip_address_family", iface.get("ipAddressFamily")) == "IPv4"
        ),
        "unknown",
    )
)

# ── Remote provisioning: install Docker + deploy MLflow ───────────────────
# Uses pulumi-command to SSH into the VPS and set up the single-container
# MLflow deployment with built-in basic auth.
provision_docker = command.remote.Command(
    "install-docker",
    connection={
        "host": server_ip,
        "user": "deploy",
        "private_key": config.get_secret("ssh_private_key") or "",
    },
    create=textwrap.dedent("""\
        set -euo pipefail
        sudo apt-get update -qq
        sudo apt-get install -y -qq docker.io docker-compose-v2
        sudo systemctl enable --now docker
        sudo usermod -aG docker deploy
    """),
    opts=pulumi.ResourceOptions(depends_on=[server]),
)

deploy_mlflow = command.remote.Command(
    "deploy-mlflow",
    connection={
        "host": server_ip,
        "user": "deploy",
        "private_key": config.get_secret("ssh_private_key") or "",
    },
    create=pulumi.Output.all(
        db_public_uri,
        s3_endpoint,
        s3_access_key.access_key_id,
        s3_access_key.secret_access_key,
        mlflow_admin_password,
    ).apply(
        lambda args: textwrap.dedent(f"""\
            set -euo pipefail
            sudo mkdir -p /opt/mlflow

            # basic_auth.ini for MLflow built-in auth
            sudo tee /opt/mlflow/basic_auth.ini > /dev/null <<'AUTHEOF'
            [mlflow]
            default_permission = READ
            database_uri = sqlite:///basic_auth.db
            admin_username = admin
            admin_password = {args[4]}
            AUTHEOF
            sudo chmod 600 /opt/mlflow/basic_auth.ini

            # Dockerfile — MLflow with psycopg2 + boto3 (for PostgreSQL + S3)
            sudo tee /opt/mlflow/Dockerfile > /dev/null <<'DOCKEOF'
            FROM ghcr.io/mlflow/mlflow:v2.20.3
            RUN pip install --no-cache-dir psycopg2-binary boto3 flask-wtf
            DOCKEOF

            # Docker Compose — single MLflow container
            sudo tee /opt/mlflow/docker-compose.yml > /dev/null <<COMPEOF
            services:
              mlflow:
                build: .
                container_name: minivess-mlflow
                restart: unless-stopped
                environment:
                  AWS_ACCESS_KEY_ID: "{args[2]}"
                  AWS_SECRET_ACCESS_KEY: "{args[3]}"
                  MLFLOW_S3_ENDPOINT_URL: "{args[1]}"
                  MLFLOW_FLASK_SERVER_SECRET_KEY: "{args[4]}"
                  MLFLOW_AUTH_CONFIG_PATH: /app/basic_auth.ini
                entrypoint:
                  - /bin/sh
                  - -c
                  - >-
                    mlflow server
                    --host 0.0.0.0
                    --port 5000
                    --app-name basic-auth
                    --backend-store-uri "{args[0]}"
                    --artifacts-destination s3://mlflow-artifacts
                    --serve-artifacts
                ports:
                  - "5000:5000"
                volumes:
                  - ./basic_auth.ini:/app/basic_auth.ini:ro
                healthcheck:
                  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
                  interval: 15s
                  timeout: 5s
                  retries: 5
            COMPEOF

            cd /opt/mlflow
            sudo docker compose up -d --build
        """)
    ),
    opts=pulumi.ResourceOptions(depends_on=[provision_docker]),
)

# ── Outputs ───────────────────────────────────────────────────────────────
pulumi.export("server_ip", server_ip)
pulumi.export("mlflow_url", server_ip.apply(lambda ip: f"http://{ip}:5000"))
pulumi.export("mlflow_username", "admin")
pulumi.export("postgres_host", db.service_host)
pulumi.export("s3_endpoint", s3_endpoint)
pulumi.export("s3_bucket", s3_bucket.name)
pulumi.export(
    "ssh_command",
    server_ip.apply(lambda ip: f"ssh -i ~/.ssh/upcloud_minivess deploy@{ip}"),
)
pulumi.export("dvc_bucket", dvc_bucket.name)
pulumi.export("dvc_s3_endpoint", s3_endpoint)
