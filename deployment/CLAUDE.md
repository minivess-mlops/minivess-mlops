# Deployment — Docker Infrastructure

## Three-Layer Docker Hierarchy

```
Layer 1: nvidia/cuda:12.6.3-runtime-ubuntu24.04  (upstream, never modified)
Layer 2: minivess-base:latest                     (THIS — all shared deps)
Layer 3: Dockerfile.{flow}                        (thin — scripts, env, CMD only)
```

**Flow Dockerfiles NEVER run `apt-get` or `uv`** — all deps belong in Dockerfile.base.

## Building

```bash
# Base image (rebuild when pyproject.toml or Dockerfile.base changes):
docker build -t minivess-base:latest -f deployment/docker/Dockerfile.base .

# Match host UID for bind-mount dev:
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
  -t minivess-base:latest -f deployment/docker/Dockerfile.base .
```

## Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Infrastructure: PostgreSQL, MinIO, MLflow, Prefect, Grafana |
| `docker-compose.flows.yml` | Per-flow services: 12 flow containers |

## Volume Mount Rules (Non-Negotiable)

Every artifact that must survive the container MUST be volume-mounted:

| Mount | Used By | Mode |
|-------|---------|------|
| `raw_data:/app/data/raw` | acquisition, data(ro) | rw/ro |
| `data_cache:/app/data` | data, train(ro), analyze(ro), hpo(ro) | varies |
| `configs_splits:/app/configs/splits` | data, train(ro), analyze(ro), hpo(ro) | varies |
| `checkpoint_cache:/app/checkpoints` | train, post_training(ro), analyze(ro), hpo | varies |
| `mlruns_data:/app/mlruns` | most flows | varies |
| `logs_data:/app/logs` | acquisition, train, hpo | rw |

**/tmp and tempfile.mkdtemp() are FORBIDDEN for artifacts.**

## Network

All services use the `minivess-network` external network:
```bash
docker network create minivess-network
```

## GPU Reservation

Only `train` and `hpo` services reserve GPU devices:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Running Flows

```bash
# Start infrastructure first:
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Run a flow:
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model train

# With Hydra overrides:
docker compose -f deployment/docker-compose.flows.yml run --rm \
  -e EXPERIMENT=debug_single_model \
  -e HYDRA_OVERRIDES="max_epochs=5,model=sam3_vanilla" \
  train
```
