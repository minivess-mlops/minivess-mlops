# Grafana LGTM Stack — Docker Compose Integration

**Phase 9 deliverable** — Add to `deployment/docker-compose.yml` (infra stack).

Reference: [Grafana Docker OTel LGTM](https://grafana.com/docs/opentelemetry/docker-lgtm/)

## Docker Compose Service

```yaml
  # Unified observability: OTel Collector + Prometheus + Tempo + Loki + Grafana
  grafana-lgtm:
    image: grafana/otel-lgtm:latest
    container_name: minivess-grafana-lgtm
    ports:
      - "${GRAFANA_LGTM_PORT:-3001}:3000"       # Grafana UI (3001 to avoid MLflow conflict)
      - "${OTEL_GRPC_PORT:-4317}:4317"           # OTLP gRPC receiver
      - "${OTEL_HTTP_PORT:-4318}:4318"           # OTLP HTTP receiver
    volumes:
      - grafana_lgtm_data:/data
      - ./grafana/dashboards:/otel-lgtm/grafana/dashboards:ro
    networks:
      - minivess
    restart: unless-stopped
    labels:
      project: minivess-mlops
      managed-by: docker-compose
```

## Volume Definition

```yaml
volumes:
  grafana_lgtm_data: {}
```

## x-common-env Addition (docker-compose.flows.yml)

```yaml
x-common-env: &common-env
  # ... existing vars ...
  OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT:-http://minivess-grafana-lgtm:4318}
```

## .env.example Addition

```bash
# ── Observability Backend (Grafana LGTM) ─────────────────
GRAFANA_LGTM_PORT=3001
OTEL_GRPC_PORT=4317
OTEL_HTTP_PORT=4318
OTEL_EXPORTER_OTLP_ENDPOINT=http://minivess-grafana-lgtm:4318
```

## Access

- **Grafana UI**: http://localhost:3001 (admin/admin)
- **OTLP HTTP endpoint**: http://localhost:4318
- **OTLP gRPC endpoint**: localhost:4317

## Custom Dashboards

Place JSON dashboard files in `deployment/grafana/dashboards/`:
- `ml-training-overview.json` — custom training monitoring dashboard
- Download NVIDIA DCGM dashboard from https://grafana.com/grafana/dashboards/12239
