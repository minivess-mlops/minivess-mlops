# NVIDIA DCGM Exporter — GPU Hardware Monitoring

**Phase 11 deliverable** — Add to `deployment/docker-compose.yml` (infra stack).

Reference: [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)

## Docker Compose Service

```yaml
  # NVIDIA GPU hardware metrics → Prometheus format at :9400
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04
    container_name: minivess-dcgm-exporter
    devices:
      - "nvidia.com/gpu=all"
    ports:
      - "${DCGM_EXPORTER_PORT:-9400}:9400"
    networks:
      - minivess
    restart: unless-stopped
    cap_drop:
      - ALL
    labels:
      project: minivess-mlops
      managed-by: docker-compose
```

## LGTM Prometheus Scrape Config

The Grafana LGTM stack's embedded Prometheus needs to know about the DCGM
Exporter endpoint. Create a custom Prometheus config overlay:

```yaml
# deployment/grafana/prometheus-extra.yml
scrape_configs:
  - job_name: dcgm-exporter
    static_configs:
      - targets: ['minivess-dcgm-exporter:9400']
    scrape_interval: 15s
```

## .env.example Addition

```bash
# ── NVIDIA DCGM GPU Metrics ─────────────────────────────
DCGM_EXPORTER_PORT=9400
```

## Grafana Dashboard

Import the official NVIDIA DCGM dashboard:
- **Dashboard ID**: 12239
- **URL**: https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/
- **Download**: Save JSON to `deployment/grafana/dashboards/nvidia-dcgm.json`

## Metrics Exposed

| Metric | Description |
|--------|-------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization % |
| `DCGM_FI_DEV_FB_USED` | Framebuffer memory used (MB) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (°C) |
| `DCGM_FI_DEV_POWER_USAGE` | Power consumption (W) |
| `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe TX throughput |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | ECC single-bit errors |

## What This Would Have Caught

In the 4-hour CPU fallback incident:
- `DCGM_FI_DEV_GPU_UTIL = 0%` for 4 hours → immediate Prometheus alert
- `DCGM_FI_DEV_FB_USED = 0 MB` → no model loaded on GPU
- Combined with Grafana dashboard → visible in seconds, not hours
