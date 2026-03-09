# 3D Slicer + MONAI Label Setup Guide

Interactive annotation for MinIVess volumes using 3D Slicer with MONAI Label
server and BentoML champion pre-segmentation.

## Architecture

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  3D Slicer    │────▶│  MONAI Label     │────▶│  BentoML Server   │
│  (desktop)    │◀────│  (Docker :8000)  │◀────│  (Docker :3333)   │
└───────────────┘     └──────────────────┘     └───────────────────┘
       │                      │
       │ Submit label         │ Save approved
       ▼                      ▼
┌───────────────┐     ┌──────────────────┐
│  Annotation   │────▶│  DVC + MLflow    │
│  Approval     │     │  (versioning)    │
│  (Prefect)    │     └──────────────────┘
└───────────────┘
```

## Prerequisites

1. Docker Compose infrastructure running (PostgreSQL, MinIO, MLflow)
2. BentoML deploy flow completed (champion model exported and serving)
3. 3D Slicer installed on the desktop machine

## Step 1: Start MONAI Label Server

```bash
# Start the MONAI Label server (reads BentoML champion for pre-segmentation)
docker compose --env-file .env -f deployment/docker-compose.flows.yml up monai-label

# Verify it's running:
curl http://localhost:8000/info
```

The server will be available at `http://localhost:${MONAI_LABEL_PORT}` (default: 8000).

## Step 2: Install 3D Slicer

1. Download from [slicer.org](https://download.slicer.org/)
2. Install for your platform (Windows, macOS, Linux)
3. Launch 3D Slicer

## Step 3: Install MONAILabel Extension

1. In 3D Slicer: **View → Extension Manager**
2. Search for **"MONAILabel"**
3. Click **Install**
4. Restart 3D Slicer when prompted

## Step 4: Connect to MONAI Label Server

1. In 3D Slicer: **Modules → Active Learning → MONAILabel**
2. Set **Server URL**: `http://localhost:8000` (or your `MONAI_LABEL_PORT`)
3. Click **Connect** — the volume list should populate

## Step 5: Annotate

1. **Browse volumes** in the left panel (loaded from `/data/studies`)
2. Click **"Get"** next to a volume — this triggers BentoML pre-segmentation
3. The champion model's prediction appears as an initial mask
4. Use **Segment Editor** tools to correct the mask:
   - **Paint** — brush tool for adding regions
   - **Erase** — remove false positives
   - **Threshold** — intensity-based refinement
   - **Scissors** — cut regions
   - **Islands** — remove disconnected components
5. Click **"Submit"** to approve the corrected mask

## Step 6: Approval Pipeline

Submitted labels trigger the annotation approval Prefect task:
- Mask validated (binary, correct spatial dimensions)
- `dvc add` versions the approved label
- MLflow logs the annotation event with metadata
- Version tag: `annotation/v{YYYY-MM-DD}-{hash[:8]}`

## Troubleshooting

### 3D Slicer cannot connect to MONAI Label

```bash
# Check if MONAI Label container is running:
docker ps | grep monai-label

# Check container logs:
docker compose --env-file .env -f deployment/docker-compose.flows.yml logs monai-label

# Verify the port is accessible:
curl http://localhost:8000/info
```

### Pre-segmentation returns empty mask

The champion model is served by BentoML. Check that BentoML is running:

```bash
# Check BentoML health:
curl http://localhost:3333/health

# If BentoML is not running, MONAI Label falls back to a zero mask.
# The annotator can still segment from scratch.
```

### Volume not showing in 3D Slicer

Volumes must be in the studies directory (mounted at `/data/studies`):

```bash
# Check the data volume:
docker compose --env-file .env -f deployment/docker-compose.flows.yml exec monai-label ls /data/studies/
```

## References

- [MONAILabel documentation](https://docs.monai.io/projects/label/en/latest/)
- [3D Slicer MONAILabel module](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer)
- [3D Slicer Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
