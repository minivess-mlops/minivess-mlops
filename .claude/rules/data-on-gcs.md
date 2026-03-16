---
paths:
  - "deployment/skypilot/**"
  - "scripts/configure_dvc_remote.py"
  - ".dvc/**"
  - "configs/dvc/**"
---

Production data storage: GCS (gs://minivess-mlops-dvc-data).
DVC remote for GCP staging/prod: gcs (Google Cloud Storage).
RunPod env: data uploaded from local disk to Network Volume (no cloud dependency).
s3://minivessdataset is a READ-ONLY public data origin, not a production backend.
NEVER use AWS S3 as infrastructure — it is not part of the architecture.
