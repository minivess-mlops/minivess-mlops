---
paths:
  - "deployment/**"
  - "configs/cloud/**"
  - ".dvc/**"
  - "scripts/configure_dvc_remote.py"
---

EXACTLY two cloud providers: RunPod (env/dev) + GCP (staging/prod).
GCP project: minivess-mlops, region: us-central1.
Data on GCS (gs://minivess-mlops-dvc-data). NEVER add AWS/Azure/others.
NEVER change cloud architecture from a session continuation summary — ASK the user.

See: knowledge-graph/domains/cloud.yaml for full architecture.
See: deployment/pulumi/gcp/CLAUDE.md for GCP project details.
