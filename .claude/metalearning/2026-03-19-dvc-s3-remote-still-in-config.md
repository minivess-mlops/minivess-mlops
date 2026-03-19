# Metalearning: DVC S3 Remote Still in Config — BANNED Provider Not Removed

**Date:** 2026-03-19
**Severity:** P0 — banned cloud provider still referenced in config
**Trigger:** `make test-prod` fails on `test_dvc_cloud_pull.py` which tests
`dvc status -r remote_storage` pointing to `s3://minivessdataset`

---

## What Happened

1. `.dvc/config` has `remote_storage` pointing to `s3://minivessdataset` (AWS S3)
2. A test (`test_dvc_cloud_pull.py`) validates this remote works
3. The test fails because no AWS credentials are configured
4. This was classified as "environment issue" and ignored

## The Real Problem

AWS S3 is a BANNED provider. Per KG `cloud.yaml`:
- `s3://minivessdataset` was removed 2026-03-16 (unauthorized 3rd provider)
- DVC remotes should be: `minio` (local), `gcs` (GCP)
- `remote_storage` and `remote_readonly` entries are STALE remnants

The metalearning doc `2026-03-16-unauthorized-aws-s3-architecture-migration.md`
documents the S3 removal, but the DVC config was never cleaned up. The test was
written to validate a remote that should no longer exist.

## Fix

1. Remove `remote_storage` and `remote_readonly` from `.dvc/config`
2. Delete `test_dvc_cloud_pull.py` (tests a banned remote)
3. Keep `minio` (local) and `gcs` (GCP) as the only DVC remotes

## Rule

When a cloud provider is BANNED, remove ALL references including:
- DVC remotes
- Tests that validate the banned remote
- Documentation that references it
- `.env.example` variables for it
