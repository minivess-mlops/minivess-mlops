# User Prompt: S3 Mounting, Testing & Simulation Plan (Verbatim)

**Date**: 2026-03-13
**Context**: During RunPod/SkyPilot integration testing (T0.1-T1.2 complete)

## Original User Message

> Well create an P0 Issue then on how to set-up the S3 to UpCloud, hopefully all programatically. Can we spin the S3 via Pulumi then so that we can build on top of Pulumi and then later add other "S3 spin scripts" for other clouds if needed? So some general S3 class with special cases for the different providers, all handled via Pulumi. And as you can authenticate to my UpCloud, plan then how to achieve this, with proper test suite to ensure that we can access that S3 from Docker running e.g. on Runpod, docker running on my local machine, and in "dev" environment without any docker. And as this S3 is the backend for DVC and MLFlow, then the S3 testing obviously should be skipped if someone wants to run this (or just test) all locally using some local dir as the mounted S3 artifact store for MLflow, and some other subdir as the Data folder simulating S3 that we for example symlink to the data folder inside the repo. Make this plan comprehensive as well for the /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/s3-mounting-testing-and-simulation-plan.md and optimize with reviewer agents until converging into a perfect plan / background research report with multi-hypothesis decision matrix on different options here, all in academic format. And start by writing my prompt verbatim to disk as appendix and start planning and do quality work. Quality over AI Slop then!

## Key Requirements Extracted

1. **Pulumi-based S3 provisioning** — spin up UpCloud S3 programmatically
2. **Multi-cloud S3 abstraction** — general S3 class with provider-specific implementations (UpCloud, AWS, GCP, MinIO)
3. **Three access environments** — Docker on RunPod, Docker locally, dev without Docker
4. **S3 for DVC + MLflow** — both use the same S3 backend
5. **Graceful local fallback** — skip S3 tests when running locally with local dirs
6. **Local S3 simulation** — symlinked data folder simulating S3
7. **Comprehensive test suite** — access tests from all 3 environments
8. **Academic format** — multi-hypothesis decision matrix
9. **Quality over AI slop** — thorough research, not superficial coverage
