# Infrastructure Optimization Research Report: Docker Pull, RunPod Provisioning, MLflow Artifact Upload

**Authors**: Claude Opus 4.6 (1M context) + Petteri Teikari
**Date**: 2026-03-17
**Issues**: #751 (Docker pull), #754 (RunPod provisioning), #755 (MLflow upload)
**Platform**: MinIVess MLOps v2 --- MONAI Ecosystem Extension for Multiphoton Biomedical Imaging

---

## Abstract

Three infrastructure bottlenecks --- 15-minute Docker image pulls from Google Artifact Registry (GAR), 40+ minute RunPod provisioning failures, and HTTP 500 errors on 900 MB MLflow checkpoint uploads through Cloud Run --- collectively consume 58% of GPU smoke test wall time and block the critical path for the MinIVess MLOps platform (#734). This report surveys the academic literature on container image distribution, GPU cold-start optimization, and cloud artifact management, then constructs multi-hypothesis design matrices for each issue. We identify 12 hypotheses for Docker pull optimization, 7 for RunPod provisioning, and 4 for MLflow upload, ranked by expected impact, implementation effort, and compatibility with the existing SkyPilot + GAR + GCP architecture. The recommended execution order follows a ralph-loop-friendly cheapest-first strategy: fix the MLflow `--serve-artifacts` flag (5 minutes), add zstd compression to the already-optimized multi-stage Docker builds (1 hour), then evaluate GKE Image Streaming and eStargz for deeper gains.

> **Correction (2026-03-17)**: The 3-tier multi-stage Docker architecture (GPU/CPU/Light
> bases, 2-stage builder→runner, uv cache mounts, optimized layer ordering) is **already
> fully implemented** per `docker-base-improvement-plan.md`. The 9 GB image size is the
> irreducible floor after multi-stage optimization (CUDA runtime + PyTorch + MONAI + deps).
> H1 below targets incremental cleanup (strip `__pycache__`, unused extras), not the
> initial multi-stage implementation which was completed in March 2026.
> See: `.claude/metalearning/2026-03-17-docker-research-report-false-premise.md`

---

## 1. Introduction

### 1.1 Platform Context

MinIVess MLOps is a model-agnostic biomedical segmentation platform built as an extension to the MONAI ecosystem. The platform employs a Docker-per-flow isolation architecture with SkyPilot as an intercloud broker ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)), orchestrating GPU workloads across RunPod (dev environment) and GCP (staging/production). Training flows execute inside Docker containers pulled from Google Artifact Registry (GAR) at `europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`.

### 1.2 Three Infrastructure Bottlenecks

Issue #734 (GPU Experiment Infrastructure) identified three blocking bottlenecks:

1. **Issue #751 --- Docker Image Pull**: The 9 GB `minivess-base` image takes ~15 minutes to pull from GAR europe-north1, consuming 58% of GCP smoke test wall time at a wasted cost of $0.055 per trial (L4 spot at $0.22/hr).

2. **Issue #754 --- RunPod Provisioning**: Three consecutive attempts stuck in `STARTING` state for 40+ minutes on EU-RO-1 datacenter with RTX 4090 and Network Volume attached, making the dev environment unusable.

3. **Issue #755 --- MLflow Upload**: 900 MB SAM3 checkpoint upload fails with HTTP 500 against MLflow on Cloud Run due to a hard 32 MB request body limit. The `GCSArtifactRepository` does not implement `MultipartUploadMixin`.

### 1.3 Research Methodology

This report follows a multi-hypothesis approach inspired by the scientific method: enumerate all plausible solutions from literature and industry practice, estimate cost/benefit for each, and execute cheapest/simplest hypotheses first with measurement gates before advancing to more complex solutions. This approach aligns with the ralph-loop infrastructure monitoring pattern used in MinIVess.

---

## 2. Issue #751: Docker Image Pull Optimization (15-minute GAR Pull)

### 2.1 Problem Statement

The `minivess-base:latest` image is approximately 9 GB (compressed). When a GCP L4 spot instance provisions, it must pull this image from GAR europe-north1. Despite same-region placement, the pull takes approximately 15 minutes, which constitutes 58% of the total smoke test wall time (~26 minutes). At L4 spot pricing of $0.22/hr, each pull wastes $0.055 in idle GPU time. Over an HPO sweep of 50 trials, this accumulates to $2.75 in pure pull overhead.

The `minivess-base` image uses a multi-stage Dockerfile (`deployment/docker/Dockerfile.base`) that already separates builder (devel) and runner (runtime) stages. The runner stage is based on `nvidia/cuda:12.6.3-runtime-ubuntu24.04` with Python 3.13 from deadsnakes PPA, the full `.venv` from uv sync, and SkyPilot SSH requirements (openssh-server, rsync, curl, patch, sudo).

### 2.2 Literature Review

The seminal work by [Harter et al. (2016). "Slacker: Fast Distribution with Lazy Docker Containers." *FAST'16*.](https://www.usenix.org/conference/fast16/technical-sessions/presentation/harter) established that **76% of container startup time is spent pulling the image, yet only 6.4% of the pulled data is actually read during startup**. This finding motivates lazy-pulling approaches where the container starts before the full image is downloaded.

Subsequent work has attacked this problem from multiple angles:

- **Deduplication**: [Zhao et al. (2020). "DupHunter: Flexible High-Performance Deduplication for Docker Registries." *ATC'20*.](https://www.usenix.org/conference/atc20/presentation/zhao) demonstrated up to 6.9x storage reduction and 2.8x GET latency improvement through a two-tier storage hierarchy with prefetch/preconstruct caching.

- **P2P Distribution**: [Wang et al. (2021). "FaaSNet: Scalable and Fast Provisioning of Custom Serverless Container Runtimes at Alibaba Cloud Function Compute." *ATC'21*.](https://www.usenix.org/conference/atc21/presentation/wang-ao) provisioned 2,500 containers on 1,000 VMs in 8.3 seconds using adaptive function trees with on-demand fetching, achieving 13.4x faster scaling than baseline and 16.3x faster than Kraken P2P registry.

- **Layer-wise Caching**: [Yu et al. (2024). "RainbowCake: Mitigating Cold-starts in Serverless with Layer-wise Container Caching and Sharing." *ASPLOS'24*.](https://openreview.net/forum?id=wFjd4zxei6) reduced function startup latency by 68% and memory waste by 77% through sharing-aware layer-wise caching decisions.

- **Lazy Pulling**: The eStargz format, based on Google CRFS's stargz (seekable tar.gz), enables containers to start without waiting for full image pull. The [containerd/stargz-snapshotter](https://github.com/containerd/stargz-snapshotter) project implements this as a containerd proxy plugin, mounting layers via FUSE with on-demand content fetching from registries.

- **CNCF Graduated Solutions**: [Dragonfly](https://d7y.io/docs/) (CNCF Graduated, January 2026) with its [Nydus](https://nydus.dev/) image acceleration service provides P2P distribution with lazy pulling, reporting 80% network latency reduction and 90% startup time savings in production environments at Ant Group ([CNCF Blog, 2023](https://www.cncf.io/blog/2023/05/01/ant-group-security-technologys-nydus-and-dragonfly-image-acceleration-practices/)).

- **Content-Addressable Chunking**: [Cerebrium](https://www.cerebrium.ai/blog/rethinking-container-image-distribution-to-eliminate-cold-starts) rethought container distribution for ML workloads, observing that on a 10 GB ML image, the container reads only ~640 MB at startup. Their content-addressable chunking system uses SHA-256 digests as chunk identities, enabling deduplication across images and on-demand fetching of individual chunks rather than entire compressed tarballs.

### 2.3 Solution Hypothesis Matrix

| ID | Solution | Expected Speedup | Complexity | GAR Compatible | SkyPilot Compatible | Prerequisites | Key Citation |
|----|----------|-----------------|------------|----------------|---------------------|---------------|--------------|
| H1 | **ALREADY IMPLEMENTED (multi-stage + uv cache mounts). Incremental: strip `__pycache__`, `.pyc` files, test dependencies; use `--no-compile` in runner; audit `.venv` for unused extras** --- Three-tier multi-stage build (GPU/CPU/Light) with 2-stage builder→runner and uv cache mounts is already in place (March 2026). This hypothesis now targets incremental cleanup only. | 40-60% size reduction (9 GB to 3.5-5.5 GB) | LOW (2 hr) | Yes | Yes | Dockerfile edit only | [Astral uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/) |
| H2 | **zstd compression** --- Build with `--output type=image,compression=zstd` via BuildKit. zstd is multi-threaded with faster decompression than gzip | 20-30% pull time reduction | LOW (1 hr) | Yes (OCI support) | Yes | BuildKit 0.10+, containerd 1.5+ on target | [AWS Blog (2024)](https://aws.amazon.com/blogs/containers/reducing-aws-fargate-startup-times-with-zstd-compressed-container-images/) |
| H3 | **GKE Image Streaming** --- GKE mounts container data layer via network mount from Artifact Registry, starting containers in seconds regardless of image size. Intelligent read-ahead in GKE 1.32+ | 5-6x speedup (191s to ~30s) | MEDIUM (4 hr) | Yes (native GAR) | No (GKE only) | GKE cluster (not bare VM), GKE 1.30.1+ | [Google Cloud Blog (2025)](https://cloud.google.com/blog/products/containers-kubernetes/improving-gke-container-image-streaming-for-faster-app-startup/) |
| H4 | **GKE Secondary Boot Disks** --- Preload base image layers onto secondary boot disks attached to node pools; GKE reads from local disk instead of registry | 29x reduction for 16 GB cached image | HIGH (8 hr) | Yes (native GAR) | No (GKE only) | GKE 1.28.3+, Terraform/Pulumi for disk management | [GKE Docs](https://cloud.google.com/kubernetes-engine/docs/how-to/data-container-image-preloading) |
| H5 | **eStargz lazy pulling** --- Convert images to eStargz format; container starts in ~3s while fetching data on-demand via FUSE | Container starts in ~3s; total I/O may increase | HIGH (12 hr) | Partial (OCI-compatible, but GAR has no native eStargz optimization) | No (requires stargz-snapshotter on host) | stargz-snapshotter daemon, containerd config | [containerd/stargz-snapshotter](https://github.com/containerd/stargz-snapshotter) |
| H6 | **BuildKit registry cache with GAR** --- Push build cache to a separate GAR image; subsequent builds pull only changed layers | Faster rebuilds, not faster first pull | LOW (2 hr) | Yes | Yes | GAR write access for cache image | [Docker Docs: Registry cache](https://docs.docker.com/build/cache/backends/registry/) |
| H7 | **RunPod base images as FROM** --- Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` as base; RunPod pre-caches these on Secure Cloud nodes, yielding near-zero pull for base layers | Near-zero pull for base layers (~4-6 GB savings) | MEDIUM (4 hr) | N/A (RunPod only) | Yes | Rebuild Dockerfile with RunPod base; Python 3.11 instead of 3.13 | [RunPod Docker Hub](https://hub.docker.com/r/runpod/pytorch) |
| H8 | **GAR remote repository as Docker Hub proxy** --- Configure GAR as a caching proxy for Docker Hub; base CUDA layers cached in same-region GAR | Faster base layer pulls via regional cache | LOW (1 hr) | Yes (native feature) | Yes | GAR remote repo config | [GAR Docs: Remote repositories](https://docs.google.com/artifact-registry/docs/repositories/remote-repo) |
| H9 | **Nydus + Dragonfly P2P** --- CNCF-graduated P2P distribution with lazy pulling; 80% network latency reduction in production | 80-90% startup time reduction at scale | VERY HIGH (40+ hr) | No (custom registry + FUSE daemon) | No (requires Nydus/Dragonfly infrastructure) | Dragonfly cluster, Nydus snapshotter on each node | [Nydus](https://nydus.dev/), [CNCF Dragonfly](https://d7y.io/docs/) |
| H10 | **ORAS for model weight separation** --- Store model weights (SAM3 ViT-32L ~1.8 GB) as separate OCI artifacts; pull base image and weights independently | Decouple weight updates from image rebuilds; parallel pulls | MEDIUM (6 hr) | Yes (OCI-compliant) | Partial (custom init script) | ORAS CLI, weight registry convention | [ORAS Project](https://oras.land/), [KAITO](https://kaito-project.github.io/kaito/docs/next/model-as-oci-artifacts/) |
| H11 | **SlimToolkit (docker-slim)** --- Dynamic analysis traces file access during container execution; removes everything not on the hot path | Up to 30x reduction (theoretical); **ML caveat: dynamic loading of `.so` files may break** | MEDIUM (4 hr) | Yes | Yes | SlimToolkit installed; thorough test coverage to exercise all code paths | [SlimToolkit](https://slimtoolkit.org/) |
| H12 | **Content-addressable chunking (Cerebrium approach)** --- Chunk image by content hash, fetch on-demand, deduplicate across versions | Eliminates redundant pulls across image versions | VERY HIGH (80+ hr) | No (custom distribution system) | No (custom runtime) | Custom image distribution infrastructure | [Cerebrium Blog (2025)](https://www.cerebrium.ai/blog/rethinking-container-image-distribution-to-eliminate-cold-starts) |

### 2.4 Recommended Execution Order (ralph-loop Friendly)

**Phase 1 --- Quick Wins (< 4 hours total, measure before proceeding)**

1. **H1: Optimize Dockerfile** --- Audit `.venv` contents with `du -sh .venv/lib/python3.13/site-packages/*` sorted by size. Strip test files, type stubs, documentation, and `__pycache__`. Measure compressed image size before/after.

2. **H2: zstd compression** --- Single-line change to build command: `docker buildx build --output type=image,compression=zstd,compression-level=3`. Measure pull time on fresh GCP L4 instance.

3. **H6: BuildKit registry cache** --- Add `--cache-to type=registry,ref=europe-north1-docker.pkg.dev/minivess-mlops/minivess/cache:latest --cache-from type=registry,ref=europe-north1-docker.pkg.dev/minivess-mlops/minivess/cache:latest` to build command. Speeds up rebuilds (not first pull).

4. **H8: GAR remote repository** --- Configure GAR as Docker Hub proxy to cache `nvidia/cuda` base layers regionally. One-time setup via `gcloud artifacts repositories create`.

**Phase 2 --- Medium Effort (4-8 hours, requires architecture decisions)**

5. **H7: RunPod base images** --- For the RunPod dev path only, create an alternative Dockerfile using `FROM runpod/pytorch:*` to leverage pre-cached layers on Secure Cloud nodes. Requires acceptance of Python 3.11 constraint on RunPod.

6. **H3: GKE Image Streaming** --- Requires migration from SkyPilot bare-VM provisioning to GKE-managed nodes. Significant architectural change but offers 5-6x speedup with zero image-side changes. Evaluate feasibility with `docs/planning/gke-migration-assessment.md`.

7. **H10: ORAS model weight separation** --- Decouple SAM3 ViT-32L weights from the base image. Publish weights as OCI artifact; pull in SkyPilot setup block.

**Phase 3 --- Research Spikes (> 12 hours, proof-of-concept only)**

8. **H5: eStargz** --- Requires stargz-snapshotter on target nodes. Test on a dedicated GKE cluster with containerd config override.

9. **H4: GKE Secondary Boot Disks** --- Requires GKE + Pulumi IaC for disk image management. Significant operational overhead but eliminates pull entirely for cached images.

10. **H9/H12: Nydus/Dragonfly or content-addressable chunking** --- Research-grade solutions. Only viable at scale (100+ nodes). Monitor Dragonfly graduation progress.

---

## 3. Issue #754: RunPod Provisioning Stuck in STARTING (40+ min)

### 3.1 Problem Statement

Three consecutive attempts to provision an RTX 4090 instance on RunPod (EU-RO-1 datacenter) via SkyPilot became stuck in `STARTING` state for 40+ minutes before manual cancellation. The configuration uses a custom Docker image (`europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`), a Network Volume (`minivess-dev`), and the `dev_runpod.yaml` SkyPilot task file. This makes the RunPod dev environment completely unusable.

### 3.2 Root Cause Analysis

Based on SkyPilot GitHub issues and RunPod documentation, the root causes fall into three categories:

**Category A --- Docker Image Issues (Most Common)**

[SkyPilot Issue #4285](https://github.com/skypilot-org/skypilot/issues/4285) documents the exact symptom: custom Docker images stuck in `STARTING` on RunPod. The root cause is that RunPod uses the cloud API to directly specify the Docker image, and if the image has a custom `ENTRYPOINT` that does not yield an SSH-accessible shell, RunPod cannot complete its initialization handshake.

[SkyPilot Issue #3879](https://github.com/skypilot-org/skypilot/issues/3879) further clarifies that RunPod does not provide an API for customizing Docker run arguments, making `ENTRYPOINT` management critical. The MinIVess `Dockerfile.base` intentionally does NOT set `ENTRYPOINT` or `CMD`, which should avoid this issue --- but the `HEALTHCHECK` directive may interfere with RunPod's readiness detection.

**Category B --- Image Pull Timeout**

A 9 GB image pull from GAR (europe-north1) to a RunPod datacenter (EU-RO-1, likely Romania) crosses cloud provider boundaries. Unlike GCP-to-GCP same-region pulls, GAR-to-RunPod traverses the public internet. If the pull exceeds RunPod's internal timeout, the pod remains in `STARTING` indefinitely.

**Category C --- Regional Availability**

[SkyPilot Issue #4265](https://github.com/skypilot-org/skypilot/issues/4265) reports RTX 4090 spot unavailability on RunPod. EU-RO-1 may have limited RTX 4090 inventory, especially with Network Volume constraints that pin the pod to a specific datacenter.

### 3.3 Solution Hypothesis Matrix

| ID | Solution | Expected Impact | Complexity | Risk | Prerequisites |
|----|----------|----------------|------------|------|---------------|
| H1 | **Verify Dockerfile has no blocking ENTRYPOINT** --- Confirm `ENTRYPOINT` and `CMD` are unset in the final stage. Remove `HEALTHCHECK` directive (may block RunPod readiness). Test with `docker inspect` | HIGH (most common root cause per SkyPilot #4285) | LOW (30 min) | LOW | `docker inspect` on built image |
| H2 | **Use RunPod base images for dev path** --- Build `FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` for RunPod-only use. RunPod pre-caches these layers on Secure Cloud, yielding near-instant base layer availability | HIGH (eliminates cross-cloud pull for base layers) | MEDIUM (4 hr) | MEDIUM (Python 3.11 constraint) | Separate Dockerfile for RunPod dev |
| H3 | **Multi-datacenter Network Volumes** --- Create Network Volumes in multiple EU datacenters (EU-RO-1, EU-NL-1, EU-SE-1). SkyPilot can failover across datacenters | MEDIUM (increases availability pool) | MEDIUM (2 hr per volume) | LOW (volume cost: $3.50/mo each) | RunPod account, `sky volumes apply` per DC |
| H4 | **SkyPilot EAGER_NEXT_REGION recovery** --- Set `recovery_strategy: EAGER_NEXT_REGION` in SkyPilot config. On provisioning failure, SkyPilot immediately tries the next region instead of retrying the same one | MEDIUM (faster recovery, not prevention) | LOW (1 hr) | LOW | SkyPilot config edit |
| H5 | **Docker image size optimization** --- Reduce image from 9 GB to < 4 GB (see Section 2 hypotheses H1/H2). Smaller image = faster pull = less likely to hit RunPod timeout | HIGH (addresses Category B root cause) | MEDIUM (see Section 2) | LOW | Dockerfile optimization |
| H6 | **Secure Cloud over Community Cloud** --- Use RunPod Secure Cloud (not Community Cloud). Secure Cloud nodes are more reliable and may pre-cache popular base images | MEDIUM (better reliability, slightly higher cost) | LOW (config change) | LOW (cost increase ~20%) | RunPod Secure Cloud access |
| H7 | **Increase SkyPilot provision_timeout** --- Set `provision_timeout: 1800` (30 min) in `~/.sky/config.yaml` to give more time for large image pulls | LOW (treats symptom, not cause) | LOW (5 min) | LOW | SkyPilot config edit |

### 3.4 GPU Cold Start Waterfall

Understanding the full cold-start timeline is essential for targeted optimization:

| Phase | Typical Duration (RunPod) | Typical Duration (GCP L4) | Bottleneck |
|-------|--------------------------|--------------------------|------------|
| Node provisioning (VM allocation) | 60-120s | 30-60s | Cloud provider capacity |
| Docker image pull | 120-600s (cross-cloud, 9 GB) | 600-900s (same-region GAR, 9 GB) | Network bandwidth, image size |
| SkyPilot SSH setup + setup block | 30-60s | 30-60s | SSH handshake, apt-get/dvc |
| DVC data pull (2.7 GB MiniVess) | 0s (cached on Network Volume) | 60-180s (GCS to local disk) | GCS bandwidth |
| Model weight download (SAM3 ViT-32L) | 0-60s (cached on Network Volume) | 30-120s (HuggingFace) | HF bandwidth, model size |
| CUDA initialization | 5-15s | 5-15s | GPU driver |
| Weight transfer to GPU VRAM | 10-30s | 10-30s | PCIe bandwidth, model size |
| **Total cold start** | **~4-14 min** | **~13-22 min** | **Image pull dominates** |

### 3.5 Academic Context: Spot Instance Recovery

The MinIVess platform uses spot/preemptible instances for cost efficiency ($0.22/hr L4 spot vs $0.70/hr on-demand). Academic research on spot instance management provides context for fault tolerance:

- [Thorpe et al. (2023). "Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs." *NSDI'23*.](https://www.usenix.org/conference/nsdi23/presentation/thorpe) introduced redundant computations in pipeline bubbles, achieving 3.7x throughput improvement over traditional checkpointing and 2.4x cost reduction vs on-demand instances.

- [Duan et al. (2024). "Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances." *NSDI'24*.](https://www.usenix.org/conference/nsdi24/presentation/duan) defined the *liveput* metric (expected throughput under preemption scenarios) and achieved up to 10x improvement over existing spot-instance systems through proactive parallelization adjustment with availability prediction.

- [Miao et al. (2024). "SpotServe: Serving Generative Large Language Models on Preemptible Instances." *ASPLOS'24*.](https://dl.acm.org/doi/10.1145/3620665.3640411) introduced dynamic reparallelization with optimal migration planning (Kuhn-Munkres algorithm) and stateful inference recovery, reducing P99 tail latency by 2.4-9.1x while saving 54% monetary cost.

- [Mao et al. (2025). "SkyServe: Serving AI Models across Regions and Clouds with Spot Instances." *EuroSys'25*.](https://dl.acm.org/doi/10.1145/3689031.3717459) implemented the SpotHedge policy --- a mixture of spot and on-demand replicas across failure domains (regions + clouds) --- reducing cost by 43% while improving P50/P90/P99 latency by 2.1-2.3x.

- [GFS (2026). "GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management." *ASPLOS'26*.](https://arxiv.org/abs/2509.11134) introduced lightweight GPU demand forecasting for proactive resource management, reducing HP task queuing time by 63.5% and spot task completion time by 14.5%.

While MinIVess does not operate at the scale where pipeline parallelism or dynamic reparallelization are applicable (single-GPU training), the key insight from this literature is that **cold-start time directly determines the cost-effectiveness of spot instances**. For single-GPU jobs where the cold-start overhead is a significant fraction of total job time (15 min pull / 26 min total = 58%), reducing pull time has outsized impact on cost efficiency.

---

## 4. Issue #755: MLflow Cloud Run Multipart Upload 500

### 4.1 Problem Statement

The MinIVess GCP staging environment runs an MLflow tracking server on Cloud Run (`deployment/docker/Dockerfile.mlflow-gcp`). The server is started with `--serve-artifacts`, which proxies all artifact uploads through the Cloud Run instance. When training completes and the SAM3 model checkpoint (~900 MB) is logged via `mlflow.log_artifact()`, the upload fails with HTTP 500.

The current SkyPilot task file (`deployment/skypilot/smoke_test_gcp.yaml`) sets:
```
MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD: "true"
MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE: "10485760"
```

These environment variables are intended to enable chunked upload, but the failure persists.

### 4.2 Root Cause Analysis

**Primary Cause: Cloud Run 32 MB Request Body Limit**

Cloud Run imposes a [hard 32 MB limit on HTTP request bodies](https://cloud.google.com/run/quotas). This limit **cannot be increased** --- it is an architectural constraint of the Cloud Run managed service. When MLflow's `--serve-artifacts` mode proxies an artifact upload, the entire file (or chunk) passes through the Cloud Run instance's HTTP handler. Any single request exceeding 32 MB triggers a 413 (Request Entity Too Large) error, which MLflow surfaces as an HTTP 500.

**Secondary Cause: GCSArtifactRepository Does Not Implement MultipartUploadMixin**

Even with `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true`, the `GCSArtifactRepository` class in MLflow does **not** implement the `MultipartUploadMixin` interface. The multipart upload feature was developed primarily for S3-compatible backends (S3, MinIO, Azure Blob). For GCS, the tracking server falls back to single-request upload, hitting the 32 MB limit.

**Tertiary Cause: Known MLflow Multipart Upload Bugs**

Even on S3-compatible backends where multipart upload *is* implemented, multiple bugs affect production use:

- [Issue #11225](https://github.com/mlflow/mlflow/issues/11225): Proxied multipart upload stores artifacts at incorrect S3 path (root of artifact store instead of run directory).
- [Issue #11268](https://github.com/mlflow/mlflow/issues/11268): Cannot use multipart upload without explicit `artifact_path` argument.
- [Issue #11828](https://github.com/mlflow/mlflow/issues/11828): Proxied multipart upload generates presigned URLs with wrong AWS region (us-east-1 instead of bucket region).

### 4.3 Solution Hypothesis Matrix

| ID | Solution | Expected Impact | Complexity | Risk | GCS Compatible | Prerequisites |
|----|----------|----------------|------------|------|----------------|---------------|
| H1 | **`--no-serve-artifacts` (bypass Cloud Run proxy)** --- Remove `--serve-artifacts` from `Dockerfile.mlflow-gcp`. Client uploads directly to GCS using ADC credentials. Cloud Run only serves metadata/tracking. **RECOMMENDED** | **RESOLVES ISSUE** (no size limit on direct GCS upload) | **LOW (15 min)** | LOW (client must have GCS write access --- already true via ADC on GCP VMs) | Yes (native GCS client) | GCP VMs with ADC; `--default-artifact-root gs://minivess-mlops-mlflow-artifacts` |
| H2 | **`MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD` on S3-compatible backend** --- Switch artifact store to MinIO (S3-compatible) where multipart upload mixin IS implemented | Partial fix (multipart works for S3, but known bugs #11225, #11268, #11828) | HIGH (8 hr) | HIGH (introduces MinIO dependency on GCP; defeats purpose of native GCS) | No (requires S3 backend) | MinIO or S3-compatible store, bug workarounds |
| H3 | **Hybrid: metadata via Cloud Run + artifacts direct to GCS** --- Use `--no-serve-artifacts` for artifact upload but keep Cloud Run for experiment tracking API. Client writes artifacts directly; Cloud Run handles metadata only | **RESOLVES ISSUE** (same as H1 with clearer separation) | LOW (15 min) | LOW | Yes | Same as H1 |
| H4 | **Separate artifact-only server on GKE/VM** --- Deploy a dedicated MLflow artifact proxy on GKE with no request body limit, fronted by an NGINX reverse proxy with `client_max_body_size 0` | RESOLVES ISSUE | HIGH (12 hr) | MEDIUM (operational overhead of separate service) | Yes | GKE cluster or dedicated VM, NGINX config |

### 4.4 MLflow Known Issues Table

| Issue | Title | Severity | Status | Relevance |
|-------|-------|----------|--------|-----------|
| [#7564](https://github.com/mlflow/mlflow/issues/7564) | MLflow returns error 504 after uploading large files (800 MB+) | HIGH | Open | Direct match --- same symptom with large checkpoints |
| [#8539](https://github.com/mlflow/mlflow/issues/8539) | Timeout errors while uploading large models to MLflow server | HIGH | Open | Timeout variant of same root cause |
| [#8963](https://github.com/mlflow/mlflow/issues/8963) | Uploading models larger than 50 MB leads to urllib3 protocol error | MEDIUM | Open | Protocol-level failure on large uploads |
| [#10332](https://github.com/mlflow/mlflow/issues/10332) | 413 Client Error: Request Entity Too Large | HIGH | Open | Exact match --- Cloud Run + GCS + large model |
| [#11225](https://github.com/mlflow/mlflow/issues/11225) | Proxied multipart upload stores artifact at incorrect S3 path | HIGH | Open | Multipart path routing bug |
| [#11268](https://github.com/mlflow/mlflow/issues/11268) | Cannot use multipart upload without artifact_path argument | MEDIUM | Open | Multipart API contract violation |
| [#11828](https://github.com/mlflow/mlflow/issues/11828) | Proxied multipart upload wrong region in presigned URL | HIGH | Open | Region mismatch in multipart presigned URLs |
| [#15455](https://github.com/mlflow/mlflow/issues/15455) | Remote tracking server - Unable to log models - Error 500 | HIGH | Open | Generic 500 error on remote model logging |

### 4.5 Recommended Fix

**H1 is the clear winner.** Changing `--serve-artifacts` to `--no-serve-artifacts` in `deployment/docker/Dockerfile.mlflow-gcp` is a one-line fix that:

1. Eliminates the Cloud Run 32 MB bottleneck entirely
2. Enables direct client-to-GCS upload with no size limit
3. Removes the need for `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD` and its associated bugs
4. Reduces Cloud Run CPU/memory usage (no longer proxying large files)
5. Is already supported by the GCP architecture (ADC credentials on GCP VMs grant GCS write access)

The only caveat is that **RunPod dev environment** does NOT have ADC credentials for GCS. However, RunPod uses file-based MLflow (`MLFLOW_TRACKING_URI: /opt/vol/mlruns`) and never talks to Cloud Run, so this change has zero impact on RunPod.

**Implementation:**

Change in `deployment/docker/Dockerfile.mlflow-gcp`:
```dockerfile
# BEFORE (blocked by 32 MB Cloud Run limit):
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--serve-artifacts"]

# AFTER (client uploads directly to GCS):
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--no-serve-artifacts"]
```

Remove from `deployment/skypilot/smoke_test_gcp.yaml`:
```yaml
# No longer needed --- client uploads directly to GCS
# MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD: "true"
# MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE: "10485760"
```

---

## 5. Multi-Hypothesis Design Matrix (Combined)

The following table ranks all hypotheses across all three issues by expected impact, implementation effort, and ralph-loop testability.

| Rank | Issue | Hypothesis | Expected Impact | Effort (hours) | Risk | ralph-loop Testable | Dependencies |
|------|-------|-----------|----------------|----------------|------|---------------------|--------------|
| 1 | #755 | H1: `--no-serve-artifacts` | **CRITICAL** --- resolves upload failure | 0.25 | LOW | Yes (log artifact size > 32 MB) | None |
| 2 | #754 | H1: Verify no blocking ENTRYPOINT | **HIGH** --- most common RunPod stuck cause | 0.5 | LOW | Yes (check `docker inspect`) | None |
| 3 | #751 | H1: Optimize Dockerfile size | **HIGH** --- 40-60% size reduction | 2.0 | LOW | Yes (measure pull time) | None |
| 4 | #751 | H2: zstd compression | **MEDIUM** --- 20-30% pull reduction | 1.0 | LOW | Yes (measure pull time) | BuildKit |
| 5 | #754 | H4: EAGER_NEXT_REGION | **MEDIUM** --- faster failover | 1.0 | LOW | Yes (observe failover in logs) | None |
| 6 | #751 | H6: BuildKit registry cache | **MEDIUM** --- faster rebuilds | 2.0 | LOW | Yes (measure rebuild time) | GAR write |
| 7 | #751 | H8: GAR remote repository | **MEDIUM** --- cache base layers | 1.0 | LOW | Yes (measure base pull time) | GAR config |
| 8 | #754 | H5: Reduce image size (cross-ref #751 H1) | **HIGH** --- faster RunPod pull | 2.0 | LOW | Yes (measure RunPod startup) | #751 H1 |
| 9 | #754 | H7: Increase provision_timeout | **LOW** --- treats symptom | 0.1 | LOW | Yes (observe timeout behavior) | None |
| 10 | #754 | H2: RunPod base images | **HIGH** --- near-zero base pull | 4.0 | MEDIUM | Yes (measure pull time) | Separate Dockerfile |
| 11 | #754 | H6: Secure Cloud | **MEDIUM** --- better reliability | 0.5 | LOW | Yes (observe startup reliability) | RunPod account tier |
| 12 | #754 | H3: Multi-DC volumes | **MEDIUM** --- wider availability | 2.0 | LOW | Yes (test across DCs) | RunPod volumes |
| 13 | #751 | H7: RunPod base images | **HIGH** (RunPod only) | 4.0 | MEDIUM | Yes (measure RunPod pull) | Python 3.11 acceptance |
| 14 | #751 | H10: ORAS model weight separation | **MEDIUM** --- decouple weights | 6.0 | MEDIUM | Partial | ORAS tooling |
| 15 | #755 | H3: Hybrid metadata+direct | **CRITICAL** (same as #755 H1) | 0.25 | LOW | Yes | None |
| 16 | #751 | H3: GKE Image Streaming | **VERY HIGH** --- 5-6x speedup | 4.0 | HIGH | Yes (GKE-only) | GKE cluster |
| 17 | #751 | H11: SlimToolkit | **HIGH** (up to 30x, ML caveat) | 4.0 | HIGH | Yes (measure size + test) | Full test coverage |
| 18 | #751 | H4: GKE Secondary Boot Disks | **VERY HIGH** --- 29x reduction | 8.0 | HIGH | Yes (GKE-only) | GKE + Pulumi IaC |
| 19 | #751 | H5: eStargz lazy pulling | **HIGH** --- ~3s container start | 12.0 | HIGH | Partial | stargz-snapshotter |
| 20 | #755 | H4: Separate artifact server | **CRITICAL** | 12.0 | MEDIUM | Yes | GKE/VM |
| 21 | #755 | H2: S3-compatible multipart | Partial (known bugs) | 8.0 | HIGH | Partial | MinIO on GCP |
| 22 | #751 | H9: Nydus + Dragonfly P2P | **VERY HIGH** at scale | 40.0 | VERY HIGH | No | Full Dragonfly infra |
| 23 | #751 | H12: Content-addressable chunking | **VERY HIGH** (theoretical) | 80.0 | VERY HIGH | No | Custom distribution system |

---

## 6. Execution Plan for ralph-loop

### Phase 1: Quick Wins (< 4 hours total, one afternoon)

**Step 1 (15 min): Fix MLflow upload (#755 H1)**
- Change `--serve-artifacts` to `--no-serve-artifacts` in `Dockerfile.mlflow-gcp`
- Remove `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD` from `smoke_test_gcp.yaml`
- Rebuild and push MLflow GCP image
- **Verification**: Log a 100 MB test artifact from a GCP VM

**Step 2 (30 min): Verify RunPod Docker image (#754 H1)**
- Run `docker inspect minivess-base:latest` to verify no ENTRYPOINT/CMD
- Remove `HEALTHCHECK` directive from `Dockerfile.base` (may interfere with RunPod readiness)
- Test RunPod provisioning with cleaned image
- **Verification**: `sky launch` with cleaned image succeeds within 5 minutes

**Step 3 (5 min): Increase provision_timeout (#754 H7)**
- Add `provision_timeout: 1800` to `~/.sky/config.yaml`
- **Verification**: Observe longer wait before failover in SkyPilot logs

**Step 4 (1 hr): Enable EAGER_NEXT_REGION (#754 H4)**
- Set `recovery_strategy: EAGER_NEXT_REGION` in SkyPilot config
- **Verification**: On simulated failure, SkyPilot logs show immediate region change

**Step 5 (2 hr): Optimize Dockerfile size (#751 H1)**
- Audit `.venv` contents: identify top-10 packages by size
- Strip `__pycache__`, test files, type stubs, documentation
- Consider `--no-compile` in runner stage (bytecodes compiled at runtime)
- Measure: `docker images minivess-base:latest --format '{{.Size}}'`
- **Verification**: Image size reduction > 30%

**Step 6 (1 hr): Enable zstd compression (#751 H2)**
- Rebuild with `--output type=image,compression=zstd,compression-level=3`
- Push to GAR and measure pull time on fresh L4 instance
- **Verification**: Pull time reduction > 15%

### Phase 2: Medium Effort (next sprint, 2-3 days)

**Step 7 (4 hr): RunPod base image variant (#754 H2, #751 H7)**
- Create `Dockerfile.base-runpod` using `FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Install only the delta (MinIVess-specific packages) on top
- Push to Docker Hub (RunPod pulls from Docker Hub faster than GAR)
- **Verification**: RunPod cold start < 5 minutes

**Step 8 (1 hr): GAR remote repository (#751 H8)**
- Create GAR remote repo proxying Docker Hub
- Configure `nvidia/cuda` pulls to go through regional cache
- **Verification**: Base layer pull time on GCP reduced

**Step 9 (2 hr): Multi-DC Network Volumes (#754 H3)**
- Create Network Volumes in EU-NL-1 and EU-SE-1
- Test failover across datacenters
- **Verification**: Successful provisioning when EU-RO-1 is full

**Step 10 (2 hr): BuildKit registry cache (#751 H6)**
- Configure GAR cache image for build layer reuse
- **Verification**: Rebuild time after dependency change < 5 minutes

### Phase 3: Research Spikes (backlog, by priority)

**Step 11 (4 hr): GKE Image Streaming evaluation (#751 H3)**
- Assess feasibility of GKE-based training (SkyPilot + GKE integration)
- Prototype with a minimal GKE Autopilot cluster
- **Verification**: Pull time < 30 seconds for minivess-base

**Step 12 (6 hr): ORAS model weight separation (#751 H10)**
- Publish SAM3 weights as OCI artifact
- Modify SkyPilot setup block to pull weights separately
- **Verification**: Image size without weights < 5 GB

**Step 13 (12 hr): eStargz proof-of-concept (#751 H5)**
- Convert minivess-base to eStargz format
- Deploy stargz-snapshotter on test node
- **Verification**: Container starts before full image pull

---

## 7. References

### Container Image Distribution

- [Harter, T., Salmon, B., Liu, R., Arpaci-Dusseau, A. C., and Arpaci-Dusseau, R. H. (2016). "Slacker: Fast Distribution with Lazy Docker Containers." *14th USENIX Conference on File and Storage Technologies (FAST'16)*.](https://www.usenix.org/conference/fast16/technical-sessions/presentation/harter)

- [Zhao, N., Albahar, H., Abraham, S., Chen, K., Tarasov, V., Skourtis, D., Rupprecht, L., Anwar, A., and Butt, A. R. (2020). "DupHunter: Flexible High-Performance Deduplication for Docker Registries." *2020 USENIX Annual Technical Conference (ATC'20)*.](https://www.usenix.org/conference/atc20/presentation/zhao)

- [Wang, A., Chang, S., Tian, H., Wang, H., Yang, H., Li, H., Du, R., and Cheng, Y. (2021). "FaaSNet: Scalable and Fast Provisioning of Custom Serverless Container Runtimes at Alibaba Cloud Function Compute." *2021 USENIX Annual Technical Conference (ATC'21)*.](https://www.usenix.org/conference/atc21/presentation/wang-ao)

- [Yu, H., Basu Roy, R., Fontenot, C., Tiwari, D., Li, J., Zhang, H., Wang, H., and Park, S.-J. (2024). "RainbowCake: Mitigating Cold-starts in Serverless with Layer-wise Container Caching and Sharing." *29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'24)*.](https://openreview.net/forum?id=wFjd4zxei6)

### Spot Instance Management

- [Yang, Z., Wu, Z., Luo, M., Chiang, W.-L., Bhardwaj, R., Kwon, W., Zhuang, S., Luan, F. S., Mittal, G., Shenker, S., and Stoica, I. (2023). "SkyPilot: An Intercloud Broker for Sky Computing." *20th USENIX Symposium on Networked Systems Design and Implementation (NSDI'23)*.](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)

- [Thorpe, J., Qiao, P., Didona, J., Bhoag, S., Dang, H. T., and Xu, H. (2023). "Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs." *20th USENIX Symposium on Networked Systems Design and Implementation (NSDI'23)*.](https://www.usenix.org/conference/nsdi23/presentation/thorpe)

- [Duan, J., Song, Z., Miao, X., Xi, X., Lin, D., Xu, H., Zhang, M., and Jia, Z. (2024). "Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances." *21st USENIX Symposium on Networked Systems Design and Implementation (NSDI'24)*.](https://www.usenix.org/conference/nsdi24/presentation/duan)

- [Miao, X., Shi, C., Duan, J., Xi, X., Lin, D., Cui, B., and Jia, Z. (2024). "SpotServe: Serving Generative Large Language Models on Preemptible Instances." *29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'24)*.](https://dl.acm.org/doi/10.1145/3620665.3640411)

- [Mao, Z., Xia, T., Wu, Z., Chiang, W.-L., Griggs, T., Bhardwaj, R., Yang, Z., Shenker, S., and Stoica, I. (2025). "SkyServe: Serving AI Models across Regions and Clouds with Spot Instances." *EuroSys'25*.](https://dl.acm.org/doi/10.1145/3689031.3717459)

- [GFS (2026). "GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management." *31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'26)*.](https://arxiv.org/abs/2509.11134)

### Cloud Infrastructure and Container Optimization

- [Google Cloud (2025). "Improving GKE Container Image Streaming for Faster App Startup." *Google Cloud Blog*.](https://cloud.google.com/blog/products/containers-kubernetes/improving-gke-container-image-streaming-for-faster-app-startup/)

- [Google Cloud (2024). "Use Secondary Boot Disks to Preload Data or Container Images." *GKE Documentation*.](https://cloud.google.com/kubernetes-engine/docs/how-to/data-container-image-preloading)

- [Google Cloud (2024). "Tips and Tricks to Reduce Cold Start Latency on GKE." *Google Cloud Blog*.](https://cloud.google.com/blog/products/containers-kubernetes/tips-and-tricks-to-reduce-cold-start-latency-on-gke)

- [AWS (2024). "Reducing AWS Fargate Startup Times with zstd Compressed Container Images." *AWS Containers Blog*.](https://aws.amazon.com/blogs/containers/reducing-aws-fargate-startup-times-with-zstd-compressed-container-images/)

- [Cerebrium (2025). "Rethinking Container Image Distribution to Eliminate Cold Starts." *Cerebrium Blog*.](https://www.cerebrium.ai/blog/rethinking-container-image-distribution-to-eliminate-cold-starts)

- [Astral (2025). "Using uv in Docker." *uv Documentation*.](https://docs.astral.sh/uv/guides/integration/docker/)

- [Docker (2025). "Registry Cache." *Docker Documentation*.](https://docs.docker.com/build/cache/backends/registry/)

### Container Image Formats and Distribution

- [containerd/stargz-snapshotter. "Fast Container Image Distribution Plugin with Lazy Pulling." *GitHub*.](https://github.com/containerd/stargz-snapshotter)

- [Nydus Project. "Acceleration Framework for Cloud-Native Distribution." *nydus.dev*.](https://nydus.dev/)

- [CNCF (2026). "Cloud Native Computing Foundation Announces Dragonfly's Graduation." *CNCF Announcements*.](https://www.cncf.io/announcements/2026/01/14/cloud-native-computing-foundation-announces-dragonflys-graduation/)

- [CNCF (2023). "Ant Group Security Technology's Nydus and Dragonfly Image Acceleration Practices." *CNCF Blog*.](https://www.cncf.io/blog/2023/05/01/ant-group-security-technologys-nydus-and-dragonfly-image-acceleration-practices/)

- [ORAS Project. "OCI Registry As Storage." *oras.land*.](https://oras.land/)

- [KAITO Project. "Model As OCI Artifacts." *KAITO Documentation*.](https://kaito-project.github.io/kaito/docs/next/model-as-oci-artifacts/)

- [SlimToolkit. "Optimize Your Containerized App Dev Experience." *slimtoolkit.org*.](https://slimtoolkit.org/)

### MLflow Issues

- [MLflow Issue #7564. "MLflow Returns Error 504 After Uploading Large Files (800 MB+)." *GitHub*.](https://github.com/mlflow/mlflow/issues/7564)

- [MLflow Issue #8539. "Timeout Errors While Uploading Large Models to MLflow Server." *GitHub*.](https://github.com/mlflow/mlflow/issues/8539)

- [MLflow Issue #8963. "Uploading Models Larger Than 50 MB Leads to urllib3 Protocol Error." *GitHub*.](https://github.com/mlflow/mlflow/issues/8963)

- [MLflow Issue #10332. "413 Client Error: Request Entity Too Large." *GitHub*.](https://github.com/mlflow/mlflow/issues/10332)

- [MLflow Issue #11225. "Proxied Multipart Upload Stores Artifact at Incorrect S3 Path." *GitHub*.](https://github.com/mlflow/mlflow/issues/11225)

- [MLflow Issue #11268. "Cannot Use Multipart Upload for Artifacts Without artifact_path Argument." *GitHub*.](https://github.com/mlflow/mlflow/issues/11268)

- [MLflow Issue #11828. "Proxied Multipart Upload Wrong Region in Presigned URL." *GitHub*.](https://github.com/mlflow/mlflow/issues/11828)

- [MLflow Issue #15455. "Remote Tracking Server - Unable to Log Models - Error 500." *GitHub*.](https://github.com/mlflow/mlflow/issues/15455)

### SkyPilot Issues

- [SkyPilot Issue #4285. "Stuck at STARTING When Launching with a Custom Image on RunPod." *GitHub*.](https://github.com/skypilot-org/skypilot/issues/4285)

- [SkyPilot Issue #3879. "Support Docker Image with Customized Entrypoint on RunPod." *GitHub*.](https://github.com/skypilot-org/skypilot/issues/3879)

- [SkyPilot Issue #4265. "RunPod 4090 Spot Not Available." *GitHub*.](https://github.com/skypilot-org/skypilot/issues/4265)

### RunPod Documentation

- [RunPod (2025). "Optimizing Docker Setup for PyTorch Training with CUDA 12.8 and Python 3.11." *RunPod Articles*.](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11)

- [RunPod (2025). "Running RunPod on SkyPilot." *RunPod Documentation*.](https://docs.runpod.io/integrations/skypilot)

- [RunPod Docker Hub. "runpod/pytorch." *Docker Hub*.](https://hub.docker.com/r/runpod/pytorch)

### Cloud Run Limits

- [Google Cloud (2026). "Cloud Run Quotas and Limits." *Google Cloud Documentation*.](https://cloud.google.com/run/quotas)

- [Dev.to (2020). "How to Overcome Cloud Run's 32 MB Request Limit." *DEV Community*.](https://dev.to/stack-labs/how-to-overcome-cloud-runs-32mb-request-limit-190j)
