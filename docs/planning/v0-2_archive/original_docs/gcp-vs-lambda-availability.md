# GCP vs Lambda Labs: Availability Self-Reflection

> **Date**: 2026-03-14
> **Conclusion**: GCP is DRAMATICALLY more available and easier to set up.

---

## The Lambda Experience (8+ hours of pain)

| Step | Time Spent | Result |
|------|-----------|--------|
| Debug RunPod Docker failures | 8 hours | Discovered RunPod = containers, not VMs |
| Switch to Lambda Labs | 30 min | Configured, launched |
| Wait for Lambda A10 GPU | ~3 hours | ALL SOLD OUT across 17 regions |
| Catch A100 briefly | 10 min | Got it! Training worked! |
| MLflow artifact upload | 2 hours | Failed — server v2.20 vs client v3.10 |
| Upgrade MLflow server | 15 min | Done, but Lambda sold out again |
| Retry with retry_until_up | 2+ hours | Still sold out. Gave up. |
| **Total Lambda debugging** | **~16 hours** | Training works but can't verify artifact fix |

### Lambda's Fundamental Problem

Lambda Labs has **great pricing** (A10 at $0.86/hr) but **terrible availability**.
All 17 regions were simultaneously sold out for single-GPU instances. When capacity
appeared, it was brief (minutes) before being snatched.

The multi-region launcher we built (17 regions, EU first) is nice engineering but
can't solve a supply problem. Lambda is building new data centers but demand
consistently exceeds supply.

## The GCP Experience (15 minutes to first launch)

| Step | Time Spent | Result |
|------|-----------|--------|
| Install gcloud CLI | Already done | User installed in parallel |
| gcloud init | 2 min | Browser auth, project selected |
| Enable APIs | 3 min | 8 APIs enabled via CLI |
| Fix pyparsing bug | 5 min | httplib2 version pinned |
| sky check gcp | 30 sec | **ENABLED** (compute + storage) |
| Launch smoke test | 10 sec | **Immediately found T4 spot capacity** |
| GPU quota request | 2 min | Submitted (auto-approval expected) |
| **Total GCP setup** | **~15 minutes** | First launch attempt in under 15 min |

### GCP's Advantages

1. **Instant availability**: SkyPilot found T4 and L4 spot instances IMMEDIATELY.
   No waiting, no sold out, no retry loops. (Blocked only by quota, not capacity.)

2. **Multiple regions tried automatically**: SkyPilot tried me-west1, asia-northeast3,
   us-west3 — all had capacity, just no quota for new accounts.

3. **Quota is a one-time setup**: Once approved (minutes to hours for small requests),
   GPU access is permanent. Lambda's availability is always uncertain.

4. **Same-region everything**: Once we add Cloud Run + GCS + Cloud SQL, MLflow artifacts
   upload in seconds (same region), not minutes (cross-Atlantic to UpCloud Helsinki).

5. **Spot recovery built-in**: SkyPilot MOUNT_CACHED + managed jobs handles preemption
   automatically. Lambda has no spot instances at all.

## The Irony

We spent 8 hours debugging RunPod, then another 8 hours waiting for Lambda GPUs.
GCP was set up in 15 minutes and immediately found GPU capacity. The only blocker
is a quota request that GCP auto-approves for small requests.

## Cost Comparison (Honest)

| Provider | GPU | $/hr | Availability | Docker |
|----------|-----|------|-------------|--------|
| RunPod RTX 4090 spot | 24 GB | $0.34 | Good | **NO** (container-based) |
| Lambda A10 on-demand | 24 GB | $0.86 | **TERRIBLE** | YES (VM) |
| **GCP T4 spot** | **16 GB** | **$0.14** | **EXCELLENT** | **YES (VM)** |
| **GCP L4 spot** | **24 GB** | **$0.22** | **EXCELLENT** | **YES (VM)** |
| GCP A100 spot | 40 GB | $1.15 | Good | YES (VM) |

**GCP T4 spot at $0.14/hr is the cheapest Docker-compatible GPU option that actually
has availability.** Lambda is cheaper per-hour for some GPUs but you can't use them
if they're sold out.

## Lesson Learned

> "The cheapest GPU is the one you can actually get."
>
> Lambda Labs' pricing is attractive but meaningless when capacity is zero.
> GCP's spot pricing is competitive AND available. For a solo researcher who
> needs GPUs NOW, not "whenever Lambda restocks," GCP is the clear winner.

## Recommendation

- **GCP**: Primary cloud for staging + prod (spot L4/T4, always available)
- **Lambda**: Keep as fallback (on-demand, when it has capacity)
- **RunPod**: Dev environment only (no Docker, consumer GPUs)
