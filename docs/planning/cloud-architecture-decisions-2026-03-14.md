# Cloud Architecture Decisions — User Verbatim (2026-03-14)

> These are the user's exact words, saved verbatim so the decisions don't need
> to be re-explained in future sessions.

## Grand Vision: Multi-Provider SkyPilot Architecture

> "The decision was to keep Runpod for "dev" environment when not using Docker
> (which we are not doing atm, but someone might want to do with this repo later),
> and then for the "prod" and "staging" with Docker obligatory, we would like to
> keep both Lambda and GCP L4 and T4. Remember that we are using SkyPilot (or dstack)
> and we should ideally have every fucking instance on every fucking supported
> provider supported via Skypilot so let's start with a manageable set with
> a) Runpod 4090 for "dev", b) current Lambda Labs for Lambda Labs support,
> c) the 2 new GCP spots. And then remember that this is an academic repo
> open-sourced so the end-users in different labs might have AWS or Azure deals
> so it would be nicest to provide them ready-made recipes for AWS and Azure to
> minimize their struggle for this excellent DevEx that we have in our CLAUDE.md"

### Summary

| Environment | Provider(s) | GPU Types | Docker | Purpose |
|-------------|-------------|-----------|--------|---------|
| **dev** | RunPod | RTX 4090 | No (runtime env) | Fast interactive iteration |
| **staging** | Lambda Labs, GCP | A10, L4 spot, T4 spot | YES (mandatory) | Docker-based testing |
| **prod** | Lambda Labs, GCP | L4/T4/A100 spot | YES (mandatory) | Full pipeline |
| **recipes** | AWS, Azure | Per user's cloud deal | YES | Ready-made YAMLs for labs |

## Consumer GPU Reality

> "As we don't have yet massive datasets, the consumer 4090 is pretty sufficient
> for our needs, but then it is not available on major cloud platforms as NVIDIA
> license does not allow its use like that, so we would like to have Runpod pricing
> with the 'whole cloud stack' using consumer GPUs but I guess those are not really
> available?"
>
> "Like the 16GB VRAM is enough for us, and the A100 seems like an overkill, but we
> should for sure support the large GPU instances as well when people have larger
> datasets. I would have just preferred that our ideal cloud 'partner' would have
> instances from 4090-like GPUs to the big boys (8 x A100/H100/etc instances)"

### Takeaway

- 16 GB VRAM is sufficient for current workloads
- GCP L4 (24 GB, $0.22/hr spot) is the closest to RTX 4090 on a major cloud
- GCP T4 (16 GB, $0.14/hr spot) is the cheapest option that fits
- RunPod keeps the consumer GPU path alive for dev
- A100/H100 support for users with larger datasets

## Priority: Fix Lambda First

> "Need to create [GCP account] later. So create a P0 Issue on this, and I will fix
> this later tonight, so let's focus first getting the Lambda Labs working if it was
> just one mlflow env variable that was blocking us. Let's make the Lambda Labs work
> before moving on to GCP"

### Action Items

1. **NOW**: Fix `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` on UpCloud
2. **NOW**: Re-run Lambda smoke test to verify artifacts upload
3. **TONIGHT (user)**: Create GCP account ($300 free credits)
4. **THIS WEEK**: Set up GCP Pulumi stack
5. **LATER**: Add AWS + Azure recipe YAMLs for academic labs

## Save User Feedback

> "Read my previous answer on the grand vision, and save all my answers verbatim
> to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning so that
> I don't have to keep on explaining the same things all over again"
