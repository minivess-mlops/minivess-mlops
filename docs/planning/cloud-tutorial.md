---
title: "Cloud Setup Tutorial — Remote GPU Training and MLflow Hosting"
status: reference
created: "2026-03-13"
---

# Cloud Setup Tutorial

Step-by-step guide for setting up cloud services used by MinIVess MLOps.
**Everything here is OPTIONAL.** Local-only training with Docker Compose works
without any cloud accounts.

**Cross-references:**
- [`.env.example`](../../.env.example) — All variables defined there, this doc explains how to get them
- [`skypilot-and-finops-complete-report.md`](skypilot-and-finops-complete-report.md) — Architecture and cost analysis
- [`mlflow-deployment-storage-analysis.md`](mlflow-deployment-storage-analysis.md) — MLflow hosting options compared (Oracle rejected 2026-03-13)
- [`pulumi-iac-implementation-guide.md`](pulumi-iac-implementation-guide.md) — Pulumi deep-dive

---

## Who Needs What?

| I want to... | Services needed | Monthly cost |
|--------------|----------------|-------------|
| Train locally on my own GPU | Nothing from this page | $0 |
| Train on cloud GPUs (RunPod) | RunPod + SkyPilot + Remote MLflow | ~$0.22-1.99/hr GPU time |
| Share MLflow experiments publicly | UpCloud (trial) / Hetzner / DagsHub | EUR 0 (trial) / EUR 3.79 |
| Automate cloud deployment | Pulumi + UpCloud/Hetzner | EUR 0 (trial) |
| Use a custom domain (mlflow.yourlab.org) | Cloudflare + domain registrar | ~$10/year for domain |

Pick only the sections you need. Each section is independent.

---

## Quick Start: One-Script Bootstrap

> **Coming soon** ([#614](https://github.com/petteriTeikari/minivess-mlops/issues/614))
> — A single interactive script that walks you through all cloud setup:
>
> ```bash
> bash scripts/setup_cloud.sh
> ```
>
> The script opens browser links, waits for you to paste tokens, verifies each
> credential via API call, and writes everything to `.env`. Each service is
> skippable. Re-running is safe (idempotent). Total time: ~10 minutes vs ~45
> minutes of manual setup.
>
> **The manual steps below document what the script does behind the scenes.**
> Read them to understand what each service is and why it's needed — the script
> automates the mechanics, not the understanding.

---

## 1. RunPod Account (Cloud GPU Provider)

**What it is:** RunPod rents GPUs by the hour. Consumer GPUs (RTX 3090/4090) are
5-28x cheaper than AWS. SkyPilot uses RunPod as its primary GPU backend.

**Cost:** Pay-as-you-go, no subscription. RTX 3090 = $0.22/hr, RTX 4090 = $0.34/hr,
A100 80GB = $1.74/hr. You load credit in advance ($10 minimum).

<details>
<summary><strong>Step-by-step: Create RunPod account and get API key</strong></summary>

1. Go to [https://www.runpod.io/](https://www.runpod.io/) and click **Sign Up**
2. Create an account (email or GitHub OAuth)
3. Add billing: **Billing** tab in the left sidebar, add a credit card or PayPal
4. Load credit: start with **$10-25** (enough for many hours of RTX 3090)
5. Get your API key:
   - Click your avatar (top-right) -> **Settings**
   - Scroll to **API Keys**
   - Click **Create API Key**
   - Copy the key (starts with `rp_...`)
6. Add to your `.env` file:
   ```bash
   RUNPOD_API_KEY=rp_your_key_here
   ```

**Verify it works:**
```bash
# After installing SkyPilot (see section 2)
sky check
# Should show: RunPod: enabled
```

</details>

---

## 2. SkyPilot (Cloud GPU Orchestrator)

**What it is:** SkyPilot is an open-source tool that launches training jobs on
any cloud. You write one YAML file, and SkyPilot finds the cheapest available GPU
across RunPod, Lambda, AWS, GCP, etc. It handles spot instances, preemption
recovery, and auto-shutdown.

**Cost:** SkyPilot itself is free and open-source. You pay only for the cloud
GPUs it provisions.

<details>
<summary><strong>Step-by-step: Install SkyPilot and configure cloud access</strong></summary>

1. Install the SkyPilot extras group:
   ```bash
   uv sync --extra sky
   ```

2. Set up cloud credentials. SkyPilot reads credentials from standard locations:

   **RunPod** (recommended first cloud):
   ```bash
   # Already done if you set RUNPOD_API_KEY in .env
   # SkyPilot reads it from the environment
   ```

   **AWS** (optional, for failover):
   ```bash
   # Configure via AWS CLI
   aws configure
   # Enter: Access Key ID, Secret Key, region (us-east-1), output (json)
   ```

   **GCP** (optional, for failover):
   ```bash
   gcloud auth application-default login
   ```

3. Verify cloud access:
   ```bash
   sky check
   ```
   You should see a table showing which clouds are enabled. At minimum, RunPod
   should show `enabled`.

4. Test with a trivial job (optional):
   ```bash
   sky jobs launch -n test-job --cloud runpod --gpus T4:1 -- echo "Hello from cloud GPU"
   ```
   This launches a T4 GPU instance, prints "Hello", and shuts down. Costs ~$0.01.

**Common issues:**

- `sky check` shows RunPod as `disabled`: Make sure `RUNPOD_API_KEY` is set in
  your current shell. Try `source .env` or `export RUNPOD_API_KEY=rp_...`.
- `sky check` hangs: SkyPilot is checking all clouds. Use `--cloud runpod` to
  check only RunPod.
- `ModuleNotFoundError: skypilot`: You need `uv sync --extra sky`, not just `uv sync`.

</details>

---

## 3. Remote MLflow Server

**What it is:** When training runs on a cloud GPU (via SkyPilot), the cloud VM
needs somewhere to log metrics and artifacts. Your local MLflow at
`http://localhost:5000` is not reachable from the internet. A remote MLflow server
is a publicly accessible MLflow instance that both your local machine and cloud
VMs can reach.

**Options** (pick one):

| Option | Cost | Setup time | Best for |
|--------|------|-----------|----------|
| **UpCloud VPS** (active — free trial) | EUR 0 for 30 days (EUR 250 credit) | 5 minutes | Free trial, Helsinki DC |
| **Hetzner VPS** (archived fallback) | EUR 3.79/month | 5 minutes | Long-term, cheapest |
| **DagsHub** | $0 (20 GB free) | 5 minutes | Quick start, zero ops |
| **Scaleway** | EUR 9.30/month | 10 minutes | Alternative EU provider |
| ~~Nebius~~ | ~~$40/month~~ | — | **Skip** — GPU-focused, overpriced for VPS |
| ~~Oracle Cloud Always Free~~ | ~~$0/month~~ | ~~2 hours~~ | **Rejected** — chronic ARM capacity shortage, garbage DevEx |
| **Your own server** | Varies | Varies | If you already have one |

### Full Provider Comparison (researched 2026-03-13)

| Provider | Plan | vCPU | RAM | Disk | EUR/month | Trial | CLI | Verdict |
|----------|------|------|-----|------|-----------|-------|-----|---------|
| **UpCloud** | DEV-2xCPU-4GB | 2 | 4 GB | 60 GB SSD | 19.42 | **EUR 250 / 30d** | `upctl` | **Active** |
| **Hetzner** | CX22 | 2 | 4 GB | 40 GB NVMe | 3.79 | None | `hcloud` | **Fallback** |
| **Scaleway** | DEV1-M + IPv4 | 3 | 4 GB | 40 GB NVMe | 17.33 | EUR 100 | `scw` | Alternative |
| **Nebius** | cpu-e2 2x8GB | 2 | 8 GB | 50 GB SSD | ~36.50 | None | `nebius` | Skip |
| **Oracle** | A1.Flex 4x24 | 4 | 24 GB | 200 GB | 0 | Always Free | `oci` | **Rejected** |

<details>
<summary><strong>Option A: DagsHub (fastest — 5 minutes, managed)</strong></summary>

DagsHub hosts a free MLflow server for you. No infrastructure to manage.

1. Sign up at [https://dagshub.com/](https://dagshub.com/) (free, GitHub OAuth)
2. Create a new repository (or connect your existing GitHub repo)
3. Go to your repo on DagsHub -> **Experiments** tab
4. Click the **Remote** button to see your MLflow tracking URI
5. It looks like: `https://dagshub.com/YOUR_USER/YOUR_REPO.mlflow`
6. Get your credentials:
   - DagsHub username = your MLflow username
   - DagsHub token (Settings -> Tokens) = your MLflow password
7. Add to `.env`:
   ```bash
   MLFLOW_TRACKING_URI_REMOTE=https://dagshub.com/YOUR_USER/YOUR_REPO.mlflow
   MLFLOW_TRACKING_USERNAME_REMOTE=YOUR_DAGSHUB_USERNAME
   MLFLOW_TRACKING_PASSWORD_REMOTE=YOUR_DAGSHUB_TOKEN
   ```

**Limitations:** 20 GB total storage (DVC + MLflow combined). No direct
PostgreSQL access for DuckDB analytics. DagsHub controls the MLflow version.

</details>

<details>
<summary><strong>Option B: UpCloud VPS (active — EUR 250 free trial, 30 days)</strong></summary>

UpCloud is a Finnish cloud provider with Helsinki datacenters. The 30-day free
trial with EUR 250 credit is enough to run the MLflow stack for the entire trial.

**Automated setup** (recommended — 5 minutes):
```bash
# 1. Get API token: https://hub.upcloud.com/account/api-tokens
# 2. Add to .env: UPCLOUD_TOKEN=ucat_...
# 3. Run:
bash scripts/upcloud-setup-script.sh
```

The script creates the server, installs Docker, deploys the MLflow stack, sets up
the firewall, and prints your access credentials. See `docs/planning/upcloud-mlflow-plan.md`.

**After the script completes**, add to `.env`:
```bash
MLFLOW_TRACKING_URI_REMOTE=http://SERVER_IP
MLFLOW_TRACKING_USERNAME_REMOTE=minivess
MLFLOW_TRACKING_PASSWORD_REMOTE=your_generated_password
```

**Trial management:** Back up your data before day 30. See the backup section in
`docs/planning/upcloud-mlflow-plan.md`. After trial, either deposit EUR 10 to keep
the server, or migrate to Hetzner (`bash scripts/hetzner-setup-script.sh`).

</details>

<details>
<summary><strong>Option C: Hetzner VPS (archived fallback — EUR 3.79/month)</strong></summary>

Hetzner is the cheapest long-term option. Use after UpCloud trial ends.

**Automated setup:**
```bash
# 1. Get API token: https://console.hetzner.cloud → Security → API Tokens
# 2. Add to .env: HETZNER_API_TOKEN=...
# 3. Run:
bash scripts/hetzner-setup-script.sh
```

See `docs/planning/hetzner-mlflow-plan.md` for details.

</details>

<details>
<summary><strong>~~Option D: Oracle Cloud Always Free (REJECTED)~~</strong></summary>

Oracle Cloud Always Free was rejected (2026-03-13) due to:
- Chronic ARM capacity shortage in Frankfurt (all 3 ADs exhausted)
- Home region is permanent — cannot change after sign-up
- Card verification blocks second account on a better region
- Terrible DevEx (PKCS#8 key format issues, intermittent auth errors, browser-only bootstrap)

See `docs/planning/oracle-config-planning.md` (status: rejected).

</details>

---

## 4. Oracle Cloud Account (Free VM for MLflow)

**What it is:** Oracle Cloud has a genuinely permanent free tier that includes a
4-core ARM server with 24 GB RAM and 200 GB storage. Unlike AWS/GCP/Azure free
tiers that expire after 12 months, Oracle's is forever.

**Cost:** $0/month. Credit card required at sign-up but never charged for Always
Free resources.

<details>
<summary><strong>Step-by-step: Create Oracle Cloud account</strong></summary>

1. Go to [https://cloud.oracle.com/](https://cloud.oracle.com/) and click **Sign Up**
2. Fill in your details. Choose your **home region** carefully — it is
   **permanent** and cannot be changed after sign-up. Pick the region
   closest to you that has good ARM capacity:

   | Region | Code | ARM availability | Best for |
   |--------|------|-----------------|----------|
   | **Milan** | `eu-milan-1` | Good | EU users (recommended) |
   | **Marseille** | `eu-marseille-1` | Good | EU users |
   | **Stockholm** | `eu-stockholm-1` | Good | Nordic users |
   | Frankfurt | `eu-frankfurt-1` | **Chronically exhausted** | Avoid — weeks-long wait |
   | **Phoenix** | `us-phoenix-1` | Good | US West users |
   | **Ashburn** | `us-ashburn-1` | Moderate | US East users |
   | **Chicago** | `us-chicago-1` | Good | US Central users |
   | **San Jose** | `us-sanjose-1` | Good | US West users |
   | **Tokyo** | `ap-tokyo-1` | Moderate | Asia-Pacific users |
   | **Seoul** | `ap-seoul-1` | Good | Asia-Pacific users |
   | **Sydney** | `ap-sydney-1` | Good | Oceania users |
   | **Mumbai** | `ap-mumbai-1` | Good | South Asia users |
   | **São Paulo** | `sa-saopaulo-1` | Good | South America users |

   **Rule of thumb:** Avoid the most popular regions (Frankfurt, Ashburn).
   Pick a region on the same continent as your SkyPilot GPU provider — RunPod
   has data centers in EU and US, so latency from any EU/US Oracle region is
   fine for MLflow metric logging.

3. Add a credit card (required for verification, never charged for free tier)
4. Wait for account activation (usually instant, sometimes up to 24 hours)

</details>

<details>
<summary><strong>Step-by-step: Find your Tenancy OCID</strong></summary>

Your Tenancy OCID identifies your entire Oracle Cloud account. You need it for
Pulumi and the OCI CLI.

1. Log in to [OCI Console](https://cloud.oracle.com/)
2. Click the **hamburger menu** (three horizontal lines, top-left corner)
3. Scroll down to **Governance & Administration** -> click **Tenancy Details**
4. On the Tenancy Details page, find **OCID** — it's a long string starting with
   `ocid1.tenancy.oc1..`
5. Click the **Copy** link next to it
6. Add to `.env`:
   ```bash
   OCI_TENANCY_OCID=ocid1.tenancy.oc1..aaaa...  # paste what you copied
   ```

</details>

<details>
<summary><strong>Step-by-step: Find your User OCID</strong></summary>

Your User OCID identifies *you* within the Oracle Cloud account.

1. In the OCI Console, click your **avatar icon** (top-right corner, looks like
   a person silhouette)
2. In the dropdown, click **User settings**
3. On the User Details page, find **OCID** under your email address
4. Click **Copy** next to it (starts with `ocid1.user.oc1..`)
5. Add to `.env`:
   ```bash
   OCI_USER_OCID=ocid1.user.oc1..aaaa...  # paste what you copied
   ```

</details>

<details>
<summary><strong>Step-by-step: Find your Compartment OCID</strong></summary>

A Compartment is like a folder for organizing your cloud resources. Every OCI
account has a "root" compartment created automatically.

1. Click the **hamburger menu** (top-left)
2. Go to **Identity & Security** -> **Compartments**
3. You should see one compartment with the same name as your tenancy
   (e.g., `petteriteikari`)
4. Click on it
5. Copy the **OCID** (starts with `ocid1.compartment.oc1..`)
6. Add to `.env`:
   ```bash
   OCI_COMPARTMENT_OCID=ocid1.compartment.oc1..aaaa...  # paste what you copied
   ```

</details>

<details>
<summary><strong>Step-by-step: Generate and upload API signing key</strong></summary>

Oracle uses RSA key pairs (like SSH keys) to authenticate API calls. You generate
a key pair on your computer, then upload the **public** half to Oracle so it can
verify your requests.

You can generate the key pair via the **OCI CLI** (recommended) or via the
**OCI Console web UI**. Both produce the same result.

---

**Option A — OCI CLI (recommended, scriptable):**

```bash
# Install OCI CLI if you don't have it
pip install oci-cli

# Generate key pair (no passphrase — press Enter twice when prompted)
oci setup keys --output-dir ~/.oci --key-name oci_api_key
```

This creates two files:
- `~/.oci/oci_api_key.pem` — the **private** key (stays on your machine, never share)
- `~/.oci/oci_api_key_public.pem` — the **public** key (you upload this to Oracle)

It also prints a fingerprint like `a8:10:05:a1:64:dc:...` — save this for later.

Now upload the **public** key to Oracle. This is the one step you must do in the
web browser (chicken-and-egg: you need a working key to call the API, but you
need the API to upload a key):

1. First, copy the public key contents to your clipboard. Run this in your
   terminal:
   ```bash
   cat ~/.oci/oci_api_key_public.pem
   ```
   Select and copy the **entire output**, including the `-----BEGIN PUBLIC KEY-----`
   and `-----END PUBLIC KEY-----` lines.

2. In the OCI Console, click your **avatar icon** (top-right corner)
3. Click **User settings**
4. Click the **Tokens and keys** tab (in the row of tabs: Details, My groups,
   My requests, My resources, **Tokens and keys**, ...)
5. Scroll down to the **API keys** section
6. Click the **Add API key** button
7. In the dialog that appears, select **Paste a public key** (the third option)
8. Paste the public key you copied in step 1 into the **Public key** text box
9. Click **Add**
10. Oracle shows a "Configuration file preview" dialog — you can close this
    (we use `.env` instead of the OCI config file)

The fingerprint shown should match what `oci setup keys` printed earlier.

---

**Option B — OCI Console (manual, no CLI needed):**

If you don't want to install the OCI CLI, Oracle can generate the key pair for you:

1. In the OCI Console: avatar -> **User settings** -> **Tokens and keys** tab
2. Scroll to **API keys** -> click **Add API key**
3. Select **Generate API key pair** (the first option)
4. Click **Download private key** — save the `.pem` file
5. Click **Add**
6. Copy the **fingerprint** from the confirmation dialog
7. Move the downloaded key to the standard location:
   ```bash
   mkdir -p ~/.oci
   mv ~/Downloads/*.pem ~/.oci/oci_api_key.pem
   chmod 600 ~/.oci/oci_api_key.pem
   ```

---

**Add to `.env` (same for both options):**
```bash
OCI_FINGERPRINT=a8:10:05:a1:64:dc:7a:a5:8b:a2:9e:2b:36:c2:a5:d3  # your fingerprint
OCI_PRIVATE_KEY_PATH=~/.oci/oci_api_key.pem
OCI_REGION=eu-frankfurt-1  # replace with your actual home region
```

**Verify it works:**
```bash
oci iam user get --user-id $OCI_USER_OCID
# Should return your user details as JSON (name, email, lifecycle-state: ACTIVE)
# If you get "NotAuthenticated": double-check fingerprint and key path
```

</details>

<details>
<summary><strong>Key rotation (after first key is working)</strong></summary>

Once you have one working API key, future key rotations are fully programmatic
(no console clicks needed):

```bash
# Generate new key pair
oci setup keys --output-dir ~/.oci --key-name oci_api_key_new

# Upload via API (uses your existing working key for auth)
oci iam user api-key upload \
  --user-id "$OCI_USER_OCID" \
  --key-file ~/.oci/oci_api_key_new_public.pem

# Update .env with new fingerprint
# (printed by both oci setup keys and oci iam user api-key upload)

# Optionally delete the old key
oci iam user api-key delete \
  --user-id "$OCI_USER_OCID" \
  --fingerprint "$OLD_FINGERPRINT"
```

The `scripts/setup_cloud.sh` bootstrap script
([#614](https://github.com/petteriTeikari/minivess-mlops/issues/614)) will
automate this entire flow, including the first-time manual upload step with
guided prompts.

</details>

<details>
<summary><strong>Gotcha: Frankfurt ARM capacity shortage</strong></summary>

Oracle Frankfurt (`eu-frankfurt-1`) is the most popular EU region and frequently
runs out of ARM (A1.Flex) capacity. If you see "Out of host capacity" when
creating your VM:

1. **Try a different region** at sign-up (Milan, Marseille, Stockholm)
2. **Retry script:** Use [hitrov/oci-arm-host-capacity](https://github.com/hitrov/oci-arm-host-capacity)
   to automatically retry provisioning until capacity opens up
3. **Start small:** Request 1 OCPU / 6 GB first (easier to provision), then
   resize later

The free tier gives you **4 OCPU + 24 GB total** across all VMs. You can split
this (e.g., 2+12 GB + 2+12 GB) or use it all in one VM.

</details>

---

## 5. Pulumi (Infrastructure-as-Code)

**What it is:** Pulumi lets you define cloud infrastructure in Python code.
Instead of clicking through the Oracle Cloud console to create a VM, network,
firewall rules, and DNS records, you write a Python program and run `pulumi up`.
It creates everything in the right order and can tear it all down with
`pulumi destroy`.

**Cost:** Free tier (1 user, unlimited resources). The Pulumi CLI is open-source.

<details>
<summary><strong>Step-by-step: Set up Pulumi</strong></summary>

1. **Create a Pulumi account** (for state storage):
   - Go to [https://app.pulumi.com/signup](https://app.pulumi.com/signup)
   - Sign up with GitHub (recommended) or email
   - The free Individual tier is all you need

2. **Get your access token:**
   - Go to [https://app.pulumi.com/account/tokens](https://app.pulumi.com/account/tokens)
   - Click **Create Token**
   - Give it a name (e.g., "minivess-dev")
   - Copy the token
   - Add to `.env`:
     ```bash
     PULUMI_ACCESS_TOKEN=pul-...your_token...
     ```

3. **Install the Pulumi CLI:**
   ```bash
   # Linux
   curl -fsSL https://get.pulumi.com | sh

   # macOS
   brew install pulumi

   # Verify
   pulumi version
   ```

4. **Log in:**
   ```bash
   pulumi login
   # Uses PULUMI_ACCESS_TOKEN from environment, or prompts for it
   ```

5. **Deploy the MLflow stack** (after the Pulumi program is implemented):
   ```bash
   cd deployment/pulumi
   pulumi stack init dev
   pulumi config set oci:region eu-frankfurt-1
   pulumi config set --secret postgres_password "$(openssl rand -base64 24)"
   pulumi up
   ```
   This creates the Oracle Cloud VM, installs Docker, deploys MLflow, and
   configures DNS — all in one command.

</details>

---

## 6. Cloudflare DNS (Custom Domain)

**What it is:** If you want your MLflow server at `mlflow.yourdomain.com` instead
of a raw IP address, you need DNS management. Cloudflare provides free DNS hosting
with bonus DDoS protection. Pulumi uses the Cloudflare API to create DNS records
automatically.

**Cost:** Free plan. You need to own a domain (~$10/year from any registrar).

<details>
<summary><strong>Step-by-step: Set up Cloudflare for your domain</strong></summary>

**Prerequisites:** You must own a domain (e.g., `minivess.fi`). Buy one from
[Namecheap](https://www.namecheap.com/), [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/),
[Porkbun](https://porkbun.com/), or any registrar.

1. **Add your domain to Cloudflare:**
   - Go to [https://dash.cloudflare.com/](https://dash.cloudflare.com/) and sign up / log in
   - Click **Add a Site**
   - Enter your domain (e.g., `minivess.fi`)
   - Select the **Free** plan
   - Cloudflare shows you two nameservers (e.g., `ana.ns.cloudflare.com`)
   - Go to your domain registrar and change the nameservers to Cloudflare's
   - Wait for DNS propagation (usually 5 minutes to 24 hours)

2. **Get your Zone ID:**
   - In Cloudflare dashboard, click your domain
   - Look at the right sidebar under **API** -> **Zone ID**
   - Copy the 32-character hex string
   - Add to `.env`:
     ```bash
     CLOUDFLARE_ZONE_ID=abc123def456...
     ```

3. **Create an API token:**
   - Click your avatar (top-right) -> **My Profile** -> **API Tokens**
   - Click **Create Token**
   - Use the **Edit zone DNS** template (or create a custom token)
   - Set permissions: **Zone** / **DNS** / **Edit**
   - Set zone resources: **Include** / **Specific zone** / your domain
   - Click **Continue to summary** -> **Create Token**
   - Copy the token (shown only once)
   - Add to `.env`:
     ```bash
     CLOUDFLARE_API_TOKEN=your_token_here
     ```

4. **Verify it works:**
   ```bash
   curl -X GET "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID" \
     -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
     -H "Content-Type: application/json" | python3 -m json.tool
   ```
   Should return your zone details without errors.

</details>

---

## 7. GitHub Secrets (For Future CI)

GitHub Actions CI is currently **explicitly disabled** in this project (CLAUDE.md
Rule #21). These secrets are documented for completeness — you only need to set
them if/when CI is re-enabled by the project maintainer.

<details>
<summary><strong>Which secrets to set in GitHub</strong></summary>

Go to your GitHub repo -> **Settings** -> **Secrets and variables** -> **Actions**.

| Secret name | Value source | Used by |
|-------------|-------------|---------|
| `RUNPOD_API_KEY` | RunPod dashboard (section 1) | SkyPilot cloud launches |
| `HF_TOKEN` | [HuggingFace tokens](https://huggingface.co/settings/tokens) | Gated model downloads |
| `OCI_PRIVATE_KEY` | `base64 -w0 ~/.oci/oci_api_key.pem` | Pulumi OCI deployments |
| `OCI_TENANCY_OCID` | OCI Console (section 4) | Pulumi OCI provider |
| `OCI_USER_OCID` | OCI Console (section 4) | Pulumi OCI provider |
| `OCI_COMPARTMENT_OCID` | OCI Console (section 4) | Pulumi OCI provider |
| `OCI_FINGERPRINT` | OCI API key setup (section 4) | Pulumi OCI provider |
| `OCI_REGION` | Your OCI home region | Pulumi OCI provider |
| `CLOUDFLARE_API_TOKEN` | Cloudflare dashboard (section 6) | DNS management |
| `CLOUDFLARE_ZONE_ID` | Cloudflare dashboard (section 6) | DNS management |
| `PULUMI_ACCESS_TOKEN` | Pulumi Cloud (section 5) | IaC state management |
| `MLFLOW_TRACKING_PASSWORD_REMOTE` | Self-generated | Remote MLflow auth |

**Remember:** None of these are needed until CI is re-enabled.

</details>

---

## Quick Reference: Which `.env` Variables Do I Need?

### Minimum (local-only training)
```bash
# These are already in .env.example — just fill in your values
HF_TOKEN=hf_...
MODEL_CACHE_HOST_PATH=/home/youruser/download_cache
```

### Add SkyPilot cloud training
```bash
# Add these to the above
RUNPOD_API_KEY=rp_...
MLFLOW_TRACKING_URI_REMOTE=https://mlflow.yourdomain.com
MLFLOW_TRACKING_USERNAME_REMOTE=minivess
MLFLOW_TRACKING_PASSWORD_REMOTE=...
```

### Add UpCloud MLflow hosting (free trial)
```bash
# Add these to the above
UPCLOUD_TOKEN=ucat_...
# Then run: bash scripts/upcloud-setup-script.sh
```

### Add Hetzner MLflow hosting (long-term fallback)
```bash
# Add these to the above
HETZNER_API_TOKEN=...
# Then run: bash scripts/hetzner-setup-script.sh
```

### Add custom domain
```bash
# Add these to the above
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_ZONE_ID=...
```
