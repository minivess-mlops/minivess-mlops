# Security Posture Assessment: LiteLLM/Trivy Supply Chain Incident

**Date**: 2026-03-25
**Trigger**: LiteLLM PyPI supply chain compromise (TeamPCP, March 24 2026)
**Scope**: Vascadia repository — dependency audit, Docker scanning, model weight security
**Status**: Assessment complete, executable actions at bottom

---

## 1. Incident Summary

On March 24, 2026, threat actor **TeamPCP** published backdoored versions of `litellm`
(1.82.7 and 1.82.8) to PyPI. The attack chain started a week earlier with the compromise
of **Trivy** (Aqua Security's vulnerability scanner) via a GitHub Actions workflow
vulnerability. Trivy was the initial vector — the compromised scanner exfiltrated
LiteLLM's `PYPI_PUBLISH` token, which was then used to push trojaned packages.

### Attack Timeline

| Date | Event |
|------|-------|
| Late Feb 2026 | TeamPCP discovers `pull_request_target` vulnerability in Trivy's CI |
| Mar 19, 17:43 UTC | Trivy v0.69.4 published (backdoored); 76/77 `trivy-action` tags force-pushed |
| Mar 23 | Checkmarx KICS GitHub Actions also compromised; C2 domain registered |
| Mar 24, 10:39 UTC | Malicious `litellm 1.82.7` on PyPI (injected into `proxy_server.py`) |
| Mar 24, 10:52 UTC | Malicious `litellm 1.82.8` on PyPI (added `litellm_init.pth` payload) |
| Mar 24, ~13:38 UTC | PyPI quarantines the packages (~3 hour exposure window) |

### Technical Payload (litellm 1.82.8)

Three-stage attack:
1. **Credential harvesting**: Sweeps SSH keys, AWS/GCP/Azure credentials, K8s secrets,
   `.env` files, shell history, crypto wallets
2. **Exfiltration**: AES-256-CBC encrypted with hardcoded RSA-4096 public key,
   POSTed to `https://models.litellm.cloud/` (lookalike domain)
3. **Persistence**: systemd backdoor (`sysmon.service`) + Kubernetes lateral movement
   deploying privileged pods to every node

**Critical technical detail**: The `litellm_init.pth` is a **Python path configuration
file** (NOT a PyTorch checkpoint). It executes on EVERY Python interpreter startup —
no import required. pip, IDEs, pytest, language servers all trigger it.

### Sources

- [Snyk: How a Poisoned Security Scanner Became the Key to Backdooring LiteLLM](https://snyk.io/articles/poisoned-security-scanner-backdooring-litellm/)
- [Wiz: TeamPCP trojanizes LiteLLM in continuation of campaign](https://www.wiz.io/blog/threes-a-crowd-teampcp-trojanizes-litellm-in-continuation-of-campaign)
- [Simon Willison: Malicious litellm_init.pth](https://simonwillison.net/2026/Mar/24/malicious-litellm/)
- [The Hacker News: TeamPCP Backdoors LiteLLM 1.82.7-1.82.8](https://thehackernews.com/2026/03/teampcp-backdoors-litellm-versions.html)
- [Aqua Security: Trivy Supply Chain Attack](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know/)
- [Microsoft: Detecting Trivy Supply Chain Compromise](https://www.microsoft.com/en-us/security/blog/2026/03/24/detecting-investigating-defending-against-trivy-supply-chain-compromise/)

---

## 2. Impact Assessment on Vascadia

### 2a. LiteLLM — MODERATE RISK (currently safe, future exposure)

| Aspect | Finding |
|--------|---------|
| **Installed version** | `litellm 1.82.0` (in `agents` optional extra) |
| **Compromised versions** | 1.82.7, 1.82.8 (yanked from PyPI) |
| **Are we affected?** | **NO** — installed version predates compromise |
| **Future risk** | `pyproject.toml` line 118: `"litellm>=1.70,<2.0"` — a fresh `uv sync --all-extras` could theoretically resolve to a compromised version if PyPI un-yanks |
| **IOC check** | No `litellm_init.pth` in site-packages. No `sysmon.service` in `~/.config/systemd/`. Clean. |
| **Action** | **P0**: Pin to `>=1.70,<=1.82.6` immediately |

### 2b. Trivy — DIRECT RISK (unpinned install script)

| Aspect | Finding |
|--------|---------|
| **Installed?** | NO — Trivy is not currently installed on the system |
| **Exposure vector** | `Makefile` line 383: `curl ... | sh` installs **latest** Trivy with NO version pin |
| **If run Mar 19-24** | Would have installed compromised v0.69.4 |
| **Affected files** | `Makefile` (install-trivy), `scripts/weekly_security_scan.sh`, `scripts/pr_readiness_check.sh` |
| **Action** | **P1**: Pin to v0.69.3 or switch to Grype |

### 2c. MLflow — NOT DIRECTLY AFFECTED

| Aspect | Finding |
|--------|---------|
| **Installed version** | MLflow 3.10.0 |
| **TeamPCP impact** | None — MLflow was NOT compromised by TeamPCP |
| **Historical CVEs** | CVE-2024-27132 (XSS), prior RCE vulns — all patched in 3.10.0 |
| **Action** | **P2**: Monitor advisories, no immediate action |

### 2d. PyTorch Model Weights — PRE-EXISTING RISK

| Aspect | Finding |
|--------|---------|
| **torch.load safety** | 6 source files use `weights_only=False` (allows arbitrary code execution) |
| **Affected files** | `checkpoint_utils.py`, `checkpoint_averaging.py`, `model_merging.py`, `subsampled_ensemble.py`, `swag.py` (x2) |
| **Risk level** | LOW — checkpoints are self-produced, not downloaded from internet |
| **SAM3 weights** | Downloaded from HuggingFace — these ARE from the internet |
| **Action** | **P2**: Audit `weights_only=False` calls, add SHA-256 verification for HF downloads |

### 2e. GitHub Actions CI — NOT AFFECTED

GitHub Actions CI is explicitly disabled (CLAUDE.md Rule #21). No `trivy-action` or
`setup-trivy` GitHub Actions in any workflow. Trivy usage is CLI-only via Makefile.

### 2f. Python `.pth` File Attack Vector

The `litellm_init.pth` attack exploits Python's path configuration file mechanism
(NOT PyTorch's `.pth` checkpoint format). Current `.pth` files in vascadia's venv
are all legitimate: `a1_coverage.pth`, `_cuda_bindings_redirector.pth`,
`distutils-precedence.pth`, `_minivess_mlops.pth`, `pytest-cov.pth`, `_virtualenv.pth`.

**Action**: **P2**: Add a pre-commit check that audits `.pth` files in site-packages.

---

## 3. AI Agent Sandbox Escape Risk

### SandboxEscapeBench (arXiv 2603.02277)

Key findings from the benchmark evaluating 9 frontier LLMs on 18 container escape scenarios:

- **GPT-5**: 0.50 success rate; **Claude Opus 4.5**: 0.49 on misconfiguration exploits
- **Easy escapes** (difficulty 1-2): Frontier models reliably exploit privileged containers,
  Docker socket exposure, capability misconfigurations (CAP_SYS_ADMIN)
- **Hard escapes** (difficulty 4-5): Zero successes on kernel exploits (Dirty COW/Pipe)
- **No novel vulnerabilities discovered** — all successes exploit documented CVEs
- **Conclusion**: "Correctly configured, up-to-date container runtimes remain appropriate
  isolation for current models"

### Claude Code Sandbox Escapes (Disclosed CVEs)

| CVE | Severity | Vector | Status |
|-----|----------|--------|--------|
| CVE-2025-59536 | Critical | Hooks in `.claude/settings.json` execute on SessionStart | Patched |
| CVE-2026-21852 | Critical | `ANTHROPIC_BASE_URL` redirects API to attacker server | Patched |
| CVE-2026-25725 | High | Bubblewrap fails to protect `.claude/settings.json` when absent at startup | Patched |
| Ona research | N/A | `/proc/self/root/` symlink bypasses path deny patterns; dynamic linker (`ld-linux-x86-64.so.2`) bypasses content-hash enforcement | Known limitation |

**Implication**: Path-based and even content-hash-based restrictions are insufficient
against reasoning agents. Defense-in-depth across exec, load, and network levels is required.

### Docker Hardening Status in Vascadia

Our `docker-compose.flows.yml` already enforces (per `test_compose_hardening.py`):
- `cap_drop: [ALL]` on all flow containers (CIS 5.3)
- `security_opt: [no-new-privileges]` (CIS 5.25)
- Resource limits (mem_limit, cpus)
- Non-root user in Dockerfile.base

**Gaps**: No seccomp custom profiles, no AppArmor, no image signing (cosign), no
runtime monitoring (Falco/Tetragon). These are appropriate for production, overkill
for academic use.

---

## 4. Security Scanner Decision Matrix

### Why Switch from Trivy?

Trivy's supply chain was compromised on March 19, 2026. While the specific attack
targeted GitHub Actions (which we don't use), the incident reveals:
1. **Organizational security posture concerns** at Aqua Security
2. **Unpinned install scripts are vulnerable** (our `Makefile` fetches latest)
3. **Trust erosion** — a security scanner that gets compromised is ironic and damaging

### Scanner Comparison

| Scanner | Type | Cost | Local | Integration | Supply Chain Risk | Verdict |
|---------|------|------|-------|-------------|-------------------|---------|
| **Grype** (Anchore) | Container + FS vuln | Free/OSS | Yes | Minutes | Low (Go binary, no Actions) | **RECOMMENDED** |
| **Trivy** (Aqua) | Multi-target | Free/OSS | Yes | Minutes | **HIGH (compromised Mar 2026)** | Replace |
| **Docker Scout** | Image vuln + SBOM | Free tier | Yes | Minutes | Medium (Docker Inc.) | Alternative |
| **Snyk Container** | Image + app deps | Freemium ($25/seat) | Limited | Hours | Low (established vendor) | Too expensive |
| **Clair** | Image layer scanner | Free/OSS | Self-hosted | Hours | Low | More complex than Grype |
| **GitHub Dependabot** | Dependency updates | Free (public) | No | Minutes | Low (GitHub infra) | CI-only (disabled for us) |
| **Cosign** (Sigstore) | Image signing | Free/OSS | Yes | Hours | Low | Complementary, not scanner |

### Python-Specific Supply Chain Tools

| Tool | What | Cost | Local | Already Active? |
|------|------|------|-------|-----------------|
| **Ruff (Bandit rules)** | Code security linting | Free | Yes | **YES** |
| **uv lockfile + hashes** | Dependency integrity | Free | Yes | **YES** |
| **pip-audit** (Google) | CVE scanning vs OSV DB | Free | Yes | **NO — add this** |
| **CycloneDX Python** | SBOM generation | Free | Yes | NO |
| **Safety CLI** | CVE scanning vs Safety DB | Freemium | Yes | NO (pip-audit is better free tier) |

---

## 5. Model Weight Security

### Threat: Malicious Model Weights on HuggingFace

~100 instances of malicious models with real payloads identified on HuggingFace.
Attack vectors: typosquatting ("fecebook" vs "facebook"), pickle deserialization
exploits, fine-tuned models with injected misinformation (PoisonGPT).

### Current State in Vascadia

| Aspect | Status |
|--------|--------|
| SAM3 weights format | Pickle (`.pt`) — via `torch.load(weights_only=False)` |
| Download source | HuggingFace Hub (gated model, requires HF_TOKEN) |
| Hash verification | **NONE** — weights downloaded and trusted as-is |
| safetensors support | Not yet (tracked in PRD `model-export-format.decision.yaml`) |

### How to Show End-Users "No Funny Business"

| Approach | Effort | Trust Level |
|----------|--------|-------------|
| **SHA-256 hash pinning** | Hours | High — published hash matches downloaded file |
| **safetensors format** | Days | Highest — no code execution during deserialization |
| **Cosign signature on Docker image** | Hours | High — proves image hasn't been tampered with |
| **SBOM with hashes** | Minutes | Medium — transparency about all dependencies |
| **Reproducible builds** | Days | Highest — anyone can rebuild and verify identical image |

### References

- [Rapid7: From .pth to p0wned — Pickle Files in AI Supply Chains](https://www.rapid7.com/blog/post/from-pth-to-p0wned-abuse-of-pickle-files-in-ai-model-supply-chains/)
- [Red Hat: Model Authenticity with Sigstore](https://next.redhat.com/2025/04/10/model-authenticity-and-transparency-with-sigstore/)
- [Dark Reading: 100 Malicious Models on Hugging Face](https://www.darkreading.com/application-security/hugging-face-ai-platform-100-malicious-code-execution-models)
- [US DoD: AI/ML Supply Chain Risks and Mitigations (March 2026)](https://media.defense.gov/2026/Mar/04/2003882809/-1/-1/0/AI_ML_SUPPLY_CHAIN_RISKS_AND_MITIGATIONS.PDF)
- [HuggingFace: Safetensors Security Audit](https://huggingface.co/blog/safetensors-security-audit)

---

## 6. Broader Context: The AI Clownpocalypse

The Reddit thread and LinkedIn discussion highlight a systemic pattern:

1. **Supply chain depth**: LiteLLM sits between your product and 100+ LLM providers.
   Trivy sits between your CI and your security posture. Compromising either gives
   access to everything downstream. The "move fast and break things" ethos creates
   attack surfaces faster than defenders can audit them.

2. **Agent permission models are fundamentally broken**: Hidden HTML comments that agents
   can see but users can't. MCP servers with no security model ("The S in MCP stands
   for security"). `ANTHROPIC_BASE_URL` redirecting API keys to attacker servers.
   The field is building reactors without control rods.

3. **Prompt injection is unsolvable with current LLM architecture**: LLMs cannot
   distinguish between command layer and content layer — it's all tokens. Every
   "safety" boundary is probabilistic, not deterministic. This makes agent systems
   fundamentally different from traditional software where authorization is boolean.

4. **The skill drain risk**: As developers rely more on LLM-generated code without
   understanding it, the ability to audit and catch supply chain attacks atrophies.
   The 50K LOC change from a "vibe coder" is the perfect vehicle for injecting payloads
   that no one will audit.

### What This Means for Vascadia

We are an academic ML platform with:
- Docker-per-flow isolation (strong)
- Pre-commit hooks and test gates (strong)
- uv lockfile with hashes (strong)
- BUT: no container scanning in CI (weak), no image signing (weak), no model weight
  verification (weak), no `.pth` file auditing (weak)

The LiteLLM incident shows that even "infrastructure plumbing" packages (3.4M daily
downloads) can be compromised. Our defense must assume any dependency could be trojaned.

---

## 7. Recommended Security Stack

| Layer | Current | Recommended | Effort | Priority |
|-------|---------|-------------|--------|----------|
| **Code linting** | Ruff (Bandit rules) | Keep | Active | N/A |
| **Dependency locking** | uv.lock + hashes | Keep | Active | N/A |
| **Dependency CVE scan** | None | **pip-audit** in pre-commit | Minutes | **P1** |
| **Container vuln scan** | Trivy (unpinned, not installed) | **Grype** (replace Trivy) | Hours | **P1** |
| **SBOM generation** | None | **Syft** + CycloneDX | Minutes | P2 |
| **Image signing** | None | **Cosign** for GAR images | Hours | P2 |
| **Model weight verification** | None | SHA-256 hash pinning for HF downloads | Hours | P2 |
| **`.pth` file auditing** | None | Pre-commit check for unexpected `.pth` files | Minutes | P2 |
| **`torch.load` safety** | 6 files use `weights_only=False` | Audit and convert | Hours | P2 |
| **Runtime monitoring** | None | Falco/Tetragon (optional) | Days | P3 |
| **GRC platform** | None | CISO Assistant (not recommended for academic repo) | Days | Not recommended |

---

## 8. Executable Actions

### P0 — Immediate (today)

- [ ] **Pin litellm**: Change `pyproject.toml` line 118 from `"litellm>=1.70,<2.0"` to
  `"litellm>=1.70,<=1.82.6"`. Run `uv lock`. Verify no `litellm_init.pth` in
  `.venv/lib/python3.13/site-packages/`.

### P1 — This Sprint

- [ ] **Replace Trivy with Grype**: Update `Makefile` `install-trivy` → `install-grype`.
  Update `scripts/weekly_security_scan.sh` and `scripts/pr_readiness_check.sh`.
  Install: `curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin v0.90.0` (pinned version).
- [ ] **Add pip-audit to pre-commit**: Add `uv run pip-audit --strict` as a pre-commit hook.
  Catches known CVEs in all installed dependencies.
- [ ] **Pin Trivy if keeping**: If not switching to Grype, pin Trivy install to
  `v0.69.3` in Makefile: `curl ... | sh -s -- -b /usr/local/bin v0.69.3`.

### P2 — Roadmap

- [ ] **Add SBOM generation**: `make sbom` target using Syft: `syft docker:minivess-base:latest -o cyclonedx-json > sbom.json`
- [ ] **Add image signing**: `cosign sign` after `docker push` to GAR.
- [ ] **SHA-256 hash pinning for SAM3 weights**: After first verified download, record
  hash in `configs/model_profiles/sam3_*.yaml`. Verify on subsequent downloads.
- [ ] **Audit `torch.load(weights_only=False)`**: Convert to `weights_only=True` where
  possible. Add `torch.serialization.add_safe_globals()` for custom types.
- [ ] **Pre-commit `.pth` file auditor**: Check `.venv/lib/*/site-packages/*.pth` against
  a known-good allowlist. Flag any new `.pth` files as potential supply chain attack.
- [ ] **safetensors migration**: Track in PRD as long-term goal for model export format.

### P3 — If Pursuing Compliance

- [ ] **Runtime monitoring**: Falco for syscall monitoring in Docker containers
  (overkill for academic use, relevant for Nature Protocols paper claims)
- [ ] **GRC platform**: CISO Assistant only if pursuing ISO 27001 or NIST AI RMF compliance

---

## 9. Decision: Trivy vs Grype vs Docker Scout

### Hypothesis Matrix

| Hypothesis | Evidence For | Evidence Against | Confidence |
|-----------|-------------|-----------------|------------|
| **H1: Switch to Grype** | Free, local, CLI-first, not compromised, Syft companion for SBOM, Go binary (no supply chain via npm/pip) | Narrower scope than Trivy (no IaC scanning), smaller community | **HIGH (0.85)** |
| **H2: Keep Trivy (pinned)** | Broadest scope (images, IaC, K8s, SBOM), largest community, well-documented | **Supply chain compromised Mar 2026**, organizational trust damaged | LOW (0.20) |
| **H3: Docker Scout** | Docker-native, zero learning curve, free tier | Vendor lock-in, limited free tier, cloud-dependent for some features | MEDIUM (0.50) |
| **H4: Multi-scanner** (Grype + Scout) | Defense-in-depth, different vuln databases | Maintenance overhead, potential conflicting results | MEDIUM (0.45) |

### Recommendation

**H1: Switch to Grype** as the primary container scanner. It is the lowest-risk option
with the highest confidence. Grype is a Go binary distributed independently (no npm/pip
supply chain vector), uses the same vulnerability databases as Trivy, and produces
compatible output formats. The companion tool Syft generates SBOMs that Grype can consume.

**Rationale**: Using a security scanner that was itself compromised 6 days ago is
a hard sell for any security-conscious project, regardless of whether the specific
attack vector (GitHub Actions) applies to us.
