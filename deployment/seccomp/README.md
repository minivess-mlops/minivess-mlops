# Seccomp Profiles — MinIVess MLOps

Linux seccomp (Secure Computing Mode) filters restrict which syscalls a container
can make. This directory holds seccomp profiles for MinIVess flow containers.

## Files

| File | Purpose |
|------|---------|
| `audit.json` | Audit profile — logs all syscalls without blocking (SCMP_ACT_LOG) |

## Workflow: Building a Per-Flow Allowlist

### Step 1 — Run with audit profile (syscall discovery)

```bash
make seccomp-audit-train
```

Or manually:
```bash
docker compose --env-file .env -f deployment/docker-compose.flows.yml \
  run --rm --security-opt seccomp=deployment/seccomp/audit.json \
  --shm-size 8g train
```

**Requires**: auditd running on host (`sudo systemctl start auditd`).

### Step 2 — Extract syscalls from audit log

```bash
sudo ausearch -m SECCOMP | grep syscall= | awk -F'syscall=' '{print $2}' | \
  cut -d' ' -f1 | sort -u
```

### Step 3 — Build per-flow allowlist

Create `deployment/seccomp/train.json` with `"defaultAction": "SCMP_ACT_ERRNO"` and
`"syscalls"` listing only the discovered syscalls with `"action": "SCMP_ACT_ALLOW"`.

Reference the Docker default profile structure:
https://docs.docker.com/engine/security/seccomp/

### Step 4 — Apply allowlist profile

```bash
docker compose --env-file .env -f deployment/docker-compose.flows.yml \
  run --rm --security-opt seccomp=deployment/seccomp/train.json \
  --shm-size 8g train
```

## Caveats

### MONAI pin_memory and IPC_LOCK

MONAI DataLoader uses `pin_memory=True` for GPU training, which calls `mlock()`.
If the allowlist profile blocks `mlock`, add `cap_add: [IPC_LOCK]` to the service
or allowlist the `mlock` syscall explicitly.

### seccomp is silently ignored with privileged: true

If a service runs with `privileged: true`, seccomp is completely bypassed with no
warning. MinIVess services never use `privileged: true` (CIS Benchmark 5.4).

### SCMP_ACT_LOG vs SCMP_ACT_KILL

- `SCMP_ACT_LOG`: logs the syscall and continues (safe for discovery)
- `SCMP_ACT_KILL`: kills the process immediately (use only in production allowlist)
- `SCMP_ACT_ERRNO`: returns EPERM (preferred over KILL for debuggability)

## Current Status

Only the audit profile exists. Per-flow allowlist profiles are operational work:
run each flow with the audit profile in staging, extract the syscall list, build
the allowlist. This is tracked in GitHub issues for each flow.
