#!/usr/bin/env bash
# Install git hooks for the minivess-mlops repository.
#
# Usage:
#   bash scripts/install-hooks.sh
#
# Installs a pre-push hook that runs scripts/pr_readiness_check.sh before
# every `git push`. This prevents pushing code that fails lint, typecheck,
# or unit tests.
#
# The hook is NOT a pre-commit hook (too slow) — it only runs on git push.
# For pre-commit hooks see .pre-commit-config.yaml.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$(git rev-parse --git-dir)/hooks"

echo "Installing git hooks into ${HOOK_DIR}..."

cat > "${HOOK_DIR}/pre-push" <<'HOOK'
#!/usr/bin/env bash
# Pre-push hook: run PR readiness check before pushing.
# Installed by scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo ""
echo "======================================================"
echo "  Pre-push gate: running PR readiness check..."
echo "======================================================"
echo ""

bash "${REPO_ROOT}/scripts/pr_readiness_check.sh"
HOOK

chmod +x "${HOOK_DIR}/pre-push"

echo ""
echo "Pre-push hook installed at ${HOOK_DIR}/pre-push"
echo "  -> Runs scripts/pr_readiness_check.sh on every 'git push'"
echo ""
echo "To bypass (emergencies only): git push --no-verify"
