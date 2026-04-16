#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup/install_uv_profile.sh [base] [transformers] [dev]
EOF
}

if [[ $# -eq 0 ]]; then
    echo "[INFO] Installing base uv environment"
    exec uv sync
fi

args=()
for profile in "$@"; do
    case "$profile" in
        -h|--help)
            usage
            exit 0
            ;;
        base)
            ;;
        transformers|dev)
            args+=(--extra "$profile")
            ;;
        *)
            echo "[ERROR] Unknown profile: $profile" >&2
            echo "Supported profiles: base transformers dev" >&2
            exit 1
            ;;
    esac
done

echo "[INFO] Installing uv environment with profiles: $*"
exec uv sync "${args[@]}"
