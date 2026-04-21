#!/usr/bin/env bash
set -euo pipefail

PRIVATE_REPO_DEFAULT="/home/soyeon/kmetbench/private/MeteorQA"
PUBLIC_REPO_DEFAULT="/home/soyeon/kmetbench/public/kmetbench-release"

private_repo="$PRIVATE_REPO_DEFAULT"
public_repo="$PUBLIC_REPO_DEFAULT"

usage() {
    cat <<'EOF'
Install local pre-push guards for the private MeteorQA repo and the public
kmetbench-release repo.

Usage:
  bash scripts/setup/install_repo_identity_guards.sh
  bash scripts/setup/install_repo_identity_guards.sh \
    --private-repo /path/to/MeteorQA \
    --public-repo /path/to/kmetbench-release
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --private-repo)
            private_repo="$2"
            shift 2
            ;;
        --public-repo)
            public_repo="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

install_guard() {
    local repo_path="$1"
    local repo_label="$2"
    local repo_role="$3"
    local remote_primary="$4"
    local remote_alt="$5"

    if [[ ! -d "$repo_path/.git" ]]; then
        echo "[WARN] Skipping $repo_path because it is not a git repo." >&2
        return 0
    fi

    git -C "$repo_path" config --local kmetbench.label "$repo_label"
    git -C "$repo_path" config --local kmetbench.role "$repo_role"
    git -C "$repo_path" config --local kmetbench.expectedRemote "$remote_primary"
    git -C "$repo_path" config --local kmetbench.expectedRemoteAlt "$remote_alt"

    local hook_path
    hook_path=$(git -C "$repo_path" rev-parse --git-path hooks/pre-push)
    if [[ "$hook_path" != /* ]]; then
        hook_path="$repo_path/$hook_path"
    fi
    mkdir -p "$(dirname "$hook_path")"

    cat > "$hook_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
repo_label=$(git config --get kmetbench.label || true)
repo_role=$(git config --get kmetbench.role || true)
expected=$(git config --get kmetbench.expectedRemote || true)
expected_alt=$(git config --get kmetbench.expectedRemoteAlt || true)
actual=$(git remote get-url --push origin 2>/dev/null || true)

if [[ -z "$expected" || -z "$actual" ]]; then
    exit 0
fi

if [[ "$actual" == "$expected" ]]; then
    exit 0
fi

if [[ -n "$expected_alt" && "$actual" == "$expected_alt" ]]; then
    exit 0
fi

echo "[BLOCKED] Refusing to push from the wrong repo identity." >&2
echo "  repo root : $repo_root" >&2
echo "  repo label: ${repo_label:-unknown}" >&2
echo "  repo role : ${repo_role:-unknown}" >&2
echo "  actual    : $actual" >&2
echo "  expected  : $expected" >&2
if [[ -n "$expected_alt" ]]; then
    echo "  alt       : $expected_alt" >&2
fi
exit 1
EOF

    chmod +x "$hook_path"
    echo "[OK] Installed pre-push guard for $repo_label"
    echo "     repo   : $repo_path"
    echo "     remote : $(git -C "$repo_path" remote get-url --push origin 2>/dev/null || echo '<missing origin>')"
}

install_guard \
    "$private_repo" \
    "MeteorQA" \
    "private" \
    "https://github.com/kmetbench/MeteorQA.git" \
    "git@github.com:kmetbench/MeteorQA.git"

install_guard \
    "$public_repo" \
    "kmetbench-release" \
    "public" \
    "https://github.com/kmetbench/kmetbench-release.git" \
    "git@github.com:kmetbench/kmetbench-release.git"
