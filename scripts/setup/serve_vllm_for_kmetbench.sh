#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "USAGE: $(basename "$0") <model-or-config> [prompt_type] [port] [tensor_parallel_size]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_INPUT="$1"
PROMPT_TYPE="${2:-advanced}"
CONFIG_DEFAULTS=$(
    cd "$REPO_ROOT" &&
    python - "$MODEL_INPUT" <<'PY'
import sys

from src.eval.model_configs import load_model_config

identifier = sys.argv[1]
try:
    config = load_model_config(identifier)
except Exception:
    raise SystemExit(0)

if config.model.engine.lower() != "vllm":
    raise SystemExit(f"Model config must use engine=vllm: {identifier}")

vllm = config.get_section("vllm")
print(config.model.name)
print(vllm.get("port", ""))
print(vllm.get("tensor_parallel_size", ""))
print(vllm.get("gpu_memory_utilization", ""))
PY
)

MODEL="$MODEL_INPUT"
CONFIG_PORT=""
CONFIG_TP_SIZE=""
CONFIG_GPU_MEMORY_UTILIZATION=""
if [[ -n "$CONFIG_DEFAULTS" ]]; then
    mapfile -t _config_lines <<< "$CONFIG_DEFAULTS"
    MODEL="${_config_lines[0]:-$MODEL_INPUT}"
    CONFIG_PORT="${_config_lines[1]:-}"
    CONFIG_TP_SIZE="${_config_lines[2]:-}"
    CONFIG_GPU_MEMORY_UTILIZATION="${_config_lines[3]:-}"
fi

PORT="${3:-${CONFIG_PORT:-8237}}"
TP_SIZE="${4:-${CONFIG_TP_SIZE:-1}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${CONFIG_GPU_MEMORY_UTILIZATION:-0.90}}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
IMAGE_LIMIT="${IMAGE_LIMIT:-4}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"

bool_is_true() {
    local value="${1:-false}"
    value="${value,,}"
    case "$value" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

MAX_MODEL_LEN="${MAX_MODEL_LEN:-${KMETBENCH_VLLM_MAX_MODEL_LEN:-}}"
if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$(
        cd "$REPO_ROOT" &&
        python - "$PROMPT_TYPE" <<'PY'
import sys
from src.eval.config import get_default_max_model_len

print(get_default_max_model_len(sys.argv[1]))
PY
    )
fi

echo "[INFO] model_input=$MODEL_INPUT"
echo "[INFO] model=$MODEL"
echo "[INFO] prompt_type=$PROMPT_TYPE"
echo "[INFO] max_model_len=$MAX_MODEL_LEN"
echo "[INFO] port=$PORT"
echo "[INFO] tensor_parallel_size=$TP_SIZE"
echo "[INFO] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
if [[ -n "$DOWNLOAD_DIR" ]]; then
    echo "[INFO] download_dir=$DOWNLOAD_DIR"
fi

if [[ "$MODEL" == "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" ]]; then
    export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
    export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
    export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
    export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
    echo "[INFO] hyperclovax_deep_gemm=$VLLM_USE_DEEP_GEMM"
    echo "[INFO] hyperclovax_deep_gemm_e8m0=$VLLM_USE_DEEP_GEMM_E8M0"
fi

if bool_is_true "$VLLM_ENFORCE_EAGER"; then
    VLLM_ENFORCE_EAGER="true"
else
    VLLM_ENFORCE_EAGER="false"
fi
echo "[INFO] enforce_eager=$VLLM_ENFORCE_EAGER"

VLLM_ARGS=(
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT}"
    --trust-remote-code
)

if [[ -n "$DOWNLOAD_DIR" ]]; then
    VLLM_ARGS+=(--download-dir "$DOWNLOAD_DIR")
fi

if bool_is_true "$VLLM_ENFORCE_EAGER"; then
    VLLM_ARGS+=(--enforce-eager)
fi

exec vllm serve "$MODEL" "${VLLM_ARGS[@]}"
