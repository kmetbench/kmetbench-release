#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "USAGE: $(basename "$0") <model> [prompt_type] [port] [tensor_parallel_size]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL="$1"
PROMPT_TYPE="${2:-advanced}"
PORT="${3:-8237}"
TP_SIZE="${4:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
IMAGE_LIMIT="${IMAGE_LIMIT:-4}"

MAX_MODEL_LEN=$(
    cd "$REPO_ROOT" &&
    python - "$PROMPT_TYPE" <<'PY'
import sys
from src.eval.config import get_default_max_model_len

print(get_default_max_model_len(sys.argv[1]))
PY
)

echo "[INFO] model=$MODEL"
echo "[INFO] prompt_type=$PROMPT_TYPE"
echo "[INFO] max_model_len=$MAX_MODEL_LEN"
echo "[INFO] port=$PORT"
echo "[INFO] tensor_parallel_size=$TP_SIZE"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT}" \
    --trust-remote-code
