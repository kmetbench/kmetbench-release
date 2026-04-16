#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
    cat <<'EOF'
Usage: bash scripts/setup/serve_vllm_for_kmetbench.sh --model-config <config> [options]

Required:
  --model-config <config>          Model config identifier under configs/models/

Optional:
  --prompt-type <advanced|reasoning>
  --port <port>
  --tensor-parallel-size <n>
  --max-model-len <n>
  --gpu-memory-utilization <float>
  --max-num-seqs <n>
  --image-limit <n>
  --visible-devices <csv>
  --dry-run
  -h, --help
EOF
}

MODEL_CONFIG=""
PROMPT_TYPE="advanced"
PORT_OVERRIDE=""
TP_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
GPU_MEMORY_UTILIZATION_OVERRIDE=""
MAX_NUM_SEQS_OVERRIDE=""
IMAGE_LIMIT_OVERRIDE=""
VISIBLE_DEVICES_OVERRIDE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --prompt-type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_OVERRIDE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN_OVERRIDE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION_OVERRIDE="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS_OVERRIDE="$2"
            shift 2
            ;;
        --image-limit)
            IMAGE_LIMIT_OVERRIDE="$2"
            shift 2
            ;;
        --visible-devices)
            VISIBLE_DEVICES_OVERRIDE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

if [[ -z "$MODEL_CONFIG" ]]; then
    echo "[ERROR] Missing --model-config" >&2
    usage >&2
    exit 1
fi

resolved_output=$(
    cd "$REPO_ROOT" &&
    python - "$MODEL_CONFIG" "$PROMPT_TYPE" <<'PY'
import sys
from urllib.parse import urlparse

from src.eval.config import get_default_max_model_len
from src.eval.model_configs import load_model_config

config = load_model_config(sys.argv[1])
prompt_type = sys.argv[2]
if config.model.engine.lower() != "vllm":
    raise SystemExit(f"Model config must use engine=vllm: {config.relative_path}")

vllm = config.get_section("vllm")
parsed = urlparse(config.model.base_url or "")

host = str(vllm.get("host") or "0.0.0.0")
port = vllm.get("port") or parsed.port or 8237
tensor_parallel_size = vllm.get("tensor_parallel_size", 1)
max_model_len = vllm.get("max_model_len") or get_default_max_model_len(prompt_type)
gpu_memory_utilization = vllm.get("gpu_memory_utilization", 0.9)
max_num_seqs = vllm.get("max_num_seqs", 4)
image_limit = vllm.get("image_limit", 4)
visible_devices = str(vllm.get("visible_devices", "") or "")
is_vlm = "1" if config.model.is_vlm else "0"
trust_remote_code = "1" if bool(vllm.get("trust_remote_code", True)) else "0"
base_url = config.model.base_url or ""

for value in (
    config.model.name,
    host,
    str(port),
    str(tensor_parallel_size),
    str(max_model_len),
    str(gpu_memory_utilization),
    str(max_num_seqs),
    str(image_limit),
    visible_devices,
    is_vlm,
    trust_remote_code,
    base_url,
):
    print(value)
PY
)

readarray -t resolved <<<"$resolved_output"

MODEL="${resolved[0]}"
HOST="${resolved[1]}"
PORT="${PORT_OVERRIDE:-${resolved[2]}}"
TP_SIZE="${TP_OVERRIDE:-${resolved[3]}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-${resolved[4]}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-${resolved[5]}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-${resolved[6]}}"
IMAGE_LIMIT="${IMAGE_LIMIT_OVERRIDE:-${resolved[7]}}"
VISIBLE_DEVICES="${VISIBLE_DEVICES_OVERRIDE:-${resolved[8]}}"
IS_VLM="${resolved[9]}"
TRUST_REMOTE_CODE="${resolved[10]}"
BASE_URL="${resolved[11]}"

echo "[INFO] model_config=$MODEL_CONFIG"
echo "[INFO] model=$MODEL"
echo "[INFO] base_url=$BASE_URL"
echo "[INFO] prompt_type=$PROMPT_TYPE"
echo "[INFO] host=$HOST"
echo "[INFO] port=$PORT"
echo "[INFO] tensor_parallel_size=$TP_SIZE"
echo "[INFO] max_model_len=$MAX_MODEL_LEN"
echo "[INFO] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "[INFO] max_num_seqs=$MAX_NUM_SEQS"
if [[ "$IS_VLM" == "1" ]]; then
    echo "[INFO] image_limit=$IMAGE_LIMIT"
fi
if [[ -n "$VISIBLE_DEVICES" ]]; then
    echo "[INFO] visible_devices=$VISIBLE_DEVICES"
fi

cmd=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ "$IS_VLM" == "1" && "$IMAGE_LIMIT" != "0" ]]; then
    cmd+=(--limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT}")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -n "$VISIBLE_DEVICES" ]]; then
        printf '[DRY_RUN] CUDA_VISIBLE_DEVICES=%q ' "$VISIBLE_DEVICES"
    else
        printf '[DRY_RUN] '
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

if [[ -n "$VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
fi

exec "${cmd[@]}"
