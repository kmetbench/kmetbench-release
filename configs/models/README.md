# Model Configs

This directory defines canonical model profiles for the unified
`scripts/eval.py` entrypoint.

Groups:

- `api/`: hosted API models accessed through an OpenAI-compatible endpoint
- `vllm/`: models served behind a local or remote vLLM OpenAI-compatible server
- `hf/`: local `transformers` models supported by the current fallback runner

Usage:

```bash
python scripts/eval.py run --list-model-configs

python scripts/eval.py \
  run \
  --model-config vllm/Qwen_Qwen3-VL-8B-Thinking \
  --protocol public \
  --prompt-type advanced \
  --data-type explicit

python scripts/setup/serve_vllm_for_kmetbench.sh \
  vllm/Qwen_Qwen3-VL-8B-Thinking \
  reasoning
```

Config resolution accepts:

- explicit relative labels such as `vllm/Qwen_Qwen3-VL-8B-Thinking`
- a bare unique stem such as `Qwen_Qwen3-VL-8B-Thinking`
- an explicit YAML path

The current unified entrypoint supports:

- `engine: api`
- `engine: vllm`
- `engine: transformers`

Model YAML may also define `prompt_overrides` keyed by runtime profile or exact
prompt type. Typical use is to keep `advanced` and `reasoning` decoding settings
separate, for example `advanced.temperature = 0.1` and
`reasoning.temperature = 1.0`.

For `vllm/` entries, the `vllm.port` and `vllm.tensor_parallel_size` fields are
treated as the default serve profile. `scripts/setup/serve_vllm_for_kmetbench.sh`
will read those values when given a model-config identifier.

Reasoning judge remains a separate workflow for now.
