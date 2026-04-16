# Model Configs

This directory defines canonical model profiles for the unified
`scripts/eval.py` public entrypoint.

Groups:

- `api/`: hosted API models accessed through an OpenAI-compatible endpoint
- `vllm/`: models served behind a local or remote vLLM OpenAI-compatible server
- `hf/`: local `transformers` models supported by the public fallback runner

Usage:

```bash
python scripts/eval.py run --list-model-configs

python scripts/eval.py run \
  --model-config vllm/Qwen_Qwen3-VL-8B-Thinking \
  --prompt-type advanced \
  --data-type explicit
```

Reasoning judge stays inside the same top-level script:

```bash
python scripts/eval.py judge --model Qwen/Qwen3-VL-8B-Thinking
```
