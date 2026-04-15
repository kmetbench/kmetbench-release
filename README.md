# K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-4285F4.svg?style=for-the-badge)](https://openreview.net/forum?id=1Gn5pKek8k)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-EA4335.svg?style=for-the-badge)](https://huggingface.co/datasets/soyeonbot/K-MetBench)
[![Code](https://img.shields.io/badge/Code-GitHub-FBBC05.svg?style=for-the-badge)](https://github.com/kmetbench/kmetbench-release)
[![Citation](https://img.shields.io/badge/Citation-BibTeX-34A853.svg?style=for-the-badge)](#citation)

This repository contains the public evaluation kit for K-MetBench, a benchmark for Korean meteorology exam questions spanning expert reasoning, geo-cultural alignment, and multimodal weather understanding.

K-MetBench evaluates 1,774 questions drawn from the Korean National Meteorological Engineer Examination. The benchmark includes 82 multimodal questions, 141 reasoning questions with expert-verified rationales, and 73 Korean-specific questions across five official subject areas: Weather Analysis and Forecast Theory (P1), Meteorological Observation Methods (P2), Atmospheric Dynamics (P3), Climatology (P4), and Atmospheric Physics (P5).

The public protocol in this repository covers two prediction settings, `explicit_advanced` and `explicit_reasoning`, followed by the public judge output `explicit_reasoning_evaluation`.

## Updates

- **[2026/04/16]** Public release repository trimmed to the evaluation kit only. Internal planning documents, export pipelines, and release-only summaries were removed from the public tree.

> Use GitHub Releases for versioned snapshots of the public eval kit.
>
> See [VERSIONS.md](./VERSIONS.md) for the current artifact tracker.

## Leaderboard

Visit the [K-MetBench leaderboard](https://kmetbench.github.io/) for the public model table, scaling plots, and citation block used in the website release.

## Getting Started

### Installation

Create the default `uv` environment and run the environment check:

```bash
uv sync
uv run python scripts/setup/env_doctor.py
```

<details>
<summary>Optional install profiles</summary>

If you want local `transformers` inference:

```bash
bash scripts/setup/install_uv_profile.sh transformers
```

If you are working in a larger private development checkout and also need compatibility with legacy private-only paths:

```bash
bash scripts/setup/install_uv_profile.sh private-compat
```

Equivalent `uv` extras:

```bash
uv sync --extra transformers
uv sync --extra private-compat
uv sync --extra dev
```
</details>

If `uv` is unavailable, use:

```bash
pip install -r requirements-eval.txt
```

### Evaluation

Fastest path: run against an existing OpenAI-compatible endpoint.

```bash
# explicit_advanced
uv run python scripts/eval/eval_openai_compatible.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://127.0.0.1:8237/v1 \
  --prompt-type advanced \
  --data-type explicit

# explicit_reasoning
uv run python scripts/eval/eval_openai_compatible.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://127.0.0.1:8237/v1 \
  --prompt-type reasoning \
  --data-type explicit
```

Local `transformers` inference:

```bash
uv run python scripts/eval/eval_transformers.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --prompt-type advanced \
  --data-type explicit
```

Local `vllm` serving helper:

```bash
# run this from a separate environment that already has vllm installed
bash scripts/setup/serve_vllm_for_kmetbench.sh \
  Qwen/Qwen3-VL-8B-Thinking \
  advanced \
  8237 \
  4

# then evaluate from the uv environment
uv run python scripts/eval/eval_openai_compatible.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://127.0.0.1:8237/v1 \
  --prompt-type advanced \
  --data-type explicit
```

Reasoning judge:

```bash
export GEMINI_API_KEY=...
uv run python scripts/eval/eval_reasoning_judge.py \
  --model Qwen/Qwen3-VL-8B-Thinking
```

<details>
<summary>All CLI arguments</summary>

**`scripts/eval/eval_openai_compatible.py`**

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-VL-8B-Thinking` | Model name passed to the OpenAI-compatible API. |
| `--image-root` | repo default | Override the image root. |
| `--explicit-data-file` | repo default | Override the explicit benchmark JSON path. |
| `--output-root` | `experiments/results/evaluation` | Override the output directory for evaluation JSON files. |
| `--api-key` | `""` | API key for the endpoint. |
| `--base-url` | `http://0.0.0.0:8192/v1` | OpenAI-compatible endpoint base URL. |
| `--num-samples` | `-1` | Limit the number of evaluated samples. |
| `--temperature` | `0.0` | Sampling temperature. |
| `--top-p` | `0.95` | Top-p sampling parameter. |
| `--prompt-type` | `advanced` | Prompt type: `advanced` or `reasoning`. |
| `--data-type` | `explicit` | Public data type. |
| `--seed` | `42` | Random seed. |
| `--quiet` | `False` | Suppress per-item output. |
| `--max-tokens` | prompt-dependent default | Maximum tokens to generate. |
| `--concurrency` | `1` | Number of concurrent API requests. |

**`scripts/eval/eval_transformers.py`**

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-VL-8B-Thinking` | Hugging Face model name for local `transformers` inference. |
| `--image-root` | repo default | Override the image root. |
| `--explicit-data-file` | repo default | Override the explicit benchmark JSON path. |
| `--output-root` | `experiments/results/evaluation` | Override the output directory for evaluation JSON files. |
| `--api-key` | `""` | Accepted by the shared parser; not used for local `transformers` runs. |
| `--base-url` | `http://0.0.0.0:8192/v1` | Accepted by the shared parser; not used for local `transformers` runs. |
| `--num-samples` | `-1` | Limit the number of evaluated samples. |
| `--temperature` | `0.0` | Sampling temperature. |
| `--top-p` | `0.95` | Top-p sampling parameter. |
| `--prompt-type` | `advanced` | Prompt type: `advanced` or `reasoning`. |
| `--data-type` | `explicit` | Public data type. |
| `--seed` | `42` | Random seed. |
| `--quiet` | `False` | Suppress per-item output. |
| `--max-tokens` | prompt-dependent default | Maximum tokens to generate. |
| `--device` | `cuda` | Target device for local inference. |

**`scripts/eval/eval_reasoning_judge.py`**

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | required | Target model whose reasoning predictions will be judged. |
| `--predictions` | latest matching run | Explicit reasoning prediction JSON path. |
| `--evaluator` | `gemini-2.5-pro` | Judge model name. |
| `--base-url` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible Gemini endpoint. |
| `--api-key` | `GEMINI_API_KEY` fallback | Judge API key. |
| `--explicit-data-file` | repo default | Override the explicit benchmark JSON path. |
| `--output-root` | `experiments/results/evaluation` | Override the output directory for judge results. |
| `--data-type` | `explicit` | Public data type. |
| `--prompt-type` | `reasoning_evaluation` | Judge prompt type. |
| `--num-samples` | `-1` | Limit the number of judged samples. |
| `--seed` | `42` | Random seed. |
| `--temperature` | `0.0` | Sampling temperature for the judge. |
| `--top-p` | `0.95` | Top-p sampling parameter for the judge. |
| `--timeout` | `300` | Request timeout in seconds. |
| `--max-parse-retries` | `3` | Maximum retries when parsing judge output fails. |
| `--wo-rationale` | `False` | Evaluate without the expert rationale block. |
| `--quiet` | `False` | Suppress per-item output. |
</details>

## Citation

```bibtex
@inproceedings{kim2026kmetbench,
title={K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology},
author={Kim, Soyeon and Kang, Cheongwoong and Lee, Myeongjin and Chang, Eun-Chul and Lee, Jaedeok and Choi, Jaesik},
booktitle={The 64th Annual Meeting of the Association for Computational Linguistics},
year={2026},
url={https://openreview.net/forum?id=1Gn5pKek8k}
}
```

## Contact

For questions or concerns regarding the dataset or code, please contact Soyeon Kim (soyeon.k@kaist.ac.kr).

## License

The code in this repository is released under the [MIT License](./LICENSE). The K-MetBench dataset is distributed separately on Hugging Face under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
