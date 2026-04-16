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

Equivalent `uv` extras:

```bash
uv sync --extra transformers
uv sync --extra dev
```
</details>

If `uv` is unavailable, use:

```bash
pip install -r requirements-eval.txt
```

### Evaluation

The public eval kit now uses a single entrypoint: `scripts/eval.py`.

List the available model configs:

```bash
uv run python scripts/eval.py run --list-model-configs
```

Fastest path: run against an existing OpenAI-compatible endpoint backed by the
Qwen3-VL-8B-Thinking config.

```bash
# explicit_advanced
uv run python scripts/eval.py run \
  --model-config vllm/Qwen_Qwen3-VL-8B-Thinking \
  --prompt-type advanced \
  --data-type explicit

# explicit_reasoning
uv run python scripts/eval.py run \
  --model-config vllm/Qwen_Qwen3-VL-8B-Thinking \
  --prompt-type reasoning \
  --data-type explicit
```

Local `transformers` fallback:

```bash
uv run python scripts/eval.py run \
  --model-config hf/meta-llama_Llama-3.2-90B-Vision-Instruct \
  --prompt-type advanced \
  --data-type explicit
```

Inspect the resolved dispatch payload before running:

```bash
uv run python scripts/eval.py run \
  --model-config vllm/Qwen_Qwen3-VL-8B-Thinking \
  --prompt-type reasoning \
  --data-type explicit \
  --dry-run
```

Reasoning judge stays in the same top-level script:

```bash
export GEMINI_API_KEY=...
uv run python scripts/eval.py judge \
  --model Qwen/Qwen3-VL-8B-Thinking
```

<details>
<summary>All CLI arguments</summary>

**`scripts/eval.py run`**

| Argument | Default | Description |
| --- | --- | --- |
| `--model-config` | required unless `--list-model-configs` | Config path or identifier under `configs/models/`. |
| `--list-model-configs` | `False` | Print the available model configs and exit. |
| `--prompt-type` | `advanced` | Prompt type: `advanced` or `reasoning`. |
| `--data-type` | `explicit` | Public data type. |
| `--image-root` | repo default | Override the image root. |
| `--explicit-data-file` | repo default | Override the explicit benchmark JSON path. |
| `--output-root` | `experiments/results/evaluation` | Override the output directory for evaluation JSON files. |
| `--api-key` | config env fallback | Override the API key instead of using the model config env rule. |
| `--base-url` | config default | Override the endpoint base URL. |
| `--num-samples` | `-1` | Limit the number of evaluated samples. |
| `--temperature` | config default or runtime default | Override temperature. |
| `--top-p` | config default or runtime default | Override top-p. |
| `--seed` | `42` | Random seed. |
| `--quiet` | `False` | Suppress per-item output. |
| `--max-tokens` | config default or prompt runtime default | Maximum tokens to generate. |
| `--concurrency` | backend default | Override concurrency for OpenAI-compatible runs. |
| `--device` | config default or `cuda` | Override device for local `transformers` runs. |
| `--dry-run` | `False` | Print the resolved dispatch payload without running evaluation. |

**`scripts/eval.py judge`**

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
