from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from src.utils.repo_layout import EVALUATION_INDEX_ROOT

from .config import (
    get_default_max_tokens,
    get_system_prompt,
    resolve_dataset_path,
    resolve_image_root,
    resolve_results_root,
)
from .data import filter_items_for_run, load_items
from .messages import build_messages
from .models import EvaluationResult, Item
from .parsing import evaluate_accuracy, extract_evaluation_info, get_safe_name, strip_think_tags
from .public_protocol import (
    PUBLIC_DATA_CHOICES,
    PUBLIC_PROMPT_CHOICES,
    build_public_metric_row,
    build_public_protocol_block,
    format_public_metric_summary,
)
from .runners import ChatGPTRunner, OpenAICompatibleRunner, TransformersRunner

PROMPT_CHOICES = [
    "standard",
    "advanced",
    "reasoning",
    "reasoning_evaluation",
    "reasoning_generate",
    "standard_no_choices",
    "standard_no_questions",
    "standard_all_answer",
]

BACKEND_OPENAI_COMPATIBLE = "openai_compatible"
BACKEND_CHATGPT = "chatgpt"
BACKEND_CHATGPT_THINKING = "chatgpt_thinking"
BACKEND_TRANSFORMERS = "transformers"


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable | None = None, **_: Any):
        return iterable


def _load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv()
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_parser(
    backend: str,
    *,
    prompt_choices: Iterable[str] | None = None,
    data_choices: Iterable[str] | None = None,
    default_prompt_type: str = "standard",
    default_data_type: str = "explicit",
) -> argparse.ArgumentParser:
    resolved_prompt_choices = list(prompt_choices or PROMPT_CHOICES)
    resolved_data_choices = list(data_choices or ("implicit", "explicit"))
    if default_prompt_type not in resolved_prompt_choices:
        raise ValueError(f"default_prompt_type must be one of {resolved_prompt_choices}")
    if default_data_type not in resolved_data_choices:
        raise ValueError(f"default_data_type must be one of {resolved_data_choices}")

    parser = argparse.ArgumentParser(description="K-MetBench evaluation entrypoint")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-8B-Instruct")
    parser.add_argument("--image-root", type=str, default=None, help="Image root. Defaults to repo-relative data/shuffled.")
    if "explicit" in resolved_data_choices:
        parser.add_argument("--explicit-data-file", type=str, default=None, help="Override explicit dataset path.")
    else:
        parser.set_defaults(explicit_data_file=None)
    if "implicit" in resolved_data_choices:
        parser.add_argument("--implicit-data-file", type=str, default=None, help="Override implicit dataset path.")
    else:
        parser.set_defaults(implicit_data_file=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Evaluation output root. Defaults to results/evaluation.",
    )
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8192/v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-type", type=str, default=default_prompt_type, choices=resolved_prompt_choices)
    parser.add_argument("--data-type", type=str, default=default_data_type, choices=resolved_data_choices)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-item output.")

    if backend == BACKEND_TRANSFORMERS:
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=None,
            help="Maximum tokens to generate. Defaults to the prompt-type runtime config.",
        )
        parser.add_argument("--device", type=str, default="cuda")
    else:
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=None,
            help="Maximum tokens to generate. Defaults to the prompt-type runtime config.",
        )

    if backend == BACKEND_OPENAI_COMPATIBLE:
        parser.add_argument("--concurrency", type=int, default=1)

    return parser


def _resolve_runtime(args: argparse.Namespace) -> tuple[Path, Path, Path, str, list[Item]]:
    if args.data_type == "implicit" and "reasoning" in args.prompt_type:
        raise ValueError("Implicit data type does not support reasoning prompt type")

    if args.max_tokens is None:
        args.max_tokens = get_default_max_tokens(args.prompt_type)

    dataset_path = resolve_dataset_path(
        args.data_type,
        explicit_override=args.explicit_data_file,
        implicit_override=args.implicit_data_file,
    )
    image_root = resolve_image_root(args.image_root)
    results_root = resolve_results_root(args.output_root)
    prompt = get_system_prompt(args.prompt_type)

    items = load_items(dataset_path, num_samples=-1, data_type=args.data_type)
    print(f"Loaded {len(items)} items from {dataset_path}")
    filtered_items = filter_items_for_run(items, args.prompt_type, args.model)
    if args.num_samples != -1:
        filtered_items = filtered_items[: args.num_samples]
    if "reasoning" in args.prompt_type:
        print(f"Filtered to {len(filtered_items)} reasoning-enabled items.")
    if len(filtered_items) != len(items):
        print(f"After modality filtering: {len(filtered_items)} items.")

    out_dir = results_root / f"{args.data_type}_{args.prompt_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return dataset_path, image_root, out_dir, prompt, filtered_items


def _effective_temperature(prompt_type: str, temperature: float | None) -> float:
    if temperature is not None:
        return temperature
    return 1.0 if "reasoning" in prompt_type else 0.1


def _print_item_trace(item: Item, text: str, predicted_answer: int, is_correct: bool) -> None:
    print("-" * 60)
    print(f"Q[{item.id}]: {item.question_text}")
    if item.question_image:
        print(f"<image>: {item.question_image}")
    for index, choice in enumerate(item.choices_text):
        marker = " *" if index + 1 == item.answer else ""
        print(f"  {index + 1}. {choice}{marker}")
        if index < len(item.choices_image) and item.choices_image[index]:
            print(f"    <image>: {item.choices_image[index]}")
    print("Images: attached" if item.multimodal else "Images: N/A (text-only)")
    print("--- OUTPUT ---")
    print(text)
    print(f"Predicted: {predicted_answer}, Correct: {item.answer}, Match: {is_correct}")
    print("-" * 60)


def _build_result(item: Item, model_key: str, generated_text: str, inference_time: float) -> EvaluationResult:
    extracted = extract_evaluation_info(generated_text)
    predicted_answer = int(extracted["predicted_answer"])
    is_correct = predicted_answer == item.answer
    return EvaluationResult(
        item_id=item.id,
        question_text=item.question_text,
        model_key=model_key,
        generated_text=generated_text,
        predicted_answer=predicted_answer,
        correct_answer=item.answer,
        is_correct=is_correct,
        inference_time=inference_time,
        part=item.part,
        multimodal=item.multimodal,
        reasoning=str(extracted["reasoning"]),
        rationale=item.rationale,
        korean=item.korean,
    )


def _save_run(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    items: list[Item],
    results: list[EvaluationResult],
    raw_results: list[dict[str, Any]] | None,
    filename_tag: str,
) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = evaluate_accuracy(results)
    benchmark_name = f"{args.data_type}_{args.prompt_type}"
    public_protocol = build_public_protocol_block(metrics, benchmark=benchmark_name)
    safe_model_name = get_safe_name(args.model)
    output_file = out_dir / f"{safe_model_name}{filename_tag}_seed{args.seed}_{timestamp}.json"

    payload: dict[str, Any] = {
        "model_key": args.model,
        "model_name": args.model,
        "timestamp": timestamp,
        "metrics": metrics,
        "public_protocol": public_protocol,
        "evaluation_settings": {
            "num_samples": len(items),
            "max_tokens": args.max_tokens,
            "temperature": _effective_temperature(args.prompt_type, args.temperature),
            "top_p": args.top_p,
            "base_url": args.base_url,
            "seed": args.seed,
            "prompt_type": args.prompt_type,
            "data_type": args.data_type,
        },
        "results": [asdict(result) for result in results],
    }
    if raw_results is not None:
        payload["raw_results"] = raw_results

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    model_accuracy = {
        "model": args.model,
        "accuracy": metrics["accuracy"],
        "inference_time": metrics["inference_time"],
        "part1": metrics["part1"],
        "part2": metrics["part2"],
        "part3": metrics["part3"],
        "part4": metrics["part4"],
        "part5": metrics["part5"],
        "text": metrics["text"],
        "multimodal": metrics["multimodal"],
        "korean": metrics["korean"],
        "non_korean": metrics["non_korean"],
    }
    index_dir = _ensure_dir(EVALUATION_INDEX_ROOT / benchmark_name)
    with (index_dir / "model_accuracies.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(model_accuracy, ensure_ascii=False) + "\n")
    public_metric_row = build_public_metric_row(
        model_key=args.model,
        benchmark=benchmark_name,
        timestamp=timestamp,
        metrics=metrics,
    )
    with (index_dir / "model_public_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(public_metric_row, ensure_ascii=False) + "\n")

    print(f"\n{args.model} Results:")
    print(f"  {format_public_metric_summary(public_protocol, correct=metrics['correct'], total=metrics['total'])}")
    print(f"Results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{args.model:15} | {format_public_metric_summary(public_protocol, correct=metrics['correct'], total=metrics['total'])}")
    print(f"Avg Inference Time: {metrics['inference_time']:.3f}s")
    return output_file


async def _run_openai_compatible_async(args: argparse.Namespace) -> int:
    _load_env()
    _set_seed(args.seed)
    _, image_root, out_dir, system_prompt, items = _resolve_runtime(args)
    if not items:
        print("No items matched the requested configuration.")
        return 1

    runner = OpenAICompatibleRunner(api_key=args.api_key, base_url=args.base_url)
    effective_temperature = _effective_temperature(args.prompt_type, args.temperature)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = []

    async def evaluate_item(item: Item) -> EvaluationResult:
        messages = build_messages(item, args.model, args.prompt_type, system_prompt, image_root)
        start_time = time.time()
        text = ""
        try:
            text = await runner.generate_async(
                messages=messages,
                model=args.model,
                temperature=effective_temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            print(f"Error getting response for item {item.id}: {exc}")
        inference_time = time.time() - start_time
        cleaned_text = strip_think_tags(text)
        result = _build_result(item, args.model, cleaned_text, inference_time)
        if not args.quiet:
            _print_item_trace(item, cleaned_text, result.predicted_answer, result.is_correct)
        return result

    async def wrapped(item: Item) -> EvaluationResult:
        async with semaphore:
            return await evaluate_item(item)

    for item in items:
        tasks.append(asyncio.create_task(wrapped(item)))

    results: list[EvaluationResult] = []
    iterator = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Processing {args.model} | {args.prompt_type} {args.data_type}",
        unit="item",
    )
    for task in iterator:
        results.append(await task)

    _save_run(
        args=args,
        out_dir=out_dir,
        items=items,
        results=results,
        raw_results=None,
        filename_tag="",
    )
    return 0


def _run_chatgpt(args: argparse.Namespace, *, thinking: bool, filename_tag: str) -> int:
    _load_env()
    _set_seed(args.seed)
    _, image_root, out_dir, system_prompt, items = _resolve_runtime(args)
    if not items:
        print("No items matched the requested configuration.")
        return 1

    runner = ChatGPTRunner(api_key=args.api_key, base_url=args.base_url, thinking=thinking)
    effective_temperature = _effective_temperature(args.prompt_type, args.temperature)
    results: list[EvaluationResult] = []
    raw_results: list[dict[str, Any]] = []

    progress = tqdm(items, desc=f"Processing {args.model} | {args.prompt_type} {args.data_type}", unit="item")
    for item in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"ID": item.id})
        messages = build_messages(item, args.model, args.prompt_type, system_prompt, image_root)
        start_time = time.time()
        text = ""
        raw: dict[str, Any] = {}
        try:
            text, raw = runner.generate(
                messages=messages,
                model=args.model,
                temperature=effective_temperature,
                top_p=args.top_p,
            )
        except Exception as exc:
            print(f"Error getting response for item {item.id}: {exc}")
        inference_time = time.time() - start_time
        cleaned_text = strip_think_tags(text)
        result = _build_result(item, args.model, cleaned_text, inference_time)
        results.append(result)
        raw_results.append(raw)
        if not args.quiet:
            _print_item_trace(item, cleaned_text, result.predicted_answer, result.is_correct)

    _save_run(
        args=args,
        out_dir=out_dir,
        items=items,
        results=results,
        raw_results=raw_results,
        filename_tag=filename_tag,
    )
    return 0


def _run_transformers(args: argparse.Namespace) -> int:
    _load_env()
    _set_seed(args.seed)
    _, image_root, out_dir, system_prompt, items = _resolve_runtime(args)
    if not items:
        print("No items matched the requested configuration.")
        return 1

    runner = TransformersRunner(model_name=args.model, device=args.device)
    effective_temperature = _effective_temperature(args.prompt_type, args.temperature)
    results: list[EvaluationResult] = []
    progress = tqdm(items, desc=f"Processing {args.model} | {args.prompt_type} {args.data_type}", unit="item")

    for item in progress:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"ID": item.id})
        messages = build_messages(item, args.model, args.prompt_type, system_prompt, image_root)
        start_time = time.time()
        text = ""
        try:
            text = runner.generate(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=effective_temperature,
                top_p=args.top_p,
            )
        except Exception as exc:
            print(f"Error in transformers inference for item {item.id}: {exc}")
        inference_time = time.time() - start_time
        cleaned_text = strip_think_tags(text)
        result = _build_result(item, args.model, cleaned_text, inference_time)
        results.append(result)
        if not args.quiet:
            _print_item_trace(item, cleaned_text, result.predicted_answer, result.is_correct)

    _save_run(
        args=args,
        out_dir=out_dir,
        items=items,
        results=results,
        raw_results=None,
        filename_tag="",
    )
    return 0


def main_openai_compatible(argv: list[str] | None = None) -> int:
    parser = _build_parser(BACKEND_OPENAI_COMPATIBLE)
    args = parser.parse_args(argv)
    return asyncio.run(_run_openai_compatible_async(args))


def main_chatgpt(argv: list[str] | None = None) -> int:
    parser = _build_parser(BACKEND_CHATGPT)
    args = parser.parse_args(argv)
    return _run_chatgpt(args, thinking=False, filename_tag="_multimodal")


def main_chatgpt_thinking(argv: list[str] | None = None) -> int:
    parser = _build_parser(BACKEND_CHATGPT_THINKING)
    args = parser.parse_args(argv)
    return _run_chatgpt(args, thinking=True, filename_tag="_multimodal_Thinking")


def main_transformers(argv: list[str] | None = None) -> int:
    parser = _build_parser(BACKEND_TRANSFORMERS)
    args = parser.parse_args(argv)
    return _run_transformers(args)


def main_public_openai_compatible(argv: list[str] | None = None) -> int:
    parser = _build_parser(
        BACKEND_OPENAI_COMPATIBLE,
        prompt_choices=PUBLIC_PROMPT_CHOICES,
        data_choices=PUBLIC_DATA_CHOICES,
        default_prompt_type="advanced",
        default_data_type="explicit",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run_openai_compatible_async(args))


def main_public_transformers(argv: list[str] | None = None) -> int:
    parser = _build_parser(
        BACKEND_TRANSFORMERS,
        prompt_choices=PUBLIC_PROMPT_CHOICES,
        data_choices=PUBLIC_DATA_CHOICES,
        default_prompt_type="advanced",
        default_data_type="explicit",
    )
    args = parser.parse_args(argv)
    return _run_transformers(args)
