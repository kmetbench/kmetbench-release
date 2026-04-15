#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.api_clients import OpenAIClient
from src.eval.config import get_system_prompt, resolve_dataset_path, resolve_results_root
from src.eval.data import load_items
from src.eval.models import ReasoningEvaluationResult
from src.eval.parsing import (
    evaluate_reasoning_scores,
    extract_reasoning_evaluation_info,
    get_safe_name,
)
from src.eval.public_protocol import (
    build_public_metric_row,
    build_public_protocol_block,
    format_public_metric_summary,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_):
        return iterable


def build_reasoning_evaluation_messages(
    *,
    system_prompt: str,
    question_text: str,
    choices_text: list[str],
    correct_answer: int,
    rationale: str,
    reasoning: str,
    predicted_answer: int,
    wo_rationale: bool,
) -> list[dict[str, str]]:
    prompt_lines = [
        "아래 정보를 바탕으로 평가를 수행하시오.",
        "",
        "--- BEGIN INPUT DATA ---",
        "",
        "### <문제 정보>",
        f"**문제 텍스트:** {question_text}",
        "",
        "**선택지:**",
    ]
    for index, choice_text in enumerate(choices_text, start=1):
        prompt_lines.append(f"{index}. {choice_text}")
    prompt_lines.extend(
        [
            "",
            f"**정답:** {correct_answer}",
            "",
            "### <전문가 검증 참조 자료>",
        ]
    )
    if wo_rationale:
        prompt_lines.append("(자료 없음)")
    else:
        prompt_lines.append(rationale or "")
    prompt_lines.extend(
        [
            "",
            "### <평가 대상 답변>",
            f"**생성된 추론 근거:** {reasoning}",
            f"**답안:** {predicted_answer}",
            "",
            "--- END INPUT DATA ---",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(prompt_lines)},
    ]


def locate_prediction_file(
    *,
    results_root: Path,
    model_name: str,
    data_type: str,
    seed: int,
) -> Path:
    result_dir = results_root / f"{data_type}_reasoning"
    safe_model_name = get_safe_name(model_name)
    seeded_matches = sorted(result_dir.glob(f"{safe_model_name}_seed{seed}_*.json"))
    if seeded_matches:
        return seeded_matches[-1]
    fallback_matches = sorted(result_dir.glob(f"{safe_model_name}_*.json"))
    if fallback_matches:
        return fallback_matches[-1]
    raise FileNotFoundError(
        f"No reasoning prediction file found for model={model_name!r} in {result_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate reasoning outputs with a judge model.")
    parser.add_argument("--model", type=str, required=True, help="Target model whose reasoning outputs will be judged.")
    parser.add_argument("--predictions", type=str, default=None, help="Explicit reasoning prediction JSON path.")
    parser.add_argument("--evaluator", type=str, default="gemini-2.5-pro")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--explicit-data-file", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--data-type", type=str, default="explicit", choices=["explicit"])
    parser.add_argument("--prompt-type", type=str, default="reasoning_evaluation", choices=["reasoning_evaluation"])
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-parse-retries", type=int, default=3)
    parser.add_argument("--wo-rationale", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing API key. Set --api-key or GEMINI_API_KEY.")

    dataset_path = resolve_dataset_path(
        args.data_type,
        explicit_override=args.explicit_data_file,
    )
    results_root = resolve_results_root(args.output_root)
    system_prompt = get_system_prompt(args.prompt_type)

    predictions_path = (
        Path(args.predictions).expanduser().resolve()
        if args.predictions
        else locate_prediction_file(
            results_root=results_root,
            model_name=args.model,
            data_type=args.data_type,
            seed=args.seed,
        )
    )

    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions_payload = json.load(handle)
    prediction_rows = list(predictions_payload.get("results", []))
    if args.num_samples != -1:
        prediction_rows = prediction_rows[: args.num_samples]
    if not prediction_rows:
        print(f"No prediction rows found in {predictions_path}")
        return 1

    input_items = load_items(dataset_path, num_samples=-1)
    input_by_id = {item.id: item for item in input_items}

    out_dir_name = f"{args.data_type}_{args.prompt_type}"
    if args.wo_rationale:
        out_dir_name += "_wo_rationale"
    out_dir = results_root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAIClient(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=max(args.max_parse_retries, 3),
    )

    model_results: list[ReasoningEvaluationResult] = []
    progress = tqdm(
        prediction_rows,
        desc=f"Judging {args.model} with {args.evaluator}",
        unit="item",
    )
    for prediction in progress:
        item_id = int(prediction["item_id"])
        input_item = input_by_id.get(item_id)
        if input_item is None:
            print(f"Skipping item {item_id}: missing source item in {dataset_path}")
            continue
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"ID": item_id})

        reasoning = str(prediction.get("reasoning", "") or "")
        if not reasoning and isinstance(prediction.get("generated_text"), str):
            reasoning = str(prediction["generated_text"])
        messages = build_reasoning_evaluation_messages(
            system_prompt=system_prompt,
            question_text=input_item.question_text,
            choices_text=input_item.choices_text,
            correct_answer=int(prediction.get("correct_answer", input_item.answer)),
            rationale=str(prediction.get("rationale", input_item.rationale) or ""),
            reasoning=reasoning,
            predicted_answer=int(prediction.get("predicted_answer", -1)),
            wo_rationale=args.wo_rationale,
        )

        generated_text = ""
        extracted: dict[str, object] = {}
        start_time = time.time()
        for attempt in range(max(1, args.max_parse_retries)):
            try:
                generated_text = client.chat_completion(
                    messages=messages,
                    model=args.evaluator,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as exc:
                if attempt == args.max_parse_retries - 1:
                    print(f"Error judging item {item_id}: {exc}")
                continue

            extracted = extract_reasoning_evaluation_info(generated_text)
            if int(extracted.get("total", -1)) != -1:
                break
        inference_time = float(prediction.get("inference_time", 0.0))

        result = ReasoningEvaluationResult(
            item_id=item_id,
            question_text=input_item.question_text,
            model_key=str(predictions_payload.get("model_key") or args.model),
            predicted_answer=int(prediction.get("predicted_answer", -1)),
            correct_answer=int(prediction.get("correct_answer", input_item.answer)),
            is_correct=bool(prediction.get("is_correct", False)),
            inference_time=inference_time,
            part=int(prediction.get("part", input_item.part)),
            multimodal=bool(prediction.get("multimodal", input_item.multimodal)),
            reasoning=reasoning,
            rationale=str(prediction.get("rationale", input_item.rationale) or ""),
            evaluator=args.evaluator,
            generated_text=generated_text,
            total=int(extracted.get("total", -1)),
            factual=int(extracted.get("factual", -1)),
            logical=int(extracted.get("logical", -1)),
            depth=int(extracted.get("depth", -1)),
            clarity=int(extracted.get("clarity", -1)),
            comment=str(extracted.get("comment", "") or ""),
            factual_error=list(extracted.get("factual_error", []) or []),
            korean=bool(prediction.get("korean", input_item.korean)),
        )
        model_results.append(result)

        if not args.quiet:
            elapsed = time.time() - start_time
            print("-" * 60)
            print(f"Q[{item_id}] pred={result.predicted_answer} gold={result.correct_answer} match={result.is_correct}")
            print(f"Judge total={result.total} factual={result.factual} logical={result.logical} depth={result.depth} clarity={result.clarity}")
            print(f"Judge latency: {elapsed:.2f}s")

    metrics = evaluate_reasoning_scores(model_results)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = get_safe_name(args.model)
    output_file = out_dir / f"{safe_model_name}_seed{args.seed}_{timestamp}.json"
    benchmark_name = f"{args.data_type}_{args.prompt_type}"
    public_protocol = build_public_protocol_block(metrics, benchmark=benchmark_name)

    payload = {
        "model_key": str(predictions_payload.get("model_key") or args.model),
        "model_name": str(predictions_payload.get("model_name") or args.model),
        "timestamp": timestamp,
        "metrics": metrics,
        "public_protocol": public_protocol,
        "evaluation_settings": {
            "num_samples": len(model_results),
            "evaluator_model": args.evaluator,
            "base_url": args.base_url,
            "seed": args.seed,
            "prompt_type": args.prompt_type,
            "data_type": args.data_type,
            "predictions_file": str(predictions_path),
            "wo_rationale": args.wo_rationale,
        },
        "results": [asdict(result) for result in model_results],
    }
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    public_metric_row = build_public_metric_row(
        model_key=str(predictions_payload.get("model_key") or args.model),
        benchmark=benchmark_name,
        timestamp=timestamp,
        metrics=metrics,
    )
    with (out_dir / "model_public_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(public_metric_row, ensure_ascii=False) + "\n")

    print(f"Judged {len(model_results)} items.")
    print(format_public_metric_summary(public_protocol, correct=metrics["correct"], total=metrics["total"]))
    print(f"Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
