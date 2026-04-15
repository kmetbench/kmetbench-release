from __future__ import annotations

import json
import re

from .models import EvaluationResult, ReasoningEvaluationResult


def get_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def get_answer_from_value(value: object) -> int:
    if value is None:
        return -1
    try:
        clean_str = str(value).strip().strip("'\"")
        digits = re.findall(r"\d+", clean_str)
        if digits:
            return int(digits[0])
    except (TypeError, ValueError):
        pass
    return -1


def find_json_string(text: str) -> str | None:
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    match = re.search(r"```\s*({.*?})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    scratchpad_end = text.rfind("</scratchpad>")
    search_start = scratchpad_end + len("</scratchpad>") if scratchpad_end != -1 else 0
    start_index = text.find("{", search_start)
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]
    return None


def _extract_choice_from_freeform_text(text: str) -> int:
    if not text:
        return -1

    tail = strip_think_tags(text)[-2000:]
    direct_patterns = [
        r"(?:정답|답)\s*(?:은|:)?\s*(?:선택지|보기|옵션|option)?\s*([1-5])",
        r"(?:final answer|answer)\s*(?:is|:)?\s*(?:option\s*)?([1-5])",
        r"(?:선택지|보기|옵션|option)\s*([1-5])(?:번)?\s*(?:이|가|는)?\s*(?:정답|답)",
        r"(?:선택지|보기|옵션|option)\s*([1-5])(?:번)?\s*(?:이|가|는)?\s*(?:가장\s+유력|가장\s+적절|정답일\s+가능성이\s+높|일\s+가능성이\s+높)",
        r"(?:option)\s*([1-5])\s*(?:is|would be)\s*(?:the\s+answer|correct|incorrect|likely)",
        r"(?:선택지)\s*([1-5])(?:번)?\s*(?:은|는|이|가)?\s*(?:옳지\s+않|틀리|맞지\s+않)",
    ]
    for pattern in direct_patterns:
        matches = list(re.finditer(pattern, tail, flags=re.IGNORECASE))
        if matches:
            return get_answer_from_value(matches[-1].group(1))

    fallback_matches = list(
        re.finditer(r"(?:선택지|보기|옵션|option)\s*([1-5])(?:번)?", tail, flags=re.IGNORECASE)
    )
    if fallback_matches:
        return get_answer_from_value(fallback_matches[-1].group(1))

    return -1


def extract_evaluation_info(text: str) -> dict[str, object]:
    outputs: dict[str, object] = {
        "predicted_answer": -1,
        "reasoning": "",
    }
    json_str = find_json_string(text)
    if not json_str:
        outputs["predicted_answer"] = _extract_choice_from_freeform_text(text)
        return outputs

    data = None
    candidates = [
        re.sub(r",\s*([}\]])", r"\1", json_str),
        re.sub(r",\s*([}\]])", r"\1", json_str.replace("'", '"')),
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(data, dict):
        outputs["predicted_answer"] = _extract_choice_from_freeform_text(text)
        return outputs

    outputs["predicted_answer"] = get_answer_from_value(data.get("정답"))
    reasoning = data.get("생성된_추론_근거")
    outputs["reasoning"] = reasoning.strip() if isinstance(reasoning, str) else ""
    if outputs["predicted_answer"] == -1:
        outputs["predicted_answer"] = _extract_choice_from_freeform_text(text)
    return outputs


def _subset_accuracy(results: list[EvaluationResult], predicate) -> float:
    subset = [result for result in results if predicate(result)]
    if not subset:
        return 0.0
    correct = sum(1 for result in subset if result.is_correct)
    return correct / len(subset)


def evaluate_accuracy(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "inference_time": 0.0,
            "part1": 0.0,
            "part2": 0.0,
            "part3": 0.0,
            "part4": 0.0,
            "part5": 0.0,
            "text": 0.0,
            "multimodal": 0.0,
            "korean": 0.0,
            "non_korean": 0.0,
        }

    correct = sum(1 for result in results if result.is_correct)
    total = len(results)
    inference_time = sum(result.inference_time for result in results) / total

    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "inference_time": inference_time,
        "part1": _subset_accuracy(results, lambda result: result.part == 1),
        "part2": _subset_accuracy(results, lambda result: result.part == 2),
        "part3": _subset_accuracy(results, lambda result: result.part == 3),
        "part4": _subset_accuracy(results, lambda result: result.part == 4),
        "part5": _subset_accuracy(results, lambda result: result.part == 5),
        "text": _subset_accuracy(results, lambda result: not result.multimodal),
        "multimodal": _subset_accuracy(results, lambda result: result.multimodal),
        "korean": _subset_accuracy(results, lambda result: result.korean),
        "non_korean": _subset_accuracy(results, lambda result: not result.korean),
    }


def evaluate_reasoning_scores(results: list[ReasoningEvaluationResult]) -> dict[str, float]:
    if not results:
        return {
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "inference_time": 0.0,
            "part1": 0.0,
            "part2": 0.0,
            "part3": 0.0,
            "part4": 0.0,
            "part5": 0.0,
            "text": 0.0,
            "multimodal": 0.0,
            "korean": 0.0,
            "non_korean": 0.0,
            "reasoning_total": 0.0,
            "factual": 0.0,
            "logical": 0.0,
            "depth": 0.0,
            "clarity": 0.0,
            "reasoning_total_correct": 0.0,
            "factual_correct": 0.0,
            "logical_correct": 0.0,
            "depth_correct": 0.0,
            "clarity_correct": 0.0,
        }

    correct = sum(1 for result in results if result.is_correct)
    total = len(results)
    inference_time = sum(result.inference_time for result in results) / total

    factual = sum(result.factual for result in results) / total
    logical = sum(result.logical for result in results) / total
    depth = sum(result.depth for result in results) / total
    clarity = sum(result.clarity for result in results) / total

    correct_results = [result for result in results if result.is_correct]
    correct_count = len(correct_results)
    factual_correct = sum(result.factual for result in correct_results) / correct_count if correct_count else 0.0
    logical_correct = sum(result.logical for result in correct_results) / correct_count if correct_count else 0.0
    depth_correct = sum(result.depth for result in correct_results) / correct_count if correct_count else 0.0
    clarity_correct = sum(result.clarity for result in correct_results) / correct_count if correct_count else 0.0

    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "inference_time": inference_time,
        "part1": _subset_accuracy(results, lambda result: result.part == 1),
        "part2": _subset_accuracy(results, lambda result: result.part == 2),
        "part3": _subset_accuracy(results, lambda result: result.part == 3),
        "part4": _subset_accuracy(results, lambda result: result.part == 4),
        "part5": _subset_accuracy(results, lambda result: result.part == 5),
        "text": _subset_accuracy(results, lambda result: not result.multimodal),
        "multimodal": _subset_accuracy(results, lambda result: result.multimodal),
        "korean": _subset_accuracy(results, lambda result: result.korean),
        "non_korean": _subset_accuracy(results, lambda result: not result.korean),
        "reasoning_total": factual + logical + depth + clarity,
        "factual": factual,
        "logical": logical,
        "depth": depth,
        "clarity": clarity,
        "reasoning_total_correct": factual_correct + logical_correct + depth_correct + clarity_correct,
        "factual_correct": factual_correct,
        "logical_correct": logical_correct,
        "depth_correct": depth_correct,
        "clarity_correct": clarity_correct,
    }


def extract_reasoning_evaluation_info(text: str) -> dict[str, object]:
    outputs: dict[str, object] = {
        "total": -1,
        "factual": -1,
        "logical": -1,
        "depth": -1,
        "clarity": -1,
        "comment": "",
        "factual_error": [],
    }
    json_str = find_json_string(text)
    if not json_str:
        return outputs

    data = None
    candidates = [
        re.sub(r",\s*([}\]])", r"\1", json_str),
        re.sub(r",\s*([}\]])", r"\1", json_str.replace("'", '"')),
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(data, dict):
        return outputs

    try:
        scores = data.get("평가_점수", {})
        if isinstance(scores, dict):
            outputs["factual"] = get_answer_from_value(scores.get("정확성"))
            outputs["logical"] = get_answer_from_value(scores.get("논리적_완결성"))
            outputs["depth"] = get_answer_from_value(scores.get("추론의_깊이"))
            outputs["clarity"] = get_answer_from_value(scores.get("표현의_명확성"))
        outputs["comment"] = str(data.get("평가_사유", "") or "")
        factual_error = data.get("사실_오류_목록", [])
        outputs["factual_error"] = factual_error if isinstance(factual_error, list) else []

        score_values = [
            int(outputs["factual"]),
            int(outputs["logical"]),
            int(outputs["depth"]),
            int(outputs["clarity"]),
        ]
        if all(score != -1 for score in score_values):
            outputs["total"] = sum(score_values)
    except Exception:
        return outputs

    return outputs
