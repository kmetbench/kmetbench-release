from __future__ import annotations

import json
from pathlib import Path

from .config import is_multimodal_model
from .models import Item


def _resolve_question(example: dict, data_type: str) -> dict:
    question = dict(example["question"])
    if data_type != "implicit":
        return question

    implicit_question = example.get("question_implicit")
    if implicit_question is None:
        return question
    if isinstance(implicit_question, dict):
        return implicit_question

    resolved = dict(question)
    resolved["text"] = str(implicit_question)
    return resolved


def _resolve_choices(example: dict, data_type: str) -> list[dict]:
    choices = [dict(choice) for choice in example["choices"]]
    if data_type != "implicit":
        return choices

    implicit_choices = example.get("choices_implicit")
    if implicit_choices is None:
        return choices

    if implicit_choices and isinstance(implicit_choices[0], dict):
        return [dict(choice) for choice in implicit_choices]

    resolved: list[dict] = []
    for index, choice_text in enumerate(implicit_choices):
        image = ""
        if index < len(choices):
            image = choices[index].get("image") or ""
        resolved.append({"text": str(choice_text), "image": image})
    return resolved


def load_items(
    file_path: str | Path,
    num_samples: int = -1,
    *,
    data_type: str = "explicit",
) -> list[Item]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    data_slice = raw if num_samples == -1 else raw[:num_samples]
    items: list[Item] = []
    for example in data_slice:
        question = _resolve_question(example, data_type)
        choices = _resolve_choices(example, data_type)
        items.append(
            Item(
                id=example["id"],
                question_text=question.get("text", ""),
                question_image=question.get("image") or "",
                choices_text=[choice.get("text", "") for choice in choices],
                choices_image=[choice.get("image") or "" for choice in choices],
                answer=example["answer"],
                source=example.get("source", ""),
                source_id=str(example.get("source_id", "")),
                question_original=example.get("question_original", ""),
                multimodal=bool(example.get("multimodal", False)),
                rationale=example.get("rationale", ""),
                part=int(example.get("part", 0)),
                korean=bool(example.get("korean", False)),
            )
        )
    return items


def filter_items_for_run(items: list[Item], prompt_type: str, model_key: str) -> list[Item]:
    filtered = items
    if "reasoning" in prompt_type:
        filtered = [item for item in filtered if item.rationale]
    if not is_multimodal_model(model_key):
        filtered = [item for item in filtered if not item.multimodal]
    return filtered
