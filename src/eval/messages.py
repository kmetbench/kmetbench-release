from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .config import is_multimodal_model
from .models import Item

PROMPT_RENDER_VARIANTS = {
    "standard_no_choices": "no_choices",
    "standard_no_questions": "no_questions",
    "standard_all_answer": "all_answer",
}


def normalize_prompt_variant(prompt_type: str) -> str:
    return PROMPT_RENDER_VARIANTS.get(prompt_type, prompt_type)


def build_messages(
    item: Item,
    model: str,
    prompt_type: str,
    system_prompt: str,
    image_root: Path | None,
) -> list[dict[str, Any]]:
    normalized_prompt_type = normalize_prompt_variant(prompt_type)
    if is_multimodal_model(model):
        return build_multimodal_messages(item, normalized_prompt_type, system_prompt, image_root)
    return build_text_only_messages(item, normalized_prompt_type, system_prompt)


def _encode_image_to_base64_url(image_path: str, image_root: Path | None) -> str | None:
    if not image_path or image_root is None:
        return None
    full_path = image_root / image_path
    if not full_path.exists():
        return None
    with full_path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_multimodal_messages(
    item: Item,
    prompt_type: str,
    system_prompt: str,
    image_root: Path | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    content: list[dict[str, Any]] = []

    if prompt_type == "no_questions":
        content.append({"type": "text", "text": "문제: "})
    else:
        content.append({"type": "text", "text": f"문제: {item.question_text}"})
        if item.question_image:
            question_url = _encode_image_to_base64_url(item.question_image, image_root)
            if question_url:
                content.append({"type": "image_url", "image_url": {"url": question_url}})

    if prompt_type == "no_choices":
        for index in range(4):
            content.append({"type": "text", "text": f"{index + 1}. "})
    elif prompt_type == "all_answer" and item.choices_text:
        answer_index = max(item.answer - 1, 0)
        answer_text = item.choices_text[answer_index]
        answer_image = item.choices_image[answer_index] if answer_index < len(item.choices_image) else ""
        for index in range(4):
            content.append({"type": "text", "text": f"{index + 1}. {answer_text}"})
            answer_url = _encode_image_to_base64_url(answer_image, image_root)
            if answer_url:
                content.append({"type": "image_url", "image_url": {"url": answer_url}})
    else:
        for index, choice_text in enumerate(item.choices_text):
            content.append({"type": "text", "text": f"{index + 1}. {choice_text}"})
            choice_image = item.choices_image[index] if index < len(item.choices_image) else ""
            choice_url = _encode_image_to_base64_url(choice_image, image_root)
            if choice_url:
                content.append({"type": "image_url", "image_url": {"url": choice_url}})

    messages.append({"role": "user", "content": content})
    return messages


def build_text_only_messages(item: Item, prompt_type: str, system_prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    content: list[dict[str, Any]] = []

    if prompt_type == "no_questions":
        content.append({"type": "text", "text": "문제: "})
    else:
        content.append({"type": "text", "text": f"문제: {item.question_text}"})

    if prompt_type == "no_choices":
        for index in range(4):
            content.append({"type": "text", "text": f"{index + 1}. "})
    elif prompt_type == "all_answer" and item.choices_text:
        answer_text = item.choices_text[max(item.answer - 1, 0)]
        for index in range(4):
            content.append({"type": "text", "text": f"{index + 1}. {answer_text}"})
    else:
        for index, choice_text in enumerate(item.choices_text):
            content.append({"type": "text", "text": f"{index + 1}. {choice_text}"})

    messages.append({"role": "user", "content": content})
    return messages
