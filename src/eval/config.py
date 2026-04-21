from __future__ import annotations

import os
from pathlib import Path

from src.utils.repo_layout import CANONICAL_DATA_FILE, EVALUATION_RESULTS_ROOT, IMAGE_ROOT

DEFAULT_EXPLICIT_DATA_FILE = CANONICAL_DATA_FILE
DEFAULT_IMPLICIT_DATA_FILE = DEFAULT_EXPLICIT_DATA_FILE
DEFAULT_IMAGE_ROOT = IMAGE_ROOT
DEFAULT_RESULTS_ROOT = EVALUATION_RESULTS_ROOT

PROMPT_RUNTIME_DEFAULTS: dict[str, dict[str, int]] = {
    "advanced": {
        "max_tokens": 4096,
        "max_model_len": 8192,
    },
    "reasoning": {
        "max_tokens": 8192,
        "max_model_len": 16384,
    },
}


def _resolve_env_path(env_name: str, default_path: Path) -> Path:
    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return default_path


def resolve_dataset_path(
    data_type: str,
    *,
    explicit_override: str | None = None,
    implicit_override: str | None = None,
) -> Path:
    if data_type == "explicit":
        if explicit_override:
            return Path(explicit_override).expanduser().resolve()
        return _resolve_env_path("KMETBENCH_EXPLICIT_DATA_FILE", DEFAULT_EXPLICIT_DATA_FILE)
    if data_type == "implicit":
        if implicit_override:
            return Path(implicit_override).expanduser().resolve()
        return _resolve_env_path("KMETBENCH_IMPLICIT_DATA_FILE", DEFAULT_IMPLICIT_DATA_FILE)
    raise ValueError(f"Unsupported data type: {data_type}")


def resolve_image_root(image_root: str | None = None) -> Path:
    if image_root:
        return Path(image_root).expanduser().resolve()
    return _resolve_env_path("KMETBENCH_IMAGE_ROOT", DEFAULT_IMAGE_ROOT)


def resolve_results_root(results_root: str | None = None) -> Path:
    if results_root:
        return Path(results_root).expanduser().resolve()
    return _resolve_env_path("KMETBENCH_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)


def resolve_prompt_runtime_profile(prompt_type: str) -> str:
    return "reasoning" if "reasoning" in prompt_type else "advanced"


def get_prompt_runtime_defaults(prompt_type: str) -> dict[str, int]:
    profile = resolve_prompt_runtime_profile(prompt_type)
    return dict(PROMPT_RUNTIME_DEFAULTS[profile])


def get_default_max_tokens(prompt_type: str) -> int:
    return get_prompt_runtime_defaults(prompt_type)["max_tokens"]


def get_default_max_model_len(prompt_type: str) -> int:
    return get_prompt_runtime_defaults(prompt_type)["max_model_len"]


def get_system_prompt_map() -> dict[str, str]:
    from .prompts import SYSTEM_PROMPT_MAP

    return dict(SYSTEM_PROMPT_MAP)


def get_system_prompt(prompt_type: str) -> str:
    prompt_map = get_system_prompt_map()
    if prompt_type not in prompt_map:
        raise ValueError(f"System prompt not found for prompt_type: {prompt_type}")
    return prompt_map[prompt_type]


def is_multimodal_model(model_key: str) -> bool:
    from .model_registry import is_multimodal_model as _is_multimodal_model

    return _is_multimodal_model(model_key)


def is_reasoning_model(model_key: str) -> bool:
    from .model_registry import is_reasoning_model as _is_reasoning_model

    return _is_reasoning_model(model_key)
