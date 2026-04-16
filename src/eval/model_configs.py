from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .parsing import get_safe_name

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_ROOT = REPO_ROOT / "configs" / "models"
MODEL_CONFIG_GROUPS = ("api", "hf", "vllm")


@dataclass
class ModelParams:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None


@dataclass
class ModelMetadata:
    release_date: str | None = None
    model_size: str | int | float | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    modality: str = "llm"
    is_reasoning: bool = False
    huggingface_url: str = ""


@dataclass
class ModelDefinition:
    name: str
    client: str
    engine: str
    base_url: str | None = None
    api_key_env: str | None = None
    is_vlm: bool = False
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class LoadedModelConfig:
    slug: str
    group: str
    path: Path
    metadata: ModelMetadata
    model: ModelDefinition
    raw: dict[str, Any]

    @property
    def relative_path(self) -> str:
        return str(self.path.relative_to(MODEL_CONFIG_ROOT))

    def get_section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _iter_model_config_paths() -> list[Path]:
    paths: list[Path] = []
    for group in MODEL_CONFIG_GROUPS:
        group_dir = MODEL_CONFIG_ROOT / group
        if not group_dir.exists():
            continue
        paths.extend(sorted(group_dir.glob("*.yaml")))
    return paths


def list_model_config_entries() -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for path in _iter_model_config_paths():
        if path.stem.startswith("_"):
            continue
        rel_no_suffix = path.relative_to(MODEL_CONFIG_ROOT).with_suffix("")
        entries.append((str(rel_no_suffix), path))
    return entries


def resolve_model_config_path(identifier: str) -> Path:
    raw_value = identifier.strip()
    if not raw_value:
        raise ValueError("model config identifier must not be empty")

    direct_path = Path(raw_value).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    root_relative = (MODEL_CONFIG_ROOT / raw_value).expanduser()
    if root_relative.exists():
        return root_relative.resolve()
    if root_relative.with_suffix(".yaml").exists():
        return root_relative.with_suffix(".yaml").resolve()

    normalized = raw_value.removesuffix(".yaml").replace("\\", "/")
    safe_name = get_safe_name(normalized)

    matches: list[Path] = []
    for label, path in list_model_config_entries():
        if normalized == label or normalized == path.stem or safe_name == path.stem:
            matches.append(path)

    if not matches:
        raise FileNotFoundError(f"Could not find model config: {identifier}")
    if len(matches) > 1:
        rendered = ", ".join(str(path.relative_to(MODEL_CONFIG_ROOT)) for path in matches)
        raise ValueError(f"Ambiguous model config {identifier!r}: {rendered}")
    return matches[0].resolve()


def load_model_config(identifier: str) -> LoadedModelConfig:
    path = resolve_model_config_path(identifier)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a mapping: {path}")

    metadata_block = _as_mapping(payload.get("metadata"), field_name="metadata")
    model_block = _as_mapping(payload.get("model"), field_name="model")
    params_block = _as_mapping(model_block.get("params"), field_name="model.params")

    name = model_block.get("name")
    client = model_block.get("client")
    engine = model_block.get("engine")
    if not isinstance(name, str) or not name:
        raise ValueError(f"model.name must be a non-empty string: {path}")
    if not isinstance(client, str) or not client:
        raise ValueError(f"model.client must be a non-empty string: {path}")
    if not isinstance(engine, str) or not engine:
        raise ValueError(f"model.engine must be a non-empty string: {path}")

    metadata = ModelMetadata(
        release_date=metadata_block.get("release_date"),
        model_size=metadata_block.get("model_size"),
        context_window=metadata_block.get("context_window"),
        max_output_tokens=metadata_block.get("max_output_tokens"),
        modality=str(metadata_block.get("modality", "llm")),
        is_reasoning=bool(metadata_block.get("is_reasoning", False)),
        huggingface_url=str(metadata_block.get("huggingface_url", "") or ""),
    )
    model = ModelDefinition(
        name=name,
        client=client,
        engine=engine,
        base_url=model_block.get("base_url"),
        api_key_env=model_block.get("api_key_env"),
        is_vlm=bool(model_block.get("is_vlm", False)),
        params=ModelParams(
            max_tokens=params_block.get("max_tokens"),
            temperature=params_block.get("temperature"),
            top_p=params_block.get("top_p"),
            reasoning_effort=params_block.get("reasoning_effort"),
        ),
    )
    return LoadedModelConfig(
        slug=path.stem,
        group=path.parent.name,
        path=path,
        metadata=metadata,
        model=model,
        raw=payload,
    )
