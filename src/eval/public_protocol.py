from __future__ import annotations

from typing import Any, Mapping

PUBLIC_PROMPT_CHOICES = ("advanced", "reasoning")
PUBLIC_DATA_CHOICES = ("explicit",)
PUBLIC_METRIC_ORDER = (
    "accuracy",
    "reasoning",
    "geo",
    "text_only",
    "multimodal",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
)

_PUBLIC_METRIC_MAP = (
    ("accuracy", "accuracy"),
    ("reasoning", "reasoning_total"),
    ("geo", "korean"),
    ("text_only", "text"),
    ("multimodal", "multimodal"),
    ("p1", "part1"),
    ("p2", "part2"),
    ("p3", "part3"),
    ("p4", "part4"),
    ("p5", "part5"),
)

_PUBLIC_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "reasoning": "Reasoning",
    "geo": "Geo",
    "text_only": "Text-Only",
    "multimodal": "Multimodal",
    "p1": "P1",
    "p2": "P2",
    "p3": "P3",
    "p4": "P4",
    "p5": "P5",
}


def build_public_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    public_metrics: dict[str, float | None] = {}
    for public_key, metric_key in _PUBLIC_METRIC_MAP:
        value = metrics.get(metric_key)
        public_metrics[public_key] = float(value) if isinstance(value, (int, float)) else None
    return public_metrics


def build_public_protocol_block(
    metrics: Mapping[str, Any],
    *,
    benchmark: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "metric_order": list(PUBLIC_METRIC_ORDER),
        "metrics": build_public_metrics(metrics),
    }


def build_public_metric_row(
    *,
    model_key: str,
    benchmark: str,
    timestamp: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "model": model_key,
        "benchmark": benchmark,
        "timestamp": timestamp,
    }
    row.update(build_public_metrics(metrics))
    return row


def format_public_metric_summary(
    protocol_block: Mapping[str, Any],
    *,
    correct: int | None = None,
    total: int | None = None,
) -> str:
    metrics = protocol_block.get("metrics", {})
    parts: list[str] = []
    for key in PUBLIC_METRIC_ORDER:
        label = _PUBLIC_METRIC_LABELS[key]
        value = metrics.get(key)
        if value is None:
            rendered = "n/a"
        else:
            rendered = f"{value:.3f}"
            if key == "accuracy" and correct is not None and total is not None:
                rendered = f"{rendered} ({correct}/{total})"
        parts.append(f"{label}: {rendered}")
    return " | ".join(parts)
