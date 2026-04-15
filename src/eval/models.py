from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Item:
    id: int
    question_text: str
    question_image: str | None
    choices_text: list[str]
    choices_image: list[str | None]
    answer: int
    source: str
    source_id: str
    question_original: str
    multimodal: bool
    rationale: str
    part: int
    korean: bool


@dataclass
class EvaluationResult:
    item_id: int
    question_text: str
    model_key: str
    generated_text: str
    predicted_answer: int
    correct_answer: int
    is_correct: bool
    inference_time: float
    part: int
    multimodal: bool
    reasoning: str
    rationale: str
    korean: bool


@dataclass
class ReasoningEvaluationResult:
    item_id: int
    question_text: str
    model_key: str
    predicted_answer: int
    correct_answer: int
    is_correct: bool
    inference_time: float
    part: int
    multimodal: bool
    reasoning: str
    rationale: str
    evaluator: str
    generated_text: str
    total: int
    factual: int
    logical: int
    depth: int
    clarity: int
    comment: str
    factual_error: list[str]
    korean: bool
