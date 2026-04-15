from __future__ import annotations

MULTIMODAL_MODELS = {
    "LGAI-EXAONE/EXAONE-4.5-33B",
    "skt/A.X-4.0-VL-Light",
    "NCSOFT/VARCO-VISION-2.0-1.7B",
    "NCSOFT/VARCO-VISION-2.0-14B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/QVQ-72B-Preview",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "OpenGVLab/InternVL3_5-38B-Instruct",
    "OpenGVLab/InternVL3_5-14B-Instruct",
    "OpenGVLab/InternVL3_5-8B-Instruct",
    "OpenGVLab/InternVL3_5-4B-Instruct",
    "OpenGVLab/InternVL3_5-2B-Instruct",
    "OpenGVLab/InternVL3_5-1B-Instruct",
    "gemini-3-pro-preview",
    "gpt-5.2",
}

REASONING_MODELS = {
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "LGAI-EXAONE/EXAONE-4.5-33B",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "microsoft/Phi-4-reasoning",
    "microsoft/Phi-4-mini-reasoning",
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/QVQ-72B-Preview",
    "OpenGVLab/InternVL3_5-38B-Instruct",
    "OpenGVLab/InternVL3_5-14B-Instruct",
    "OpenGVLab/InternVL3_5-8B-Instruct",
    "OpenGVLab/InternVL3_5-4B-Instruct",
    "OpenGVLab/InternVL3_5-2B-Instruct",
    "OpenGVLab/InternVL3_5-1B-Instruct",
    "CohereLabs/command-a-reasoning-08-2025",
    "gemini-3-pro-preview",
    "gpt-5.2",
}


def is_multimodal_model(model_key: str) -> bool:
    return model_key in MULTIMODAL_MODELS


def is_reasoning_model(model_key: str) -> bool:
    return model_key in REASONING_MODELS
