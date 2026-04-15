from __future__ import annotations

import logging
import time
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, api_key: str = "", base_url: str = "http://0.0.0.0:8192/v1", timeout: int = 60, max_retries: int = 3):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.max_retries = max_retries

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int | None = None,
    ) -> str:
        if model is None:
            raise ValueError("model must be provided")

        for attempt in range(self.max_retries):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(
                    **request_kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("Attempt %s failed: %s", attempt + 1, exc)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)


class ChatGPTClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://0.0.0.0:8192/v1",
        timeout: int = 60,
        max_retries: int = 10,
        thinking: bool = False,
    ):
        from openai import OpenAI

        if "google" in base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.thinking = thinking

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> tuple[str, Any]:
        if model is None:
            raise ValueError("model must be provided")

        for attempt in range(self.max_retries):
            try:
                if self.thinking:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        reasoning_effort="high",
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort="none",
                    )
                return response.choices[0].message.content or "", response
            except Exception as exc:
                logger.warning("Attempt %s failed: %s", attempt + 1, exc)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
