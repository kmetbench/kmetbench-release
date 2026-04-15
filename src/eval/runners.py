from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .api_clients import ChatGPTClient, OpenAIClient
from .model_registry import is_multimodal_model
from .transformers_utils import (
    convert_openai_to_llama32_format,
    convert_openai_to_skt_format,
    get_ax4_answer,
    get_exaone45_answer,
    get_llama32_answer,
    get_text_causal_lm_answer,
)


class OpenAICompatibleRunner:
    def __init__(self, *, api_key: str, base_url: str, timeout: int = 300, max_retries: int = 3):
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def generate(self, *, messages: list[dict[str, Any]], model: str, temperature: float, top_p: float, max_tokens: int) -> str:
        return self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    async def generate_async(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        return await asyncio.to_thread(
            self.generate,
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


class ChatGPTRunner:
    def __init__(self, *, api_key: str, base_url: str, thinking: bool, timeout: int = 300, max_retries: int = 10):
        self.client = ChatGPTClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            thinking=thinking,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
    ) -> tuple[str, dict[str, Any]]:
        text, raw = self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        raw_dict = raw.model_dump() if hasattr(raw, "model_dump") else {}
        return text, raw_dict


class TransformersRunner:
    def __init__(self, *, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.mode = None
        self._torch = None
        self._prepare_helpers()

    def _prepare_helpers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, MllamaForConditionalGeneration

        self._torch = torch
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoProcessor = AutoProcessor
        self._MllamaForConditionalGeneration = MllamaForConditionalGeneration
        self._convert_openai_to_llama32_format = convert_openai_to_llama32_format
        self._convert_openai_to_skt_format = convert_openai_to_skt_format
        self._get_llama32_answer = get_llama32_answer
        self._get_ax4_answer = get_ax4_answer
        self._get_exaone45_answer = get_exaone45_answer
        self._get_text_causal_lm_answer = get_text_causal_lm_answer

    def _patch_ax4_cached_modules(self) -> None:
        from transformers.dynamic_module_utils import get_cached_module_file

        module_file_path = get_cached_module_file(
            self.model_name,
            "processing_ax4vl.py",
            force_download=False,
        )
        if not module_file_path:
            return

        module_dir = Path(module_file_path).parent
        patch_marker = "# Fix for relative import issue with model names containing dots"
        for filename in ("processing_ax4vl.py", "image_processing_ax4vl.py"):
            file_path = module_dir / filename
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            if patch_marker in content:
                continue
            old_import = "from .configuration_ax4vl import AX4VLConfig"
            new_import = (
                "import os\n"
                "import sys\n"
                "# Fix for relative import issue with model names containing dots\n"
                "_current_dir = os.path.dirname(os.path.abspath(__file__))\n"
                "if _current_dir not in sys.path:\n"
                "    sys.path.insert(0, _current_dir)\n"
                "from configuration_ax4vl import AX4VLConfig"
            )
            file_path.write_text(content.replace(old_import, new_import, 1), encoding="utf-8")

    def _load_model(self) -> None:
        model_name_lower = self.model_name.lower()
        if model_name_lower == "meta-llama/llama-3.2-90b-vision-instruct":
            self.mode = "llama32"
            self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.model = self._MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self._torch.bfloat16 if self._torch.cuda.is_available() else self._torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return

        if model_name_lower == "skt/a.x-4.0-vl-light":
            self.mode = "ax4"
            self._patch_ax4_cached_modules()
            self.processor = self._AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = self._AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self._torch.bfloat16 if self._torch.cuda.is_available() else self._torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            return

        if model_name_lower == "lgai-exaone/exaone-4.5-33b":
            from huggingface_hub import snapshot_download
            from transformers import Exaone4_5_ForConditionalGeneration

            self.mode = "exaone45"
            model_source = self.model_name
            local_only_kwargs: dict[str, Any] = {}
            try:
                model_source = snapshot_download(self.model_name, local_files_only=True)
                local_only_kwargs["local_files_only"] = True
            except Exception:
                model_source = self.model_name

            self.processor = self._AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
                use_fast=False,
                **local_only_kwargs,
            )
            self.model = Exaone4_5_ForConditionalGeneration.from_pretrained(
                model_source,
                dtype=self._torch.bfloat16 if self._torch.cuda.is_available() else self._torch.float32,
                device_map="auto",
                trust_remote_code=True,
                **local_only_kwargs,
            )
            # The current EXAONE 4.5 fork wires generation through `self.model.embed_tokens`
            # and `self.model.rotary_emb`, while the base model stores them under
            # `language_model`. Mirror the references so `generate()` works for text-only prompts.
            if not hasattr(self.model.model, "embed_tokens"):
                self.model.model.embed_tokens = self.model.model.language_model.embed_tokens
            if not hasattr(self.model.model, "rotary_emb"):
                self.model.model.rotary_emb = self.model.model.language_model.rotary_emb
            # The auxiliary MTP path breaks generation in the current EXAONE 4.5 fork.
            # We only need the main next-token head for benchmark inference.
            self.model.config.num_nextn_predict_layers = 0
            if getattr(self.processor.tokenizer, "pad_token_id", None) is None:
                self.processor.tokenizer.pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            return

        if is_multimodal_model(self.model_name):
            raise ValueError(
                "Transformers runner currently supports only "
                "'meta-llama/Llama-3.2-90B-Vision-Instruct', 'skt/A.X-4.0-VL-Light', "
                "'LGAI-EXAONE/EXAONE-4.5-33B', and generic text-only causal LM models."
            )

        self.mode = "text"
        self.tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = self._AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._torch.bfloat16 if self._torch.cuda.is_available() else self._torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        if self.model is None:
            self._load_model()

        if self.mode == "llama32":
            prompt = self._convert_openai_to_llama32_format(messages)
            return self._get_llama32_answer(
                prompt,
                self.tokenizer,
                self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=self.device,
            )

        if self.mode == "ax4":
            skt_messages, pil_images = self._convert_openai_to_skt_format(messages)
            return self._get_ax4_answer(
                messages=skt_messages,
                processor=self.processor,
                model=self.model,
                images=pil_images,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=self.device,
            )

        if self.mode == "exaone45":
            return self._get_exaone45_answer(
                messages=messages,
                processor=self.processor,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=self.device,
            )

        if self.mode == "text":
            return self._get_text_causal_lm_answer(
                messages=messages,
                tokenizer=self.tokenizer,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=self.device,
            )

        raise RuntimeError("Transformers runner was not initialized correctly.")
