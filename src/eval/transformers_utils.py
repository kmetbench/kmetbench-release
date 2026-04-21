from __future__ import annotations

import base64
import re
import sys
from io import BytesIO
from typing import Any


def get_llama32_answer(prompt, tokenizer, model, max_tokens=2048, temperature=0.0, top_p=0.95, device="cuda"):
    import torch
    from transformers import TextStreamer

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer if sys.stdout.isatty() else None,
        )
    result = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    return result.strip()


def get_ax4_answer(
    messages,
    processor,
    model,
    images,
    max_new_tokens=2048,
    temperature=0.5,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
    device="cuda",
):
    import torch

    if not hasattr(processor, "apply_chat_template"):
        raise TypeError(f"processor must be a Processor object, got {type(processor)}: {processor}")

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images is None or (isinstance(images, list) and not images):
        images = None

    inputs = processor(images=images, text=[text_prompt], padding=True, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "top_p": top_p,
                "temperature": temperature,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
        )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_ids = inputs.input_ids
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)
    if input_ids.shape[0] != generated_ids.shape[0]:
        input_ids = input_ids[0:1] if input_ids.shape[0] > 0 else input_ids
        generated_ids = generated_ids[0:1] if generated_ids.shape[0] > 0 else generated_ids

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    if not generated_ids_trimmed:
        return ""

    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not response:
        return ""
    return response[0].strip() if response[0] else ""


def get_hyperclovax_answer(
    messages,
    processor,
    model,
    max_new_tokens=2048,
    temperature=0.0,
    top_p=0.95,
    repetition_penalty=1.0,
    device="cuda",
):
    import torch

    hcx_messages = convert_openai_to_hyperclovax_format(messages)
    model_inputs = processor.apply_chat_template(
        hcx_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(device=device)
    else:
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    input_ids = model_inputs["input_ids"]
    prompt_length = input_ids.shape[-1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
    }
    if not generation_kwargs["do_sample"]:
        generation_kwargs.pop("top_p", None)
        generation_kwargs.pop("temperature", None)

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)

    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    generated_tokens = output_ids[:, prompt_length:]
    decoded = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not decoded:
        return ""
    return decoded[0].strip()


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                continue
            if item:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content).strip()


def _build_fallback_prompt(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        text = _flatten_message_content(message.get("content"))
        if text:
            rendered.append(f"{role}: {text}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def get_text_causal_lm_answer(
    messages,
    tokenizer,
    model,
    max_tokens=2048,
    temperature=0.0,
    top_p=0.95,
    device="cuda",
):
    import torch

    normalized_messages = [
        {
            "role": message.get("role", "user"),
            "content": _flatten_message_content(message.get("content")),
        }
        for message in messages
    ]

    model_device = next(model.parameters()).device
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            inputs = tokenizer.apply_chat_template(
                normalized_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            else:
                inputs = {key: value.to(model_device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[-1]
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model_device)
            inputs = {"input_ids": input_ids}
            prompt_length = input_ids.shape[-1]
    else:
        prompt = _build_fallback_prompt(normalized_messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        prompt_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    generated_tokens = output[0][prompt_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()


def normalize_messages_for_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "user")),
            "content": _flatten_message_content(message.get("content")),
        }
        for message in messages
    ]


def get_exaone45_answer(
    messages,
    processor,
    model,
    max_tokens=2048,
    temperature=0.0,
    top_p=0.95,
    device="cuda",
):
    import torch

    normalized_messages = normalize_messages_for_chat(messages)
    text_prompt = processor.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
    model_device = next(model.parameters()).device
    inputs = processor(text=[text_prompt], return_tensors="pt").to(model_device)

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None)
        or getattr(processor.tokenizer, "eos_token_id", None),
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = output[:, prompt_length:]
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not decoded:
        return ""
    return decoded[0].strip()


def convert_openai_to_llama32_format(messages):
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = _flatten_message_content(msg.get("content"))
        if role == "system":
            prompt += f"[INST] {content} [/INST]\n"
        elif role in {"user", "assistant"}:
            prompt += f"{content}\n"
    return prompt


def convert_openai_to_skt_format(openai_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    from PIL import Image

    skt_messages: list[dict[str, Any]] = []
    pil_images: list[Image.Image] = []

    for message in openai_messages:
        role = message["role"]
        content = message.get("content")

        if role == "system":
            skt_messages.append(message)
            continue

        if role == "user" and isinstance(content, list):
            new_content_list = []
            for item in content:
                if item.get("type") == "text":
                    new_content_list.append(item)
                elif item.get("type") == "image_url":
                    new_content_list.append({"type": "image"})
                    base64_url = item.get("image_url", {}).get("url")
                    if base64_url:
                        image = _decode_base64_to_pil(base64_url)
                        if image:
                            pil_images.append(image)
            skt_messages.append({"role": role, "content": new_content_list})
        elif role == "assistant":
            skt_messages.append(message)

    return skt_messages, pil_images


def convert_openai_to_hyperclovax_format(openai_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hcx_messages: list[dict[str, Any]] = []
    image_index = 0

    for message in openai_messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        if isinstance(content, list):
            new_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    text = str(item).strip()
                    if text:
                        new_content.append({"type": "text", "text": text})
                    continue
                if item.get("type") == "text":
                    text = str(item.get("text", "") or "")
                    if text:
                        new_content.append({"type": "text", "text": text})
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if url:
                        new_content.append(
                            {
                                "type": "image",
                                "filename": f"image_{image_index}.png",
                                "image": url,
                            }
                        )
                        image_index += 1
            if new_content:
                hcx_messages.append({"role": role, "content": new_content})
            continue

        text = _flatten_message_content(content)
        if text:
            hcx_messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    return hcx_messages


def _decode_base64_to_pil(base64_url: str) -> Image.Image | None:
    from PIL import Image

    try:
        img_data = re.sub(r"^data:image/.+;base64,", "", base64_url)
        img_bytes = base64.b64decode(img_data)
        return Image.open(BytesIO(img_bytes))
    except Exception as exc:
        print(f"Error decoding base64 image: {exc}")
        return None
