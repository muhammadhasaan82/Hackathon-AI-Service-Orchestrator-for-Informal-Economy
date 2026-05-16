"""
Unsloth Client — Local Gemma inference through Unsloth FastLanguageModel.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, AsyncIterator, Optional

import torch
from unsloth import FastLanguageModel

from .base_llm import BaseLLM, GenerationConfig, LLMResponse

logger = logging.getLogger("core.unsloth")

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class UnslothClient(BaseLLM):
    """Gemma LLM client loaded locally with Unsloth."""

    provider_name = "unsloth"
    health_service_name = "unsloth"

    def __init__(
        self,
        model_id: str,
        generation_config: GenerationConfig,
        max_seq_length: int = 8192,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
    ):
        if not model_id:
            raise ValueError("UNSLOTH_MODEL_ID is required for the Unsloth provider.")

        super().__init__(model_id, generation_config)
        self.max_seq_length = max_seq_length
        self.dtype = self._resolve_dtype(dtype)
        self.load_in_4bit = load_in_4bit
        self.hf_token = hf_token
        self._generation_lock = threading.Lock()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=self.hf_token,
        )
        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device
        logger.info(
            "UnslothClient initialized: model=%s max_seq_length=%s load_in_4bit=%s device=%s",
            self.model_id,
            self.max_seq_length,
            self.load_in_4bit,
            self.device,
        )

    def _resolve_dtype(self, dtype: Optional[str]) -> Optional[torch.dtype]:
        if not dtype:
            return None
        resolved = _DTYPE_MAP.get(dtype.lower())
        if resolved is None:
            raise ValueError(f"Unsupported UNSLOTH_DTYPE='{dtype}'. Use float16, bfloat16, or float32.")
        return resolved

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        messages = []

        if history:
            for turn in history:
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        user_content = prompt
        if system_instruction:
            user_content = f"{system_instruction}\n\n{prompt}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def _format_plain_prompt(self, messages: list[dict]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user").title()
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _encode_messages(self, messages: list[dict]) -> torch.Tensor:
        try:
            if getattr(self.tokenizer, "chat_template", None):
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                return input_ids.to(self.device)
        except Exception as exc:
            logger.warning("Chat template encoding failed; falling back to plain prompt: %s", exc)

        plain_prompt = self._format_plain_prompt(messages)
        encoded = self.tokenizer(
            plain_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        return encoded["input_ids"].to(self.device)

    def _build_generation_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        generation_kwargs = {
            "max_new_tokens": overrides.pop("max_new_tokens", self.generation_config.max_output_tokens),
            "temperature": overrides.pop("temperature", self.generation_config.temperature),
            "top_p": overrides.pop("top_p", self.generation_config.top_p),
            "top_k": overrides.pop("top_k", self.generation_config.top_k),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if generation_kwargs["temperature"] and generation_kwargs["temperature"] > 0:
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)
            generation_kwargs.pop("top_k", None)

        if "repeat_penalty" in overrides and "repetition_penalty" not in overrides:
            overrides["repetition_penalty"] = overrides.pop("repeat_penalty")

        for ignored_key in ("num_ctx", "format"):
            overrides.pop(ignored_key, None)

        generation_kwargs.update(overrides)
        return generation_kwargs

    def _generate_sync(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        if tools:
            logger.warning("UnslothClient received tools, but local Hugging Face generation does not execute tools.")

        start = time.time()
        messages = self._build_messages(prompt, system_instruction, history)
        input_ids = self._encode_messages(messages)
        generation_kwargs = self._build_generation_kwargs(kwargs)

        with self._generation_lock:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    **generation_kwargs,
                )

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(generated_ids.shape[-1])

        return LLMResponse(
            text=text,
            model_name=self.model_id,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            latency_ms=self._measure_latency(start),
            raw_response={"model_id": self.model_id},
            finish_reason="stop",
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            system_instruction,
            history,
            tools,
            **kwargs,
        )

    def _extract_json_text(self, text: str) -> str:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")

        candidates = []
        if object_start >= 0 and object_end > object_start:
            candidates.append(cleaned[object_start:object_end + 1])
        if array_start >= 0 and array_end > array_start:
            candidates.append(cleaned[array_start:array_end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                continue

        return cleaned

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        schema_text = json.dumps(response_schema or {}, ensure_ascii=False)
        structured_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON. Do not include markdown fences or explanation.\n"
            f"JSON schema/context:\n{schema_text}"
        )

        response = await self.generate(
            prompt=structured_prompt,
            system_instruction=system_instruction,
            **kwargs,
        )
        response.text = self._extract_json_text(response.text)
        return response

    async def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        response = await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
            **kwargs,
        )
        yield response.text

    def check_health(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def list_models(self) -> list[str]:
        return [self.model_id]
