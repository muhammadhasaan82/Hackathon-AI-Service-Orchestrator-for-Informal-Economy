"""
Transformers Client — CPU-safe LLM inference via HuggingFace Transformers.

This client does NOT require Unsloth or CUDA. It runs on plain CPU-only
machines (slow but functional) and automatically uses CUDA if available.
Used as the default backend so the API can start on CPU-only VMs.

Heavy imports (torch, transformers) are kept inside __init__ so the
module itself can be imported on machines where these are not present.
"""

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any, AsyncIterator, Optional

from .base_llm import BaseLLM, GenerationConfig, LLMResponse

logger = logging.getLogger("core.transformers")


class TransformersClient(BaseLLM):
    """LLM client using HuggingFace transformers (CPU/GPU compatible)."""

    provider_name = "transformers"
    health_service_name = "transformers"

    def __init__(
        self,
        model_id: str,
        generation_config: GenerationConfig,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        max_seq_length: int = 8192,
        trust_remote_code: bool = False,
    ):
        if not model_id:
            raise ValueError(
                "MODEL_NAME is required for the transformers backend "
                "(e.g., MODEL_NAME=google/gemma-3-4b-it)."
            )

        super().__init__(model_id, generation_config)
        self.max_seq_length = max_seq_length
        self._generation_lock = threading.Lock()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers backend requires `transformers` and `torch`. "
                "Install with `uv pip install transformers torch`."
            ) from e

        # Resolve device — auto-detect if not specified
        if device:
            resolved_device = device
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"

        # Resolve dtype
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype:
            resolved_dtype = dtype_map.get(torch_dtype.lower())
            if resolved_dtype is None:
                raise ValueError(
                    f"Unsupported torch_dtype='{torch_dtype}'. "
                    "Use float16, bfloat16, or float32."
                )
        elif resolved_device == "cuda":
            resolved_dtype = torch.float16
        else:
            # CPU works best with fp32 (fp16 is slow on CPU)
            resolved_dtype = torch.float32

        load_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        if load_in_4bit and resolved_device == "cuda":
            # Only attempt 4-bit on GPU; CPU does not support bitsandbytes
            load_kwargs["load_in_4bit"] = True

        logger.info(
            "Loading transformers model: %s (device=%s, dtype=%s)",
            model_id, resolved_device, resolved_dtype,
        )

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if resolved_device == "cuda":
            try:
                self.model = self.model.to("cuda")
            except Exception as e:
                logger.warning("Failed to move model to CUDA: %s. Falling back to CPU.", e)
                resolved_device = "cpu"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._torch = torch
        self.device = resolved_device

        logger.info(
            "TransformersClient initialized: model=%s device=%s",
            self.model_id, self.device,
        )

    # ── helpers ──────────────────────────────────────────────────
    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if history:
            for turn in history:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_sync(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Synchronous generation. Callers run this in a thread executor."""
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers without a chat template
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            parts.append("ASSISTANT:")
            input_text = "\n".join(parts)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temp = temperature if temperature is not None else self.generation_config.temperature
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.generation_config.max_output_tokens,
            "temperature": max(temp, 1e-5),
            "top_p": top_p if top_p is not None else self.generation_config.top_p,
            "top_k": top_k if top_k is not None else self.generation_config.top_k,
            "do_sample": temp > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with self._generation_lock:
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort extraction of a JSON object from raw model text."""
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group() if match else (text or "{}")

    # ── BaseLLM interface ────────────────────────────────────────
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        start = time.time()
        messages = self._build_messages(prompt, system_instruction, history)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: self._generate_sync(
                messages,
                max_new_tokens=kwargs.get("max_output_tokens"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
            ),
        )

        return LLMResponse(
            text=text,
            model_name=self.model_id,
            usage={},
            latency_ms=self._measure_latency(start),
        )

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        json_instr = (
            "Respond ONLY with valid JSON. No prose, no markdown fences. "
            "Output a single JSON object."
        )
        sys_inst = (
            f"{system_instruction}\n\n{json_instr}"
            if system_instruction
            else json_instr
        )

        response = await self.generate(prompt, system_instruction=sys_inst, **kwargs)
        response.text = self._extract_json(response.text)
        return response

    async def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield the full generation as a single chunk (CPU-friendly)."""
        response = await self.generate(prompt, system_instruction, history, **kwargs)
        yield response.text

    def check_health(self) -> bool:
        """Health check — returns True once model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None
