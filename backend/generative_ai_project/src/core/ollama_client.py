"""
Ollama Client — Local Gemma 4 LLM via Ollama REST API.

Fully open-source, no API keys. Supports streaming, tool calling,
and structured JSON output via Ollama's native API.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

import httpx

from .base_llm import BaseLLM, GenerationConfig, LLMResponse

logger = logging.getLogger("core.ollama")


class OllamaClient(BaseLLM):
    """Gemma 4 LLM client via Ollama REST API."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        generation_config: GenerationConfig,
        num_ctx: int = 8192,
    ):
        super().__init__(model_id, generation_config)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        logger.info(f"OllamaClient initialized: model={model_id} url={self.base_url}")

    def _build_options(self, **overrides) -> dict:
        """Build Ollama generation options from config."""
        opts = {
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
            "num_predict": self.generation_config.max_output_tokens,
            "num_ctx": self.num_ctx,
            "repeat_penalty": 1.1,
        }
        opts.update(overrides)
        return opts

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Ollama /api/chat."""
        start = time.time()

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if history:
            for turn in history:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": self._build_options(**kwargs),
        }
        if tools:
            payload["tools"] = tools

        try:
            response = await self._async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "")
            usage = {}
            if "eval_count" in data:
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }

            return LLMResponse(
                text=text,
                model_name=self.model_id,
                usage=usage,
                latency_ms=self._measure_latency(start),
                raw_response=data,
                finish_reason=data.get("done_reason", "stop"),
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate structured JSON output via Ollama's format parameter."""
        start = time.time()

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": self._build_options(**kwargs),
        }

        try:
            response = await self._async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "{}")
            usage = {}
            if "eval_count" in data:
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }

            return LLMResponse(
                text=text,
                model_name=self.model_id,
                usage=usage,
                latency_ms=self._measure_latency(start),
                raw_response=data,
            )

        except Exception as e:
            logger.error(f"Ollama structured generation failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens from Ollama."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if history:
            for turn in history:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "options": self._build_options(**kwargs),
        }

        try:
            async with self._async_client.stream("POST", "/api/chat", json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    def check_health(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = self._client.get("/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        try:
            r = self._client.get("/api/tags")
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []
