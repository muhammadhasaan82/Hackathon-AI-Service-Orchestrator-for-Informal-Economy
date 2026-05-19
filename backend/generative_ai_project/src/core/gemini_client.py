"""
Gemini Client — Google Gemini / Gemma model integration.

Uses the google-genai SDK for both Gemini cloud models and
Gemma open-source models. Supports function calling, streaming,
and structured JSON output.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types

from .base_llm import BaseLLM, GenerationConfig, LLMResponse

logger = logging.getLogger("core.gemini")


class GeminiClient(BaseLLM):
    """Google Gemini / Gemma LLM client."""

    def __init__(self, model_id: str, api_key: str, generation_config: GenerationConfig):
        super().__init__(model_id, generation_config)
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized GeminiClient with model: {model_id}")

    def _build_config(self, **overrides) -> types.GenerateContentConfig:
        """Build generation config from base + overrides."""
        params = {
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
            "max_output_tokens": self.generation_config.max_output_tokens,
            "candidate_count": self.generation_config.candidate_count,
        }
        params.update(overrides)

        config_kwargs = {}
        if "system_instruction" in overrides and overrides["system_instruction"]:
            config_kwargs["system_instruction"] = overrides.pop("system_instruction")
        if "tools" in overrides and overrides["tools"]:
            config_kwargs["tools"] = overrides.pop("tools")
        if "response_mime_type" in overrides:
            config_kwargs["response_mime_type"] = overrides.pop("response_mime_type")
        if "response_schema" in overrides:
            config_kwargs["response_schema"] = overrides.pop("response_schema")

        return types.GenerateContentConfig(
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            max_output_tokens=params["max_output_tokens"],
            candidate_count=params["candidate_count"],
            **config_kwargs,
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        start = time.time()

        config = self._build_config(
            system_instruction=system_instruction,
            tools=tools,
            **kwargs,
        )

        # Build contents
        contents = []
        if history:
            for turn in history:
                contents.append(
                    types.Content(
                        role=turn.get("role", "user"),
                        parts=[types.Part.from_text(text=turn["content"])],
                    )
                )
        contents.append(prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )

            text = response.text or ""
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return LLMResponse(
                text=text,
                model_name=self.model_id,
                usage=usage,
                latency_ms=self._measure_latency(start),
                raw_response=response,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else "stop",
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate structured JSON output."""
        start = time.time()

        config = self._build_config(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            **kwargs,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config,
            )

            text = response.text or "{}"
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return LLMResponse(
                text=text,
                model_name=self.model_id,
                usage=usage,
                latency_ms=self._measure_latency(start),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Gemini structured generation failed: {e}")
            raise

    async def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        config = self._build_config(
            system_instruction=system_instruction,
            **kwargs,
        )

        contents = []
        if history:
            for turn in history:
                contents.append(
                    types.Content(
                        role=turn.get("role", "user"),
                        parts=[types.Part.from_text(text=turn["content"])],
                    )
                )
        contents.append(prompt)

        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise
