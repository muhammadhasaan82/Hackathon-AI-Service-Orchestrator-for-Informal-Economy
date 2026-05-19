"""
Inference Engine — Unified inference pipeline.
"""

import logging
from typing import Optional
from ..core.base_llm import BaseLLM, LLMResponse

logger = logging.getLogger("inference.engine")


class InferenceEngine:
    """Unified inference with context assembly and tool execution."""

    def __init__(self, llm: BaseLLM, max_steps: int = 6):
        self.llm = llm
        self.max_steps = max_steps

    async def run(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> LLMResponse:
        return await self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history,
        )
