"""
Base LLM — Abstract interface for all language model providers.

Every LLM client implements this interface so the rest of the system
is completely decoupled from any specific provider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional
import time


@dataclass
class GenerationConfig:
    """Generation parameters — loaded from model_config.yaml at runtime."""
    temperature: float = 0.15
    top_p: float = 0.85
    top_k: int = 40
    max_output_tokens: int = 4096
    candidate_count: int = 1


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model_name: str
    usage: dict = field(default_factory=dict)     # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float = 0.0
    raw_response: Any = None                       # provider-specific raw object
    finish_reason: str = "stop"


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_id: str, generation_config: GenerationConfig):
        self.model_id = model_id
        self.generation_config = generation_config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the model."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response constrained to a JSON schema."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

    def _measure_latency(self, start: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start) * 1000
