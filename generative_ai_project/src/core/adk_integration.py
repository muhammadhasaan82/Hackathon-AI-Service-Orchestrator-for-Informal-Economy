"""
ADK V2 Integration — Routes Google ADK agent calls to local UnslothClient.

Implements a custom ADK BaseLlm so ADK orchestration works fully in-process,
without LiteLLM/Ollama or any external API. Falls back to LiteLLM only when
an in-process LLM is not provided (kept for backward compatibility).
"""

import logging
from typing import AsyncIterator, Optional

from .base_llm import BaseLLM

logger = logging.getLogger("core.adk")

UNSLOTH_ADK_MODEL_NAME = "unsloth_inprocess"


def _build_unsloth_adk_model(unsloth_client: BaseLLM):
    """Construct a Pydantic-compatible ADK BaseLlm bound to UnslothClient."""
    from google.adk.models.base_llm import BaseLlm as AdkBaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types as genai_types

    class UnslothAdkModel(AdkBaseLlm):
        """ADK BaseLlm wrapper that delegates generation to UnslothClient."""

        model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

        @classmethod
        def supported_models(cls) -> list[str]:
            return [r"unsloth.*", UNSLOTH_ADK_MODEL_NAME]

        async def generate_content_async(
            self,
            llm_request: LlmRequest,
            stream: bool = False,
        ) -> AsyncIterator[LlmResponse]:
            system_instruction = _extract_system_instruction(llm_request)
            history = _extract_history(llm_request)
            prompt = history.pop()["content"] if history else ""
            overrides = _extract_generation_overrides(llm_request)

            response = await unsloth_client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                history=history or None,
                **overrides,
            )

            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=response.text or "")],
                ),
                partial=False,
            )

    return UnslothAdkModel(model=UNSLOTH_ADK_MODEL_NAME)


def _extract_system_instruction(llm_request) -> Optional[str]:
    cfg = getattr(llm_request, "config", None)
    sys_instr = getattr(cfg, "system_instruction", None) if cfg else None
    if not sys_instr:
        return None
    if isinstance(sys_instr, str):
        return sys_instr
    parts = getattr(sys_instr, "parts", None) or []
    text = "\n".join(getattr(p, "text", "") for p in parts if getattr(p, "text", ""))
    return text or None


def _extract_history(llm_request) -> list[dict]:
    history = []
    for content in getattr(llm_request, "contents", None) or []:
        role = getattr(content, "role", "user") or "user"
        role = "assistant" if role == "model" else role
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
        if text:
            history.append({"role": role, "content": text})
    return history


def _extract_generation_overrides(llm_request) -> dict:
    cfg = getattr(llm_request, "config", None)
    if not cfg:
        return {}
    overrides = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("max_output_tokens", "max_new_tokens"),
    ):
        value = getattr(cfg, src, None)
        if value is not None:
            overrides[dst] = value
    return overrides


def create_adk_agent(
    name: str,
    instruction: str,
    llm: Optional[BaseLLM] = None,
    model_string: Optional[str] = None,
    tools: Optional[list] = None,
    sub_agents: Optional[list] = None,
):
    """
    Create a Google ADK V2 agent.

    Prefers an in-process Unsloth model when ``llm`` is provided. Falls back
    to a LiteLLM-backed ``model_string`` for backward compatibility.
    """
    try:
        from google.adk.agents import Agent

        if llm is not None:
            adk_model = _build_unsloth_adk_model(llm)
        else:
            from google.adk.models.lite_llm import LiteLlm
            if not model_string:
                raise ValueError("Either `llm` or `model_string` must be provided.")
            adk_model = LiteLlm(model=model_string)

        agent_kwargs = {
            "name": name,
            "model": adk_model,
            "instruction": instruction,
        }
        if tools:
            agent_kwargs["tools"] = tools
        if sub_agents:
            agent_kwargs["sub_agents"] = sub_agents

        agent = Agent(**agent_kwargs)
        logger.info(
            "ADK agent created: %s (model=%s)",
            name,
            getattr(adk_model, "model", "unknown"),
        )
        return agent

    except ImportError:
        logger.warning(
            "google-adk not installed. Using fallback agent wrapper. "
            "Install with: uv pip install google-adk litellm"
        )
        return FallbackAgent(name=name, instruction=instruction, llm=llm)


class FallbackAgent:
    """Fallback agent when google-adk is not installed."""

    def __init__(self, name: str, instruction: str, llm: Optional[BaseLLM] = None):
        self.name = name
        self.instruction = instruction
        self.llm = llm
        logger.info(f"FallbackAgent created: {name}")


def build_adk_agent_tree(configs: dict, llm: BaseLLM, model_string: Optional[str] = None):
    """
    Build the complete ADK V2 agent hierarchy from config.

    Creates: Orchestrator → [Intent, Discovery, Ranking, Booking, FollowUp]

    ``llm`` is the in-process UnslothClient instance; ADK calls are routed to
    it through ``_build_unsloth_adk_model``. ``model_string`` is only used as a
    fallback when ``llm`` is ``None`` (e.g. LiteLLM-based providers).
    """
    agents_cfg = configs["agents"]["agents"]
    prompts = configs["prompts"]

    def _make(name, instruction, sub=None):
        return create_adk_agent(
            name=name,
            instruction=instruction,
            llm=llm,
            model_string=model_string,
            sub_agents=sub,
        )

    intent_agent = _make(
        "intent_agent",
        prompts.get("intent_extraction", "Extract user intent."),
    )
    discovery_agent = _make("discovery_agent", agents_cfg["discovery"]["role"])
    ranking_agent = _make("ranking_agent", agents_cfg["ranking"]["role"])
    booking_agent = _make("booking_agent", agents_cfg["booking"]["role"])
    followup_agent = _make("followup_agent", agents_cfg["followup"]["role"])

    return _make(
        "orchestrator",
        prompts.get("orchestrator", "Route user requests."),
        sub=[intent_agent, discovery_agent, ranking_agent, booking_agent, followup_agent],
    )
