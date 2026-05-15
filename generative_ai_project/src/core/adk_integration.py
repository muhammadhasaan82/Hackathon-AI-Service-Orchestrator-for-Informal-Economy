"""
ADK V2 Integration — Google Agent Development Kit bridge.

Wraps our agents in ADK V2 format using LiteLLM for Ollama connectivity.
Enables MCP tool servers and A2A protocol stubs.
"""

import logging
from typing import Optional

logger = logging.getLogger("core.adk")


def create_adk_agent(
    name: str,
    instruction: str,
    model_string: str = "ollama_chat/gemma4:e4b",
    tools: Optional[list] = None,
    sub_agents: Optional[list] = None,
):
    """
    Create a Google ADK V2 agent with LiteLLM model bridge.

    This allows local Gemma 4 (via Ollama) to power ADK agents.
    """
    try:
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm

        agent_kwargs = {
            "name": name,
            "model": LiteLlm(model=model_string),
            "instruction": instruction,
        }
        if tools:
            agent_kwargs["tools"] = tools
        if sub_agents:
            agent_kwargs["sub_agents"] = sub_agents

        agent = Agent(**agent_kwargs)
        logger.info(f"ADK agent created: {name} (model={model_string})")
        return agent

    except ImportError:
        logger.warning(
            "google-adk not installed. Using fallback agent wrapper. "
            "Install with: pip install google-adk litellm"
        )
        return FallbackAgent(name=name, instruction=instruction)


class FallbackAgent:
    """Fallback agent when google-adk is not installed."""

    def __init__(self, name: str, instruction: str):
        self.name = name
        self.instruction = instruction
        logger.info(f"FallbackAgent created: {name}")


def build_adk_agent_tree(configs: dict, model_string: str):
    """
    Build the complete ADK V2 agent hierarchy from config.

    Creates: Orchestrator → [Intent, Discovery, Ranking, Booking, FollowUp]
    """
    agents_cfg = configs["agents"]["agents"]
    prompts = configs["prompts"]

    # Build sub-agents
    intent_agent = create_adk_agent(
        name="intent_agent",
        instruction=prompts.get("intent_extraction", "Extract user intent."),
        model_string=model_string,
    )

    discovery_agent = create_adk_agent(
        name="discovery_agent",
        instruction=agents_cfg["discovery"]["role"],
        model_string=model_string,
    )

    ranking_agent = create_adk_agent(
        name="ranking_agent",
        instruction=agents_cfg["ranking"]["role"],
        model_string=model_string,
    )

    booking_agent = create_adk_agent(
        name="booking_agent",
        instruction=agents_cfg["booking"]["role"],
        model_string=model_string,
    )

    followup_agent = create_adk_agent(
        name="followup_agent",
        instruction=agents_cfg["followup"]["role"],
        model_string=model_string,
    )

    # Root orchestrator
    root_agent = create_adk_agent(
        name="orchestrator",
        instruction=prompts.get("orchestrator", "Route user requests."),
        model_string=model_string,
        sub_agents=[intent_agent, discovery_agent, ranking_agent, booking_agent, followup_agent],
    )

    return root_agent
