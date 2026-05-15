"""
Model Factory — Creates LLM instances from YAML configuration.

Reads model_config.yaml and instantiates the Ollama client for
local Gemma 4 inference. Fully open-source, no API keys.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from .base_llm import BaseLLM, GenerationConfig

logger = logging.getLogger("core.factory")

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file."""
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(purpose: str = "default", config_override: Optional[dict] = None) -> BaseLLM:
    """
    Factory method to create an LLM instance.

    Returns an OllamaClient connected to the local Gemma 4 model.
    """
    config = _load_yaml("model_config.yaml")

    provider_name = os.getenv("MODEL_PROVIDER", config.get("active_provider", "ollama"))
    provider_config = config["providers"].get(provider_name)

    if not provider_config:
        raise ValueError(f"Unknown model provider: {provider_name}")

    # Build generation config from YAML
    gen_params = provider_config.get("generation_config", {})
    if config_override:
        gen_params.update(config_override)

    generation_config = GenerationConfig(
        temperature=gen_params.get("temperature", 0.10),
        top_p=gen_params.get("top_p", 0.80),
        top_k=gen_params.get("top_k", 30),
        max_output_tokens=gen_params.get("max_output_tokens", 4096),
        candidate_count=gen_params.get("candidate_count", 1),
    )

    model_id = os.getenv("OLLAMA_MODEL", provider_config["model_id"])

    # Determine base URL
    base_url_env = provider_config.get("base_url_env", "OLLAMA_BASE_URL")
    base_url = os.getenv(base_url_env, provider_config.get("base_url_default", "http://localhost:11434"))

    num_ctx = gen_params.get("num_ctx", 8192)

    # Instantiate Ollama client
    from .ollama_client import OllamaClient
    return OllamaClient(
        model_id=model_id,
        base_url=base_url,
        generation_config=generation_config,
        num_ctx=num_ctx,
    )


def get_adk_model_string() -> str:
    """Get the LiteLLM model string for ADK V2 integration."""
    config = _load_yaml("model_config.yaml")
    adk_config = config.get("adk", {})
    return adk_config.get("model_string", "ollama_chat/gemma4:e4b")


def load_config(filename: str) -> dict:
    """Public helper to load any config YAML."""
    return _load_yaml(filename)


def load_all_configs() -> dict:
    """Load all configuration files into a single dict."""
    configs = {
        "model": _load_yaml("model_config.yaml"),
        "agents": _load_yaml("agents_config.yaml"),
        "scoring": _load_yaml("scoring_config.yaml"),
        "prompts": _load_yaml("prompts_config.yaml"),
        "logging": _load_yaml("logging_config.yaml"),
    }
    # Optional configs
    ft_path = _CONFIG_DIR / "fine_tuning_config.yaml"
    if ft_path.exists():
        configs["fine_tuning"] = _load_yaml("fine_tuning_config.yaml")
    return configs
