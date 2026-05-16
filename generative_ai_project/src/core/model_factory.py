"""
Model Factory — Creates LLM instances from YAML configuration.

Reads model_config.yaml and instantiates the configured local LLM client.
Default runtime is Unsloth for local Gemma inference.
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

    Returns the configured local LLM client.
    """
    config = _load_yaml("model_config.yaml")

    provider_name = os.getenv("MODEL_PROVIDER", config.get("active_provider", "unsloth"))
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

    model_id_env = provider_config.get("model_id_env")
    if model_id_env:
        model_id = os.getenv(model_id_env, provider_config.get("model_id", "")).strip()
    elif provider_name.startswith("ollama"):
        model_id = os.getenv("OLLAMA_MODEL", provider_config["model_id"])
    else:
        model_id = provider_config.get("model_id", "").strip()

    if provider_name == "unsloth":
        from .unsloth_client import UnslothClient

        max_seq_length_env = provider_config.get("max_seq_length_env", "UNSLOTH_MAX_SEQ_LENGTH")
        load_in_4bit_env = provider_config.get("load_in_4bit_env", "UNSLOTH_LOAD_IN_4BIT")
        dtype_env = provider_config.get("dtype_env", "UNSLOTH_DTYPE")
        hf_token_env = provider_config.get("hf_token_env", "HF_TOKEN")

        max_seq_length = int(os.getenv(max_seq_length_env, gen_params.get("max_seq_length", 8192)))
        load_in_4bit = os.getenv(
            load_in_4bit_env,
            str(provider_config.get("load_in_4bit", True)),
        ).lower() in ("1", "true", "yes", "on")
        dtype = os.getenv(dtype_env, provider_config.get("dtype"))
        hf_token = os.getenv(hf_token_env) or None

        return UnslothClient(
            model_id=model_id,
            generation_config=generation_config,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
        )

    if provider_name.startswith("ollama"):
        base_url_env = provider_config.get("base_url_env", "OLLAMA_BASE_URL")
        base_url = os.getenv(base_url_env, provider_config.get("base_url_default", "http://localhost:11434"))
        num_ctx = gen_params.get("num_ctx", 8192)

        from .ollama_client import OllamaClient
        return OllamaClient(
            model_id=model_id,
            base_url=base_url,
            generation_config=generation_config,
            num_ctx=num_ctx,
        )

    raise ValueError(f"Unsupported model provider: {provider_name}")


def get_adk_model_string() -> str:
    """Get the LiteLLM model string for ADK V2 integration."""
    config = _load_yaml("model_config.yaml")
    adk_config = config.get("adk", {})
    return adk_config.get("model_string", "")


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
