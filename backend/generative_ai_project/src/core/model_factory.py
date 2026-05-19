"""
Model Factory — Creates LLM instances from YAML configuration.

Reads model_config.yaml and instantiates the configured local LLM client.

Backend selection (priority order):
    1. MODEL_BACKEND env var          (preferred — new style)
    2. MODEL_PROVIDER env var         (legacy — kept for back-compat)
    3. active_provider in YAML        (config-level default)
    4. Hard-coded fallback            ("transformers" — CPU-safe)

Heavy backend modules (unsloth, transformers, ollama_client) are
imported lazily so the API can start on CPU-only machines without
pulling in CUDA-only dependencies.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from .base_llm import BaseLLM, GenerationConfig

logger = logging.getLogger("core.factory")

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

# Allowed backend names. ``ollama_large`` resolves to the same client as
# ``ollama`` but uses a different YAML provider section.
_VALID_BACKENDS = {"transformers", "ollama", "ollama_large", "unsloth"}


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file."""
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_backend(config: dict) -> str:
    """Resolve which backend to use based on env vars and YAML."""
    raw = (
        os.getenv("MODEL_BACKEND")
        or os.getenv("MODEL_PROVIDER")
        or config.get("active_provider")
        or "transformers"
    )
    backend = raw.strip().lower()

    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown MODEL_BACKEND='{backend}'. "
            f"Valid options: {sorted(_VALID_BACKENDS)}."
        )
    return backend


def _resolve_model_id(provider_config: dict, backend: str) -> str:
    """Resolve the model id, respecting MODEL_NAME, backend-specific env, then YAML."""
    # MODEL_NAME is the unified, backend-agnostic env var
    unified = os.getenv("MODEL_NAME")
    if unified:
        return unified.strip()

    # Fall back to backend-specific env vars for compatibility
    model_id_env = provider_config.get("model_id_env")
    if model_id_env:
        env_value = os.getenv(model_id_env, "").strip()
        if env_value:
            return env_value

    if backend.startswith("ollama"):
        env_value = os.getenv("OLLAMA_MODEL", "").strip()
        if env_value:
            return env_value

    # Fall back to YAML default
    return str(provider_config.get("model_id", "")).strip()


def _ensure_cuda_available(backend_label: str) -> None:
    """Raise a clear error if a GPU-only backend is selected on a CPU-only host."""
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"MODEL_BACKEND={backend_label} requires PyTorch. "
            "Install with `uv pip install torch`."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"MODEL_BACKEND={backend_label} requires a CUDA GPU but none was detected.\n"
            "  → For CPU-only machines, set MODEL_BACKEND=ollama (recommended) "
            "or MODEL_BACKEND=transformers in your .env file.\n"
            "  → Verify your GPU with `nvidia-smi` if you expected one to be available."
        )


def get_model(purpose: str = "default", config_override: Optional[dict] = None) -> BaseLLM:
    """
    Factory method to create an LLM instance.

    Returns the configured local LLM client. Defaults to the CPU-safe
    ``transformers`` backend so the API can start on CPU-only VMs.
    """
    config = _load_yaml("model_config.yaml")
    backend = _resolve_backend(config)

    providers = config.get("providers", {})
    provider_config = providers.get(backend)
    if not provider_config:
        # Fall back to a sane default config for backends that may not be
        # explicitly listed in YAML.
        provider_config = providers.get("transformers", {}) if backend == "transformers" else {}
        if not provider_config:
            raise ValueError(
                f"No provider configuration found for MODEL_BACKEND='{backend}'. "
                f"Add a `providers.{backend}` section in config/model_config.yaml."
            )

    # Build generation config from YAML
    gen_params = dict(provider_config.get("generation_config", {}))
    if config_override:
        gen_params.update(config_override)

    generation_config = GenerationConfig(
        temperature=gen_params.get("temperature", 0.10),
        top_p=gen_params.get("top_p", 0.80),
        top_k=gen_params.get("top_k", 30),
        max_output_tokens=gen_params.get("max_output_tokens", 4096),
        candidate_count=gen_params.get("candidate_count", 1),
    )

    model_id = _resolve_model_id(provider_config, backend)

    # ── transformers (CPU-safe default) ──────────────────────────
    if backend == "transformers":
        from .transformers_client import TransformersClient

        device = os.getenv("MODEL_DEVICE") or provider_config.get("device") or None
        torch_dtype = os.getenv("MODEL_DTYPE") or provider_config.get("torch_dtype") or None
        load_in_4bit_env = os.getenv("MODEL_LOAD_IN_4BIT")
        load_in_4bit = (
            load_in_4bit_env.lower() in ("1", "true", "yes", "on")
            if load_in_4bit_env is not None
            else bool(provider_config.get("load_in_4bit", False))
        )
        hf_token = os.getenv("HF_TOKEN") or None
        max_seq_length = int(
            os.getenv("MODEL_MAX_SEQ_LENGTH", gen_params.get("max_seq_length", 8192))
        )
        trust_remote_code = bool(provider_config.get("trust_remote_code", False))

        return TransformersClient(
            model_id=model_id,
            generation_config=generation_config,
            device=device,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
            max_seq_length=max_seq_length,
            trust_remote_code=trust_remote_code,
        )

    # ── ollama (recommended for CPU-only VMs) ────────────────────
    if backend.startswith("ollama"):
        from .ollama_client import OllamaClient

        base_url_env = provider_config.get("base_url_env", "OLLAMA_BASE_URL")
        base_url = os.getenv(
            base_url_env,
            provider_config.get("base_url_default", "http://localhost:11434"),
        )
        num_ctx = gen_params.get("num_ctx", 8192)

        return OllamaClient(
            model_id=model_id,
            base_url=base_url,
            generation_config=generation_config,
            num_ctx=num_ctx,
        )

    # ── unsloth (GPU-only — lazy imported, CUDA-checked) ─────────
    if backend == "unsloth":
        # Verify CUDA BEFORE importing unsloth, because unsloth itself
        # imports CUDA-bound code at module load time.
        _ensure_cuda_available("unsloth")

        from .unsloth_client import UnslothClient

        max_seq_length_env = provider_config.get("max_seq_length_env", "UNSLOTH_MAX_SEQ_LENGTH")
        load_in_4bit_env = provider_config.get("load_in_4bit_env", "UNSLOTH_LOAD_IN_4BIT")
        dtype_env = provider_config.get("dtype_env", "UNSLOTH_DTYPE")
        hf_token_env = provider_config.get("hf_token_env", "HF_TOKEN")

        max_seq_length = int(
            os.getenv(max_seq_length_env, gen_params.get("max_seq_length", 8192))
        )
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

    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")


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
        "guardrails": _load_yaml("guardrails_config.yaml"),
        "logging": _load_yaml("logging_config.yaml"),
    }
    # Optional configs
    ft_path = _CONFIG_DIR / "fine_tuning_config.yaml"
    if ft_path.exists():
        configs["fine_tuning"] = _load_yaml("fine_tuning_config.yaml")
    return configs
