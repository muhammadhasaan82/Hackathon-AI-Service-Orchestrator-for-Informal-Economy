import os
import contextvars
import logging
from typing import Optional

logger = logging.getLogger("core.runtime_config")
_llm_calls_this_request: contextvars.ContextVar[int] = contextvars.ContextVar(
    "llm_calls_this_request",
    default=0,
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def active_model_backend(default: str = "ollama") -> str:
    return (os.getenv("MODEL_PROVIDER") or os.getenv("MODEL_BACKEND") or default).strip().lower()


def reset_llm_call_budget() -> None:
    _llm_calls_this_request.set(0)


def consume_llm_call_budget(call_site: str, backend: Optional[str] = None) -> bool:
    model_backend = (backend or active_model_backend()).strip().lower()
    if not model_backend.startswith("ollama"):
        return True

    max_calls = env_int("MAX_OLLAMA_CALLS_PER_REQUEST", 1)
    used = _llm_calls_this_request.get()
    if used >= max_calls:
        logger.info(
            "Skipping Ollama call for %s because request budget is exhausted (%s/%s)",
            call_site,
            used,
            max_calls,
        )
        return False

    _llm_calls_this_request.set(used + 1)
    logger.info(
        "Ollama call budget consumed by %s (%s/%s)",
        call_site,
        used + 1,
        max_calls,
    )
    return True
