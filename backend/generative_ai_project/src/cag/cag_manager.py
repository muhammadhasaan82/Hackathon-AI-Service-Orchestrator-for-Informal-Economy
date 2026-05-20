"""
CAG Manager — Cache-Augmented Generation.

Preloads static "Golden Knowledge" into the LLM's context window
using a high-performance Rust KV-cache (via PyO3). Falls back to
a Python dict if the Rust module isn't compiled.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("cag.manager")

_GOLDEN_KNOWLEDGE_PATH = Path(__file__).parent.parent.parent / "data" / "cache" / "golden_knowledge.yaml"
_AGENT_SECTIONS = {
    "intent": ["service_policies", "city_mappings", "emergency_handling", "language_conventions", "safety_guardrails"],
    "ranking": ["pricing_heuristics", "provider_ranking_rules", "city_mappings", "emergency_handling", "safety_guardrails"],
    "booking": ["service_policies", "pricing_heuristics", "escalation_rules", "emergency_handling"],
    "followup": ["service_policies", "emergency_handling", "escalation_rules", "language_conventions"],
    "orchestrator": ["service_policies", "city_mappings", "emergency_handling", "provider_ranking_rules", "safety_guardrails"],
    "faq": ["faq_policies", "service_policies", "pricing_heuristics", "language_conventions"],
}


class CAGManager:
    """
    Cache-Augmented Generation manager.

    Loads golden knowledge at startup and injects it into the
    LLM context window for every request.
    """

    def __init__(self, golden_knowledge_path: Optional[str] = None):
        self._cache = self._init_cache()
        self._entry_sections: dict[str, str] = {}
        self._raw_data: dict = {}
        self._load_golden_knowledge(golden_knowledge_path)

    def _init_cache(self):
        """Initialize cache — Rust KVCache if available, else Python dict."""
        try:
            from cag_cache import KVCache
            cache = KVCache()
            logger.info("CAG: Using Rust KVCache (PyO3) — high performance mode")
            return cache
        except ImportError:
            logger.warning(
                "CAG: Rust cag_cache module not found. Using Python fallback. "
                "To build: cd rust_cag && maturin develop --release"
            )
            return PythonKVCache()

    def _load_golden_knowledge(self, path: Optional[str] = None):
        """Load golden knowledge from YAML into the cache."""
        knowledge_path = Path(path) if path else _GOLDEN_KNOWLEDGE_PATH
        if not knowledge_path.exists():
            logger.warning(f"Golden knowledge file not found: {knowledge_path}")
            return

        with open(knowledge_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return
        self._raw_data = data

        entries = []
        for section, content in data.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    cache_key = f"{section}.{key}"
                    entries.append((cache_key, self._serialize_cache_value(value)))
                    self._entry_sections[cache_key] = section
            elif isinstance(content, list):
                for i, item in enumerate(content):
                    cache_key = f"{section}[{i}]"
                    entries.append((cache_key, self._serialize_cache_value(item)))
                    self._entry_sections[cache_key] = section
            else:
                entries.append((section, self._serialize_cache_value(content)))
                self._entry_sections[section] = section

        if hasattr(self._cache, "batch_set"):
            count = self._cache.batch_set(entries)
        else:
            for k, v in entries:
                self._cache.set(k, v)
            count = len(entries)

        logger.info(f"CAG: Loaded {count} golden knowledge entries")

    def _serialize_cache_value(self, value) -> str:
        """Serialize YAML values for Rust/Python KV caches without losing structure."""
        if isinstance(value, (dict, list)):
            return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).strip()
        return str(value)

    def get_context(
        self,
        keys: Optional[list[str]] = None,
        separator: str = "\n",
        max_chars: Optional[int] = None,
        max_items: Optional[int] = None,
        sections: Optional[list[str]] = None,
    ) -> str:
        """
        Get golden knowledge as a context string for LLM injection.

        If keys specified, returns only those entries.
        Otherwise, returns all cached knowledge.
        """
        if os.getenv("ENABLE_CAG", "true").strip().lower() not in ("1", "true", "yes", "on"):
            return ""

        if keys:
            if hasattr(self._cache, "batch_get"):
                results = self._cache.batch_get(keys)
                return self._format_entries(results, separator, max_chars, max_items, sections)
            else:
                parts = []
                for key in keys:
                    val = self._cache.get(key)
                    if val:
                        parts.append((key, val))
                return self._format_entries(parts, separator, max_chars, max_items, sections)
        else:
            if hasattr(self._cache, "get_all"):
                entries = self._cache.get_all()
            else:
                entries = []
            return self._format_entries(entries, separator, max_chars, max_items, sections)

    def get_context_for_agent(self, agent_name: str, separator: str = "\n") -> str:
        max_chars = int(os.getenv("CAG_MAX_CONTEXT_CHARS", "2500"))
        max_items = int(os.getenv("CAG_MAX_CONTEXT_ITEMS", "20"))
        sections = _AGENT_SECTIONS.get(agent_name, _AGENT_SECTIONS.get("orchestrator"))
        context = self.get_context(
            separator=separator,
            max_chars=max_chars,
            max_items=max_items,
            sections=sections,
        )
        logger.info(
            "CAG context prepared for %s | sections=%s chars=%s",
            agent_name,
            ",".join(sections or []),
            len(context),
        )
        return context

    def _format_entries(
        self,
        entries: list[tuple[str, str]],
        separator: str,
        max_chars: Optional[int],
        max_items: Optional[int],
        sections: Optional[list[str]],
    ) -> str:
        allowed_sections = set(sections or [])
        parts = []
        current_chars = 0
        for key, value in entries:
            section = self._entry_sections.get(key, key.split(".", 1)[0].split("[", 1)[0])
            if allowed_sections and section not in allowed_sections:
                continue
            part = f"[{key}]: {value}"
            projected = current_chars + len(part) + (len(separator) if parts else 0)
            if max_items is not None and len(parts) >= max_items:
                break
            if max_chars is not None and projected > max_chars:
                break
            parts.append(part)
            current_chars = projected
        return separator.join(parts)

    def set(self, key: str, value: str):
        """Add or update a cache entry."""
        self._cache.set(key, value)

    def get(self, key: str) -> Optional[str]:
        """Get a single cache entry."""
        return self._cache.get(key)

    def get_faq_policy(self, policy_key: str) -> Optional[str]:
        """Retrieve a specific FAQ policy entry by key (e.g. 'pricing_policy')."""
        structured = self.get_structured_faq_policy(policy_key)
        if structured.get("found") and isinstance(structured.get("policy"), dict):
            policy = structured["policy"]
            return (
                policy.get("short_answer")
                or policy.get("detailed_answer")
                or self._cache.get(f"faq_policies.{policy_key}")
            )
        return self._cache.get(f"faq_policies.{policy_key}")

    def get_structured_faq_policy(self, policy_key: str) -> dict:
        """Retrieve structured FAQ policy metadata for orchestration."""
        policy = (
            self._raw_data
            .get("faq_policies", {})
            .get(policy_key)
            if isinstance(self._raw_data.get("faq_policies"), dict)
            else None
        )
        if isinstance(policy, dict):
            return {
                "policy_key": policy_key,
                "policy": policy,
                "source": "cag",
                "structured": True,
                "found": True,
            }

        legacy = self._cache.get(f"faq_policies.{policy_key}")
        if legacy:
            return {
                "policy_key": policy_key,
                "policy": {
                    "short_answer": legacy,
                    "detailed_answer": legacy,
                    "escalation_required": False,
                    "dynamic_fields": [],
                    "related_topics": [],
                    "followup_prompts": [],
                    "safe_constraints": [],
                },
                "source": "cag",
                "structured": False,
                "found": True,
            }
        return {
            "policy_key": policy_key,
            "policy": None,
            "source": "cag",
            "structured": False,
            "found": False,
        }

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return self._cache.len()


class PythonKVCache:
    """Pure Python fallback for the Rust KVCache."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str):
        self._store[key] = value

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def batch_set(self, entries: list[tuple[str, str]]) -> int:
        for k, v in entries:
            self._store[k] = v
        return len(entries)

    def batch_get(self, keys: list[str]) -> list[tuple[str, str]]:
        return [(k, self._store[k]) for k in keys if k in self._store]

    def get_all(self) -> list[tuple[str, str]]:
        return list(self._store.items())

    def evict(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self):
        self._store.clear()

    def len(self) -> int:
        return len(self._store)

    def build_context(self, separator: str = "\n") -> str:
        return separator.join(f"[{k}]: {v}" for k, v in self._store.items())
