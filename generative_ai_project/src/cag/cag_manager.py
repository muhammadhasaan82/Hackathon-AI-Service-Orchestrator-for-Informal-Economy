"""
CAG Manager — Cache-Augmented Generation.

Preloads static "Golden Knowledge" into the LLM's context window
using a high-performance Rust KV-cache (via PyO3). Falls back to
a Python dict if the Rust module isn't compiled.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("cag.manager")

_GOLDEN_KNOWLEDGE_PATH = Path(__file__).parent.parent.parent / "data" / "cache" / "golden_knowledge.yaml"


class CAGManager:
    """
    Cache-Augmented Generation manager.

    Loads golden knowledge at startup and injects it into the
    LLM context window for every request.
    """

    def __init__(self, golden_knowledge_path: Optional[str] = None):
        self._cache = self._init_cache()
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

        entries = []
        for section, content in data.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    entries.append((f"{section}.{key}", str(value)))
            elif isinstance(content, list):
                for i, item in enumerate(content):
                    entries.append((f"{section}[{i}]", str(item)))
            else:
                entries.append((section, str(content)))

        if hasattr(self._cache, "batch_set"):
            count = self._cache.batch_set(entries)
        else:
            for k, v in entries:
                self._cache.set(k, v)
            count = len(entries)

        logger.info(f"CAG: Loaded {count} golden knowledge entries")

    def get_context(self, keys: Optional[list[str]] = None, separator: str = "\n") -> str:
        """
        Get golden knowledge as a context string for LLM injection.

        If keys specified, returns only those entries.
        Otherwise, returns all cached knowledge.
        """
        if keys:
            if hasattr(self._cache, "batch_get"):
                results = self._cache.batch_get(keys)
                return separator.join(f"[{k}]: {v}" for k, v in results)
            else:
                parts = []
                for key in keys:
                    val = self._cache.get(key)
                    if val:
                        parts.append(f"[{key}]: {val}")
                return separator.join(parts)
        else:
            if hasattr(self._cache, "build_context"):
                return self._cache.build_context(separator)
            else:
                return self._cache.build_context(separator)

    def set(self, key: str, value: str):
        """Add or update a cache entry."""
        self._cache.set(key, value)

    def get(self, key: str) -> Optional[str]:
        """Get a single cache entry."""
        return self._cache.get(key)

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
