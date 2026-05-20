"""
Lightweight semantic matching helpers.

The matcher is intentionally optional and CPU-safe:
- configured examples are embedded once at construction time;
- request-time work embeds only the incoming short message;
- missing sentence-transformers support degrades to lexical similarity.
"""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher
from typing import Any, Optional

logger = logging.getLogger("core.semantic_matcher")


class SemanticMatchResult(dict):
    """Dictionary result for easy JSON serialization."""


class SemanticFAQMatcher:
    """Cached FAQ semantic matcher with lexical fallback."""

    def __init__(self, examples_by_class: dict[str, list[str]], config: Optional[dict] = None):
        self.config = config or {}
        self.examples_by_class = {
            key: [str(item) for item in values if str(item).strip()]
            for key, values in (examples_by_class or {}).items()
            if isinstance(values, list)
        }
        self.enabled = bool(self.config.get("enabled", True))
        self.threshold = float(self.config.get("threshold", 0.62))
        self.high_confidence_threshold = float(self.config.get("high_confidence_threshold", 0.76))
        self.max_request_chars = int(self.config.get("max_request_chars", 300))
        self.lexical_fallback_enabled = bool(self.config.get("lexical_fallback_enabled", True))
        self.model_id = self.config.get("model_id", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._example_embeddings: dict[str, list[list[float]]] = {}
        self.mode = "disabled"

        if self.enabled:
            self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device="cpu")
            self._example_embeddings = {
                faq_class: self._embed_many(examples)
                for faq_class, examples in self.examples_by_class.items()
            }
            self.mode = "semantic"
            logger.info(
                "SemanticFAQMatcher initialized | model=%s classes=%d",
                self.model_id,
                len(self._example_embeddings),
            )
        except Exception as exc:
            self._model = None
            self._example_embeddings = {}
            self.mode = "lexical" if self.lexical_fallback_enabled else "disabled"
            logger.warning(
                "Semantic FAQ matcher unavailable; mode=%s error=%s",
                self.mode,
                exc,
            )

    def match(self, text: str) -> Optional[SemanticMatchResult]:
        query = str(text or "").strip()[: self.max_request_chars]
        if not query:
            return None

        if self.mode == "semantic" and self._model is not None:
            result = self._semantic_match(query)
            if result and result["score"] >= self.threshold:
                return result

        if self.lexical_fallback_enabled:
            result = self._lexical_match(query)
            if result and result["score"] >= self.threshold:
                return result

        return None

    def _semantic_match(self, query: str) -> Optional[SemanticMatchResult]:
        query_embedding = self._embed_one(query)
        best: Optional[SemanticMatchResult] = None
        for faq_class, embeddings in self._example_embeddings.items():
            examples = self.examples_by_class.get(faq_class, [])
            for index, embedding in enumerate(embeddings):
                score = _cosine(query_embedding, embedding)
                if best is None or score > best["score"]:
                    best = SemanticMatchResult(
                        faq_class=faq_class,
                        score=round(float(score), 4),
                        source="semantic",
                        example=examples[index] if index < len(examples) else "",
                    )
        return best

    def _lexical_match(self, query: str) -> Optional[SemanticMatchResult]:
        normalized_query = _normalize(query)
        best: Optional[SemanticMatchResult] = None
        for faq_class, examples in self.examples_by_class.items():
            for example in examples:
                normalized_example = _normalize(example)
                seq = SequenceMatcher(None, normalized_query, normalized_example).ratio()
                token_score = _token_overlap(normalized_query, normalized_example)
                score = max(seq, token_score)
                if best is None or score > best["score"]:
                    best = SemanticMatchResult(
                        faq_class=faq_class,
                        score=round(float(score), 4),
                        source="lexical",
                        example=example,
                    )
        return best

    def _embed_many(self, values: list[str]) -> list[list[float]]:
        if not values:
            return []
        encoded = self._model.encode(values, normalize_embeddings=True, show_progress_bar=False)
        return _to_vector_list(encoded)

    def _embed_one(self, value: str) -> list[float]:
        encoded = self._model.encode([value], normalize_embeddings=True, show_progress_bar=False)
        vectors = _to_vector_list(encoded)
        return vectors[0] if vectors else []


def _to_vector_list(encoded: Any) -> list[list[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list):
        return []
    if encoded and isinstance(encoded[0], (int, float)):
        return [[float(v) for v in encoded]]
    return [[float(v) for v in row] for row in encoded if isinstance(row, list)]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^0-9A-Za-z]+", " ", str(value or "").casefold()).split())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

