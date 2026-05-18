"""
Reranker — Cross-encoder reranking using BAAI/bge-reranker-base.

Takes top-K candidates from vector search and rescores them
with a cross-encoder for higher precision ranking.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("rag.reranker")

# Lazy load to avoid slow imports
_reranker = None


def _get_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    model_name = os.getenv("RERANKER_MODEL", model_name)
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {model_name} on {device}")
        _reranker = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded.")
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "text",
    top_n: int = 10,
    model_name: str = "BAAI/bge-reranker-base",
    device: str = "cpu",
) -> list[dict]:
    """
    Rerank candidates using cross-encoder scoring.

    Args:
        query: The user's search query
        candidates: List of dicts from vector search, each with a text field
        text_key: Key in each dict containing the text to rerank against
        top_n: Number of top results to return after reranking
        model_name: Cross-encoder model name
        device: "cpu" or "cuda"

    Returns:
        Reranked list of candidates (top_n), each with added 'rerank_score'
    """
    if not candidates:
        return []

    model = _get_reranker(model_name, device)

    # Build query-document pairs
    texts = [c.get(text_key, "") for c in candidates]
    pairs = [[query, text] for text in texts]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach scores and sort
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort by rerank score descending
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(candidates)} candidates → top {top_n} "
        f"(best score: {ranked[0]['rerank_score']:.4f})"
    )

    return ranked[:top_n]
