"""
Embedder — BAAI/bge-m3 embedding generation.

Creates 1024-dimensional vector representations for provider
chunks and user queries. Supports English, Urdu, and Roman Urdu.
Uses sentence-transformers for local inference.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("rag.embedder")

# Lazy load to avoid slow imports at startup
_model = None


def _get_model(model_name: str = "BAAI/bge-m3", device: str = "cpu"):
    """Lazy-load the embedding model."""
    global _model
    model_name = os.getenv("EMBEDDING_MODEL", model_name)
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name} on {device}")
        _model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 128,
    normalize: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    For BGE models, queries should be prefixed with
    "Represent this sentence:" for best results.

    Returns: np.ndarray of shape (len(texts), 1024)
    """
    model = _get_model(model_name, device)

    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 100,
    )
    logger.info(f"Generated embeddings: shape={embeddings.shape}")
    return embeddings


def embed_query(
    query: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate embedding for a single search query.

    Adds the BGE instruction prefix for retrieval queries.
    """
    model = _get_model(model_name, device)
    # BGE models work better with instruction prefix for queries
    instruction = "Represent this sentence for searching relevant service providers: "
    embedding = model.encode(
        instruction + query,
        normalize_embeddings=True,
    )
    return embedding


def save_embeddings(embeddings: np.ndarray, path: str):
    """Save embeddings to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info(f"Saved embeddings to {path}")


def load_embeddings(path: str) -> Optional[np.ndarray]:
    """Load pre-computed embeddings from disk."""
    p = Path(path)
    if p.exists():
        embeddings = np.load(str(p))
        logger.info(f"Loaded embeddings from {path}: shape={embeddings.shape}")
        return embeddings
    return None
