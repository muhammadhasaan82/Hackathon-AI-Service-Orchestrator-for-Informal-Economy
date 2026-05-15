"""
Indexer — Index all 50K providers into Weaviate vector store.
"""

import logging
from pathlib import Path
from typing import Optional

from ..processing.preprocessor import load_providers
from ..processing.chunking import build_chunks
from .embedder import embed_texts, save_embeddings, load_embeddings
from .vector_store import WeaviateVectorStore

logger = logging.getLogger("rag.indexer")

_EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "embeddings" / "provider_embeddings.npy"


def index_providers(
    csv_path: Optional[str] = None,
    force_rebuild: bool = False,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 128,
    device: str = "cpu",
) -> WeaviateVectorStore:
    """
    Full indexing pipeline:
    1. Load CSV → DataFrame
    2. Build text chunks
    3. Generate or load embeddings
    4. Upsert into Weaviate

    Returns the populated WeaviateVectorStore.
    """
    store = WeaviateVectorStore()

    if store.count > 0 and not force_rebuild:
        logger.info(f"Weaviate already contains {store.count} documents. Skipping indexing.")
        return store

    if force_rebuild and store.count > 0:
        logger.info("Force rebuild: deleting existing collection")
        store.delete_collection()
        store = WeaviateVectorStore()

    # Step 1: Load and preprocess
    df = load_providers(csv_path)

    # Step 2: Build chunks
    chunks = build_chunks(df)

    # Step 3: Generate or load embeddings
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = None
    if not force_rebuild:
        embeddings = load_embeddings(str(_EMBEDDINGS_PATH))

    if embeddings is None or len(embeddings) != len(texts):
        logger.info("Generating embeddings (this may take several minutes for 50K records)...")
        embeddings = embed_texts(
            texts,
            model_name=embedding_model,
            batch_size=batch_size,
            device=device,
        )
        save_embeddings(embeddings, str(_EMBEDDINGS_PATH))

    # Step 4: Insert into Weaviate
    store.add_documents(
        ids=ids,
        texts=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    logger.info(f"Indexing complete: {store.count} documents in Weaviate")
    return store
