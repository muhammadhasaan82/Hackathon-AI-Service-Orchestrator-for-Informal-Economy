"""
Retriever — 3-phase Agentic RAG retrieval.

Phase 1: Coarse metadata filter (city, category, availability)
Phase 2: Semantic similarity via BGE embeddings + Weaviate
Phase 3: Cross-encoder reranking for precision
"""

import logging
import math
from typing import Optional

from .embedder import embed_query
from .vector_store import WeaviateVectorStore
from .reranker import rerank

logger = logging.getLogger("rag.retriever")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class ProviderRetriever:
    """
    3-phase provider retrieval:
    1. Coarse filter: Weaviate metadata filter (city, category, availability)
    2. Semantic: BGE embedding similarity via Weaviate vector search
    3. Rerank: Cross-encoder rescoring of top-K results
    """

    def __init__(self, vector_store: WeaviateVectorStore, config: Optional[dict] = None):
        self.vector_store = vector_store
        self.config = config or {}
        # Reranking config from model_config.yaml
        model_cfg = self.config.get("model", {})
        self.reranking_config = model_cfg.get("reranking", {})

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        city: Optional[str] = None,
        area: Optional[str] = None,
        availability_filter: Optional[list[str]] = None,
        price_range: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        n_results: int = 20,
    ) -> list[dict]:
        """
        3-phase search for providers.

        Returns list of provider dicts with retrieval_score, rerank_score,
        and distance_km fields.
        """
        if availability_filter is None:
            agent_config = self.config.get("agents", {}).get("agents", {}).get("discovery", {})
            search_conf = agent_config.get("search_config", {})
            avail_conf = search_conf.get("availability_filter", {})
            availability_filter = avail_conf.get("include", ["Available", "Available Soon"])

        # Build semantic query
        search_query = query
        if category:
            search_query = f"{category} service provider"
        if city:
            search_query += f" in {city}"
        if area:
            search_query += f" {area}"

        logger.info(f"Retriever search: query='{search_query}' category={category} city={city}")

        # ── Phase 1+2: Weaviate hybrid search (metadata filter + vector) ──
        query_embedding = embed_query(search_query)
        coarse_k = self.reranking_config.get("top_k_input", 50)

        results = self.vector_store.hybrid_search(
            query_embedding=query_embedding.tolist(),
            category=category,
            city=city,
            area=area,
            availability=availability_filter,
            price_range=price_range,
            n_results=coarse_k,
        )

        # Add retrieval scores
        for result in results:
            result["retrieval_score"] = 1.0 - result.get("distance", 0.0)

        # ── Phase 3: Cross-encoder reranking ──────────────────────────────
        rerank_enabled = self.reranking_config.get("enabled", True)
        rerank_top_n = self.reranking_config.get("top_n_output", 10)

        if rerank_enabled and len(results) > 1:
            rerank_model = self.reranking_config.get("model_name", "BAAI/bge-reranker-base")
            rerank_device = self.reranking_config.get("device", "cpu")

            results = rerank(
                query=search_query,
                candidates=results,
                text_key="text",
                top_n=min(rerank_top_n, n_results),
                model_name=rerank_model,
                device=rerank_device,
            )
        else:
            results = results[:n_results]

        # Enrich with distance if user coordinates provided
        for result in results:
            meta = result.get("metadata", {})
            if user_lat is not None and user_lon is not None:
                provider_lat = meta.get("latitude", 0.0)
                provider_lon = meta.get("longitude", 0.0)
                result["distance_km"] = haversine_distance(
                    user_lat, user_lon, provider_lat, provider_lon
                )
            else:
                result["distance_km"] = None

        logger.info(f"Retriever returned {len(results)} candidates (reranked={rerank_enabled})")
        return results

    def search_fallback(self, query: str, n_results: int = 20) -> list[dict]:
        """Fallback search without filters."""
        logger.info(f"Fallback search: query='{query}'")
        query_embedding = embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            n_results=n_results,
        )
        for result in results:
            result["retrieval_score"] = 1.0 - result.get("distance", 0.0)
            result["distance_km"] = None
        return results
