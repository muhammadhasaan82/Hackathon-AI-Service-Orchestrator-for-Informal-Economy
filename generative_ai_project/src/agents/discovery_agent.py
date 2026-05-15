"""
Discovery Agent — Provider search via Agentic RAG.

Searches the 50K provider dataset using semantic similarity
+ metadata filtering to find relevant candidates.
"""

import logging
from typing import Optional

from ..rag.retriever import ProviderRetriever

logger = logging.getLogger("agents.discovery")


class DiscoveryAgent:
    """Searches for providers matching the user's intent."""

    def __init__(self, retriever: ProviderRetriever, agents_config: dict):
        self.retriever = retriever
        self.config = agents_config.get("agents", {}).get("discovery", {})
        self.search_config = self.config.get("search_config", {})

    async def discover(
        self,
        intent: dict,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> list[dict]:
        """
        Discover providers matching the extracted intent.

        Phase 1: Filtered search (category + city + availability)
        Phase 2: Fallback to broader search if no results
        """
        category = intent.get("service_type")
        city = intent.get("city")
        area = intent.get("area")
        price_pref = intent.get("price_preference")

        # Build availability filter from config
        avail_filter = self.search_config.get("availability_filter", {})
        include_avail = avail_filter.get("include", ["Available", "Available Soon"])

        max_candidates = self.search_config.get("max_candidates", 50)

        logger.info(
            f"Discovery: category={category} city={city} area={area} "
            f"availability={include_avail} max={max_candidates}"
        )

        # Build query string
        query_parts = []
        if category:
            query_parts.append(category)
        if area:
            query_parts.append(f"in {area}")
        if city:
            query_parts.append(city)
        query = " ".join(query_parts) if query_parts else "service provider"

        # Phase 1: Filtered search
        candidates = self.retriever.search(
            query=query,
            category=category,
            city=city,
            area=area,
            availability_filter=include_avail,
            price_range=None,  # don't filter by price, just preference
            user_lat=user_lat,
            user_lon=user_lon,
            n_results=max_candidates,
        )

        # Phase 2: Broader search if too few results
        if len(candidates) < 3:
            logger.info(f"Only {len(candidates)} results with filters. Trying broader search...")
            # Relax availability filter
            broader = self.retriever.search(
                query=query,
                category=category,
                city=city,
                area=None,  # remove area filter
                availability_filter=["Available", "Available Soon", "Busy"],
                user_lat=user_lat,
                user_lon=user_lon,
                n_results=max_candidates,
            )
            # Merge, avoiding duplicates
            seen_ids = {c["id"] for c in candidates}
            for b in broader:
                if b["id"] not in seen_ids:
                    candidates.append(b)
                    seen_ids.add(b["id"])

        # Phase 3: Absolute fallback
        if len(candidates) == 0:
            logger.warning("No results with any filters. Using fallback search.")
            candidates = self.retriever.search_fallback(
                query=f"{category or 'service provider'} {city or ''}",
                n_results=20,
            )

        logger.info(f"Discovery found {len(candidates)} candidates")
        return candidates
