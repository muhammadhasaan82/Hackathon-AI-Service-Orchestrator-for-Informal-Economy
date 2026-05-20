"""Optional Google Maps Places enrichment."""

from __future__ import annotations

import os
from typing import Any


FALLBACK_MESSAGE = "Google Maps disabled or API key missing. Using mock provider dataset."


def google_maps_enabled() -> bool:
    enabled = os.getenv("ENABLE_GOOGLE_MAPS", "false").strip().lower() in {"1", "true", "yes", "on"}
    return enabled and bool(os.getenv("GOOGLE_MAPS_API_KEY", "").strip())


async def search_nearby_places(
    lat: float,
    lng: float,
    keyword: str,
    radius: int = 5000,
    limit: int = 5,
) -> dict[str, Any]:
    """Search Google Places Nearby Search API when explicitly enabled."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not google_maps_enabled():
        return {"enabled": False, "results": [], "message": FALLBACK_MESSAGE}

    params = {
        "location": f"{lat},{lng}",
        "radius": int(radius),
        "keyword": keyword,
        "key": api_key,
    }
    import httpx

    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("results", [])[:limit]:
        location = item.get("geometry", {}).get("location", {})
        results.append({
            "name": item.get("name"),
            "rating": item.get("rating"),
            "user_ratings_total": item.get("user_ratings_total"),
            "address": item.get("vicinity"),
            "vicinity": item.get("vicinity"),
            "lat": location.get("lat"),
            "lng": location.get("lng"),
            "place_id": item.get("place_id"),
            "source": "google_maps",
        })

    return {
        "enabled": True,
        "results": results,
        "message": "Google Maps Places results returned.",
    }
