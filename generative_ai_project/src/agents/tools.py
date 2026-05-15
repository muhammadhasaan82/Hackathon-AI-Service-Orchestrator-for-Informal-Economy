"""
Agent Tools — Function definitions for agent tool use.

Each tool is a standalone function that agents can call.
Tools encapsulate deterministic logic (search, score, book)
while the LLM handles probabilistic reasoning.
"""

import logging
import math
import time
import uuid
from typing import Optional

logger = logging.getLogger("agents.tools")


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in kilometers between two coordinates."""
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
    return round(R * c, 2)


def score_provider(provider: dict, scoring_config: dict, user_lat: Optional[float] = None, user_lon: Optional[float] = None, price_preference: Optional[str] = None) -> dict:
    """
    Compute composite score for a provider using configurable weights.

    This is deterministic — same inputs always produce same score.
    The LLM then reasons about these scores to generate recommendations.
    """
    weights = scoring_config.get("weights", {})
    avail_scores = scoring_config.get("availability_scores", {})
    dist_config = scoring_config.get("distance", {})
    rating_config = scoring_config.get("rating", {})
    exp_config = scoring_config.get("experience", {})
    resp_config = scoring_config.get("response_time", {})

    meta = provider.get("metadata", provider)

    # ── Distance Score ──────────────────────────────────────
    if user_lat is not None and user_lon is not None:
        dist_km = calculate_distance(
            user_lat, user_lon,
            meta.get("latitude", 0), meta.get("longitude", 0),
        )
        decay = dist_config.get("decay_constant_km", 5.0)
        max_dist = dist_config.get("max_distance_km", 50.0)
        distance_score = math.exp(-dist_km / decay) if dist_km <= max_dist else 0.0
    else:
        dist_km = dist_config.get("default_distance_km", 10.0)
        distance_score = 0.5  # neutral when no location

    # ── Rating Score ────────────────────────────────────────
    rating = meta.get("rating", 3.0)
    min_r = rating_config.get("min_value", 1.0)
    max_r = rating_config.get("max_value", 5.0)
    rating_score = (rating - min_r) / (max_r - min_r)

    # ── Availability Score ──────────────────────────────────
    availability = meta.get("availability", "Offline")
    availability_score = avail_scores.get(availability, 0.0)

    # ── Experience Score ────────────────────────────────────
    exp_years = meta.get("experience_years", 0)
    max_cap = exp_config.get("max_years_cap", 20)
    experience_score = min(exp_years / max_cap, 1.0)

    # ── Response Time Score ─────────────────────────────────
    resp_time = meta.get("response_time_minutes", 60)
    ideal = resp_config.get("ideal_minutes", 5)
    max_resp = resp_config.get("max_minutes", 120)
    if resp_time <= ideal:
        response_score = 1.0
    elif resp_time >= max_resp:
        response_score = 0.0
    else:
        response_score = max(0, 1.0 - (resp_time - ideal) / (max_resp - ideal))

    # ── Composite Score ─────────────────────────────────────
    composite = (
        weights.get("distance", 0.30) * distance_score
        + weights.get("rating", 0.25) * rating_score
        + weights.get("availability", 0.20) * availability_score
        + weights.get("experience", 0.15) * experience_score
        + weights.get("response_time", 0.10) * response_score
    )

    # ── Verification Bonus ──────────────────────────────────
    if meta.get("verified_provider", False):
        composite += scoring_config.get("verification_bonus", 0.05)

    # ── Price Preference Modifier ───────────────────────────
    if price_preference:
        price_range = meta.get("price_range", "Medium")
        modifiers = scoring_config.get("price_modifiers", {}).get(price_range, {})
        pref_key = {
            "budget": "budget_conscious",
            "moderate": "budget_conscious",
            "premium": "premium_seeker",
        }.get(price_preference, "budget_conscious")
        composite += modifiers.get(pref_key, 0.0)

    composite = max(0.0, min(1.0, composite))

    return {
        "composite_score": round(composite, 4),
        "breakdown": {
            "distance": {"score": round(distance_score, 4), "weight": weights.get("distance", 0.30), "distance_km": round(dist_km, 2) if user_lat else None},
            "rating": {"score": round(rating_score, 4), "weight": weights.get("rating", 0.25), "value": rating},
            "availability": {"score": round(availability_score, 4), "weight": weights.get("availability", 0.20), "status": availability},
            "experience": {"score": round(experience_score, 4), "weight": weights.get("experience", 0.15), "years": exp_years},
            "response_time": {"score": round(response_score, 4), "weight": weights.get("response_time", 0.10), "minutes": resp_time},
        },
        "verified": meta.get("verified_provider", False),
    }


def generate_time_slot(time_preference: Optional[str] = None) -> str:
    """Generate a simulated time slot based on preference."""
    import datetime
    now = datetime.datetime.now()

    if time_preference:
        pref_lower = time_preference.lower()
        if any(w in pref_lower for w in ["abhi", "now", "asap", "urgent"]):
            slot = now + datetime.timedelta(minutes=30)
        elif any(w in pref_lower for w in ["kal", "tomorrow"]):
            slot = now + datetime.timedelta(days=1)
            if any(w in pref_lower for w in ["subah", "morning"]):
                slot = slot.replace(hour=10, minute=0)
            elif any(w in pref_lower for w in ["dopahar", "afternoon"]):
                slot = slot.replace(hour=14, minute=0)
            elif any(w in pref_lower for w in ["shaam", "evening"]):
                slot = slot.replace(hour=18, minute=0)
            else:
                slot = slot.replace(hour=10, minute=0)
        elif any(w in pref_lower for w in ["morning", "subah"]):
            slot = now.replace(hour=10, minute=0) if now.hour < 10 else now + datetime.timedelta(days=1)
            slot = slot.replace(hour=10, minute=0)
        elif any(w in pref_lower for w in ["evening", "shaam"]):
            slot = now.replace(hour=18, minute=0) if now.hour < 18 else now + datetime.timedelta(days=1)
            slot = slot.replace(hour=18, minute=0)
        else:
            slot = now + datetime.timedelta(hours=2)
    else:
        slot = now + datetime.timedelta(hours=2)

    return slot.strftime("%Y-%m-%d %I:%M %p")


def format_provider_summary(provider: dict, score_result: Optional[dict] = None) -> str:
    """Format a provider as a human-readable summary."""
    meta = provider.get("metadata", provider)
    lines = [
        f"📌 {meta.get('provider_name', 'Unknown')}",
        f"   Category: {meta.get('category', 'N/A')}",
        f"   Location: {meta.get('area', '')}, {meta.get('city', '')}",
        f"   Rating: ⭐ {meta.get('rating', 'N/A')}/5.0",
        f"   Experience: {meta.get('experience_years', 0)} years ({meta.get('completed_jobs', 0)} jobs)",
        f"   Availability: {meta.get('availability', 'Unknown')}",
        f"   Response Time: {meta.get('response_time_minutes', 'N/A')} min",
        f"   Price: {meta.get('price_range', 'N/A')}",
        f"   Verified: {'✅ Yes' if meta.get('verified_provider') else '❌ No'}",
        f"   Languages: {meta.get('languages_supported', 'N/A')}",
    ]
    if score_result:
        dist = score_result["breakdown"]["distance"].get("distance_km")
        if dist is not None:
            lines.append(f"   Distance: {dist} km")
        lines.append(f"   Match Score: {score_result['composite_score']:.0%}")
    return "\n".join(lines)
