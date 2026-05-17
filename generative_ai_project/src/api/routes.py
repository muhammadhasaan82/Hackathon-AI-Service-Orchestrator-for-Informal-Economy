"""
API Routes — Complete endpoint definitions for mobile application.

Groups:
    /api/v1/chat           — Conversational AI
    /api/v1/providers      — Browse, search, detail
    /api/v1/bookings       — CRUD + history
    /api/v1/sessions       — Session management
    /api/v1/discovery      — Categories, cities, areas
    /api/v1/health         — System health
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    ChatRequest, ChatResponse, HealthResponse,
    ProviderSearchRequest, ProviderListResponse, ProviderDetailResponse,
    CreateBookingRequest, UpdateBookingRequest, BookingListResponse,
    SessionResponse, CategoryInfo, CityInfo,
)

logger = logging.getLogger("api.routes")

# Lazy import for batch distance calculation (avoids circular import at module level)
# We import inside the endpoint functions where needed.

router = APIRouter(prefix="/api/v1", tags=["Service Orchestrator"])

# ── Dependencies set by app.py during lifespan ──────────────
_orchestrator = None
_vector_store = None
_session_store = None
_start_time = None


def set_dependencies(orchestrator, vector_store, session_store, start_time):
    global _orchestrator, _vector_store, _session_store, _start_time
    _orchestrator = orchestrator
    _vector_store = vector_store
    _session_store = session_store
    _start_time = start_time


def _check_init():
    if _orchestrator is None:
        raise HTTPException(503, detail="System not initialized. Ensure Unsloth model is loaded and Weaviate is running.")


# ═══════════════════════════════════════════════════════════════
# 1. CHAT — Conversational AI
# ═══════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint for the mobile app.

    Send a natural language message (English, Urdu, or Roman Urdu).
    Returns: AI response + booking details + provider recommendations.
    """
    _check_init()
    session_id = request.session_id or str(uuid.uuid4())

    result = await _orchestrator.process_message(
        session_id=session_id,
        user_message=request.message,
        user_lat=request.user_lat,
        user_lon=request.user_lon,
    )

    return ChatResponse(
        response=result.get("response", ""),
        session_id=session_id,
        status=result.get("status", "unknown"),
        agent_trace=result.get("agent_trace", []),
        booking=result.get("booking"),
        followup=result.get("followup"),
        providers=result.get("providers"),
        intent=result.get("intent"),
        latency_ms=result.get("latency_ms"),
    )


# ═══════════════════════════════════════════════════════════════
# 2. PROVIDERS — Browse, Search, Detail
# ═══════════════════════════════════════════════════════════════

@router.post("/providers/search", response_model=ProviderListResponse)
async def search_providers(request: ProviderSearchRequest):
    """
    Search providers with filters.

    Mobile app uses this for the "Browse" and "Search" screens.
    Supports filtering by category, city, area, rating, price range,
    and sorting by score, rating, distance, or price.
    """
    _check_init()
    from ..rag.embedder import embed_query

    # Build search query from filters
    search_parts = []
    if request.query:
        search_parts.append(request.query)
    if request.category:
        search_parts.append(f"{request.category} service provider")
    if request.city:
        search_parts.append(f"in {request.city}")
    if request.area:
        search_parts.append(request.area)

    search_query = " ".join(search_parts) if search_parts else "service provider"
    query_embedding = embed_query(search_query)

    # Search Weaviate
    results = _vector_store.hybrid_search(
        query_embedding=query_embedding.tolist(),
        category=request.category,
        city=request.city,
        area=request.area,
        price_range=request.price_range,
        availability=[request.availability] if request.availability else None,
        n_results=request.top_n * request.page,
    )

    # Post-processing filters
    filtered = results
    if request.min_rating:
        filtered = [r for r in filtered if r.get("metadata", {}).get("rating", 0) >= request.min_rating]
    if request.verified_only:
        filtered = [r for r in filtered if r.get("metadata", {}).get("verified_provider", False)]

    # Add distance if user coordinates provided
    # THREADING: batch_calculate_distances runs haversine for all
    # providers concurrently. This is the main CPU cost in this
    # endpoint when processing 100+ results from Weaviate.
    if request.user_lat and request.user_lon:
        from ..agents.tools import batch_calculate_distances
        distances = batch_calculate_distances(
            filtered, request.user_lat, request.user_lon,
        )
        for r, dist in zip(filtered, distances):
            r["distance_km"] = dist

    # Sort
    sort_key = {
        "score": lambda x: x.get("rerank_score", x.get("retrieval_score", 0)),
        "rating": lambda x: x.get("metadata", {}).get("rating", 0),
        "distance": lambda x: x.get("distance_km", 9999),
        "price": lambda x: {"Low": 1, "Medium": 2, "High": 3}.get(
            x.get("metadata", {}).get("price_range", "Medium"), 2
        ),
    }.get(request.sort_by, lambda x: x.get("rerank_score", 0))

    reverse = request.sort_by not in ("distance", "price")
    filtered.sort(key=sort_key, reverse=reverse)

    # Paginate
    per_page = request.top_n
    start = (request.page - 1) * per_page
    page_results = filtered[start:start + per_page]

    # Format for mobile
    providers = []
    for r in page_results:
        meta = r.get("metadata", {})
        providers.append({
            "provider_id": meta.get("provider_id"),
            "provider_name": meta.get("provider_name", "Unknown"),
            "category": meta.get("category", ""),
            "city": meta.get("city", ""),
            "area": meta.get("area", ""),
            "rating": meta.get("rating", 0),
            "availability": meta.get("availability", "Unknown"),
            "experience_years": meta.get("experience_years", 0),
            "price_range": meta.get("price_range", "N/A"),
            "verified_provider": meta.get("verified_provider", False),
            "response_time_minutes": meta.get("response_time_minutes", 0),
            "distance_km": r.get("distance_km"),
            "score": r.get("rerank_score", r.get("retrieval_score", 0)),
        })

    return ProviderListResponse(
        providers=providers,
        total=len(filtered),
        page=request.page,
        per_page=per_page,
        has_next=start + per_page < len(filtered),
    )


@router.get("/providers/{provider_id}")
async def get_provider_detail(provider_id: int):
    """
    Get detailed information about a specific provider.

    Mobile app uses this for the "Provider Detail" screen.
    """
    _check_init()
    from ..rag.embedder import embed_query

    results = _vector_store.search(
        query_embedding=embed_query(f"provider {provider_id}").tolist(),
        n_results=100,
    )

    for r in results:
        meta = r.get("metadata", {})
        if meta.get("provider_id") == provider_id:
            return {
                "provider_id": meta.get("provider_id"),
                "provider_name": meta.get("provider_name"),
                "category": meta.get("category"),
                "city": meta.get("city"),
                "area": meta.get("area"),
                "full_location": meta.get("full_location", ""),
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "rating": meta.get("rating"),
                "availability": meta.get("availability"),
                "experience_years": meta.get("experience_years"),
                "completed_jobs": meta.get("completed_jobs"),
                "response_time_minutes": meta.get("response_time_minutes"),
                "price_range": meta.get("price_range"),
                "verified_provider": meta.get("verified_provider"),
                "languages_supported": meta.get("languages_supported", ""),
                "phone_number": meta.get("phone_number"),
                "email": meta.get("email"),
            }

    raise HTTPException(404, detail=f"Provider {provider_id} not found")


@router.get("/providers/nearby")
async def get_nearby_providers(
    lat: float = Query(..., description="User latitude"),
    lon: float = Query(..., description="User longitude"),
    radius_km: float = Query(10.0, description="Search radius in km"),
    category: Optional[str] = Query(None, description="Filter by category"),
    top_n: int = Query(20, ge=1, le=50),
):
    """
    Find providers near the user's GPS location.

    Mobile app uses this for the "Nearby" screen with location permissions.
    """
    _check_init()
    from ..rag.embedder import embed_query

    search_query = f"{category + ' ' if category else ''}service provider nearby"
    query_embedding = embed_query(search_query)

    results = _vector_store.hybrid_search(
        query_embedding=query_embedding.tolist(),
        category=category,
        n_results=200,
    )

    # Filter by radius
    # THREADING: Calculate distances for all results concurrently,
    # then filter by radius. This avoids a serial loop over 200
    # results (the max Weaviate returns for nearby search).
    from ..agents.tools import batch_calculate_distances
    all_distances = batch_calculate_distances(results, lat, lon)

    nearby = []
    for r, dist in zip(results, all_distances):
        if dist <= radius_km:
            r["distance_km"] = round(dist, 2)
            nearby.append(r)

    # Sort by distance
    nearby.sort(key=lambda x: x["distance_km"])
    nearby = nearby[:top_n]

    providers = []
    for r in nearby:
        meta = r.get("metadata", {})
        providers.append({
            "provider_id": meta.get("provider_id"),
            "provider_name": meta.get("provider_name"),
            "category": meta.get("category"),
            "city": meta.get("city"),
            "area": meta.get("area"),
            "rating": meta.get("rating"),
            "availability": meta.get("availability"),
            "price_range": meta.get("price_range"),
            "verified_provider": meta.get("verified_provider"),
            "distance_km": r["distance_km"],
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
        })

    return {"providers": providers, "total": len(providers), "radius_km": radius_km}


# ═══════════════════════════════════════════════════════════════
# 3. BOOKINGS — CRUD + History
# ═══════════════════════════════════════════════════════════════

@router.post("/bookings")
async def create_booking(request: CreateBookingRequest):
    """
    Create a new booking directly (without going through chat).

    Mobile app uses this from the "Book Now" button on provider detail screen.
    """
    _check_init()
    provider_meta = {"provider_id": request.provider_id, "provider_name": "Provider"}

    # Try to find provider details from vector store
    from ..rag.embedder import embed_query
    results = _vector_store.search(
        query_embedding=embed_query(f"provider {request.provider_id}").tolist(),
        n_results=100,
    )
    for r in results:
        meta = r.get("metadata", {})
        if meta.get("provider_id") == request.provider_id:
            provider_meta = meta
            break

    booking = _orchestrator.booking_store.create_booking(
        session_id=request.session_id,
        provider=provider_meta,
        service_type=request.service_type,
        location=request.location,
        scheduled_time=request.scheduled_time,
        user_notes=request.user_notes,
        status="CONFIRMED",
    )

    return booking


@router.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    """Get booking details by ID."""
    _check_init()
    booking = _orchestrator.booking_store.get_booking(booking_id)
    if not booking:
        raise HTTPException(404, detail="Booking not found")
    return booking


@router.patch("/bookings/{booking_id}")
async def update_booking(booking_id: str, request: UpdateBookingRequest):
    """
    Update booking details (reschedule or add notes).

    Mobile app uses this for the "Edit Booking" screen.
    """
    _check_init()
    booking = _orchestrator.booking_store.get_booking(booking_id)
    if not booking:
        raise HTTPException(404, detail="Booking not found")

    if request.status:
        valid = _orchestrator.booking_store.update_status(
            booking_id, request.status, f"Updated via mobile: {request.user_notes or ''}"
        )
        if not valid:
            raise HTTPException(400, detail=f"Invalid status transition to '{request.status}'")

    return _orchestrator.booking_store.get_booking(booking_id)


@router.post("/bookings/{booking_id}/cancel")
async def cancel_booking(booking_id: str):
    """Cancel a booking."""
    _check_init()
    success = _orchestrator.booking_store.update_status(
        booking_id, "CANCELLED", "Cancelled by user via mobile app"
    )
    if not success:
        raise HTTPException(404, detail="Booking not found")
    return {"booking_id": booking_id, "status": "CANCELLED"}


@router.get("/bookings/session/{session_id}", response_model=BookingListResponse)
async def get_session_bookings(session_id: str):
    """
    Get all bookings for a session/user.

    Mobile app uses this for the "My Bookings" screen.
    """
    _check_init()
    bookings = _orchestrator.booking_store.get_bookings_by_session(session_id)
    return BookingListResponse(bookings=bookings, total=len(bookings))


# ═══════════════════════════════════════════════════════════════
# 4. SESSIONS — Session Management
# ═══════════════════════════════════════════════════════════════

@router.post("/sessions")
async def create_session():
    """
    Create a new session.

    Mobile app calls this on first launch to get a session_id.
    Store this session_id persistently on the device.
    """
    if _session_store is None:
        raise HTTPException(503, detail="System not initialized")

    session_id = str(uuid.uuid4())
    session = _session_store.create_session(session_id)
    return {"session_id": session_id, "created": True}


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session state and metadata."""
    if _session_store is None:
        raise HTTPException(503, detail="System not initialized")

    session = _session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    meta = session.get("metadata", {})
    return SessionResponse(
        session_id=session_id,
        turn_count=meta.get("turn_count", 0),
        created_at=meta.get("created_at", 0),
        last_active=meta.get("last_active", 0),
        current_intent=session.get("agent_state", {}).get("intent"),
    )


@router.get("/sessions/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    max_turns: int = Query(20, ge=1, le=100),
):
    """
    Get conversation history for a session.

    Mobile app uses this to restore the chat screen after app restart.
    """
    if _session_store is None:
        raise HTTPException(503, detail="System not initialized")

    history = _session_store.get_history(session_id, max_turns=max_turns)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history."""
    if _session_store is None:
        raise HTTPException(503, detail="System not initialized")

    _session_store.delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


@router.get("/agents/trace/{session_id}")
async def get_trace(session_id: str):
    """Get the full reasoning trace for a session (debug/transparency)."""
    if _session_store is None:
        raise HTTPException(503, detail="System not initialized")

    session = _session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    return {
        "session_id": session_id,
        "reasoning_trace": session.get("reasoning_trace", []),
        "conversation_history": session.get("conversation_history", []),
        "agent_state": session.get("agent_state", {}),
    }


# ═══════════════════════════════════════════════════════════════
# 5. DISCOVERY — Categories, Cities, Areas
# ═══════════════════════════════════════════════════════════════

CATEGORY_ICONS = {
    "Plumber": "🔧", "Electrician": "⚡", "AC Technician": "❄️",
    "Beautician": "💄", "Tutor": "📚", "Home Tutor": "📖",
    "Carpenter": "🪚", "Mechanic": "🔩", "Cleaning Service": "🧹",
    "Appliance Repair": "🔌", "Mobile Repair": "📱",
    "Computer Technician": "💻", "Painter": "🎨",
    "Water Tank Cleaner": "💧",
}


@router.get("/discovery/categories")
async def get_categories():
    """
    Get all service categories with metadata.

    Mobile app uses this for the home screen category grid.
    """
    _check_init()
    categories = _orchestrator.service_categories
    result = []
    for cat in categories:
        result.append({
            "name": cat,
            "icon": CATEGORY_ICONS.get(cat, "🔧"),
            "description": None,
        })
    return {"categories": result, "total": len(result)}


@router.get("/discovery/cities")
async def get_cities():
    """
    Get all served cities with their areas.

    Mobile app uses this for city/area dropdowns.
    """
    _check_init()
    from ..processing.preprocessor import load_providers

    df = load_providers()
    cities_data = []
    for city in _orchestrator.cities:
        city_df = df[df["city"].str.lower() == city.lower()]
        areas = sorted(city_df["area"].dropna().unique().tolist()) if "area" in city_df.columns else []
        cities_data.append({
            "name": city,
            "areas": areas[:50],
            "provider_count": len(city_df),
        })

    return {"cities": cities_data, "total": len(cities_data)}


@router.get("/discovery/cities/{city_name}/areas")
async def get_city_areas(city_name: str):
    """
    Get all areas within a specific city.

    Mobile app uses this when user selects a city for area auto-complete.
    """
    _check_init()
    from ..processing.preprocessor import load_providers

    df = load_providers()
    city_df = df[df["city"].str.lower() == city_name.lower()]

    if city_df.empty:
        raise HTTPException(404, detail=f"City '{city_name}' not found")

    areas = sorted(city_df["area"].dropna().unique().tolist()) if "area" in city_df.columns else []

    return {"city": city_name, "areas": areas, "total": len(areas)}


# ═══════════════════════════════════════════════════════════════
# 6. HEALTH & SYSTEM
# ═══════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    System health check.

    Mobile app calls this on startup to verify backend is reachable.
    """
    return HealthResponse(
        status="healthy" if _orchestrator else "degraded",
        version="2.0.0",
        vector_store_count=_vector_store.count if _vector_store else 0,
        uptime_seconds=time.time() - _start_time if _start_time else 0,
        services={
            "llm": _orchestrator.llm.check_health() if _orchestrator and hasattr(_orchestrator.llm, "check_health") else False,
            "weaviate": _vector_store is not None,
            "session_store": _session_store is not None,
            "booking_store": _orchestrator.booking_store is not None if _orchestrator else False,
        },
    )
