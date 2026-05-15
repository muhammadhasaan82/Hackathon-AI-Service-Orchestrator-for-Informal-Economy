"""
API Routes — FastAPI endpoint definitions.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException

from .schemas import ChatRequest, ChatResponse, HealthResponse, ProviderSearchRequest

logger = logging.getLogger("api.routes")

router = APIRouter(prefix="/api/v1", tags=["Service Orchestrator"])

# These get set by app.py during lifespan
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


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Processes user message through the full agentic pipeline:
    Intent → Discovery → Ranking → Booking → Follow-up
    """
    if _orchestrator is None:
        raise HTTPException(503, "System not initialized")

    session_id = request.session_id or str(uuid.uuid4())

    result = await _orchestrator.process_message(
        session_id=session_id,
        user_message=request.message,
    )

    return ChatResponse(
        response=result.get("response", ""),
        session_id=session_id,
        status=result.get("status", "unknown"),
        agent_trace=result.get("agent_trace", []),
        booking=result.get("booking"),
        providers=result.get("providers"),
        intent=result.get("intent"),
        latency_ms=result.get("latency_ms"),
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    import time
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_count=_vector_store.count if _vector_store else 0,
        uptime_seconds=time.time() - _start_time if _start_time else 0,
    )


@router.get("/agents/trace/{session_id}")
async def get_trace(session_id: str):
    """Get the reasoning trace for a session."""
    if _session_store is None:
        raise HTTPException(503, "System not initialized")

    session = _session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": session_id,
        "reasoning_trace": session.get("reasoning_trace", []),
        "conversation_history": session.get("conversation_history", []),
        "agent_state": session.get("agent_state", {}),
    }


@router.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    """Get booking status."""
    if _orchestrator is None:
        raise HTTPException(503, "System not initialized")

    booking = _orchestrator.booking_store.get_booking(booking_id)
    if not booking:
        raise HTTPException(404, "Booking not found")
    return booking


@router.post("/bookings/{booking_id}/cancel")
async def cancel_booking(booking_id: str):
    """Cancel a booking."""
    if _orchestrator is None:
        raise HTTPException(503, "System not initialized")

    success = _orchestrator.booking_store.update_status(booking_id, "CANCELLED")
    if not success:
        raise HTTPException(404, "Booking not found")
    return {"booking_id": booking_id, "status": "CANCELLED"}


@router.get("/providers/categories")
async def get_categories():
    """Get available service categories."""
    if _orchestrator is None:
        raise HTTPException(503, "System not initialized")
    return {"categories": _orchestrator.service_categories}


@router.get("/providers/cities")
async def get_cities():
    """Get available cities."""
    if _orchestrator is None:
        raise HTTPException(503, "System not initialized")
    return {"cities": _orchestrator.cities}
