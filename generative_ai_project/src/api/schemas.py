"""
API Schemas — Pydantic request/response models.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    message: str = Field(..., description="User message in English, Urdu, or Roman Urdu")
    user_lat: Optional[float] = Field(None, description="User latitude for distance calculation")
    user_lon: Optional[float] = Field(None, description="User longitude for distance calculation")


class AgentTraceEntry(BaseModel):
    agent: str
    action: str
    status: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class ProviderResult(BaseModel):
    provider_name: str
    score: float
    score_breakdown: Optional[dict] = None


class BookingInfo(BaseModel):
    booking_id: str
    provider_name: str
    service_type: str
    location: str
    scheduled_time: str
    status: str
    price_range: Optional[str] = None
    provider_phone: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str
    agent_trace: list[dict] = Field(default_factory=list)
    booking: Optional[dict] = None
    providers: Optional[list[dict]] = None
    intent: Optional[dict] = None
    latency_ms: Optional[float] = None


class ProviderSearchRequest(BaseModel):
    category: Optional[str] = None
    city: Optional[str] = None
    area: Optional[str] = None
    query: Optional[str] = None
    top_n: int = Field(10, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_count: int
    uptime_seconds: float
