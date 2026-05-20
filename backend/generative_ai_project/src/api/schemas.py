"""
API Schemas — Pydantic request/response models.

Covers all mobile application endpoints:
- Chat (conversational AI)
- Providers (browse, search, details)
- Bookings (CRUD + history)
- Sessions (management)
- User (profile helpers)
- Discovery (categories, cities, areas)
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class AuthSignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)


class AuthResponse(BaseModel):
    user: dict
    access_token: str
    token_type: str = "bearer"


# ═══════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════

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
    status: str = Field(..., description="PENDING | CONFIRMED | IN_PROGRESS | COMPLETED | CANCELLED")
    price_range: Optional[str] = None
    provider_phone: Optional[str] = None
    provider_email: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = Field(
        ...,
        description=(
            "awaiting_booking_confirmation | booking_confirmed | booking_cancelled | "
            "needs_clarification | faq_answered | hybrid_route | ambiguity_detected | "
            "human_handoff_recommended | no_results | input_rejected | error"
        ),
    )
    agent_trace: list[dict] = Field(default_factory=list)
    booking: Optional[dict] = None
    followup: Optional[dict] = Field(
        None,
        description=(
            "Follow-up notification plan with scheduled reminders, "
            "status timeline, and immediate notification payload. "
            "The mobile app uses this to schedule local push notifications."
        ),
    )
    providers: Optional[list[dict]] = None
    intent: Optional[dict] = None
    routing: Optional[dict] = Field(
        None,
        description="Safe routing diagnostics: confidence, route source, policy source, ambiguity flags.",
    )
    latency_ms: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
# Provider Search & Details
# ═══════════════════════════════════════════════════════════════

class ProviderSearchRequest(BaseModel):
    category: Optional[str] = None
    city: Optional[str] = None
    area: Optional[str] = None
    query: Optional[str] = None
    price_range: Optional[str] = Field(None, description="Low | Medium | High")
    availability: Optional[str] = Field(None, description="Available | Available Soon")
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    verified_only: bool = Field(False, description="Only return verified providers")
    sort_by: Optional[str] = Field("score", description="score | rating | distance | price")
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    top_n: int = Field(5, ge=1, le=50)
    page: int = Field(1, ge=1, description="Page number for pagination")


class ProviderDetailResponse(BaseModel):
    provider_id: int
    provider_name: str
    category: str
    city: str
    area: str
    full_location: Optional[str] = None
    latitude: float
    longitude: float
    rating: float
    availability: str
    experience_years: int
    completed_jobs: int
    response_time_minutes: int
    price_range: str
    verified_provider: bool
    languages_supported: str
    phone_number: str
    email: str
    distance_km: Optional[float] = None
    score: Optional[float] = None


class ProviderListResponse(BaseModel):
    providers: list[dict]
    total: int
    page: int
    per_page: int
    has_next: bool


# ═══════════════════════════════════════════════════════════════
# Bookings
# ═══════════════════════════════════════════════════════════════

class CreateBookingRequest(BaseModel):
    session_id: str
    provider_id: int
    service_type: str
    location: str
    scheduled_time: Optional[str] = None
    user_notes: Optional[str] = None
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None


class UpdateBookingRequest(BaseModel):
    scheduled_time: Optional[str] = None
    user_notes: Optional[str] = None
    status: Optional[str] = Field(None, description="PENDING | CONFIRMED | IN_PROGRESS | COMPLETED | CANCELLED")


class BookingListResponse(BaseModel):
    bookings: list[dict]
    total: int


class BookingEventResponse(BaseModel):
    event_id: int
    booking_id: str
    event_type: str
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    details: Optional[str] = None
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# Session
# ═══════════════════════════════════════════════════════════════

class SessionResponse(BaseModel):
    session_id: str
    turn_count: int
    created_at: float
    last_active: float
    current_intent: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════
# Discovery (Categories, Cities, Areas)
# ═══════════════════════════════════════════════════════════════

class CategoryInfo(BaseModel):
    name: str
    description: Optional[str] = None
    provider_count: int = 0
    icon: Optional[str] = None


class CityInfo(BaseModel):
    name: str
    areas: list[str] = Field(default_factory=list)
    provider_count: int = 0


# ═══════════════════════════════════════════════════════════════
# Health & System
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_count: int
    uptime_seconds: float
    services: Optional[dict[str, bool]] = None


# ═══════════════════════════════════════════════════════════════
# Follow-Up & Notifications
# ═══════════════════════════════════════════════════════════════

class FollowUpReminderRequest(BaseModel):
    """Request to generate a reminder for a specific booking and tier."""
    tier: str = Field("final_reminder", description="Reminder tier: early_heads_up | preparation | final_reminder")
    language: str = Field("English", description="Response language: English | Roman Urdu")


class FollowUpStatusRequest(BaseModel):
    """Request to generate a status update notification."""
    event_type: str = Field(
        ...,
        description="Status event: provider_notified | provider_preparing | provider_en_route | service_started | service_completed",
    )
    language: str = Field("English", description="Response language")


class NotificationActionButton(BaseModel):
    """A tappable action button in a mobile notification."""
    action: str = Field(..., description="Action identifier: view_booking | call_provider | reschedule | cancel | rate_service")
    label: str = Field(..., description="Display label for the button")
    icon: str = Field("📋", description="Emoji icon for the button")


class ScheduledReminder(BaseModel):
    """A single scheduled reminder in the follow-up plan."""
    reminder_id: str
    tier_name: str
    fire_at_utc: float = Field(..., description="Unix timestamp when the reminder should fire")
    fire_at_iso: str = Field(..., description="ISO-8601 formatted fire time")
    minutes_before_appointment: float
    priority: str = Field("normal", description="Notification priority: low | normal | high")
    channel: str = Field("push", description="Delivery channel: push | sms | in_app")
    icon: str
    description: str
    action_buttons: list[str] = Field(default_factory=list)
    notification_text: Optional[dict] = Field(None, description="LLM-generated title + body")


class StatusTimelineEvent(BaseModel):
    """A status update event in the booking lifecycle timeline."""
    event_id: str
    event_type: str
    fire_at_utc: float
    fire_at_iso: str
    status_transition: str
    icon: str
    description: str
    message_key: str
    action_buttons: list[dict] = Field(default_factory=list)


class FollowUpPlanResponse(BaseModel):
    """Complete follow-up plan returned by schedule_followup."""
    followup_id: Optional[str] = None
    booking_id: str
    scheduled_reminders: list[dict] = Field(default_factory=list)
    status_timeline: list[dict] = Field(default_factory=list)
    post_completion_actions: list[dict] = Field(default_factory=list)
    immediate_notification: Optional[dict] = None
    total_scheduled: int = 0
    metadata: Optional[dict] = None

