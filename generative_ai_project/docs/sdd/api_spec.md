# API Specification — SDD Document (Mobile Application)

## Base URL
`http://localhost:8000/api/v1`

---

## 1. Chat (Conversational AI)

### POST /chat
Main conversational endpoint. Mobile app sends user messages here.

The guarded flow is multi-turn:

1. user asks for a service
2. API returns ranked options and `awaiting_booking_confirmation`
3. user confirms a specific option in the same session
4. API returns `booking_confirmed` with booking and follow-up payloads

**Request:**
```json
{
  "session_id": "uuid (optional, auto-generated if missing)",
  "message": "Mujhe kal subah G-13 mein AC technician chahiye",
  "user_lat": 33.6844,
  "user_lon": 73.0479
}
```

**Response:**
```json
{
  "response": "...",
  "session_id": "uuid",
  "status": "awaiting_booking_confirmation | booking_confirmed | booking_cancelled | needs_clarification | human_handoff_recommended | no_results | input_rejected | error",
  "agent_trace": [...],
  "booking": {...},
  "followup": {...},
  "providers": [...],
  "intent": {...},
  "latency_ms": 1234.56
}
```

**Example first-turn response semantics:**

- `awaiting_booking_confirmation`: ranked providers are ready and booking has not yet been created
- `providers`: top shortlisted providers with deterministic score breakdowns
- `booking`: `null`

**Example confirmation message:**

```json
{
  "session_id": "same-session-id",
  "message": "book option 1"
}
```

---

## 2. Providers

### POST /providers/search
Search and filter providers. Used by Browse/Search screens.

**Request:**
```json
{
  "category": "Plumber",
  "city": "Karachi",
  "area": "DHA",
  "query": "pipe leak emergency",
  "price_range": "Low",
  "availability": "Available",
  "min_rating": 4.0,
  "verified_only": false,
  "sort_by": "score | rating | distance | price",
  "user_lat": 24.8607,
  "user_lon": 67.0011,
  "top_n": 10,
  "page": 1
}
```

**Response:**
```json
{
  "providers": [
    {
      "provider_id": 1001,
      "provider_name": "Ali Ahmad",
      "category": "Plumber",
      "city": "Karachi",
      "area": "DHA",
      "rating": 4.7,
      "availability": "Available",
      "price_range": "Low",
      "verified_provider": true,
      "distance_km": 2.3,
      "score": 0.85
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 10,
  "has_next": true
}
```

### GET /providers/{provider_id}
Full provider detail. Used by Provider Detail screen.

### GET /providers/nearby?lat=33.68&lon=73.04&radius_km=10&category=Plumber&top_n=20
Find providers near GPS coordinates. Used by Nearby screen.

---

## 3. Bookings

### POST /bookings
Create booking from provider detail "Book Now" button.

This endpoint is explicit and deterministic. Unlike `/chat`, it does not require a separate conversational confirmation turn because the caller is already making a direct booking action.

**Request:**
```json
{
  "session_id": "uuid",
  "provider_id": 1001,
  "service_type": "Plumber",
  "location": "DHA, Karachi",
  "scheduled_time": "tomorrow 10 AM",
  "user_notes": "Pipe leak in kitchen"
}
```

### GET /bookings/{booking_id}
Get booking status.

### PATCH /bookings/{booking_id}
Update booking (reschedule, add notes).

**Request:**
```json
{
  "scheduled_time": "Monday 2 PM",
  "user_notes": "Updated note"
}
```

### POST /bookings/{booking_id}/cancel
Cancel a booking.

### GET /bookings/session/{session_id}
Get all bookings for a user/session. Used by "My Bookings" screen.

---

## 4. Sessions

### POST /sessions
Create a new session on first app launch. Store the `session_id` on device.

### GET /sessions/{session_id}
Get session metadata (turn count, current intent).

### GET /sessions/{session_id}/history?max_turns=20
Get conversation history. Used to restore chat screen after app restart.

### DELETE /sessions/{session_id}
Delete session and clear history.

### GET /agents/trace/{session_id}
Get full reasoning trace (transparency/debug).

---

## 5. Discovery

### GET /discovery/categories
All service categories with icons. Used by home screen category grid.

### GET /discovery/cities
All cities with areas and provider counts. Used by city/area dropdowns.

### GET /discovery/cities/{city_name}/areas
Areas within a city. Used for area auto-complete.

---

## 6. Health

### GET /health
System health check with service status.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "vector_store_count": 50000,
  "uptime_seconds": 3600.0,
  "services": {
    "llm": true,
    "weaviate": true,
    "session_store": true,
    "booking_store": true
  }
}
```

---

## Error Codes
| Code | Meaning |
|------|---------|
| 200  | Success |
| 400  | Invalid request (bad status transition) |
| 404  | Resource not found |
| 503  | Service not initialized |
| 500  | Internal error |
