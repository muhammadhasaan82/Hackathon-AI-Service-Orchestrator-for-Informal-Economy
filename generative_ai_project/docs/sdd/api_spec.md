# API Specification — SDD Document

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### POST /chat
Main conversational endpoint. Processes user message through the full agentic pipeline.

**Request**:
```json
{
  "session_id": "optional-uuid",
  "message": "Mujhe kal subah G-13 mein AC technician chahiye",
  "user_lat": 33.6844,
  "user_lon": 73.0479
}
```

**Response**:
```json
{
  "response": "...",
  "session_id": "uuid",
  "status": "booking_confirmed | needs_clarification | no_results | error",
  "agent_trace": [...],
  "booking": {...},
  "providers": [...],
  "intent": {...},
  "latency_ms": 1234.56
}
```

### GET /health
System health check.

**Response**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "vector_store_count": 50000,
  "uptime_seconds": 3600.0
}
```

### GET /agents/trace/{session_id}
Full reasoning trace for a session.

### GET /bookings/{booking_id}
Booking details.

### POST /bookings/{booking_id}/cancel
Cancel a booking.

### GET /providers/categories
List available service categories.

### GET /providers/cities
List served cities.

## Error Codes
| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Resource not found |
| 503 | Service not initialized |
| 500 | Internal error |
