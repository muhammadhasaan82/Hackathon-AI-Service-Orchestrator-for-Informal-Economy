# VM Smoke Tests

Use this guide after the backend is already running on your VM or deployed URL.

These smoke tests are intentionally dependency-free and use Python standard
library only. They do not install packages.

## What Is Covered

The script checks:

- `GET /api/v1/health`
- `GET /api/v1/discovery/categories`
- `GET /api/v1/discovery/cities`
- `POST /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/chat`
- `WS /api/v1/ws/chat`
- `GET /api/v1/providers`
- `GET /api/v1/providers/{provider_id}` when provider search returns an id
- `POST /api/v1/bookings` and booking reads when provider search returns an id

## Run Against Local VM Backend

From the repo root:

```bash
python scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

## Run Against GCP URL

```bash
python scripts/smoke_api.py --base-url https://YOUR_GCP_BACKEND_URL
```

## Skip Optional Checks

Skip WebSocket if your proxy/load balancer is not configured for WebSocket yet:

```bash
python scripts/smoke_api.py --base-url https://YOUR_GCP_BACKEND_URL --skip-websocket
```

Skip direct booking creation if you only want read/chat checks:

```bash
python scripts/smoke_api.py --base-url https://YOUR_GCP_BACKEND_URL --skip-booking
```

## Custom Chat Message

```bash
python scripts/smoke_api.py \
  --base-url http://127.0.0.1:8000 \
  --message "I need an electrician in Lahore today"
```

## Manual cURL Checks

Health:

```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

Chat:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke-vm","message":"I need a plumber in Karachi today"}'
```

Hybrid FAQ plus booking route:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke-hybrid","message":"I need plumber in Karachi. What are charges?"}'
```

Expected signals in the JSON response:

- `status` is usually `awaiting_booking_confirmation`, `needs_clarification`, or another booking status, not a dead-end FAQ-only route.
- `routing.primary_intent` is `booking`.
- `routing.secondary_intents` includes `faq_pricing`.
- `routing.should_continue_booking` is `true`.

Conversation memory continuation:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke-memory","message":"Need plumber in Karachi"}'

curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke-memory","message":"Available in DHA?"}'
```

The second response should reuse the previous `service_type` and `city`
when the session store is working.

Ambiguity handling:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke-ambiguity","message":"lumber"}'
```

Expected signal: `status` is `ambiguity_detected`, and the response asks
the user to choose between valid dataset categories.

Provider list:

```bash
curl -s "http://127.0.0.1:8000/api/v1/providers?query=plumber&city=Karachi&limit=3"
```

Discovery:

```bash
curl -s http://127.0.0.1:8000/api/v1/discovery/categories
curl -s http://127.0.0.1:8000/api/v1/discovery/cities
```

## Expected Result

The script prints `[OK]` lines for each endpoint and finishes with:

```text
[DONE] Smoke tests completed successfully.
```

If a dependency service is not ready, such as Weaviate or the model backend,
the failing endpoint will print `[FAIL]` with the backend response.
