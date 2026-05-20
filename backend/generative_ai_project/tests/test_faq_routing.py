"""
Integration tests for the FAQ routing system.

Tests the full FAQ pipeline: user message → FAQ classification →
dataset/CAG response.  These are smoke tests against the running API.

Run with:
    SMOKE_BASE_URL=http://20.17.177.214:8000 python -m pytest tests/test_faq_routing.py -v
"""

import os

import httpx
import pytest


SMOKE_BASE_URL = os.getenv("SMOKE_BASE_URL")


def _chat(message: str, timeout: float = 15.0) -> dict:
    """Send a chat message and return the response payload."""
    with httpx.Client(base_url=SMOKE_BASE_URL.rstrip("/"), timeout=timeout) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": message},
        )
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
    return response.json()


# ── FAQ — Cities ────────────────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_which_cities():
    payload = _chat("Which cities are available?")
    assert payload.get("status") == "faq_answered", f"Expected faq_answered, got {payload.get('status')}"
    response = payload["response"].lower()
    for city in ["islamabad", "karachi", "lahore", "rawalpindi", "peshawar", "faisalabad"]:
        assert city in response, f"City '{city}' not in FAQ response"


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_cities_urdu():
    payload = _chat("kis city mein available hai?")
    assert payload.get("status") == "faq_answered"
    assert "islamabad" in payload["response"].lower() or "karachi" in payload["response"].lower()


# ── FAQ — Services ──────────────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_what_services():
    payload = _chat("What services do you offer?")
    assert payload.get("status") == "faq_answered"
    response = payload["response"].lower()
    assert "plumber" in response or "electrician" in response


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_services_urdu():
    payload = _chat("kya service provide karte ho?")
    assert payload.get("status") == "faq_answered"


# ── FAQ — Areas ─────────────────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_areas_in_lahore():
    payload = _chat("areas in Lahore?")
    assert payload.get("status") == "faq_answered"
    assert "lahore" in payload["response"].lower()


# ── FAQ — Service Coverage ──────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_plumber_in_karachi():
    payload = _chat("Do you provide plumber in Karachi?")
    assert payload.get("status") == "faq_answered"
    response = payload["response"].lower()
    assert "plumber" in response
    assert "karachi" in response


# ── FAQ — Provider Count ───────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_how_many_providers():
    payload = _chat("How many providers do you have?")
    assert payload.get("status") == "faq_answered"
    assert "50,000" in payload["response"] or "50000" in payload["response"]


# ── FAQ — Policy (CAG) ─────────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_pricing():
    payload = _chat("What are your charges?")
    assert payload.get("status") == "faq_answered"
    assert len(payload["response"]) > 20  # should be a real policy answer


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_booking_process():
    payload = _chat("How does booking work?")
    assert payload.get("status") == "faq_answered"
    assert "book" in payload["response"].lower() or "service" in payload["response"].lower()


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_timings():
    payload = _chat("What are your timings?")
    assert payload.get("status") == "faq_answered"


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_cancellation():
    payload = _chat("Can I cancel my booking?")
    assert payload.get("status") == "faq_answered"
    assert "cancel" in payload["response"].lower()


# ── Booking request — should NOT be FAQ ─────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_booking_request_not_faq():
    """A clear booking request should go through the normal pipeline, not FAQ."""
    payload = _chat("I need a plumber in DHA Karachi today urgent", timeout=30.0)
    assert payload.get("status") != "faq_answered", \
        f"Booking request incorrectly routed to FAQ: {payload.get('status')}"


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_booking_request_urdu_not_faq():
    """Urdu booking request should not be FAQ."""
    payload = _chat("mujhe Islamabad mein electrician chahiye", timeout=30.0)
    assert payload.get("status") != "faq_answered", \
        f"Urdu booking request incorrectly routed to FAQ: {payload.get('status')}"


# ── Latency check ──────────────────────────────────────────────

@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API")
def test_faq_latency_under_500ms():
    """FAQ answers should be fast (< 500ms) since they're deterministic."""
    payload = _chat("What services are available?")
    assert payload.get("status") == "faq_answered"
    assert payload.get("latency_ms", 9999) < 500, \
        f"FAQ latency too high: {payload.get('latency_ms')}ms"
