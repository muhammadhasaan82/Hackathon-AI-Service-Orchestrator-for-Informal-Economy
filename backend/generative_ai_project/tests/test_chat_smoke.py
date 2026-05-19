"""
Smoke test for the running FastAPI chat endpoint.

This test expects the API to already be running. It is skipped unless
SMOKE_BASE_URL is set, for example:
    SMOKE_BASE_URL=http://localhost:8000 python -m pytest tests/test_chat_smoke.py -v
"""

import os

import httpx
import pytest


SMOKE_BASE_URL = os.getenv("SMOKE_BASE_URL")


@pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL to a running API base URL")
def test_chat_provider_search_smoke_cpu_latency():
    with httpx.Client(base_url=SMOKE_BASE_URL.rstrip("/"), timeout=10.0) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "I need a plumber in DHA Karachi today urgent"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") != "error"
    assert payload.get("providers") is not None
    assert payload.get("latency_ms") is not None
    assert payload["latency_ms"] < 2000
