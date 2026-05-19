"""
Regression tests for provider detail lookup.

These tests keep the provider detail endpoint tied to the exact
provider_id returned by search results.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import routes


class FakeVectorStore:
    def __init__(self, provider):
        self.provider = provider
        self.seen_provider_id = None

    def get_provider_by_id(self, provider_id):
        self.seen_provider_id = provider_id
        return self.provider


def test_get_provider_detail_uses_exact_provider_id(monkeypatch):
    provider = {
        "metadata": {
            "provider_id": 6315,
            "provider_name": "Ali Plumbing",
            "category": "Plumber",
            "city": "Karachi",
            "area": "DHA",
            "full_location": "DHA, Karachi",
            "latitude": 24.8,
            "longitude": 67.1,
            "rating": 4.8,
            "availability": "Available",
            "experience_years": 10,
            "completed_jobs": 250,
            "response_time_minutes": 20,
            "price_range": "Medium",
            "verified_provider": True,
            "languages_supported": "Urdu, English",
            "phone_number": "+92-300-1234567",
            "email": "ali@example.com",
        }
    }
    fake_store = FakeVectorStore(provider)
    monkeypatch.setattr(routes, "_orchestrator", object())
    monkeypatch.setattr(routes, "_vector_store", fake_store)

    payload = asyncio.run(routes.get_provider_detail(6315))

    assert fake_store.seen_provider_id == 6315
    assert payload["provider_id"] == 6315
    assert payload["provider_name"] == "Ali Plumbing"
    assert payload["phone_number"] == "+92-300-1234567"


def test_get_provider_detail_returns_404_when_provider_missing(monkeypatch):
    fake_store = FakeVectorStore(None)
    monkeypatch.setattr(routes, "_orchestrator", object())
    monkeypatch.setattr(routes, "_vector_store", fake_store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.get_provider_detail(9999))

    assert exc_info.value.status_code == 404
