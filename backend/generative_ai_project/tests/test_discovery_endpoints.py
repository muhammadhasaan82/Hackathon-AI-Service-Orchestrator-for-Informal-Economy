"""
Integration tests for discovery endpoints.

Run with:
    SMOKE_BASE_URL=http://20.17.177.214:8000 python -m pytest tests/test_discovery_endpoints.py -v
"""

import os

import httpx
import pytest


SMOKE_BASE_URL = os.getenv("SMOKE_BASE_URL")
skip_no_url = pytest.mark.skipif(not SMOKE_BASE_URL, reason="Set SMOKE_BASE_URL")


def _get(path: str, params: dict = None) -> dict:
    with httpx.Client(base_url=SMOKE_BASE_URL.rstrip("/"), timeout=10.0) as c:
        r = c.get(path, params=params)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def _get_status(path: str) -> int:
    with httpx.Client(base_url=SMOKE_BASE_URL.rstrip("/"), timeout=10.0) as c:
        r = c.get(path)
    return r.status_code


# ── Cities ──────────────────────────────────────────────────

@skip_no_url
def test_cities_returns_6():
    data = _get("/api/v1/discovery/cities")
    assert data["total"] == 6
    assert len(data["cities"]) == 6


@skip_no_url
def test_cities_have_required_fields():
    data = _get("/api/v1/discovery/cities")
    for city in data["cities"]:
        assert "name" in city
        assert "areas" in city
        assert "provider_count" in city
        assert city["provider_count"] > 0


@skip_no_url
def test_cities_include_all_expected():
    data = _get("/api/v1/discovery/cities")
    names = [c["name"] for c in data["cities"]]
    for expected in ["Faisalabad", "Islamabad", "Karachi", "Lahore", "Peshawar", "Rawalpindi"]:
        assert expected in names


# ── Categories ──────────────────────────────────────────────

@skip_no_url
def test_categories_returns_14():
    data = _get("/api/v1/discovery/categories")
    assert data["total"] == 14
    assert len(data["categories"]) == 14


@skip_no_url
def test_categories_have_required_fields():
    data = _get("/api/v1/discovery/categories")
    for cat in data["categories"]:
        assert "name" in cat
        assert "icon" in cat
        assert "provider_count" in cat
        assert cat["provider_count"] > 0


@skip_no_url
def test_categories_include_key_services():
    data = _get("/api/v1/discovery/categories")
    names = [c["name"] for c in data["categories"]]
    for expected in ["Plumber", "Electrician", "AC Technician", "Carpenter", "Beautician"]:
        assert expected in names


# ── Areas ───────────────────────────────────────────────────

@skip_no_url
def test_karachi_areas():
    data = _get("/api/v1/discovery/cities/Karachi/areas")
    assert data["city"] == "Karachi"
    assert data["total"] == 7
    names = [a["name"] for a in data["areas"]]
    for expected in ["Clifton", "DHA", "Gulshan"]:
        assert expected in names


@skip_no_url
def test_rawalpindi_areas():
    data = _get("/api/v1/discovery/cities/Rawalpindi/areas")
    assert data["city"] == "Rawalpindi"
    assert data["total"] == 4
    names = [a["name"] for a in data["areas"]]
    for expected in ["Bahria Town", "Chaklala", "PWD", "Saddar"]:
        assert expected in names


@skip_no_url
def test_areas_have_provider_count():
    data = _get("/api/v1/discovery/cities/Lahore/areas")
    for area in data["areas"]:
        assert "name" in area
        assert "provider_count" in area
        assert area["provider_count"] > 0


@skip_no_url
def test_unknown_city_returns_404():
    status = _get_status("/api/v1/discovery/cities/Mumbai/areas")
    assert status == 404


@skip_no_url
def test_city_name_case_insensitive():
    data = _get("/api/v1/discovery/cities/karachi/areas")
    assert data["city"] == "Karachi"


# ── Coverage ────────────────────────────────────────────────

@skip_no_url
def test_coverage_endpoint():
    data = _get("/api/v1/discovery/coverage")
    assert data["total_providers"] == 50000
    assert data["total_cities"] == 6
    assert data["total_categories"] == 14
    assert len(data["cities"]) == 6
    assert len(data["categories"]) == 14


@skip_no_url
def test_provider_counts_sum_to_total():
    cities = _get("/api/v1/discovery/cities")
    coverage = _get("/api/v1/discovery/coverage")
    total = sum(c["provider_count"] for c in cities["cities"])
    assert total == coverage["total_providers"]
