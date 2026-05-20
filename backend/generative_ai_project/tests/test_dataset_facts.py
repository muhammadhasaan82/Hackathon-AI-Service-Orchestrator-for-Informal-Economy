"""
Unit tests for DatasetFacts — verifies dynamic fact derivation
from the providers DataFrame.

These tests load the actual CSV so they validate real data, not mocks.
Run with:
    cd backend/generative_ai_project
    python -m pytest tests/test_dataset_facts.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.processing.preprocessor import load_providers
from src.core.dataset_facts import DatasetFacts


@pytest.fixture(scope="module")
def facts():
    """Load providers once for all tests in this module."""
    df = load_providers()
    return DatasetFacts(df)


# ── City tests ──────────────────────────────────────────────────

def test_cities_returns_list(facts):
    cities = facts.get_available_cities()
    assert isinstance(cities, list)
    assert len(cities) > 0


def test_cities_count_is_6(facts):
    """The dataset has exactly 6 cities."""
    cities = facts.get_available_cities()
    assert len(cities) == 6


def test_known_cities_present(facts):
    """All expected cities are in the dataset."""
    cities = facts.get_available_cities()
    for expected in ["Islamabad", "Karachi", "Lahore", "Rawalpindi", "Peshawar", "Faisalabad"]:
        assert expected in cities, f"Expected city '{expected}' not found"


# ── Category tests ──────────────────────────────────────────────

def test_categories_returns_list(facts):
    categories = facts.get_available_categories()
    assert isinstance(categories, list)
    assert len(categories) > 0


def test_categories_count_is_14(facts):
    """The dataset has exactly 14 service categories."""
    categories = facts.get_available_categories()
    assert len(categories) == 14


def test_known_categories_present(facts):
    """Key categories are in the dataset."""
    categories = facts.get_available_categories()
    for expected in ["Plumber", "Electrician", "AC Technician", "Carpenter", "Beautician"]:
        assert expected in categories, f"Expected category '{expected}' not found"


# ── Provider count tests ────────────────────────────────────────

def test_total_providers_is_50000(facts):
    """The dataset has exactly 50,000 providers."""
    assert facts.get_total_provider_count() == 50000


def test_city_provider_count_positive(facts):
    for city in facts.get_available_cities():
        count = facts.get_city_provider_count(city)
        assert count > 0, f"City '{city}' has 0 providers"


def test_category_provider_count_positive(facts):
    for cat in facts.get_available_categories():
        count = facts.get_category_provider_count(cat)
        assert count > 0, f"Category '{cat}' has 0 providers"


def test_city_counts_sum_to_total(facts):
    total = sum(facts.get_city_provider_count(c) for c in facts.get_available_cities())
    assert total == facts.get_total_provider_count()


# ── Area tests ──────────────────────────────────────────────────

def test_areas_lahore_not_empty(facts):
    areas = facts.get_areas_by_city("Lahore")
    assert len(areas) > 0, "Lahore should have areas"


def test_areas_unknown_city_empty(facts):
    areas = facts.get_areas_by_city("Mumbai")
    assert areas == []


# ── Fuzzy matching tests ────────────────────────────────────────

def test_fuzzy_match_plumber(facts):
    """'plumber' should exact-match to 'Plumber'."""
    result = facts.fuzzy_match_category("plumber")
    assert result == "Plumber"


def test_fuzzy_match_ac(facts):
    """'ac technician' should match."""
    result = facts.fuzzy_match_category("ac technician")
    assert result == "AC Technician"


def test_fuzzy_match_city_karachi(facts):
    """'karachi' should match 'Karachi'."""
    result = facts.fuzzy_match_city("karachi")
    assert result == "Karachi"


def test_fuzzy_match_city_unknown(facts):
    """Unknown city should return None."""
    result = facts.fuzzy_match_city("mumbai")
    assert result is None


# ── has_city / has_category ─────────────────────────────────────

def test_has_city_islamabad(facts):
    found, canonical = facts.has_city("islamabad")
    assert found is True
    assert canonical == "Islamabad"


def test_has_city_unknown(facts):
    found, canonical = facts.has_city("Tokyo")
    assert found is False
    assert canonical is None


def test_has_category_electrician(facts):
    found, canonical = facts.has_category("electrician")
    assert found is True
    assert canonical == "Electrician"


def test_has_category_unknown(facts):
    found, canonical = facts.has_category("astronaut")
    assert found is False
    assert canonical is None


# ── Coverage tests ──────────────────────────────────────────────

def test_category_city_coverage(facts):
    coverage = facts.get_category_city_coverage("Plumber")
    assert len(coverage) > 0
    for city, count in coverage.items():
        assert count > 0


def test_service_counts_by_city(facts):
    counts = facts.get_service_counts_by_city()
    assert len(counts) == 6
    assert all(v > 0 for v in counts.values())
