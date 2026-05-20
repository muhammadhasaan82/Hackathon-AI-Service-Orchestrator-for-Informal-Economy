"""
Unit contracts for hybrid FAQ routing.

These tests avoid network/model startup and exercise the deterministic
contracts used by the orchestrator.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.faq_agent import FAQAgent
from src.core.semantic_matcher import SemanticFAQMatcher


class FakeFacts:
    def __init__(self):
        self._categories = ["Plumber", "Carpenter", "Electrician", "AC Technician"]
        self._cities = ["Karachi", "Lahore"]
        self._areas = {"Karachi": ["DHA", "Clifton"], "Lahore": ["Gulberg"]}
        self._city_category_counts = {
            ("Karachi", "Plumber"): 3,
            ("Karachi", "AC Technician"): 2,
            ("Lahore", "Electrician"): 4,
        }

    def get_available_cities(self):
        return list(self._cities)

    def get_available_categories(self):
        return list(self._categories)

    def get_areas_by_city(self, city):
        return list(self._areas.get(city, []))

    def get_total_provider_count(self):
        return 9

    def get_city_provider_count(self, city):
        return sum(count for (c, _), count in self._city_category_counts.items() if c == city)

    def get_category_provider_count(self, category):
        return sum(count for (_, cat), count in self._city_category_counts.items() if cat == category)

    def get_category_city_coverage(self, category):
        return {
            city: count
            for (city, cat), count in self._city_category_counts.items()
            if cat == category
        }

    def get_availability_summary(self, city=None, category=None, area=None):
        return {"Available": 2, "Available Soon": 1}

    def has_category(self, value):
        for category in self._categories:
            if category.casefold() == str(value).casefold():
                return True, category
        return False, None

    def match_category(self, text, **kwargs):
        lowered = str(text).casefold()
        for category in self._categories:
            if category.casefold() in lowered or category.casefold().rstrip("n") in lowered:
                return {
                    "matched": True,
                    "value": category,
                    "score": 1.0,
                    "source": "exact",
                    "candidates": [{"value": category, "score": 1.0}],
                    "ambiguous": False,
                }
        return {
            "matched": False,
            "value": None,
            "score": 0.0,
            "source": "none",
            "candidates": [],
            "ambiguous": False,
        }

    def match_city(self, text, **kwargs):
        lowered = str(text).casefold()
        for city in self._cities:
            if city.casefold() in lowered:
                return {
                    "matched": True,
                    "value": city,
                    "score": 1.0,
                    "source": "exact",
                    "candidates": [{"value": city, "score": 1.0}],
                    "ambiguous": False,
                }
        return {
            "matched": False,
            "value": None,
            "score": 0.0,
            "source": "none",
            "candidates": [],
            "ambiguous": False,
        }


class FakeCAG:
    def get_structured_faq_policy(self, key):
        policies = {
            "pricing_policy": {
                "short_answer": "Charges depend on service, area, urgency, and provider price range.",
                "detailed_answer": "Detailed pricing policy.",
                "followup_prompts": ["Tell me your service and city."],
            },
            "availability_policy": {
                "short_answer": "Availability is checked during provider search.",
                "detailed_answer": "Detailed availability policy.",
                "followup_prompts": ["Tell me your service and city."],
            },
        }
        return {
            "policy_key": key,
            "policy": policies.get(key),
            "source": "cag",
            "structured": True,
            "found": key in policies,
        }


def _faq_config():
    return {
        "faq_patterns": {
            "faq_pricing": {
                "patterns": ["charges?", "fees?", "\\brate\\b", "kitne.*paise"],
                "source": "cag",
                "cag_key": "pricing_policy",
            },
            "faq_availability": {
                "patterns": ["available", "availability"],
                "source": "cag",
                "cag_key": "availability_policy",
            },
            "dataset_service_coverage": {
                "patterns": ["do you provide", "available in"],
                "source": "dataset",
            },
        },
        "booking_signals": ["need", "chahiye", "find"],
        "routing": {
            "regex_confidence": 0.92,
            "booking_signal_confidence": 0.76,
            "dataset_match_min_confidence": 0.70,
        },
        "hybrid": {"enabled": True, "continue_booking_status": "hybrid_route"},
        "semantic": {
            "enabled": False,
            "threshold": 0.45,
            "lexical_fallback_enabled": True,
            "examples": {},
        },
        "ambiguity": {
            "enabled": True,
            "entity_threshold": 0.58,
            "close_match_margin": 0.12,
            "max_candidates": 3,
            "ask_template": "Did you mean {options}?",
            "term_candidates": {"lumber": ["Plumber", "Carpenter"]},
        },
        "response_templates": {},
    }


def _agent():
    return FAQAgent(FakeFacts(), FakeCAG(), _faq_config())


def test_hybrid_booking_and_pricing_continues_booking_flow():
    result = asyncio.run(_agent().try_handle("I need plumber in Karachi. What are charges?", {}))

    assert result["status"] == "hybrid_route"
    assert result["primary_intent"] == "booking"
    assert result["should_continue_booking"] is True
    assert "faq_pricing" in result["secondary_intents"]
    assert result["routing_confidence"] >= 0.9


def test_memory_context_allows_availability_continuation():
    session = {
        "agent_state": {
            "intent": {"service_type": "Plumber", "city": "Karachi"}
        }
    }

    result = asyncio.run(_agent().try_handle("Available in DHA?", session))

    assert result["status"] == "hybrid_route"
    assert result["should_continue_booking"] is True
    assert result["entity_context"]["service_type"] == "Plumber"
    assert result["entity_context"]["city"] == "Karachi"
    assert result["entity_context"]["area"] == "DHA"


def test_configured_ambiguity_does_not_silently_guess():
    result = asyncio.run(_agent().try_handle("lumber", {}))

    assert result["status"] == "ambiguity_detected"
    assert result["explainability"]["ambiguity_detected"] is True
    assert "Plumber" in result["response"]
    assert "Carpenter" in result["response"]


def test_semantic_matcher_lexical_fallback_for_roman_urdu():
    matcher = SemanticFAQMatcher(
        {
            "faq_availability": ["kal available hoge", "abhi koi available hai"],
            "faq_pricing": ["kitne paise lagenge", "plumber ka rate"],
        },
        {"enabled": False, "threshold": 0.4, "lexical_fallback_enabled": True},
    )

    result = matcher.match("kal available hoge?")

    assert result["faq_class"] == "faq_availability"
    assert result["score"] >= 0.4

