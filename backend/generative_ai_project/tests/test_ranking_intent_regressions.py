"""
Regression tests for deterministic CPU chat paths.

Run with:
    python -m pytest tests/test_ranking_intent_regressions.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.intent_agent import IntentAgent
from src.agents.ranking_agent import RankingAgent


def test_deterministic_reasoning_ignores_nested_breakdown_values():
    agent = RankingAgent(
        llm=None,
        scoring_config={},
        prompts_config={},
        agents_config={"agents": {"ranking": {"top_n": 5}}},
    )
    provider = {
        "metadata": {"provider_name": "Test Provider"},
        "score_result": {
            "composite_score": 0.82,
            "breakdown": {
                "rating": 0.35,
                "distance": {"raw_km": 2.0, "score": 0.25},
                "availability": 0.30,
                "verified": True,
                "notes": "close match",
                "response_time": [0.2],
            },
        },
    }

    text = agent._deterministic_reasoning([provider], intent={})

    assert "rating=0.35" in text
    assert "availability=0.30" in text
    assert "distance=" not in text
    assert "verified=" not in text


def test_deterministic_reasoning_uses_fallback_when_no_numeric_strengths():
    agent = RankingAgent(
        llm=None,
        scoring_config={},
        prompts_config={},
        agents_config={"agents": {"ranking": {"top_n": 5}}},
    )
    provider = {
        "metadata": {"provider_name": "Fallback Provider"},
        "score_result": {
            "composite_score": 0.73,
            "breakdown": {
                "distance": {"raw_km": 2.0},
                "verified": True,
                "notes": "available",
            },
        },
    }

    text = agent._deterministic_reasoning([provider], intent={})

    assert "rating, availability, response time" in text


def test_rule_based_area_city_time_and_urgency_extraction():
    agent = IntentAgent(
        llm=None,
        prompts_config={},
        agents_config={
            "agents": {
                "intent": {
                    "deterministic_hints": {
                        "time_phrases": {"today": ["today"]},
                        "urgency_phrases": {"urgent": ["urgent"]},
                        "area_prepositions": ["in"],
                    }
                }
            }
        },
    )

    intent = agent._rule_based_intent(
        "I need a plumber in DHA Karachi today urgent",
        categories=["Plumber"],
        cities=["Karachi"],
    )

    assert intent["service_type"] == "Plumber"
    assert intent["city"] == "Karachi"
    assert intent["area"] == "DHA"
    assert intent["time_preference"] == "today"
    assert intent["urgency"] == "urgent"
    assert intent["needs_clarification"] == []
