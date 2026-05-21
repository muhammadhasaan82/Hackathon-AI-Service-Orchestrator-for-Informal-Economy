"""
Unit tests for Urdu intent extraction and Roman Urdu response translations.

Run with:
    python -m pytest tests/test_urdu_roman_fixes.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    class MockPytest:
        class Mark:
            @staticmethod
            def asyncio(func):
                return func
        mark = Mark()
    pytest = MockPytest()

from src.core.guardrails import translate_response
from src.agents.intent_agent import IntentAgent


def test_roman_urdu_response_translation():
    # Test case 1: "I ranked these providers..." -> "Maine providers ko rating..."
    text1 = "I ranked these providers using deterministic scoring: rating, distance, availability, experience, completed jobs, verification, and price fit."
    result1 = translate_response(text1, "roman_urdu")
    assert "Maine providers ko rating, availability, experience, response time aur price ke basis par rank kiya hai." in result1

    # Test case 2: "strongest signals" -> "sab se strong signals"
    text2 = "1. Pro scored 0.90; strongest signals: rating=0.90."
    result2 = translate_response(text2, "roman_urdu")
    assert "sab se strong signals" in result2

    # Test case 3: "BOOKING CONFIRMATION REQUIRED" -> "BOOKING CONFIRM KARNA ZAROORI HAI"
    text3 = "⏳ BOOKING CONFIRMATION REQUIRED\n"
    result3 = translate_response(text3, "roman_urdu")
    assert "BOOKING CONFIRM KARNA ZAROORI HAI" in result3

    # Test case 4: "I’ve shortlisted 3 providers..." -> "Maine 3 providers shortlist..."
    text4 = "I’ve shortlisted 3 providers. Reply with 'book option 1', 'book option 2', or the provider name to confirm the booking."
    result4 = translate_response(text4, "roman_urdu")
    assert "Maine 3 providers shortlist kiye hain. Booking confirm karne ke liye 'book option 1', 'book option 2', ya provider ka naam reply karein." in result4

    # Test case 5: Keeps provider names, city names, numbers unchanged
    text5 = "I’ve shortlisted 12 providers. Reply with 'book option 1', 'book option 2', or the provider name to confirm the booking."
    result5 = translate_response(text5, "roman_urdu")
    assert "Maine 12 providers shortlist kiye hain." in result5


@pytest.mark.asyncio
async def test_urdu_intent_extraction_plumber():
    agent = IntentAgent(
        llm=None, # Bypasses LLM since we test deterministic mapping success
        prompts_config={},
        agents_config={
            "agents": {
                "intent": {
                    "deterministic_hints": {}
                }
            }
        },
    )

    # Input in Urdu script
    intent1 = await agent.extract_intent(
        "مجھے کراچی کلفٹن میں پلمبر چاہیے",
        service_categories=["Plumber", "Mechanic"],
        cities=["Karachi", "Lahore"],
        areas=["Clifton", "DHA"]
    )

    assert intent1["service_type"] == "Plumber"
    assert intent1["city"] == "Karachi"
    assert intent1["area"] == "Clifton"
    assert intent1["confidence"]["service_type"] == 1.0
    assert intent1["confidence"]["city"] == 1.0
    assert intent1["confidence"]["area"] == 1.0
    assert intent1["needs_clarification"] == []


@pytest.mark.asyncio
async def test_roman_urdu_intent_extraction_plumber():
    agent = IntentAgent(
        llm=None,
        prompts_config={},
        agents_config={
            "agents": {
                "intent": {
                    "deterministic_hints": {}
                }
            }
        },
    )

    # Input in Roman Urdu
    intent2 = await agent.extract_intent(
        "mujhe plumber chahiye Karachi Clifton mein",
        service_categories=["Plumber", "Mechanic"],
        cities=["Karachi", "Lahore"],
        areas=["Clifton", "DHA"]
    )

    assert intent2["service_type"] == "Plumber"
    assert intent2["city"] == "Karachi"
    assert intent2["area"] == "Clifton"
    assert intent2["confidence"]["service_type"] == 1.0
    assert intent2["confidence"]["city"] == 1.0
    assert intent2["confidence"]["area"] == 1.0
    assert intent2["needs_clarification"] == []
