"""
Intent Agent — Natural Language Understanding via LLM.

Extracts structured intent from user messages in English,
Urdu, or Roman Urdu. The model THINKS about what the user
needs rather than pattern-matching.
"""

import json
import logging
from typing import Optional

from ..core.base_llm import BaseLLM

logger = logging.getLogger("agents.intent")


class IntentAgent:
    """Extracts structured intent from natural language input."""

    def __init__(self, llm: BaseLLM, prompts_config: dict, agents_config: dict, guardrails=None):
        self.llm = llm
        self.prompt_template = prompts_config.get("intent_extraction", "")
        self.config = agents_config.get("agents", {}).get("intent", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.70)
        self.guardrails = guardrails

    async def extract_intent(
        self,
        user_message: str,
        service_categories: list[str],
        cities: list[str],
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Extract structured intent from user message.

        The LLM reasons about the message and returns structured JSON.
        If confidence is below threshold, it flags fields needing clarification.
        """
        prompt = self.prompt_template.format(
            user_message=user_message,
            service_categories=", ".join(service_categories),
            cities=", ".join(cities),
        )

        if conversation_history:
            history = conversation_history[-4:]
            history_text = "\n".join(
                f"{turn.get('role', 'user').title()}: {turn.get('content', '')[:200]}"
                for turn in history
                if turn.get("content")
            )
            if history_text:
                prompt = f"{prompt}\n\nConversation history:\n{history_text}"

        logger.info(f"Extracting intent from: '{user_message}'")

        system_instruction = (
            "You are an expert intent parser for a service booking system in Pakistan. "
            "You understand English, Urdu, and Roman Urdu (romanized Urdu). "
            "Always return valid JSON with the required fields."
        )
        if self.guardrails:
            system_instruction = self.guardrails.compose_system_instruction("intent", system_instruction)

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "service_type": {"type": ["string", "null"]},
                    "city": {"type": ["string", "null"]},
                    "area": {"type": ["string", "null"]},
                    "time_preference": {"type": ["string", "null"]},
                    "urgency": {"type": ["string", "null"]},
                    "language_detected": {"type": ["string", "null"]},
                    "price_preference": {"type": ["string", "null"]},
                    "confidence": {"type": ["object", "null"]},
                },
            },
            system_instruction=system_instruction,
        )

        try:
            intent = json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            text = response.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                intent = json.loads(text[start:end])
            else:
                logger.error(f"Failed to parse intent JSON: {response.text}")
                intent = self._empty_intent()

        # Normalize and validate
        intent = self._normalize_intent(intent, service_categories, cities)

        # Check confidence
        intent["needs_clarification"] = self._check_clarification(intent)

        logger.info(f"Extracted intent: {json.dumps(intent, indent=2)}")
        return intent

    def _normalize_intent(self, intent: dict, categories: list[str], cities: list[str]) -> dict:
        """Normalize extracted fields to match known values."""
        # Fuzzy match service type
        service = intent.get("service_type")
        if service:
            service_lower = service.lower().strip()
            for cat in categories:
                if cat.lower() in service_lower or service_lower in cat.lower():
                    intent["service_type"] = cat
                    break

        # Fuzzy match city
        city = intent.get("city")
        if city:
            city_lower = city.lower().strip()
            for c in cities:
                if c.lower() == city_lower or city_lower in c.lower():
                    intent["city"] = c
                    break

        # Ensure all fields exist
        defaults = {
            "service_type": None,
            "city": None,
            "area": None,
            "time_preference": None,
            "urgency": "normal",
            "language_detected": "english",
            "price_preference": None,
            "confidence": {},
        }
        for key, default in defaults.items():
            if key not in intent:
                intent[key] = default

        return intent

    def _check_clarification(self, intent: dict) -> list[str]:
        """Determine which fields need clarification."""
        missing = []
        if not intent.get("service_type"):
            missing.append("service_type")
        if not intent.get("city"):
            missing.append("city")
        return missing

    def _empty_intent(self) -> dict:
        """Return an empty intent structure."""
        return {
            "service_type": None,
            "city": None,
            "area": None,
            "time_preference": None,
            "urgency": "normal",
            "language_detected": "english",
            "price_preference": None,
            "confidence": {},
            "needs_clarification": ["service_type", "city"],
        }

    def get_clarification_message(self, missing_fields: list[str], prompts_config: dict) -> str:
        """Get appropriate clarification message for missing fields."""
        clarifications = prompts_config.get("clarification", {})

        if "service_type" in missing_fields and "city" in missing_fields:
            return clarifications.get("missing_both", "Please tell me what service you need and your location.")
        elif "service_type" in missing_fields:
            return clarifications.get("missing_service", "What kind of service do you need?")
        elif "city" in missing_fields:
            return clarifications.get("missing_location", "Which city are you in?")
        return "Could you provide more details?"
