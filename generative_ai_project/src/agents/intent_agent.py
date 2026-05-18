"""
Intent Agent — Natural Language Understanding via LLM.

Extracts structured intent from user messages in English,
Urdu, or Roman Urdu. The model THINKS about what the user
needs rather than pattern-matching.
"""

import json
import logging
import re
import time
from typing import Optional

from ..core.base_llm import BaseLLM
from ..core.runtime_config import consume_llm_call_budget, env_bool

logger = logging.getLogger("agents.intent")

_SERVICE_ALIASES = {
    "Plumber": ["plumber", "plumbing", "pipe", "leak", "leakage", "pani", "nalka", "tank leak"],
    "Electrician": ["electrician", "bijli", "electric", "switch", "socket", "wiring", "fan", "pankha", "spark"],
    "AC Technician": ["ac", "a/c", "air conditioner", "split ac", "cooling", "gas refill", "ac technician"],
    "Carpenter": ["carpenter", "wood", "darzi", "furniture", "darwaza", "almari"],
    "Mechanic": ["mechanic", "car repair", "bike repair", "gaari", "gari", "motorcycle"],
    "Cleaning Service": ["cleaning", "safai", "cleaner", "deep clean", "maid"],
    "Appliance Repair": ["appliance", "fridge", "washing machine", "microwave", "refrigerator"],
    "Mobile Repair": ["mobile", "phone repair", "screen", "battery", "charging port"],
    "Computer Technician": ["computer", "laptop", "pc", "printer", "windows"],
    "Painter": ["painter", "paint", "rang", "wall paint"],
    "Tutor": ["tutor", "teacher", "tuition", "home tutor", "math tutor", "english tutor"],
    "Beautician": ["beautician", "makeup", "salon", "mehndi", "facial"],
    "Water Tank Cleaner": ["water tank", "tank cleaning", "tank cleaner"],
}

_TIME_PATTERNS = [
    (r"\b(today|aaj)\b", "today"),
    (r"\b(tomorrow|kal)\b", "tomorrow"),
    (r"\b(morning|subah)\b", "morning"),
    (r"\b(afternoon|dupehar|dopahar)\b", "afternoon"),
    (r"\b(evening|shaam)\b", "evening"),
    (r"\b(night|raat)\b", "night"),
    (r"\b(now|abhi|foran|jaldi|asap)\b", "as soon as possible"),
]

_URGENT_TERMS = ["urgent", "emergency", "foran", "jaldi", "abhi", "immediately", "asap", "leak", "spark", "smoke"]
_ROMAN_URDU_HINTS = ["mujhe", "chahiye", "karwana", "hai", "kal", "aaj", "abhi", "foran", "jaldi", "mein", "kaam"]


class IntentAgent:
    """Extracts structured intent from natural language input."""

    def __init__(self, llm: BaseLLM, prompts_config: dict, agents_config: dict, guardrails=None, cag_manager=None):
        self.llm = llm
        self.prompt_template = prompts_config.get("intent_extraction", "")
        self.config = agents_config.get("agents", {}).get("intent", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.70)
        self.guardrails = guardrails
        self.cag_manager = cag_manager

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
        start = time.time()
        rule_intent = self._rule_based_intent(user_message, service_categories, cities)
        if not env_bool("ENABLE_LLM_INTENT", True) or not rule_intent.get("needs_clarification"):
            logger.info(
                "Intent extraction completed via rules in %.1fms | service=%s city=%s missing=%s",
                (time.time() - start) * 1000,
                rule_intent.get("service_type"),
                rule_intent.get("city"),
                rule_intent.get("needs_clarification"),
            )
            return rule_intent

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
        system_instruction = self._with_cag("intent", system_instruction)

        if not consume_llm_call_budget("intent.extract"):
            logger.info("Intent LLM fallback skipped; returning rule-based intent.")
            return rule_intent

        llm_start = time.time()
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
        logger.info("Intent LLM fallback completed in %.1fms", (time.time() - llm_start) * 1000)

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
        intent = self._merge_rule_intent(rule_intent, intent)

        # Check confidence
        intent["needs_clarification"] = self._check_clarification(intent)

        logger.info(f"Extracted intent: {json.dumps(intent, indent=2)}")
        logger.info("Intent extraction total time %.1fms", (time.time() - start) * 1000)
        return intent

    def _rule_based_intent(self, user_message: str, categories: list[str], cities: list[str]) -> dict:
        text = (user_message or "").strip()
        lowered = text.lower()
        service_type = self._extract_service(lowered, categories)
        city = self._extract_city(lowered, cities)
        area = self._extract_area(text, city)
        time_preference = self._extract_time(lowered)
        urgency = "urgent" if any(term in lowered for term in _URGENT_TERMS) else "normal"
        language = "roman urdu" if any(f" {hint} " in f" {lowered} " for hint in _ROMAN_URDU_HINTS) else "english"
        price_preference = self._extract_price(lowered)

        confidence = {
            "service_type": 0.90 if service_type else 0.20,
            "city": 0.90 if city else 0.20,
            "area": 0.75 if area else 0.30,
            "time_preference": 0.75 if time_preference else 0.40,
            "urgency": 0.80,
        }
        intent = {
            "service_type": service_type,
            "city": city,
            "area": area,
            "time_preference": time_preference,
            "urgency": urgency,
            "language_detected": language,
            "price_preference": price_preference,
            "confidence": confidence,
        }
        intent["needs_clarification"] = self._check_clarification(intent)
        return intent

    def _extract_service(self, lowered: str, categories: list[str]) -> Optional[str]:
        for category in categories:
            if category.lower() in lowered:
                return category
        for canonical, aliases in _SERVICE_ALIASES.items():
            if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
                for category in categories:
                    if category.lower() == canonical.lower():
                        return category
                return canonical
        return None

    def _extract_city(self, lowered: str, cities: list[str]) -> Optional[str]:
        for city in cities:
            if re.search(rf"\b{re.escape(city.lower())}\b", lowered):
                return city
        aliases = {
            "isb": "Islamabad",
            "rwp": "Rawalpindi",
            "khi": "Karachi",
            "lhr": "Lahore",
        }
        for alias, canonical in aliases.items():
            if re.search(rf"\b{alias}\b", lowered):
                for city in cities:
                    if city.lower() == canonical.lower():
                        return city
                return canonical
        return None

    def _extract_area(self, text: str, city: Optional[str]) -> Optional[str]:
        lowered = text.lower()
        patterns = [
            r"\bin\s+([A-Za-z0-9\-/ ]{2,40})(?:\s+(?:mein|me|today|tomorrow|kal|aaj|urgent|chahiye|for|ke liye)\b|$)",
            r"\bmein\s+([A-Za-z0-9\-/ ]{2,40})(?:\s+(?:chahiye|service|kaam|urgent)\b|$)",
            r"\b([A-Za-z]-?\d{1,2}(?:/\d{1,2})?)\b",
            r"\b(DHA(?:\s+Phase\s+\d+)?|Gulberg|Bahria Town|Johar Town|Clifton|G-13|F-10)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                area = match.group(1).strip(" ,.-")
                if city:
                    area = re.sub(rf"\b{re.escape(city)}\b", "", area, flags=re.IGNORECASE).strip(" ,.-")
                if not area or (city and area.lower() == city.lower()):
                    continue
                return area
        if city and city.lower() in lowered:
            return None
        return None

    def _extract_time(self, lowered: str) -> Optional[str]:
        values = []
        for pattern, value in _TIME_PATTERNS:
            if re.search(pattern, lowered):
                values.append(value)
        return " ".join(values) if values else None

    def _extract_price(self, lowered: str) -> Optional[str]:
        if any(term in lowered for term in ["cheap", "sasta", "low budget", "budget"]):
            return "Low"
        if any(term in lowered for term in ["best", "premium", "high quality"]):
            return "High"
        return None

    def _merge_rule_intent(self, rule_intent: dict, llm_intent: dict) -> dict:
        for field in ["service_type", "city", "area", "time_preference", "urgency", "language_detected", "price_preference"]:
            if not llm_intent.get(field) and rule_intent.get(field):
                llm_intent[field] = rule_intent[field]
        confidence = llm_intent.get("confidence") or {}
        for key, value in (rule_intent.get("confidence") or {}).items():
            confidence.setdefault(key, value)
        llm_intent["confidence"] = confidence
        return llm_intent

    def _with_cag(self, agent_name: str, system_instruction: str) -> str:
        if not self.cag_manager:
            return system_instruction
        context = self.cag_manager.get_context_for_agent(agent_name)
        logger.info(
            "CAG injection for %s LLM call | injected=%s chars=%s",
            agent_name,
            bool(context),
            len(context),
        )
        if not context:
            return system_instruction
        return f"{system_instruction}\n\n## DOMAIN KNOWLEDGE\n{context}"

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
