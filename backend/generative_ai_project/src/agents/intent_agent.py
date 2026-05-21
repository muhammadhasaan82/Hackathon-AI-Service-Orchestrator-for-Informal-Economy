"""
Intent Agent - model-first natural language understanding.

The probabilistic path asks the configured LLM to extract intent from the
message. The deterministic path is only a config-driven fallback for offline
CPU tests, depleted LLM budgets, or invalid model JSON.
"""

import json
import logging
import time
from difflib import SequenceMatcher
from typing import Any, Optional

from ..core.base_llm import BaseLLM
from ..core.runtime_config import consume_llm_call_budget, env_bool

logger = logging.getLogger("agents.intent")

_DEFAULT_RESPONSE_SCHEMA = {
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
}

_INTENT_FIELDS = (
    "service_type",
    "city",
    "area",
    "time_preference",
    "urgency",
    "language_detected",
    "price_preference",
)


class IntentAgent:
    """Extracts structured intent from natural language input."""

    def __init__(
        self,
        llm: BaseLLM,
        prompts_config: dict,
        agents_config: dict,
        guardrails=None,
        cag_manager=None,
    ):
        self.llm = llm
        self.prompt_template = prompts_config.get("intent_extraction", "")
        self.config = agents_config.get("agents", {}).get("intent", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.70)
        self.deterministic_config = self.config.get("deterministic_hints", {})
        self.normalization_config = self.config.get("normalization", {})
        self.guardrails = guardrails
        self.cag_manager = cag_manager

    def _extract_deterministic_urdu_roman(
        self,
        text: str,
        categories: list[str],
        cities: list[str],
        areas: Optional[list[str]] = None,
    ) -> dict:
        import re
        text_lower = text.lower()
        
        extracted_service = None
        extracted_city = None
        extracted_area = None
        
        # Urdu/Roman Urdu/English service keyword mapping
        urdu_service_map = {
            "پلمبر": "Plumber",
            "plumber": "Plumber",
            "الیکٹریشن": "Electrician",
            "electrician": "Electrician",
            "اے سی": "AC Technician",
            "ac": "AC Technician",
            "ac technician": "AC Technician",
            "مکینک": "Mechanic",
            "mechanic": "Mechanic",
            "ٹیوٹر": "Tutor",
            "tutor": "Tutor",
            "کارپینٹر": "Carpenter",
            "carpenter": "Carpenter",
            "پینٹر": "Painter",
            "painter": "Painter",
            "صفائی": "Cleaning Service",
            "cleaning": "Cleaning Service",
            "موبائل": "Mobile Repair",
            "mobile repair": "Mobile Repair",
            "موبائل ریپیئر": "Mobile Repair",
            "کمپیوٹر": "Computer Technician",
            "computer technician": "Computer Technician",
            "کمپیوٹر ٹیکنیشن": "Computer Technician"
        }

        # Urdu/Roman/English city mapping
        urdu_city_map = {
            "کراچی": "Karachi",
            "karachi": "Karachi",
            "لاہور": "Lahore",
            "lahore": "Lahore",
            "اسلام آباد": "Islamabad",
            "اسلام‌آباد": "Islamabad",
            "islamabad": "Islamabad",
            "فیصل آباد": "Faisalabad",
            "فیصل‌آباد": "Faisalabad",
            "faisalabad": "Faisalabad",
            "پشاور": "Peshawar",
            "peshawar": "Peshawar",
            "راولپنڈی": "Rawalpindi",
            "rawalpindi": "Rawalpindi"
        }

        # Urdu/Roman/English area mapping
        urdu_area_map = {
            # Karachi
            "کلفٹن": "Clifton",
            "clifton": "Clifton",
            "ڈی ایچ اے": "DHA",
            "dha": "DHA",
            "گلشن": "Gulshan",
            "gulshan": "Gulshan",
            "کورنگی": "Korangi",
            "korangi": "Korangi",
            "ملیر": "Malir",
            "malir": "Malir",
            "ناظم آباد": "Nazimabad",
            "ناظم‌آباد": "Nazimabad",
            "nazimabad": "Nazimabad",
            "نارتھ ناظم آباد": "North Nazimabad",
            "نارتھ ناظم‌آباد": "North Nazimabad",
            "north nazimabad": "North Nazimabad",
            
            # Islamabad / Rawalpindi / Lahore
            "بحریہ ٹاؤن": "Bahria Town",
            "بحریہ‌ٹاؤن": "Bahria Town",
            "bahria town": "Bahria Town",
            "bahria": "Bahria Town",
            "بلیو ایریا": "Blue Area",
            "blue area": "Blue Area",
            "ایف 10": "F-10",
            "ایف-10": "F-10",
            "f-10": "F-10",
            "f10": "F-10",
            "جی 13": "G-13",
            "جی-13": "G-13",
            "g-13": "G-13",
            "g13": "G-13",
            "آئی 8": "I-8",
            "آئی-8": "I-8",
            "i-8": "I-8",
            "i8": "I-8",
            
            # Faisalabad
            "ڈی گراؤنڈ": "D Ground",
            "d ground": "D Ground",
            "d-ground": "D Ground",
            "مدینہ ٹاؤن": "Madina Town",
            "madina town": "Madina Town",
            "پیپلز کالونی": "Peoples Colony",
            "peoples colony": "Peoples Colony",
            
            # Lahore
            "گلبرگ": "Gulberg",
            "gulberg": "Gulberg",
            "جوہر ٹاؤن": "Johar Town",
            "johar town": "Johar Town",
            "ماڈل ٹاؤن": "Model Town",
            "model town": "Model Town",
            
            # Peshawar
            "کینٹ": "Cantt",
            "cantt": "Cantt",
            "حیات آباد": "Hayatabad",
            "hayatabad": "Hayatabad",
            "یونیورسٹی ٹاؤن": "University Town",
            "university town": "University Town",
            
            # Rawalpindi
            "چکلالہ": "Chaklala",
            "chaklala": "Chaklala",
            "پی ڈبلیو ڈی": "PWD",
            "pwd": "PWD",
            "صدر": "Saddar",
            "saddar": "Saddar"
        }

        # 1. Match service
        sorted_service_keys = sorted(urdu_service_map.keys(), key=len, reverse=True)
        for key in sorted_service_keys:
            is_match = False
            if any('\u0600' <= char <= '\u06FF' for char in key):
                if key in text:
                    is_match = True
            else:
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower):
                    is_match = True
            if is_match:
                extracted_service = urdu_service_map[key]
                break
                
        # 2. Match city
        sorted_city_keys = sorted(urdu_city_map.keys(), key=len, reverse=True)
        for key in sorted_city_keys:
            is_match = False
            if any('\u0600' <= char <= '\u06FF' for char in key):
                if key in text:
                    is_match = True
            else:
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower):
                    is_match = True
            if is_match:
                extracted_city = urdu_city_map[key]
                break
                
        # 3. Match area
        sorted_area_keys = sorted(urdu_area_map.keys(), key=len, reverse=True)
        for key in sorted_area_keys:
            is_match = False
            if any('\u0600' <= char <= '\u06FF' for char in key):
                if key in text:
                    is_match = True
            else:
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower):
                    is_match = True
            if is_match:
                extracted_area = urdu_area_map[key]
                break
                
        return {
            "service_type": extracted_service,
            "city": extracted_city,
            "area": extracted_area
        }

    async def extract_intent(
        self,
        user_message: str,
        service_categories: list[str],
        cities: list[str],
        areas: Optional[list[str]] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Extract structured intent from a user message.

        LLM extraction is attempted first when enabled. A deterministic,
        config-driven fallback is retained for reliability and tests.
        """
        start = time.time()
        
        # Determine language
        from ..core.guardrails import detect_language
        lang = "urdu" if any("\u0600" <= char <= "\u06ff" for char in user_message) else detect_language(user_message)
        lang = lang.lower()
        
        # Scan history for user turns to see if conversation language is Urdu or Roman Urdu
        history_lang = None
        if conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "user":
                    content = turn.get("content", "")
                    curr_lang = "urdu" if any("\u0600" <= char <= "\u06ff" for char in content) else detect_language(content)
                    if curr_lang in ("urdu", "roman_urdu"):
                        history_lang = curr_lang.lower()
                        break
        effective_lang = lang if lang in ("urdu", "roman_urdu") else (history_lang or lang)
        effective_lang = effective_lang.lower()
        
        # Try deterministic Urdu/Roman Urdu mapping on current message
        det_map = self._extract_deterministic_urdu_roman(user_message, service_categories, cities, areas)
        
        # Scan history for service_type, city, area
        history_service = None
        history_city = None
        history_area = None
        if conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "user":
                    hist_msg = turn.get("content", "")
                    hist_det = self._extract_deterministic_urdu_roman(hist_msg, service_categories, cities, areas)
                    if not history_service and hist_det["service_type"]:
                        history_service = hist_det["service_type"]
                    if not history_city and hist_det["city"]:
                        history_city = hist_det["city"]
                    if not history_area and hist_det["area"]:
                        history_area = hist_det["area"]

        service_found = det_map["service_type"] or history_service
        city_found = det_map["city"] or history_city
        area_found = det_map["area"] or history_area

        # High-performance LLM Bypass if service, city, or area is present
        if effective_lang in ("urdu", "roman_urdu") and (service_found or city_found or area_found):
            logger.info("Deterministic mapping found fields. Bypassing LLM call. intent_source='deterministic_bypass'")
            intent = {
                "service_type": service_found,
                "city": city_found,
                "area": area_found,
                "time_preference": None,
                "urgency": "normal",
                "language_detected": effective_lang,
                "price_preference": None,
                "confidence": {
                    "service_type": 1.0 if service_found else 0.20,
                    "city": 1.0 if city_found else 0.20,
                    "area": 1.0 if area_found else 0.30,
                    "time_preference": 1.0,
                    "urgency": 1.0,
                    "language_detected": 1.0,
                    "price_preference": 1.0,
                },
                "intent_source": "deterministic_bypass",
                "needs_clarification": []
            }
            intent["needs_clarification"] = self._check_clarification(intent)
            return intent

        fallback_intent = self._deterministic_intent(user_message, service_categories, cities, areas)
        
        # Override/prefer deterministic mapping values in fallback_intent
        for field in ["service_type", "city", "area"]:
            if det_map[field]:
                fallback_intent[field] = det_map[field]
                fallback_intent["confidence"][field] = 1.0

        if self.llm is None or not env_bool("ENABLE_LLM_INTENT", True):
            logger.info(
                "Intent extraction completed via deterministic fallback in %.1fms | service=%s city=%s missing=%s",
                (time.time() - start) * 1000,
                fallback_intent.get("service_type"),
                fallback_intent.get("city"),
                fallback_intent.get("needs_clarification"),
            )
            fallback_intent["intent_source"] = "deterministic_fallback"
            fallback_intent["language_detected"] = effective_lang
            fallback_intent["needs_clarification"] = self._check_clarification(fallback_intent)
            return fallback_intent

        if not consume_llm_call_budget("intent.extract"):
            logger.info("Intent LLM skipped by call budget; returning deterministic fallback.")
            fallback_intent["intent_source"] = "deterministic_fallback"
            fallback_intent["language_detected"] = effective_lang
            fallback_intent["needs_clarification"] = self._check_clarification(fallback_intent)
            return fallback_intent

        prompt = self._build_prompt(
            user_message=user_message,
            service_categories=service_categories,
            cities=cities,
            areas=areas,
            conversation_history=conversation_history,
        )
        system_instruction = (
            "You are an expert intent parser for a service booking system in Pakistan. "
            "You understand English, Urdu, and Roman Urdu. Return valid JSON only. "
            "Use null for unknown fields instead of guessing."
        )
        if self.guardrails:
            system_instruction = self.guardrails.compose_system_instruction("intent", system_instruction)
        system_instruction = self._with_cag("intent", system_instruction)

        logger.info("Extracting intent with model from: %r", user_message)
        try:
            llm_start = time.time()
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_schema=_DEFAULT_RESPONSE_SCHEMA,
                system_instruction=system_instruction,
            )
            logger.info("Intent LLM completed in %.1fms", (time.time() - llm_start) * 1000)
            model_intent = self._parse_intent_response(response.text)
        except Exception as exc:
            logger.warning("Intent LLM failed; falling back to deterministic extraction: %s", exc)
            fallback_intent["intent_source"] = "deterministic_fallback"
            fallback_intent["language_detected"] = effective_lang
            fallback_intent["needs_clarification"] = self._check_clarification(fallback_intent)
            return fallback_intent

        intent = self._normalize_intent(model_intent, service_categories, cities, areas)
        intent = self._merge_deterministic_intent(fallback_intent, intent)
        
        # Override/prefer deterministic mapping values in LLM/merged intent
        overridden = False
        for field in ["service_type", "city", "area"]:
            if det_map[field]:
                if intent.get(field) != det_map[field]:
                    overridden = True
                intent[field] = det_map[field]
                intent["confidence"][field] = 1.0
                
        if overridden:
            intent["intent_source"] = "llm_with_deterministic_override"
            logger.info("Deterministic mapping corrected LLM output. intent_source='llm_with_deterministic_override'")
        else:
            intent["intent_source"] = "llm"
            
        intent["language_detected"] = effective_lang
        intent["needs_clarification"] = self._check_clarification(intent)

        logger.info("Extracted intent: %s", json.dumps(intent, ensure_ascii=False))
        logger.info("Intent extraction total time %.1fms", (time.time() - start) * 1000)
        return intent

    def _build_prompt(
        self,
        user_message: str,
        service_categories: list[str],
        cities: list[str],
        areas: Optional[list[str]],
        conversation_history: Optional[list[dict]],
    ) -> str:
        area_limit = int(self.normalization_config.get("max_area_prompt_items", 80))
        area_values = ", ".join((areas or [])[:area_limit])
        template = self.prompt_template or (
            "Extract service booking intent as JSON.\n"
            "Services: {service_categories}\n"
            "Cities: {cities}\n"
            "Known areas: {areas}\n"
            "User message: {user_message}"
        )
        prompt = template.format(
            user_message=user_message,
            service_categories=", ".join(service_categories),
            cities=", ".join(cities),
            areas=area_values,
        )

        if conversation_history:
            history = conversation_history[-4:]
            history_text = "\n".join(
                f"{turn.get('role', 'user').title()}: {str(turn.get('content', ''))[:200]}"
                for turn in history
                if turn.get("content")
            )
            if history_text:
                prompt = f"{prompt}\n\nConversation history:\n{history_text}"
        return prompt

    def _parse_intent_response(self, text: str) -> dict:
        if not text:
            return self._empty_intent()
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else self._empty_intent()
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(text[index:])
                    return payload if isinstance(payload, dict) else self._empty_intent()
                except json.JSONDecodeError:
                    continue
        logger.error("Failed to parse intent JSON: %s", text)
        return self._empty_intent()

    def _deterministic_intent(
        self,
        user_message: str,
        categories: list[str],
        cities: list[str],
        areas: Optional[list[str]] = None,
    ) -> dict:
        text = user_message or ""
        service_type, service_score = self._match_catalog(
            text,
            categories,
            self._aliases("category_aliases"),
            threshold=float(self.normalization_config.get("service_similarity_threshold", 0.72)),
        )
        city, city_score = self._match_catalog(
            text,
            cities,
            self._aliases("city_aliases"),
            threshold=float(self.normalization_config.get("city_similarity_threshold", 0.78)),
        )
        area, area_score = self._match_catalog(
            text,
            areas or [],
            {},
            threshold=float(self.normalization_config.get("area_similarity_threshold", 0.84)),
        )
        if not area:
            area = self._extract_area_candidate(text, city, service_type, categories, cities)
            area_score = 0.66 if area else 0.30

        time_preference, time_score = self._extract_config_signal(text, "time_phrases")
        urgency, urgency_score = self._extract_urgency(text)
        language, language_score = self._detect_language(text)
        price_preference, price_score = self._extract_config_signal(text, "price_preferences")

        intent = {
            "service_type": service_type,
            "city": city,
            "area": area,
            "time_preference": time_preference,
            "urgency": urgency or "normal",
            "language_detected": language,
            "price_preference": price_preference,
            "confidence": {
                "service_type": service_score if service_type else 0.20,
                "city": city_score if city else 0.20,
                "area": area_score if area else 0.30,
                "time_preference": time_score if time_preference else 0.40,
                "urgency": urgency_score,
                "language_detected": language_score,
                "price_preference": price_score if price_preference else 0.40,
            },
        }
        intent["needs_clarification"] = self._check_clarification(intent)
        return intent

    def _rule_based_intent(
        self,
        user_message: str,
        categories: list[str],
        cities: list[str],
        areas: Optional[list[str]] = None,
    ) -> dict:
        """Backward-compatible alias for older regression tests."""
        return self._deterministic_intent(user_message, categories, cities, areas)

    def _normalize_intent(
        self,
        intent: Optional[dict],
        categories: list[str],
        cities: list[str],
        areas: Optional[list[str]] = None,
    ) -> dict:
        payload = self._intent_defaults()
        if isinstance(intent, dict):
            payload.update({key: value for key, value in intent.items() if key in payload or key == "needs_clarification"})

        confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
        payload["confidence"] = {key: self._coerce_confidence(value) for key, value in confidence.items()}

        service = payload.get("service_type")
        if service and categories:
            match, score = self._match_catalog(
                str(service),
                categories,
                self._aliases("category_aliases"),
                threshold=float(self.normalization_config.get("service_similarity_threshold", 0.72)),
            )
            payload["service_type"] = match
            payload["confidence"].setdefault("service_type", score if match else 0.20)

        city = payload.get("city")
        if city and cities:
            match, score = self._match_catalog(
                str(city),
                cities,
                self._aliases("city_aliases"),
                threshold=float(self.normalization_config.get("city_similarity_threshold", 0.78)),
            )
            payload["city"] = match
            payload["confidence"].setdefault("city", score if match else 0.20)

        area = payload.get("area")
        if area:
            cleaned_area = self._clean_text_value(area)
            match, score = self._match_catalog(
                cleaned_area,
                areas or [],
                {},
                threshold=float(self.normalization_config.get("area_similarity_threshold", 0.84)),
            )
            payload["area"] = match or cleaned_area
            payload["confidence"].setdefault("area", score if match else 0.62)

        for field in _INTENT_FIELDS:
            if isinstance(payload.get(field), str):
                payload[field] = self._clean_text_value(payload[field])

        payload["urgency"] = (payload.get("urgency") or "normal").lower()
        payload["language_detected"] = payload.get("language_detected") or "unknown"
        payload["needs_clarification"] = self._check_clarification(payload)
        return payload

    def _merge_deterministic_intent(self, fallback_intent: dict, model_intent: dict) -> dict:
        merged = dict(model_intent)
        for field in _INTENT_FIELDS:
            if not merged.get(field) and fallback_intent.get(field):
                merged[field] = fallback_intent[field]

        confidence = dict(merged.get("confidence") or {})
        for key, value in (fallback_intent.get("confidence") or {}).items():
            confidence.setdefault(key, value)
        merged["confidence"] = confidence
        return merged

    def _match_catalog(
        self,
        text: str,
        choices: list[str],
        aliases: dict[str, list[str]],
        threshold: float,
    ) -> tuple[Optional[str], float]:
        if not text or not choices:
            return None, 0.0

        normalized_text = self._normalize_text(text)
        best_choice = None
        best_score = 0.0

        for choice in choices:
            phrases = [choice]
            phrases.extend(aliases.get(choice, []))
            phrases.extend(aliases.get(choice.lower(), []))
            for phrase in phrases:
                normalized_phrase = self._normalize_text(phrase)
                if not normalized_phrase:
                    continue
                if self._contains_phrase(normalized_text, normalized_phrase):
                    score = 1.0 if self._normalize_text(choice) == normalized_phrase else 0.90
                else:
                    score = self._window_similarity(normalized_text, normalized_phrase)
                if score > best_score:
                    best_choice = choice
                    best_score = score

        if best_score >= threshold:
            return best_choice, round(float(best_score), 2)
        return None, round(float(best_score), 2)

    def _extract_config_signal(self, text: str, config_key: str) -> tuple[Optional[str], float]:
        normalized_text = self._normalize_text(text)
        signals = self.deterministic_config.get(config_key, {}) or {}
        for canonical, phrases in signals.items():
            if any(self._contains_phrase(normalized_text, self._normalize_text(phrase)) for phrase in phrases):
                return canonical, 0.76
        return None, 0.40

    def _extract_urgency(self, text: str) -> tuple[str, float]:
        urgency, score = self._extract_config_signal(text, "urgency_phrases")
        return urgency or "normal", score if urgency else 0.58

    def _detect_language(self, text: str) -> tuple[str, float]:
        if any("\u0600" <= char <= "\u06ff" for char in text):
            return "urdu", 0.90
        language, score = self._extract_config_signal(text, "language_hints")
        return language or "unknown", score if language else 0.45

    def _extract_area_candidate(
        self,
        text: str,
        city: Optional[str],
        service_type: Optional[str],
        categories: list[str],
        cities: list[str],
    ) -> Optional[str]:
        tokens = [token.strip(" ,.;:!?()[]{}\"'") for token in text.split()]
        normalized_tokens = [self._normalize_text(token) for token in tokens]
        prepositions = {
            self._normalize_text(value)
            for value in self.deterministic_config.get("area_prepositions", [])
        }
        stop_terms = {
            self._normalize_text(value)
            for value in self.deterministic_config.get("area_stop_terms", [])
        }
        boundary_tokens = set(stop_terms)
        for phrase in [*(categories or []), *(cities or []), city or "", service_type or ""]:
            boundary_tokens.update(self._normalize_text(phrase).split())
        boundary_tokens.update(self._all_signal_tokens())

        for index, token in enumerate(normalized_tokens):
            if token not in prepositions:
                continue
            candidate = []
            for raw_token, normalized in zip(tokens[index + 1 :], normalized_tokens[index + 1 :]):
                if not normalized or normalized in prepositions or normalized in boundary_tokens:
                    break
                candidate.append(raw_token)
                if len(candidate) >= int(self.normalization_config.get("max_area_tokens", 5)):
                    break
            area = self._clean_text_value(" ".join(candidate))
            if area and self._normalize_text(area) != self._normalize_text(city or ""):
                return area
        return None

    def _all_signal_tokens(self) -> set[str]:
        tokens = set()
        for value in self.deterministic_config.values():
            if isinstance(value, dict):
                for phrases in value.values():
                    if isinstance(phrases, list):
                        for phrase in phrases:
                            tokens.update(self._normalize_text(phrase).split())
            elif isinstance(value, list):
                for phrase in value:
                    tokens.update(self._normalize_text(phrase).split())
        return tokens

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

    def _aliases(self, key: str) -> dict[str, list[str]]:
        values = self.deterministic_config.get(key, {}) or {}
        return {
            str(canonical): [str(alias) for alias in aliases]
            for canonical, aliases in values.items()
            if isinstance(aliases, list)
        }

    def _window_similarity(self, normalized_text: str, normalized_phrase: str) -> float:
        text_tokens = normalized_text.split()
        phrase_tokens = normalized_phrase.split()
        if not text_tokens or not phrase_tokens:
            return 0.0

        best = SequenceMatcher(None, normalized_text, normalized_phrase).ratio()
        phrase_len = len(phrase_tokens)
        for size in {phrase_len - 1, phrase_len, phrase_len + 1}:
            if size <= 0:
                continue
            for index in range(0, max(0, len(text_tokens) - size) + 1):
                window = " ".join(text_tokens[index : index + size])
                best = max(best, SequenceMatcher(None, window, normalized_phrase).ratio())
        return float(best)

    def _contains_phrase(self, normalized_text: str, normalized_phrase: str) -> bool:
        if not normalized_text or not normalized_phrase:
            return False
        return f" {normalized_phrase} " in f" {normalized_text} "

    def _normalize_text(self, value: Any) -> str:
        text = str(value or "").casefold()
        normalized_chars = []
        for char in text:
            normalized_chars.append(char if char.isalnum() else " ")
        return " ".join("".join(normalized_chars).split())

    def _clean_text_value(self, value: Any) -> Optional[str]:
        text = " ".join(str(value or "").strip(" ,.;:!?()[]{}\"'").split())
        return text or None

    def _coerce_confidence(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(score, 1.0))

    def _intent_defaults(self) -> dict:
        return {
            "service_type": None,
            "city": None,
            "area": None,
            "time_preference": None,
            "urgency": "normal",
            "language_detected": "unknown",
            "price_preference": None,
            "confidence": {},
            "intent_source": "llm",
        }

    def _check_clarification(self, intent: dict) -> list[str]:
        """Determine which required fields need clarification."""
        missing = []
        if not intent.get("service_type"):
            missing.append("service_type")
        if not intent.get("city"):
            missing.append("city")
        return missing

    def _empty_intent(self) -> dict:
        intent = self._intent_defaults()
        intent["needs_clarification"] = ["service_type", "city"]
        return intent

    def get_clarification_message(self, missing_fields: list[str], prompts_config: dict) -> str:
        """Get an appropriate clarification message for missing fields."""
        clarifications = prompts_config.get("clarification", {})

        if "service_type" in missing_fields and "city" in missing_fields:
            return clarifications.get("missing_both", "Please tell me what service you need and your location.")
        if "service_type" in missing_fields:
            return clarifications.get("missing_service", "What kind of service do you need?")
        if "city" in missing_fields:
            return clarifications.get("missing_location", "Which city are you in?")
        return clarifications.get("fallback", "Could you provide more details?")
