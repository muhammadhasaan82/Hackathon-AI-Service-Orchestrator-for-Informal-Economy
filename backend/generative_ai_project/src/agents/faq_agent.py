"""
FAQ Agent - hybrid FAQ and booking route decomposition.

The agent is deterministic-first:
1. regex patterns from YAML;
2. dataset-derived entity matching and ambiguity checks;
3. cached MiniLM semantic matching as fallback;
4. structured CAG policy retrieval.

It does not call a generative LLM. If a message contains both FAQ and
booking components, it returns an FAQ answer plus a signal that the
orchestrator should continue the booking workflow.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from ..core.semantic_matcher import SemanticFAQMatcher

logger = logging.getLogger("agents.faq")


class FAQAgent:
    """Lightweight FAQ router with hybrid intent decomposition."""

    def __init__(
        self,
        dataset_facts,
        cag_manager,
        faq_config: dict,
        llm=None,
    ):
        self.facts = dataset_facts
        self.cag = cag_manager
        self.config = faq_config or {}
        self._llm = llm

        self.routing_config = self.config.get("routing", {})
        self.hybrid_config = self.config.get("hybrid", {})
        self.semantic_config = self.config.get("semantic", {})
        self.ambiguity_config = self.config.get("ambiguity", {})
        self._templates = self.config.get("response_templates", {})
        self._faq_classes = self.config.get("faq_patterns", {})
        self._regex_confidence = float(self.routing_config.get("regex_confidence", 0.92))

        self._compiled_patterns: list[tuple[str, re.Pattern, dict]] = []
        for faq_class, class_cfg in self._faq_classes.items():
            for pattern_str in class_cfg.get("patterns", []):
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    self._compiled_patterns.append((faq_class, compiled, class_cfg))
                except re.error as exc:
                    logger.warning("Invalid FAQ regex '%s' in %s: %s", pattern_str, faq_class, exc)

        self._booking_signals = {
            self._normalize_text(signal)
            for signal in self.config.get("booking_signals", [])
            if str(signal).strip()
        }
        self._semantic_matcher = SemanticFAQMatcher(
            examples_by_class=self._semantic_examples(),
            config=self.semantic_config,
        )

        logger.info(
            "FAQAgent initialized | classes=%d patterns=%d booking_signals=%d semantic_mode=%s",
            len(self._faq_classes),
            len(self._compiled_patterns),
            len(self._booking_signals),
            self._semantic_matcher.mode,
        )

    async def try_handle(self, user_message: str, session: dict) -> Optional[dict]:
        """Return FAQ/hybrid routing result, ambiguity clarification, or None."""
        start = time.time()
        normalized = str(user_message or "").strip()
        if not normalized:
            return None

        memory_context = self._memory_context(session)
        entity_context = self._entity_context(normalized, memory_context)
        ambiguity = self._detect_ambiguity(normalized, entity_context)
        if ambiguity:
            latency_ms = (time.time() - start) * 1000
            result = self._ambiguity_result(ambiguity, entity_context, latency_ms)
            logger.info(
                "FAQ ambiguity_detected | candidates=%s latency=%.1fms",
                [item.get("value") for item in ambiguity.get("candidates", [])],
                latency_ms,
            )
            return result

        faq_matches = self._faq_matches(normalized)
        if not faq_matches:
            logger.debug("FAQ skip: no route for '%s'", normalized[:80])
            return None

        has_booking_signal = self._has_booking_signal(normalized)
        should_continue_booking = self._should_continue_booking(
            faq_matches=faq_matches,
            entity_context=entity_context,
            has_booking_signal=has_booking_signal,
        )

        responses = []
        for match in faq_matches:
            response = self._compose_response(match, normalized, entity_context)
            if response:
                responses.append(response)

        if not responses:
            logger.debug("FAQ skip: no composable response for '%s'", normalized[:80])
            return None

        latency_ms = (time.time() - start) * 1000
        response_text = self._join_responses(responses)
        routing_confidence = self._routing_confidence(faq_matches, entity_context, has_booking_signal)
        status = self.hybrid_config.get("continue_booking_status", "hybrid_route") if should_continue_booking else "faq_answered"
        primary_intent = "booking" if should_continue_booking else "faq"
        secondary_intents = [match["faq_class"] for match in faq_matches]
        explainability = self._explainability(
            faq_matches=faq_matches,
            entity_context=entity_context,
            has_booking_signal=has_booking_signal,
            should_continue_booking=should_continue_booking,
            latency_ms=latency_ms,
        )

        logger.info(
            "FAQ route | status=%s primary=%s secondary=%s confidence=%.2f latency=%.1fms why=%s",
            status,
            primary_intent,
            secondary_intents,
            routing_confidence,
            latency_ms,
            explainability.get("why_selected"),
        )

        return {
            "response": response_text,
            "status": status,
            "faq_class": secondary_intents[0] if secondary_intents else None,
            "faq_source": faq_matches[0].get("source") if faq_matches else None,
            "faq_matches": faq_matches,
            "primary_intent": primary_intent,
            "secondary_intents": secondary_intents,
            "should_continue_booking": should_continue_booking,
            "routing_confidence": routing_confidence,
            "extraction_confidence": entity_context.get("extraction_confidence", 0.0),
            "dataset_match_confidence": entity_context.get("dataset_match_confidence", 0.0),
            "entity_context": entity_context,
            "explainability": explainability,
            "latency_ms": latency_ms,
        }

    def _faq_matches(self, text: str) -> list[dict]:
        matches = self._classify_by_pattern(text)
        if matches:
            return matches

        semantic_match = self._semantic_matcher.match(text)
        if not semantic_match:
            return []

        faq_class = semantic_match["faq_class"]
        class_cfg = self._faq_classes.get(faq_class, {})
        return [{
            "faq_class": faq_class,
            "topic": self._topic_for_class(faq_class),
            "source": semantic_match.get("source", "semantic"),
            "route_source": semantic_match.get("source", "semantic"),
            "confidence": float(semantic_match["score"]),
            "semantic_match_score": float(semantic_match["score"]),
            "policy_source": class_cfg.get("source", "dataset"),
            "cag_key": class_cfg.get("cag_key"),
            "config": class_cfg,
            "matched_pattern": semantic_match.get("example"),
        }]

    def _classify_by_pattern(self, text: str) -> list[dict]:
        text_lower = text.lower()
        matches: list[dict] = []
        seen = set()
        for faq_class, pattern, class_cfg in self._compiled_patterns:
            if faq_class in seen:
                continue
            if pattern.search(text_lower):
                seen.add(faq_class)
                matches.append({
                    "faq_class": faq_class,
                    "topic": self._topic_for_class(faq_class),
                    "source": "regex",
                    "route_source": "regex",
                    "confidence": self._regex_confidence,
                    "semantic_match_score": 0.0,
                    "policy_source": class_cfg.get("source", "dataset"),
                    "cag_key": class_cfg.get("cag_key"),
                    "config": class_cfg,
                    "matched_pattern": pattern.pattern,
                })
        return matches

    def _compose_response(self, match: dict, user_message: str, entity_context: dict) -> Optional[str]:
        class_cfg = match.get("config", {})
        source = class_cfg.get("source", "dataset")
        faq_class = match.get("faq_class")
        if source == "dataset":
            return self._compose_dataset_response(faq_class, user_message, entity_context)
        if source == "cag":
            return self._compose_cag_response(faq_class, class_cfg, user_message, entity_context)
        return None

    def _compose_dataset_response(self, faq_class: str, user_message: str, entity_context: dict) -> Optional[str]:
        if faq_class == "dataset_available_cities":
            cities = self.facts.get_available_cities()
            template = self._templates.get("cities_list", "Available cities: {cities}.")
            return template.strip().format(cities=", ".join(cities))

        if faq_class == "dataset_available_services":
            categories = self.facts.get_available_categories()
            template = self._templates.get("categories_list", "Available services: {categories}.")
            return template.strip().format(categories=", ".join(categories), count=len(categories))

        if faq_class == "dataset_city_areas":
            return self._compose_city_areas_response(entity_context)

        if faq_class == "dataset_service_coverage":
            return self._compose_coverage_response(entity_context)

        if faq_class == "dataset_provider_count":
            return self._compose_count_response(entity_context)

        return None

    def _compose_city_areas_response(self, entity_context: dict) -> str:
        city = entity_context.get("city")
        if city:
            areas = self.facts.get_areas_by_city(city)
            if areas:
                template = self._templates.get("city_areas", "Areas in {city}: {areas}.")
                return template.strip().format(city=city, areas=", ".join(areas))
            return self._unknown_city_response(city)

        cities = self.facts.get_available_cities()
        lines = []
        for city_name in cities:
            areas = self.facts.get_areas_by_city(city_name)
            lines.append(f"- {city_name}: {', '.join(areas)}")
        return "Here are the areas we cover:\n" + "\n".join(lines) + "\n\nWhich city are you interested in?"

    def _compose_coverage_response(self, entity_context: dict) -> str:
        city = entity_context.get("city")
        category = entity_context.get("service_type")
        area = entity_context.get("area")

        if category and city:
            count = self.facts._city_category_counts.get((city, category), 0)
            if count > 0:
                scope = f"{area}, {city}" if area else city
                availability = self.facts.get_availability_summary(city=city, category=category, area=area)
                availability_text = self._availability_text(availability)
                return (
                    f"Yes, {category} service is available in {scope} with {count} providers. "
                    f"{availability_text} Tell me your preferred time, or I can show the best matches."
                )
            return (
                f"{category} is not currently available in {city}. "
                f"Available cities are: {', '.join(self.facts.get_available_cities())}."
            )

        if category:
            coverage = self.facts.get_category_city_coverage(category)
            if coverage:
                city_list = ", ".join(f"{city_name} ({count})" for city_name, count in sorted(coverage.items()))
                return f"Yes, {category} service is available in: {city_list}. Which city do you need service in?"

        if city:
            count = self.facts.get_city_provider_count(city)
            categories = self.facts.get_available_categories()
            return (
                f"Yes, services are available in {city} with {count} providers. "
                f"Available services: {', '.join(categories)}. Which service do you need?"
            )

        cities = self.facts.get_available_cities()
        categories = self.facts.get_available_categories()
        return (
            f"We serve {len(cities)} cities ({', '.join(cities)}) "
            f"with {len(categories)} service categories. Tell me your city and required service."
        )

    def _compose_count_response(self, entity_context: dict) -> str:
        city = entity_context.get("city")
        category = entity_context.get("service_type")
        if city and category:
            count = self.facts._city_category_counts.get((city, category), 0)
            return f"We have {count} {category} providers in {city}."
        if city:
            count = self.facts.get_city_provider_count(city)
            return f"We have {count} service providers in {city}."
        if category:
            count = self.facts.get_category_provider_count(category)
            return f"We have {count} {category} providers across all cities."
        total = self.facts.get_total_provider_count()
        return (
            f"We have {total:,} service providers across "
            f"{len(self.facts.get_available_cities())} cities and "
            f"{len(self.facts.get_available_categories())} service categories."
        )

    def _compose_cag_response(
        self,
        faq_class: str,
        class_cfg: dict,
        user_message: str,
        entity_context: dict,
    ) -> Optional[str]:
        cag_key = class_cfg.get("cag_key", "")
        if not cag_key:
            return None

        policy_result = self.cag.get_structured_faq_policy(cag_key)
        if not policy_result.get("found"):
            logger.warning("CAG policy missing | key=faq_policies.%s", cag_key)
            return None

        policy = policy_result.get("policy") or {}
        wants_detail = bool(re.search(r"\b(detail|explain|full|complete|tafseel)\b", user_message, re.IGNORECASE))
        answer = policy.get("detailed_answer") if wants_detail else policy.get("short_answer")
        answer = answer or policy.get("detailed_answer")
        if not answer:
            return None

        answer = " ".join(str(answer).split())
        if faq_class == "faq_availability":
            availability = self.facts.get_availability_summary(
                city=entity_context.get("city"),
                category=entity_context.get("service_type"),
                area=entity_context.get("area"),
            )
            availability_text = self._availability_text(availability)
            if availability_text:
                answer = f"{answer} {availability_text}"

        if not self._should_suppress_followup(entity_context):
            prompts = policy.get("followup_prompts") or []
            if prompts:
                answer = f"{answer} {prompts[0]}"

        return answer

    def _entity_context(self, text: str, memory_context: dict) -> dict:
        category_threshold = float(self.routing_config.get("dataset_match_min_confidence", 0.70))
        city_threshold = 0.78
        ambiguity_threshold = float(self.ambiguity_config.get("entity_threshold", 0.58))
        margin = float(self.ambiguity_config.get("close_match_margin", 0.12))
        max_candidates = int(self.ambiguity_config.get("max_candidates", 3))

        category_match = self.facts.match_category(
            text,
            threshold=category_threshold,
            ambiguity_threshold=ambiguity_threshold,
            close_match_margin=margin,
            max_candidates=max_candidates,
        )
        city_match = self.facts.match_city(
            text,
            threshold=city_threshold,
            ambiguity_threshold=ambiguity_threshold,
            close_match_margin=margin,
            max_candidates=max_candidates,
        )

        service_type = category_match.get("value") or memory_context.get("service_type")
        city = city_match.get("value") or memory_context.get("city")
        area = self._extract_area_from_message(text, city) or memory_context.get("area")

        scores = [
            category_match.get("score", 0.0) if category_match.get("value") else 0.0,
            city_match.get("score", 0.0) if city_match.get("value") else 0.0,
        ]
        if memory_context.get("service_type") and not category_match.get("value"):
            scores.append(0.74)
        if memory_context.get("city") and not city_match.get("value"):
            scores.append(0.74)

        return {
            "service_type": service_type,
            "city": city,
            "area": area,
            "category_match": category_match,
            "city_match": city_match,
            "dataset_match_confidence": round(max(scores or [0.0]), 4),
            "extraction_confidence": round(max(scores or [0.0]), 4),
            "memory_context_used": bool(memory_context),
            "memory_context": memory_context,
        }

    def _memory_context(self, session: dict) -> dict:
        agent_state = (session or {}).get("agent_state", {}) or {}
        intent = agent_state.get("intent") or {}
        return {
            key: intent.get(key)
            for key in ("service_type", "city", "area", "time_preference", "urgency", "price_preference")
            if intent.get(key)
        }

    def _detect_ambiguity(self, text: str, entity_context: dict) -> Optional[dict]:
        if not self.ambiguity_config.get("enabled", True):
            return None

        configured = self._configured_ambiguity(text)
        if configured:
            return configured

        for key in ("category_match", "city_match"):
            match = entity_context.get(key) or {}
            if match.get("ambiguous"):
                return {
                    "type": "service_type" if key == "category_match" else "city",
                    "candidates": match.get("candidates", []),
                    "score": match.get("score", 0.0),
                    "source": match.get("source", "fuzzy"),
                }
        return None

    def _configured_ambiguity(self, text: str) -> Optional[dict]:
        normalized = self._normalize_text(text)
        term_candidates = self.ambiguity_config.get("term_candidates", {}) or {}
        for term, candidates in term_candidates.items():
            if not self._contains_phrase(normalized, self._normalize_text(term)):
                continue
            valid_candidates = []
            for candidate in candidates or []:
                found, canonical = self.facts.has_category(candidate)
                if found and canonical:
                    valid_candidates.append({"value": canonical, "score": 0.75})
            if len(valid_candidates) > 1:
                return {
                    "type": "service_type",
                    "candidates": valid_candidates[: int(self.ambiguity_config.get("max_candidates", 3))],
                    "score": 0.75,
                    "source": "configured_ambiguity",
                }
        return None

    def _ambiguity_result(self, ambiguity: dict, entity_context: dict, latency_ms: float) -> dict:
        options = [item.get("value") for item in ambiguity.get("candidates", []) if item.get("value")]
        if len(options) > 1:
            joined = " or ".join(options)
        else:
            joined = options[0] if options else "one of the available services"
        template = self.ambiguity_config.get("ask_template", "Did you mean {options}?")
        response = template.format(options=joined)
        return {
            "response": response,
            "status": "ambiguity_detected",
            "faq_class": None,
            "faq_source": ambiguity.get("source"),
            "faq_matches": [],
            "primary_intent": "clarification",
            "secondary_intents": [],
            "should_continue_booking": False,
            "routing_confidence": float(ambiguity.get("score", 0.0)),
            "extraction_confidence": entity_context.get("extraction_confidence", 0.0),
            "dataset_match_confidence": entity_context.get("dataset_match_confidence", 0.0),
            "entity_context": entity_context,
            "explainability": {
                "route_source": ambiguity.get("source", "fuzzy"),
                "regex_match": False,
                "semantic_match": False,
                "fuzzy_match_used": ambiguity.get("source") in ("fuzzy", "configured_ambiguity"),
                "ambiguity_detected": True,
                "why_selected": "Ambiguous dataset entity match requires user clarification.",
                "latency_ms": latency_ms,
            },
            "latency_ms": latency_ms,
        }

    def _should_continue_booking(self, faq_matches: list[dict], entity_context: dict, has_booking_signal: bool) -> bool:
        if not self.hybrid_config.get("enabled", True):
            return False
        has_context = bool(entity_context.get("service_type") or entity_context.get("city") or entity_context.get("area"))
        if has_booking_signal and has_context:
            return True
        continuation_topics = {"faq_availability", "dataset_service_coverage", "dataset_city_areas"}
        if has_context and any(match.get("faq_class") in continuation_topics for match in faq_matches):
            return True
        return False

    def _has_booking_signal(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        tokens = set(normalized.split())
        if tokens & self._booking_signals:
            return True
        return any(self._contains_phrase(normalized, signal) for signal in self._booking_signals if " " in signal)

    def _routing_confidence(self, faq_matches: list[dict], entity_context: dict, has_booking_signal: bool) -> float:
        faq_conf = max((float(match.get("confidence", 0.0)) for match in faq_matches), default=0.0)
        dataset_conf = float(entity_context.get("dataset_match_confidence", 0.0))
        booking_conf = float(self.routing_config.get("booking_signal_confidence", 0.76)) if has_booking_signal else 0.0
        return round(max(faq_conf, dataset_conf, booking_conf), 4)

    def _explainability(
        self,
        faq_matches: list[dict],
        entity_context: dict,
        has_booking_signal: bool,
        should_continue_booking: bool,
        latency_ms: float,
    ) -> dict:
        route_sources = sorted({match.get("route_source", match.get("source")) for match in faq_matches if match.get("source")})
        semantic_scores = [float(match.get("semantic_match_score", 0.0)) for match in faq_matches]
        faq_confidence = max((float(match.get("confidence", 0.0)) for match in faq_matches), default=0.0)
        dataset_score = float(entity_context.get("dataset_match_confidence", 0.0))
        fuzzy_used = any(
            (entity_context.get(key) or {}).get("source") == "fuzzy"
            for key in ("category_match", "city_match")
        )
        return {
            "route_source": "+".join(route_sources) if route_sources else "none",
            "regex_match": "regex" in route_sources,
            "semantic_match": any(source in ("semantic", "lexical") for source in route_sources),
            "faq_confidence": round(faq_confidence, 4),
            "semantic_match_score": round(max(semantic_scores or [0.0]), 4),
            "dataset_match_score": round(dataset_score, 4),
            "fuzzy_match_used": fuzzy_used,
            "ambiguity_detected": False,
            "policy_source": self._policy_source(faq_matches),
            "has_booking_signal": has_booking_signal,
            "should_continue_booking": should_continue_booking,
            "memory_context_used": entity_context.get("memory_context_used", False),
            "latency_ms": latency_ms,
            "why_selected": self._why_selected(route_sources, has_booking_signal, should_continue_booking),
        }

    def _policy_source(self, faq_matches: list[dict]) -> str:
        sources = {match.get("policy_source") for match in faq_matches if match.get("policy_source")}
        if "cag" in sources:
            return "cag"
        if "dataset" in sources:
            return "dataset"
        return "none"

    def _why_selected(self, route_sources: list[str], has_booking_signal: bool, should_continue_booking: bool) -> str:
        if should_continue_booking:
            return "FAQ component matched and booking context is available, so answer FAQ and continue workflow."
        if "regex" in route_sources:
            return "FAQ regex pattern matched configured topic."
        if any(source in ("semantic", "lexical") for source in route_sources):
            return "Semantic FAQ fallback matched configured examples."
        if has_booking_signal:
            return "Booking signal detected."
        return "FAQ route selected by configured matching."

    def _extract_area_from_message(self, text: str, city: Optional[str]) -> Optional[str]:
        if not city:
            return None
        text_lower = text.lower()
        for area in sorted(self.facts.get_areas_by_city(city), key=len, reverse=True):
            if area.lower() in text_lower:
                return area
        return None

    def _availability_text(self, availability: dict[str, int]) -> str:
        if not availability:
            return ""
        parts = [f"{status}: {count}" for status, count in sorted(availability.items()) if count]
        return f"Current availability mix is {', '.join(parts)}." if parts else ""

    def _unknown_city_response(self, city: str) -> str:
        return (
            f"Sorry, {city} is not in our coverage area. "
            f"Available cities: {', '.join(self.facts.get_available_cities())}."
        )

    def _should_suppress_followup(self, entity_context: dict) -> bool:
        return bool(entity_context.get("service_type") and entity_context.get("city"))

    def _join_responses(self, responses: list[str]) -> str:
        cleaned = []
        seen = set()
        for response in responses:
            text = " ".join(str(response).split())
            key = text.casefold()
            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)
        return "\n\n".join(cleaned)

    def _semantic_examples(self) -> dict[str, list[str]]:
        configured = self.semantic_config.get("examples", {}) or {}
        examples: dict[str, list[str]] = {
            faq_class: [str(item) for item in values]
            for faq_class, values in configured.items()
            if isinstance(values, list)
        }
        for faq_class, class_cfg in self._faq_classes.items():
            values = examples.setdefault(faq_class, [])
            values.append(self._topic_for_class(faq_class).replace("_", " "))
            for pattern in class_cfg.get("patterns", []):
                values.append(self._pattern_to_example(pattern))
        return {
            faq_class: [value for value in dict.fromkeys(values) if value.strip()]
            for faq_class, values in examples.items()
        }

    def _pattern_to_example(self, pattern: str) -> str:
        text = str(pattern).replace("\\b", " ").replace("(?i)", " ")
        for char in "\\^$?+*()[]{}|.":
            text = text.replace(char, " ")
        return " ".join(text.split())

    def _topic_for_class(self, faq_class: str) -> str:
        return str(faq_class or "").removeprefix("faq_").removeprefix("dataset_")

    def _normalize_text(self, value: str) -> str:
        normalized_chars = []
        for char in str(value or "").casefold():
            normalized_chars.append(char if char.isalnum() else " ")
        return " ".join("".join(normalized_chars).split())

    def _contains_phrase(self, normalized_text: str, normalized_phrase: str) -> bool:
        if not normalized_text or not normalized_phrase:
            return False
        return f" {normalized_phrase} " in f" {normalized_text} "
