"""
Orchestrator — Root agent for the full agentic pipeline.

Routes: Intent → Discovery → Ranking → Booking → Follow-up
Integrates: Agentic RAG + Reranking + CAG + Context Engineering + OpenTelemetry
"""

import json
import logging
import time
from typing import Optional

from ..core.base_llm import BaseLLM
from ..core.model_factory import load_all_configs
from ..core.runtime_config import reset_llm_call_budget
from ..rag.retriever import ProviderRetriever
from ..rag.vector_store import WeaviateVectorStore
from ..state.session_store import SessionStore
from ..state.booking_store import BookingStore
from ..cag.cag_manager import CAGManager
from ..guardrails import GuardrailEngine
from ..prompts.context_engine import ContextEngine
from ..observability.tracing import trace_span
from ..processing.preprocessor import load_providers, get_service_categories, get_cities, get_areas_by_city
from .intent_agent import IntentAgent
from .discovery_agent import DiscoveryAgent
from .ranking_agent import RankingAgent
from .booking_agent import BookingAgent
from .followup_agent import FollowUpAgent
from .tools import format_provider_summary

logger = logging.getLogger("agents.orchestrator")

class Orchestrator:
    """Root orchestrator managing the Hybrid AI Knowledge Engine pipeline."""

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: WeaviateVectorStore,
        session_store: Optional[SessionStore] = None,
        booking_store: Optional[BookingStore] = None,
        cag_manager: Optional[CAGManager] = None,
        configs: Optional[dict] = None,
    ):
        self.llm = llm
        self.configs = configs or load_all_configs()
        self.session_store = session_store or SessionStore()
        self.booking_store = booking_store or BookingStore()
        self.cag_manager = cag_manager or CAGManager()
        self.guardrails = GuardrailEngine(self.configs.get("guardrails", {}))

        prompts = self.configs["prompts"]
        agents_cfg = self.configs["agents"]
        scoring = self.configs["scoring"]

        # Context engine for prompt assembly
        self.context_engine = ContextEngine(prompts)

        # Load dataset metadata
        self.providers_df = load_providers()
        self.service_categories = get_service_categories(self.providers_df)
        self.cities = get_cities(self.providers_df)
        self.areas_by_city = get_areas_by_city(self.providers_df)
        self.areas = sorted({area for areas in self.areas_by_city.values() for area in areas})
        self.greeting_config = (
            agents_cfg.get("agents", {})
            .get("orchestrator", {})
            .get("greeting", {})
        )

        # Initialize retriever with full config for reranking
        retriever = ProviderRetriever(vector_store, self.configs)

        # Initialize sub-agents
        self.intent_agent = IntentAgent(llm, prompts, agents_cfg, guardrails=self.guardrails, cag_manager=self.cag_manager)
        self.discovery_agent = DiscoveryAgent(retriever, agents_cfg)
        self.ranking_agent = RankingAgent(llm, scoring, prompts, agents_cfg, guardrails=self.guardrails, cag_manager=self.cag_manager)
        self.booking_agent = BookingAgent(llm, self.booking_store, prompts, agents_cfg, guardrails=self.guardrails, cag_manager=self.cag_manager)
        self.followup_agent = FollowUpAgent(llm, self.booking_store, prompts, agents_cfg, guardrails=self.guardrails, cag_manager=self.cag_manager)

        logger.info(
            f"Orchestrator initialized | {len(self.service_categories)} categories | "
            f"{len(self.cities)} cities | CAG entries: {self.cag_manager.size}"
        )

    def _provider_payload(self, providers: Optional[list[dict]], limit: int = 3) -> list[dict]:
        payload = []
        for provider in (providers or [])[:limit]:
            meta = provider.get("metadata", provider)
            score_result = provider.get("score_result", {})
            payload.append({
                "provider_id": meta.get("provider_id"),
                "provider_name": meta.get("provider_name"),
                "score": score_result.get("composite_score"),
                "score_breakdown": score_result.get("breakdown"),
            })
        return payload

    def _provider_options_text(self, ranked_providers: list[dict]) -> str:
        provider_summaries = []
        for i, provider in enumerate(ranked_providers[:3], 1):
            summary = format_provider_summary(provider, provider.get("score_result"))
            provider_summaries.append(f"\n{'─' * 40}\nOption {i}:\n{summary}")
        return "".join(provider_summaries)

    def _return_payload(self, payload: dict) -> dict:
        return self.guardrails.sanitize_payload(payload) if self.guardrails else payload

    def _log_response_generation(self, start: float, status: str, response_text: str) -> None:
        logger.info(
            "Timing response_generation=%.1fms status=%s chars=%s",
            (time.time() - start) * 1000,
            status,
            len(response_text or ""),
        )

    def _is_simple_greeting(self, user_message: str) -> bool:
        if not self.greeting_config.get("enabled", True):
            return False
        normalized = self._normalize_short_text(user_message)
        phrases = self.greeting_config.get("phrases", [])
        return normalized in {self._normalize_short_text(phrase) for phrase in phrases}

    def _greeting_clarification(self, user_message: str) -> str:
        normalized = self._normalize_short_text(user_message)
        markers = self.greeting_config.get("salam_markers", [])
        if any(self._normalize_short_text(marker) in normalized for marker in markers):
            return self.greeting_config.get(
                "salam_message",
                "Wa alaikum salam! Aap ko kis service ki zaroorat hai aur kis city/area mein?",
            )
        return self.greeting_config.get(
            "default_message",
            "Hi! Please tell me what service you need and your city/area.",
        )

    def _normalize_short_text(self, value: str) -> str:
        normalized_chars = []
        for char in (value or "").casefold():
            normalized_chars.append(char if char.isalnum() else " ")
        return " ".join("".join(normalized_chars).split())

    async def process_message(
        self,
        session_id: str,
        user_message: str,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> dict:
        """Process a user message through the full agentic pipeline."""
        start_time = time.time()
        reset_llm_call_budget()
        trace = []
        followup_result = None

        self.session_store.get_or_create(session_id)
        self.session_store.add_turn(session_id, "user", user_message)
        session = self.session_store.get_or_create(session_id)

        try:
            input_guardrail = self.guardrails.evaluate_input(user_message)
            if input_guardrail:
                trace.append({
                    "agent": "guardrails",
                    "action": "evaluate_input",
                    "status": input_guardrail.get("status"),
                    "result": {"policy": input_guardrail.get("policy")},
                })
                self.session_store.add_trace(session_id, "guardrails", "evaluate_input", input_guardrail)
                response_start = time.time()
                response_text = self.guardrails.sanitize_text(input_guardrail.get("message"))
                self._log_response_generation(response_start, input_guardrail.get("status", "input_rejected"), response_text)
                self.session_store.add_turn(session_id, "assistant", response_text)
                return self._return_payload({
                    "response": response_text,
                    "agent_trace": trace,
                    "session_id": session_id,
                    "status": input_guardrail.get("status", "input_rejected"),
                    "providers": self._provider_payload(session.get("agent_state", {}).get("ranked_providers")),
                    "intent": session.get("agent_state", {}).get("intent"),
                    "latency_ms": (time.time() - start_time) * 1000,
                })

            if self._is_simple_greeting(user_message):
                response_start = time.time()
                response_text = self.guardrails.sanitize_text(self._greeting_clarification(user_message))
                self._log_response_generation(response_start, "needs_clarification", response_text)
                self.session_store.add_turn(session_id, "assistant", response_text)
                latency_ms = (time.time() - start_time) * 1000
                logger.info("Greeting handled without LLM in %.1fms", latency_ms)
                return self._return_payload({
                    "response": response_text,
                    "agent_trace": [
                        {"agent": "orchestrator", "action": "greeting_short_circuit", "status": "done"}
                    ],
                    "session_id": session_id,
                    "status": "needs_clarification",
                    "providers": self._provider_payload(session.get("agent_state", {}).get("ranked_providers")),
                    "intent": session.get("agent_state", {}).get("intent"),
                    "latency_ms": latency_ms,
                })

            agent_state = session.get("agent_state", {})
            ranked_providers = agent_state.get("ranked_providers") or []
            current_intent = agent_state.get("intent") or {}

            if agent_state.get("awaiting_booking_confirmation"):
                resolution = self.guardrails.resolve_booking_confirmation(user_message, ranked_providers)
                self.session_store.add_trace(session_id, "guardrails", "resolve_booking_confirmation", resolution)

                if resolution.get("action") == "reject":
                    session["agent_state"]["awaiting_booking_confirmation"] = False
                    self.session_store.save_session(session_id, session)
                    response_start = time.time()
                    response_text = self.guardrails.sanitize_text(self.guardrails.rejection_message())
                    self._log_response_generation(response_start, "booking_cancelled", response_text)
                    self.session_store.add_turn(session_id, "assistant", response_text)
                    return self._return_payload({
                        "response": response_text,
                        "agent_trace": trace,
                        "session_id": session_id,
                        "status": "booking_cancelled",
                        "providers": self._provider_payload(ranked_providers),
                        "intent": current_intent,
                        "latency_ms": (time.time() - start_time) * 1000,
                    })

                if resolution.get("action") == "confirm":
                    selected_provider = resolution.get("provider")
                    with trace_span("booking.create", {"selected_index": resolution.get("selected_index")}) as span:
                        trace.append({"agent": "booking", "action": "create_booking", "status": "running"})
                        booking = await self.booking_agent.create_booking(
                            session_id=session_id,
                            provider=selected_provider,
                            intent=current_intent,
                            status="CONFIRMED",
                        )
                        trace[-1]["status"] = "done"
                        trace[-1]["result"] = {
                            "booking_id": booking["booking_id"],
                            "provider": booking["provider_name"],
                            "status": booking["status"],
                        }
                        self.session_store.add_trace(session_id, "booking", "create", trace[-1]["result"])

                    with trace_span("followup.schedule", {"booking_id": booking["booking_id"]}) as span:
                        trace.append({"agent": "followup", "action": "schedule_followup", "status": "running"})
                        try:
                            followup_result = await self.followup_agent.schedule_followup(
                                booking=booking,
                                intent=current_intent,
                            )
                            trace[-1]["status"] = "done"
                            trace[-1]["result"] = {
                                "followup_id": followup_result.get("followup_id"),
                                "total_scheduled": followup_result.get("total_scheduled", 0),
                            }
                            self.session_store.add_trace(session_id, "followup", "schedule", trace[-1]["result"])
                        except Exception as e:
                            logger.warning(f"Follow-up scheduling failed (non-critical): {e}")
                            followup_result = {
                                "followup_id": None,
                                "booking_id": booking["booking_id"],
                                "scheduled_reminders": [],
                                "status_timeline": [],
                                "post_completion_actions": [],
                                "immediate_notification": None,
                                "total_scheduled": 0,
                            }
                            trace[-1]["status"] = "degraded"
                            trace[-1]["error"] = str(e)

                    session["agent_state"]["selected_provider"] = selected_provider
                    session["agent_state"]["booking"] = booking
                    session["agent_state"]["followup"] = followup_result
                    session["agent_state"]["awaiting_booking_confirmation"] = False
                    self.session_store.save_session(session_id, session)

                    response_start = time.time()
                    selected_summary = format_provider_summary(selected_provider, selected_provider.get("score_result"))
                    response_text = (
                        f"{'═' * 40}\n"
                        f"✅ BOOKING CONFIRMED\n"
                        f"{'═' * 40}\n"
                        f"{booking.get('confirmation_message', '')}\n"
                        f"\nSelected provider:\n{selected_summary}\n"
                    )
                    if followup_result.get("immediate_notification"):
                        response_text += (
                            f"\n{'═' * 40}\n"
                            f"🔔 FOLLOW-UP NOTIFICATIONS SCHEDULED\n"
                            f"{'═' * 40}\n"
                            f"{followup_result['immediate_notification'].get('body', 'Reminders will be sent before your appointment.')}\n"
                        )

                    response_text = self.guardrails.sanitize_text(response_text)
                    self._log_response_generation(
                        response_start,
                        self.guardrails.booking_status("booking_confirmed", "booking_confirmed"),
                        response_text,
                    )
                    self.session_store.add_turn(session_id, "assistant", response_text)
                    return self._return_payload({
                        "response": response_text,
                        "agent_trace": trace,
                        "session_id": session_id,
                        "status": self.guardrails.booking_status("booking_confirmed", "booking_confirmed"),
                        "booking": booking,
                        "followup": followup_result,
                        "providers": self._provider_payload(ranked_providers),
                        "intent": current_intent,
                        "latency_ms": (time.time() - start_time) * 1000,
                    })

                if resolution.get("action") in ("confirm_ambiguous", "missing_context"):
                    response_start = time.time()
                    top_score = ranked_providers[0].get("score_result", {}).get("composite_score") if ranked_providers else None
                    response_text = self.guardrails.missing_selection_message()
                    if resolution.get("action") == "confirm_ambiguous":
                        response_text = self.guardrails.confirmation_prompt(len(ranked_providers[:3]), top_score)
                    response_text = self.guardrails.sanitize_text(response_text)
                    self._log_response_generation(
                        response_start,
                        self.guardrails.booking_status("recommendation_ready", "awaiting_booking_confirmation"),
                        response_text,
                    )
                    self.session_store.add_turn(session_id, "assistant", response_text)
                    return self._return_payload({
                        "response": response_text,
                        "agent_trace": trace,
                        "session_id": session_id,
                        "status": self.guardrails.booking_status("recommendation_ready", "awaiting_booking_confirmation"),
                        "providers": self._provider_payload(ranked_providers),
                        "intent": current_intent,
                        "latency_ms": (time.time() - start_time) * 1000,
                    })

            with trace_span("intent.extract", {"user_message": user_message[:200]}) as span:
                phase_start = time.time()
                trace.append({"agent": "intent", "action": "extract_intent", "status": "running"})
                intent = await self.intent_agent.extract_intent(
                    user_message=user_message,
                    service_categories=self.service_categories,
                    cities=self.cities,
                    areas=self.areas,
                    conversation_history=self.session_store.get_history(session_id),
                )

                existing_intent = session["agent_state"].get("intent")
                if existing_intent:
                    for key, val in intent.items():
                        if val is None and existing_intent.get(key):
                            intent[key] = existing_intent[key]

                intent_review = self.guardrails.evaluate_intent(intent)
                intent["needs_clarification"] = intent_review.get("needs_clarification", [])
                session["agent_state"]["intent"] = intent
                session["metadata"]["language"] = intent.get("language_detected", session["metadata"].get("language", "english"))
                self.session_store.save_session(session_id, session)

                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "service_type": intent.get("service_type"),
                    "city": intent.get("city"),
                    "area": intent.get("area"),
                }
                self.session_store.add_trace(session_id, "intent", "extract_intent", trace[-1]["result"])
                logger.info("Timing intent=%.1fms", (time.time() - phase_start) * 1000)

            if intent_review.get("should_escalate"):
                response_start = time.time()
                handoff_message = (
                    self.configs.get("agents", {})
                    .get("agents", {})
                    .get("orchestrator", {})
                    .get("escalation", {})
                    .get("human_handoff_message", "I want to be accurate, so I recommend a human operator review this request.")
                )
                handoff_message = self.guardrails.sanitize_text(handoff_message)
                self._log_response_generation(response_start, "human_handoff_recommended", handoff_message)
                self.session_store.add_turn(session_id, "assistant", handoff_message)
                return self._return_payload({
                    "response": handoff_message,
                    "agent_trace": trace,
                    "session_id": session_id,
                    "status": "human_handoff_recommended",
                    "intent": intent,
                    "latency_ms": (time.time() - start_time) * 1000,
                })

            if intent.get("needs_clarification"):
                response_start = time.time()
                clarification = self.intent_agent.get_clarification_message(intent["needs_clarification"], self.configs["prompts"])
                clarification = self.guardrails.sanitize_text(clarification)
                self._log_response_generation(response_start, "needs_clarification", clarification)
                trace.append({"agent": "intent", "action": "ask_clarification", "fields": intent["needs_clarification"]})
                self.session_store.add_turn(session_id, "assistant", clarification)
                return self._return_payload({
                    "response": clarification,
                    "agent_trace": trace,
                    "session_id": session_id,
                    "status": "needs_clarification",
                    "intent": intent,
                    "latency_ms": (time.time() - start_time) * 1000,
                })

            with trace_span("discovery.search", {"category": intent.get("service_type"), "city": intent.get("city")}) as span:
                phase_start = time.time()
                trace.append({"agent": "discovery", "action": "search_providers", "status": "running"})
                candidates = await self.discovery_agent.discover(
                    intent=intent,
                    user_lat=user_lat,
                    user_lon=user_lon,
                )
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {"candidates_found": len(candidates)}
                self.session_store.add_trace(session_id, "discovery", "search", {"count": len(candidates)})
                logger.info("Timing retrieval=%.1fms candidates=%s", (time.time() - phase_start) * 1000, len(candidates))

            if not candidates:
                response_start = time.time()
                msg = "I couldn't find any providers matching your request. Please try a different service or location."
                msg = self.guardrails.sanitize_text(msg)
                self._log_response_generation(response_start, "no_results", msg)
                self.session_store.add_turn(session_id, "assistant", msg)
                return self._return_payload({
                    "response": msg,
                    "agent_trace": trace,
                    "session_id": session_id,
                    "status": "no_results",
                    "intent": intent,
                    "latency_ms": (time.time() - start_time) * 1000,
                })

            with trace_span("ranking.score", {"candidates": len(candidates)}) as span:
                phase_start = time.time()
                trace.append({"agent": "ranking", "action": "rank_providers", "status": "running"})
                ranking_result = await self.ranking_agent.rank(
                    candidates=candidates,
                    intent=intent,
                    user_lat=user_lat,
                    user_lon=user_lon,
                )
                ranked_providers = ranking_result["ranked"]
                reasoning = ranking_result["reasoning"]
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "total_evaluated": ranking_result["total_evaluated"],
                    "top_providers": [
                        {
                            "name": provider.get("metadata", provider).get("provider_name"),
                            "score": provider["score_result"]["composite_score"],
                        }
                        for provider in ranked_providers
                    ],
                }
                self.session_store.add_trace(session_id, "ranking", "rank", trace[-1]["result"])
                logger.info("Timing ranking=%.1fms ranked=%s", (time.time() - phase_start) * 1000, len(ranked_providers))

            session["agent_state"]["ranked_providers"] = ranked_providers
            session["agent_state"]["selected_provider"] = ranked_providers[0] if ranked_providers else None
            session["agent_state"]["booking"] = None
            session["agent_state"]["followup"] = None
            session["agent_state"]["awaiting_booking_confirmation"] = False
            self.session_store.save_session(session_id, session)

            top_score = ranked_providers[0].get("score_result", {}).get("composite_score") if ranked_providers else None
            providers_payload = self._provider_payload(ranked_providers)

            if self.guardrails.booking_mode() == "confirmation_required":
                response_start = time.time()
                session["agent_state"]["awaiting_booking_confirmation"] = True
                self.session_store.save_session(session_id, session)
                confirmation_prompt = self.guardrails.confirmation_prompt(len(providers_payload), top_score)
                response_text = (
                    f"{reasoning}\n"
                    f"{self._provider_options_text(ranked_providers)}\n"
                    f"\n{'═' * 40}\n"
                    f"⏳ BOOKING CONFIRMATION REQUIRED\n"
                    f"{'═' * 40}\n"
                    f"{confirmation_prompt}\n"
                )
                response_text = self.guardrails.sanitize_text(response_text)
                self._log_response_generation(
                    response_start,
                    self.guardrails.booking_status("recommendation_ready", "awaiting_booking_confirmation"),
                    response_text,
                )
                self.session_store.add_turn(session_id, "assistant", response_text)
                return self._return_payload({
                    "response": response_text,
                    "agent_trace": trace,
                    "session_id": session_id,
                    "status": self.guardrails.booking_status("recommendation_ready", "awaiting_booking_confirmation"),
                    "providers": providers_payload,
                    "intent": intent,
                    "latency_ms": (time.time() - start_time) * 1000,
                })

            with trace_span("booking.create") as span:
                trace.append({"agent": "booking", "action": "create_booking", "status": "running"})
                booking = await self.booking_agent.create_booking(
                    session_id=session_id,
                    provider=ranked_providers[0],
                    intent=intent,
                )
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "booking_id": booking["booking_id"],
                    "provider": booking["provider_name"],
                    "status": booking["status"],
                }
                self.session_store.add_trace(session_id, "booking", "create", trace[-1]["result"])

            with trace_span("followup.schedule", {"booking_id": booking["booking_id"]}) as span:
                trace.append({"agent": "followup", "action": "schedule_followup", "status": "running"})
                try:
                    followup_result = await self.followup_agent.schedule_followup(
                        booking=booking,
                        intent=intent,
                    )
                    trace[-1]["status"] = "done"
                    trace[-1]["result"] = {
                        "followup_id": followup_result.get("followup_id"),
                        "total_scheduled": followup_result.get("total_scheduled", 0),
                    }
                    self.session_store.add_trace(session_id, "followup", "schedule", trace[-1]["result"])
                except Exception as e:
                    logger.warning(f"Follow-up scheduling failed (non-critical): {e}")
                    followup_result = {
                        "followup_id": None,
                        "booking_id": booking["booking_id"],
                        "scheduled_reminders": [],
                        "status_timeline": [],
                        "post_completion_actions": [],
                        "immediate_notification": None,
                        "total_scheduled": 0,
                    }
                    trace[-1]["status"] = "degraded"
                    trace[-1]["error"] = str(e)

            session["agent_state"]["booking"] = booking
            session["agent_state"]["followup"] = followup_result
            self.session_store.save_session(session_id, session)

            response_start = time.time()
            response_text = (
                f"{reasoning}\n"
                f"{self._provider_options_text(ranked_providers)}\n"
                f"\n{'═' * 40}\n"
                f"📋 BOOKING CONFIRMED\n"
                f"{'═' * 40}\n"
                f"{booking.get('confirmation_message', '')}\n"
            )
            if followup_result.get("immediate_notification"):
                response_text += (
                    f"\n{'═' * 40}\n"
                    f"🔔 FOLLOW-UP NOTIFICATIONS SCHEDULED\n"
                    f"{'═' * 40}\n"
                    f"{followup_result['immediate_notification'].get('body', 'Reminders will be sent before your appointment.')}\n"
                )
                if followup_result.get("scheduled_reminders"):
                    response_text += f"📅 {len(followup_result['scheduled_reminders'])} reminder(s) scheduled before your appointment.\n"

            response_text = self.guardrails.sanitize_text(response_text)
            self._log_response_generation(
                response_start,
                self.guardrails.booking_status("booking_confirmed", "booking_confirmed"),
                response_text,
            )
            self.session_store.add_turn(session_id, "assistant", response_text)
            return self._return_payload({
                "response": response_text,
                "agent_trace": trace,
                "session_id": session_id,
                "status": self.guardrails.booking_status("booking_confirmed", "booking_confirmed"),
                "booking": booking,
                "followup": followup_result,
                "providers": providers_payload,
                "intent": intent,
                "latency_ms": (time.time() - start_time) * 1000,
            })

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            trace.append({"agent": "orchestrator", "action": "error", "error": str(e)})
            response_start = time.time()
            response_text = self.guardrails.sanitize_text("I encountered an issue processing your request. Please try again.")
            self._log_response_generation(response_start, "error", response_text)
            return self._return_payload({
                "response": response_text,
                "agent_trace": trace,
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            })
