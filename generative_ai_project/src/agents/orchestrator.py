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
from ..rag.retriever import ProviderRetriever
from ..rag.vector_store import WeaviateVectorStore
from ..state.session_store import SessionStore
from ..state.booking_store import BookingStore
from ..cag.cag_manager import CAGManager
from ..prompts.context_engine import ContextEngine
from ..observability.tracing import trace_span
from ..processing.preprocessor import load_providers, get_service_categories, get_cities
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

        prompts = self.configs["prompts"]
        agents_cfg = self.configs["agents"]
        scoring = self.configs["scoring"]

        # Context engine for prompt assembly
        self.context_engine = ContextEngine(prompts)

        # Load dataset metadata
        df = load_providers()
        self.service_categories = get_service_categories(df)
        self.cities = get_cities(df)

        # Initialize retriever with full config for reranking
        retriever = ProviderRetriever(vector_store, self.configs)

        # Initialize sub-agents
        self.intent_agent = IntentAgent(llm, prompts, agents_cfg)
        self.discovery_agent = DiscoveryAgent(retriever, agents_cfg)
        self.ranking_agent = RankingAgent(llm, scoring, prompts, agents_cfg)
        self.booking_agent = BookingAgent(llm, self.booking_store, prompts, agents_cfg)
        self.followup_agent = FollowUpAgent(llm, self.booking_store, prompts, agents_cfg)

        logger.info(
            f"Orchestrator initialized | {len(self.service_categories)} categories | "
            f"{len(self.cities)} cities | CAG entries: {self.cag_manager.size}"
        )

    async def process_message(self, session_id: str, user_message: str) -> dict:
        """Process a user message through the full agentic pipeline."""
        start_time = time.time()
        session = self.session_store.get_or_create(session_id)
        self.session_store.add_turn(session_id, "user", user_message)

        trace = []
        booking = None
        ranked_providers = None

        try:
            # ═══ STEP 1: Intent Understanding ═══════════════════
            with trace_span("intent.extract", {"user_message": user_message[:200]}) as span:
                trace.append({"agent": "intent", "action": "extract_intent", "status": "running"})

                intent = await self.intent_agent.extract_intent(
                    user_message=user_message,
                    service_categories=self.service_categories,
                    cities=self.cities,
                    conversation_history=self.session_store.get_history(session_id),
                )

                # Merge with existing intent for multi-turn
                existing_intent = session["agent_state"].get("intent")
                if existing_intent:
                    for key, val in intent.items():
                        if val is None and existing_intent.get(key):
                            intent[key] = existing_intent[key]

                session["agent_state"]["intent"] = intent
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "service_type": intent.get("service_type"),
                    "city": intent.get("city"),
                    "area": intent.get("area"),
                }
                self.session_store.add_trace(session_id, "intent", "extract_intent", trace[-1]["result"])

            # Check clarification
            missing = intent.get("needs_clarification", [])
            if missing:
                clarification = self.intent_agent.get_clarification_message(missing, self.configs["prompts"])
                trace.append({"agent": "intent", "action": "ask_clarification", "fields": missing})
                self.session_store.add_turn(session_id, "assistant", clarification)
                return {
                    "response": clarification, "agent_trace": trace,
                    "session_id": session_id, "status": "needs_clarification",
                    "latency_ms": (time.time() - start_time) * 1000,
                }

            # ═══ STEP 2: Provider Discovery (Agentic RAG + Reranking) ═══
            with trace_span("discovery.search", {"category": intent.get("service_type"), "city": intent.get("city")}) as span:
                trace.append({"agent": "discovery", "action": "search_providers", "status": "running"})
                candidates = await self.discovery_agent.discover(intent=intent)
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {"candidates_found": len(candidates)}
                self.session_store.add_trace(session_id, "discovery", "search", {"count": len(candidates)})

            if not candidates:
                msg = "I couldn't find any providers matching your request. Please try a different service or location."
                self.session_store.add_turn(session_id, "assistant", msg)
                return {
                    "response": msg, "agent_trace": trace,
                    "session_id": session_id, "status": "no_results",
                    "latency_ms": (time.time() - start_time) * 1000,
                }

            # ═══ STEP 3: Ranking (Deterministic + LLM Reasoning) ═══
            with trace_span("ranking.score", {"candidates": len(candidates)}) as span:
                trace.append({"agent": "ranking", "action": "rank_providers", "status": "running"})
                ranking_result = await self.ranking_agent.rank(candidates=candidates, intent=intent)
                ranked_providers = ranking_result["ranked"]
                reasoning = ranking_result["reasoning"]
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "total_evaluated": ranking_result["total_evaluated"],
                    "top_providers": [
                        {"name": p.get("metadata", p).get("provider_name"), "score": p["score_result"]["composite_score"]}
                        for p in ranked_providers
                    ],
                }
                self.session_store.add_trace(session_id, "ranking", "rank", trace[-1]["result"])

            session["agent_state"]["ranked_providers"] = ranked_providers
            session["agent_state"]["selected_provider"] = ranked_providers[0] if ranked_providers else None

            # ═══ STEP 4: Booking ═══════════════════════════════
            with trace_span("booking.create") as span:
                trace.append({"agent": "booking", "action": "create_booking", "status": "running"})
                best_provider = ranked_providers[0]
                booking = await self.booking_agent.create_booking(
                    session_id=session_id, provider=best_provider, intent=intent,
                )
                trace[-1]["status"] = "done"
                trace[-1]["result"] = {
                    "booking_id": booking["booking_id"],
                    "provider": booking["provider_name"],
                    "status": booking["status"],
                }
                self.session_store.add_trace(session_id, "booking", "create", trace[-1]["result"])

            session["agent_state"]["booking"] = booking

            # ═══ STEP 5: Follow-Up Scheduling ══════════════════════
            # The follow-up agent THINKS about urgency, timing, and
            # language to generate a personalized notification plan.
            # DETERMINISTIC: schedule structure (tiers, timeline) from config
            # PROBABILISTIC: notification text generated by LLM
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
                        "reminders_count": len(followup_result.get("scheduled_reminders", [])),
                        "status_events_count": len(followup_result.get("status_timeline", [])),
                        "immediate_notification": bool(followup_result.get("immediate_notification")),
                    }
                    self.session_store.add_trace(session_id, "followup", "schedule", trace[-1]["result"])
                except Exception as e:
                    # Follow-up is non-critical — don't fail the booking
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

            session["agent_state"]["followup"] = followup_result

            # ═══ STEP 6: Build Response ═════════════════════════
            provider_summaries = []
            for i, p in enumerate(ranked_providers[:3], 1):
                summary = format_provider_summary(p, p.get("score_result"))
                provider_summaries.append(f"\n{'─' * 40}\nOption {i}:\n{summary}")

            response_text = (
                f"{reasoning}\n"
                f"{''.join(provider_summaries)}\n"
                f"\n{'═' * 40}\n"
                f"📋 BOOKING CONFIRMED\n"
                f"{'═' * 40}\n"
                f"{booking.get('confirmation_message', '')}\n"
            )

            # Append follow-up summary to the response text
            if followup_result.get("immediate_notification"):
                immediate = followup_result["immediate_notification"]
                response_text += (
                    f"\n{'═' * 40}\n"
                    f"🔔 FOLLOW-UP NOTIFICATIONS SCHEDULED\n"
                    f"{'═' * 40}\n"
                    f"{immediate.get('body', 'Reminders will be sent before your appointment.')}\n"
                )
                n_reminders = len(followup_result.get("scheduled_reminders", []))
                if n_reminders > 0:
                    response_text += (
                        f"📅 {n_reminders} reminder(s) scheduled before your appointment.\n"
                    )

            self.session_store.add_turn(session_id, "assistant", response_text)

            return {
                "response": response_text,
                "agent_trace": trace,
                "session_id": session_id,
                "status": "booking_confirmed",
                "booking": booking,
                "followup": followup_result,
                "providers": [
                    {
                        "provider_name": p.get("metadata", p).get("provider_name"),
                        "score": p["score_result"]["composite_score"],
                        "score_breakdown": p["score_result"]["breakdown"],
                    }
                    for p in ranked_providers[:3]
                ],
                "intent": intent,
                "latency_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            trace.append({"agent": "orchestrator", "action": "error", "error": str(e)})
            return {
                "response": "I encountered an issue processing your request. Please try again.",
                "agent_trace": trace,
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }
