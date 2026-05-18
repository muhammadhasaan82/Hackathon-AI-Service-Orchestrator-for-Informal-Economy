"""
Ranking Agent — Probabilistic scoring + LLM reasoning.

Computes deterministic composite scores from config weights,
then uses the LLM to generate natural language reasoning
explaining WHY each provider was ranked.

THREADING NOTE:
    Scoring is the most CPU-intensive step in the pipeline when
    evaluating 50+ candidates. batch_score_providers parallelizes
    the per-provider math (haversine + weighted composite) across
    the shared thread pool. The sort and LLM call remain sequential.
"""

import json
import logging
import time
from typing import Optional

from ..core.base_llm import BaseLLM
from ..core.runtime_config import consume_llm_call_budget, env_bool, env_int
from .tools import batch_score_providers, format_provider_summary

logger = logging.getLogger("agents.ranking")


class RankingAgent:
    """Scores, ranks, and explains provider recommendations."""

    def __init__(
        self,
        llm: BaseLLM,
        scoring_config: dict,
        prompts_config: dict,
        agents_config: dict,
        guardrails=None,
        cag_manager=None,
    ):
        self.llm = llm
        self.scoring_config = scoring_config
        self.prompts_config = prompts_config
        self.config = agents_config.get("agents", {}).get("ranking", {})
        self.top_n = env_int("TOP_K_RESULTS", self.config.get("top_n", 3))
        self.guardrails = guardrails
        self.cag_manager = cag_manager

    async def rank(
        self,
        candidates: list[dict],
        intent: dict,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> dict:
        """
        Rank candidates and generate reasoning.

        1. Compute deterministic scores for all candidates
        2. Sort by composite score
        3. Ask LLM to generate reasoning for top-N

        Returns:
            {
                "ranked": [...top_n providers with scores...],
                "reasoning": "natural language explanation",
                "total_evaluated": int,
            }
        """
        price_pref = intent.get("price_preference")

        # Step 1: Score all candidates deterministically (CONCURRENT)
        # batch_score_providers runs score_provider() for each candidate
        # in parallel threads. Results preserve input order, so the
        # subsequent sort produces identical rankings regardless of
        # thread scheduling.
        score_start = time.time()
        score_results = batch_score_providers(
            candidates=candidates,
            scoring_config=self.scoring_config,
            user_lat=user_lat,
            user_lon=user_lon,
            price_preference=price_pref,
        )

        # Attach scores back to candidates (single-threaded, safe)
        scored = []
        for candidate, score_result in zip(candidates, score_results):
            candidate["score_result"] = score_result
            scored.append(candidate)

        score_ms = (time.time() - score_start) * 1000
        logger.info(f"Concurrent scoring completed in {score_ms:.1f}ms for {len(candidates)} candidates")

        # Step 2: Sort by composite score
        scored.sort(key=lambda x: x["score_result"]["composite_score"], reverse=True)

        # Apply minimum threshold
        min_score = self.scoring_config.get("thresholds", {}).get("min_composite_score", 0.25)
        qualified = [s for s in scored if s["score_result"]["composite_score"] >= min_score]

        if not qualified:
            qualified = scored[:self.top_n]  # Show best available even if below threshold

        top_providers = qualified[:self.top_n]

        # Step 3: Generate ranking reasoning only when enabled.
        reasoning_start = time.time()
        if env_bool("ENABLE_LLM_RANKING_REASONING", False) and consume_llm_call_budget("ranking.reasoning"):
            reasoning = await self._generate_reasoning(top_providers, intent)
            reasoning_mode = "llm"
        else:
            reasoning = self._deterministic_reasoning(top_providers, intent)
            reasoning_mode = "deterministic"
        logger.info(
            "Ranking reasoning completed in %.1fms | mode=%s",
            (time.time() - reasoning_start) * 1000,
            reasoning_mode,
        )

        logger.info(
            f"Ranked {len(scored)} candidates → top {len(top_providers)} "
            f"(scores: {[p['score_result']['composite_score'] for p in top_providers]})"
        )

        return {
            "ranked": top_providers,
            "reasoning": reasoning,
            "total_evaluated": len(scored),
        }

    def _deterministic_reasoning(self, providers: list[dict], intent: dict) -> str:
        if not providers:
            return "I could not find enough matching providers for this request."

        lines = ["I ranked these providers using deterministic scoring: rating, distance, availability, experience, completed jobs, verification, and price fit."]
        for i, provider in enumerate(providers, 1):
            meta = provider.get("metadata", provider)
            score = provider.get("score_result", {}).get("composite_score", 0)
            breakdown = provider.get("score_result", {}).get("breakdown", {})
            strengths = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)[:3] if isinstance(breakdown, dict) else []
            strengths_text = ", ".join(f"{k}={v:.2f}" for k, v in strengths if isinstance(v, (int, float))) or "balanced score"
            lines.append(
                f"{i}. {meta.get('provider_name', 'Provider')} scored {score:.2f}; strongest signals: {strengths_text}."
            )
        return "\n".join(lines)

    async def _generate_reasoning(self, providers: list[dict], intent: dict) -> str:
        """Use LLM to generate natural language reasoning for the rankings."""
        # Build provider summaries for the prompt
        summaries = []
        for i, p in enumerate(providers, 1):
            summary = format_provider_summary(p, p.get("score_result"))
            summaries.append(f"#{i}:\n{summary}")

        provider_text = "\n\n".join(summaries)

        # Detect language for response
        lang = intent.get("language_detected", "english")
        if lang.lower() in ("roman urdu", "urdu"):
            response_lang = "Roman Urdu (romanized Urdu in Latin script)"
        else:
            response_lang = "English"

        prompt_template = self.prompts_config.get("ranking_reasoning", "Explain the ranking.")
        prompt = prompt_template.format(
            weights=json.dumps(self.scoring_config.get("weights", {})),
            user_preferences=json.dumps({
                "service": intent.get("service_type"),
                "location": f"{intent.get('area', '')} {intent.get('city', '')}".strip(),
                "price": intent.get("price_preference"),
                "urgency": intent.get("urgency"),
            }),
            ranked_providers=provider_text,
            top_n=len(providers),
            response_language=response_lang,
        )

        system_instruction = (
            "You are a helpful service recommendation assistant. "
            "Explain your rankings clearly and concisely. "
            "Focus on why each provider is a good match."
        )
        if self.guardrails:
            system_instruction = self.guardrails.compose_system_instruction("ranking", system_instruction)
        system_instruction = self._with_cag("ranking", system_instruction)

        response = await self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
        )
        return self.guardrails.sanitize_text(response.text) if self.guardrails else response.text

    def _with_cag(self, agent_name: str, system_instruction: str) -> str:
        if not self.cag_manager:
            return system_instruction
        context = self.cag_manager.get_context_for_agent(agent_name)
        logger.info(
            "CAG injection for %s ranking LLM call | injected=%s chars=%s",
            agent_name,
            bool(context),
            len(context),
        )
        if not context:
            return system_instruction
        return f"{system_instruction}\n\n## DOMAIN KNOWLEDGE\n{context}"
