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
    ):
        self.llm = llm
        self.scoring_config = scoring_config
        self.prompts_config = prompts_config
        self.config = agents_config.get("agents", {}).get("ranking", {})
        self.top_n = self.config.get("top_n", 3)

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

        # Step 3: Generate LLM reasoning
        reasoning = await self._generate_reasoning(top_providers, intent)

        logger.info(
            f"Ranked {len(scored)} candidates → top {len(top_providers)} "
            f"(scores: {[p['score_result']['composite_score'] for p in top_providers]})"
        )

        return {
            "ranked": top_providers,
            "reasoning": reasoning,
            "total_evaluated": len(scored),
        }

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

        response = await self.llm.generate(
            prompt=prompt,
            system_instruction=(
                "You are a helpful service recommendation assistant. "
                "Explain your rankings clearly and concisely. "
                "Focus on why each provider is a good match."
            ),
        )
        return response.text
