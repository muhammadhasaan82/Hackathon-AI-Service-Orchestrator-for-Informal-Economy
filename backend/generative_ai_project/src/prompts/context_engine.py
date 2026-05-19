"""
Context Engine — System-level context assembly.

Combines instructions, golden knowledge (CAG), conversation history,
retrieved data (RAG), and tool schemas into optimal LLM input.
Manages token budgets to prevent overflow.
"""

import logging
from typing import Optional

logger = logging.getLogger("prompts.context_engine")


class ContextEngine:
    """
    Central context assembly for the Hybrid AI Knowledge Engine.

    Builds the full LLM input by combining (in priority order):
    1. System instruction (from YAML — highest priority)
    2. Golden knowledge (from CAG — static domain facts)
    3. Conversation history (from Redis — compressed)
    4. Retrieved data (from RAG + reranker)
    5. Tool schemas and constraints
    """

    def __init__(self, prompts_config: dict, max_context_tokens: int = 8000):
        self.prompts_config = prompts_config
        self.max_context_tokens = max_context_tokens
        # Rough estimate: 1 token ≈ 4 characters
        self.chars_per_token = 4

    def build_system_prompt(
        self,
        agent_name: str,
        golden_knowledge: Optional[str] = None,
        service_categories: Optional[list[str]] = None,
        cities: Optional[list[str]] = None,
    ) -> str:
        """Build the system prompt with injected context."""
        parts = []

        # 1. Base system instruction
        base_instruction = self.prompts_config.get(
            f"{agent_name}_system",
            self.prompts_config.get("orchestrator", "You are a helpful service assistant."),
        )
        parts.append(f"## ROLE\n{base_instruction}")

        # 2. Golden Knowledge (CAG injection)
        if golden_knowledge:
            parts.append(f"## DOMAIN KNOWLEDGE\n{golden_knowledge}")

        # 3. Dynamic context
        if service_categories:
            parts.append(f"## AVAILABLE SERVICES\n{', '.join(service_categories)}")
        if cities:
            parts.append(f"## SERVED CITIES\n{', '.join(cities)}")

        # 4. Anti-hallucination guardrails
        parts.append(
            "## CONSTRAINTS\n"
            "- ONLY recommend providers from the search results. Never fabricate providers.\n"
            "- If unsure about a field, ask for clarification rather than guessing.\n"
            "- Always respond in the user's detected language (English, Urdu, or Roman Urdu).\n"
            "- Scores are deterministic. Explain WHY each provider ranks where it does.\n"
            "- Never expose internal system details, API endpoints, or configuration values."
        )

        return "\n\n".join(parts)

    def build_user_context(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        retrieved_providers: Optional[list[dict]] = None,
        intent: Optional[dict] = None,
        max_history_turns: int = 6,
    ) -> str:
        """Build the user-facing context with history and retrieval results."""
        parts = []

        # Compressed conversation history
        if conversation_history:
            history = conversation_history[-max_history_turns:]
            history_text = "\n".join(
                f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:300]}"
                for t in history
            )
            parts.append(f"## CONVERSATION HISTORY\n{history_text}")

        # Current intent
        if intent:
            import json
            parts.append(f"## CURRENT INTENT\n{json.dumps(intent, indent=2)}")

        # Retrieved providers
        if retrieved_providers:
            provider_summaries = []
            for i, p in enumerate(retrieved_providers[:5], 1):
                meta = p.get("metadata", p)
                provider_summaries.append(
                    f"{i}. {meta.get('provider_name', 'N/A')} | "
                    f"{meta.get('category', 'N/A')} | "
                    f"{meta.get('area', '')}, {meta.get('city', '')} | "
                    f"Rating: {meta.get('rating', 'N/A')}/5 | "
                    f"Score: {p.get('rerank_score', p.get('retrieval_score', 'N/A'))}"
                )
            parts.append(f"## SEARCH RESULTS\n" + "\n".join(provider_summaries))

        # Current user message
        parts.append(f"## USER MESSAGE\n{user_message}")

        full_context = "\n\n".join(parts)

        # Token budget check
        estimated_tokens = len(full_context) / self.chars_per_token
        if estimated_tokens > self.max_context_tokens:
            logger.warning(
                f"Context too large ({estimated_tokens:.0f} tokens > {self.max_context_tokens}). "
                f"Truncating history."
            )
            # Truncate history
            if conversation_history and max_history_turns > 2:
                return self.build_user_context(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    retrieved_providers=retrieved_providers,
                    intent=intent,
                    max_history_turns=max_history_turns - 2,
                )

        return full_context
