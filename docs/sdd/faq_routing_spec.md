# FAQ Routing Spec

## Responsibilities

The FAQ routing layer decomposes each user message into zero or more FAQ topics and a possible booking workflow. It must support pure FAQ answers, pure booking requests, and hybrid requests such as "Need plumber in Karachi, what are charges?" without forcing a single route.

## Ownership Boundaries

- Python `FAQAgent` owns request-time routing, confidence scoring, answer composition, and orchestration hints.
- `DatasetFacts` owns dataset-derived service, city, area, provider-count, and availability facts.
- `CAGManager` owns structured policy retrieval.
- The orchestrator owns final workflow composition and decides whether to continue provider discovery after FAQ answering.
- Rust CAG is the future owner of cached policy lookup, routing helpers, semantic FAQ retrieval, and aggregation.

## Data Contracts

`FAQAgent.try_handle(message, session)` returns either `None` or:

```json
{
  "response": "string",
  "status": "faq_answered | hybrid_route | ambiguity_detected",
  "primary_intent": "faq | booking | hybrid | clarification",
  "secondary_intents": ["faq_pricing"],
  "faq_matches": [
    {
      "faq_class": "faq_pricing",
      "topic": "pricing",
      "source": "regex | semantic",
      "confidence": 0.92,
      "semantic_match_score": 0.0,
      "policy_source": "cag"
    }
  ],
  "should_continue_booking": true,
  "routing_confidence": 0.86,
  "explainability": {
    "route_source": "hybrid",
    "regex_match": true,
    "semantic_match": false,
    "fuzzy_match_used": true,
    "ambiguity_detected": false
  }
}
```

## Config Contracts

`config/faq_config.yaml` defines:

- `faq_patterns`: deterministic regex classes and CAG or dataset source.
- `booking_signals`: language-neutral booking indicators.
- `routing`: thresholds for hybrid confidence and route decisions.
- `semantic`: MiniLM model id, enabled flag, cache behavior, and cosine threshold.
- `ambiguity`: fuzzy/semantic ambiguity thresholds and margin.
- `response_templates`: response text with dynamic placeholders only.

No provider count, city list, service list, price, or availability fact may be hardcoded in Python code.

## Response Semantics

- Pure FAQ: answer the FAQ and include an action-oriented next step.
- Hybrid FAQ plus booking: answer FAQ components first, then continue booking discovery or clarification.
- Ambiguous entity: ask a direct disambiguation question and do not silently pick a service or city.
- Unknown FAQ: return `None` so intent extraction can continue.

## Fallback Behavior

1. Regex classification runs first.
2. Semantic FAQ matching runs only when regex finds no FAQ topic or when configured to enrich hybrid routes.
3. Dataset fuzzy matching enriches extracted city/category/area confidence.
4. LLM generation is not used for FAQ routing.
5. If semantic dependencies are unavailable, the agent falls back to lexical similarity and logs degraded mode.

## Latency Expectations

- Regex route: target under 50 ms.
- Cached semantic route: target under 500 ms, acceptable under 1.5 s on CPU.
- FAQ routing must not call Ollama or Transformers.

## CPU Constraints

MiniLM embeddings are precomputed at startup for FAQ policy examples and reused. Request-time embedding is limited to one short message vector when semantic fallback is needed. No model is loaded per request.

## Extensibility Expectations

New FAQ topics are added through YAML patterns, policy entries, examples, and templates. The agent must not need code changes for new cities, services, or policy wording.

## Failure Handling

- Invalid regex patterns are skipped with warnings.
- Missing CAG policy returns a graceful dataset or template fallback when possible.
- Missing semantic model degrades to deterministic matching.
- Low-confidence or ambiguous routes ask for clarification.

## Observability Requirements

Every FAQ route must log:

- `faq_route`
- `regex_match`
- `semantic_match`
- `faq_confidence`
- `semantic_match_score`
- `route_source`
- `policy_source`
- `latency_ms`
- `ambiguity_detected`

