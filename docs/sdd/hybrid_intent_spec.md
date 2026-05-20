# Hybrid Intent Spec

## Responsibilities

The hybrid intent layer decomposes user messages into booking and FAQ components, keeps confidence metadata, and preserves workflow continuity. It prevents binary FAQ-or-booking routing.

## Ownership Boundaries

- `FAQAgent` owns FAQ topic detection and hybrid route hints.
- `IntentAgent` owns booking field extraction.
- `Orchestrator` owns route composition and final response ordering.
- `SessionStore` owns memory used for follow-up turns.

## Data Contracts

Hybrid route summary:

```json
{
  "primary_intent": "booking",
  "secondary_intents": ["faq_pricing", "faq_availability"],
  "routing_confidence": 0.88,
  "extraction_confidence": 0.81,
  "dataset_match_confidence": 0.93,
  "should_continue_booking": true,
  "ambiguity_detected": false,
  "route_source": "regex+dataset",
  "components": {
    "faq": [...],
    "booking": {
      "service_type": "Plumber",
      "city": "Karachi",
      "area": "DHA"
    }
  }
}
```

## Config Contracts

Hybrid routing thresholds live in `faq_config.yaml`:

- `hybrid.min_booking_signal_score`
- `hybrid.min_faq_confidence`
- `hybrid.answer_then_continue`
- `routing.min_confidence`
- `ambiguity.margin`

## Response Semantics

- FAQ components are answered first.
- Booking flow then continues using current or remembered service/city/area.
- If required booking fields are missing, the assistant asks one concise clarification after answering FAQ.
- Hybrid traces must be returned in `agent_trace` and optional `routing`.

## Fallback Behavior

If FAQ confidence is low but booking confidence is strong, continue booking and skip uncertain FAQ. If both are low, ask clarification.

## Latency Expectations

Hybrid decomposition should add less than 100 ms over FAQ route unless semantic fallback is invoked.

## CPU Constraints

The hybrid layer uses deterministic and cached semantic signals only. It must not call generative LLMs.

## Extensibility Expectations

New secondary intents are introduced through FAQ config and CAG policy keys. Orchestrator response composition should not require code changes for every new topic.

## Failure Handling

Ambiguous service or city matches block booking and ask the user to choose from candidates. Unknown FAQ topics do not block booking.

## Observability Requirements

Trace fields:

- `hybrid_route`
- `primary_intent`
- `secondary_intents`
- `routing_confidence`
- `extraction_confidence`
- `dataset_match_confidence`
- `route_source`
- `why_selected`

