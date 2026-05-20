# Conversation Memory Spec

## Responsibilities

Conversation memory carries service, city, area, FAQ topic, booking stage, provider shortlist, and selected provider across turns. It allows messages like "Available in DHA?" after "Need plumber in Karachi" to reuse prior context.

## Ownership Boundaries

- `SessionStore` owns durable session state and history.
- `Orchestrator` owns merging current extraction with remembered context.
- `FAQAgent` may read remembered context but must not persist state directly unless the orchestrator asks it to.
- `BookingStore` owns finalized booking records.

## Data Contracts

Session `agent_state` includes:

```json
{
  "intent": {},
  "last_faq_topics": [],
  "pipeline_stage": "faq | discovery | ranking | booking",
  "ranked_providers": [],
  "selected_provider": null,
  "awaiting_booking_confirmation": false,
  "routing": {}
}
```

## Config Contracts

- session TTL from `SessionStore`
- max conversation turns injected into intent
- memory merge rules from agent config
- whether FAQ context can reuse prior service/city/area

## Response Semantics

Memory is used silently when confidence is high. If remembered context conflicts with new user input, new explicit user input wins. If conflict is ambiguous, ask a clarification question.

## Fallback Behavior

If session state is missing or Redis is unavailable, in-memory fallback preserves behavior for a single process. If no memory exists, standard clarification applies.

## Latency Expectations

Memory lookup should be a single Redis or in-memory read and stay below 50 ms.

## CPU Constraints

Memory merge is dictionary-level deterministic logic. It must not trigger embedding or LLM calls.

## Extensibility Expectations

The memory object may later include user preferences, provider feedback, notification settings, and analytics without changing the `/chat` API shape.

## Failure Handling

Corrupt or missing session state should create a clean session and log the reset. Memory must not fabricate missing service/city/area values.

## Observability Requirements

Trace fields:

- `memory_context_used`
- reused fields
- overwritten fields
- conflict detected
- booking stage
- provider shortlist count

