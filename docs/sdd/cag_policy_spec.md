# CAG Policy Spec

## Responsibilities

CAG stores stable operational policy knowledge such as pricing policy, cancellation policy, booking process, escalation rules, language style, and guardrails. FAQ retrieval must return structured policies rather than opaque text blobs.

## Ownership Boundaries

- `data/cache/golden_knowledge.yaml` owns source policy content.
- `CAGManager` owns Python-side structured lookup and Rust/PyO3 fallback integration.
- Rust CAG is the long-term owner of policy retrieval, cached context building, semantic helpers, and aggregation.
- FAQ and orchestrator layers consume structured policies and choose user-facing templates.

## Data Contracts

Structured FAQ policy:

```yaml
short_answer: string
detailed_answer: string
escalation_required: boolean
dynamic_fields: []
related_topics: []
followup_prompts: []
safe_constraints: []
localization:
  default: english
  roman_urdu: string
tone:
  default: concise_helpful
```

`CAGManager.get_structured_faq_policy(policy_key)` returns:

```json
{
  "policy_key": "pricing_policy",
  "policy": {...},
  "source": "cag",
  "found": true
}
```

## Config Contracts

- `ENABLE_CAG` controls CAG lookup.
- `CAG_MAX_CONTEXT_CHARS` and `CAG_MAX_CONTEXT_ITEMS` limit prompt injection.
- FAQ config maps each FAQ class to a `cag_key`.

## Response Semantics

FAQ responses prefer `short_answer` for fast chat. `detailed_answer` may be used for explicit "details" requests. `followup_prompts` can be appended when they naturally advance booking.

## Fallback Behavior

If a structured policy is missing, the manager falls back to the legacy string policy. If no policy exists, FAQ routing should log `policy_missing` and continue to booking or clarification rather than fabricate policy.

## Latency Expectations

CAG lookup must be in-memory and normally below 10 ms.

## CPU Constraints

CAG must not invoke an LLM during lookup. Rust cache is preferred in production, Python dict fallback is acceptable.

## Extensibility Expectations

Policy schemas should allow dynamic fields, localization, tone adaptation, and future multilingual variants without code changes.

## Failure Handling

- Invalid policy schema is loaded as legacy text and logged.
- Rust cache import failure degrades to Python fallback.
- Missing policy keys are observable and non-fatal.

## Observability Requirements

Each policy retrieval trace should record:

- `CAG_policy_used`
- policy key
- structured vs legacy
- found/missing
- latency

