# Observability Spec

## Responsibilities

The orchestration layer must explain why each route was selected, what facts were used, which policy was used, and how long each layer took. Logs and traces must support debugging without exposing secrets or hidden prompts.

## Ownership Boundaries

- Agents emit structured route and confidence details.
- Orchestrator aggregates agent traces and returns safe explainability metadata.
- OpenTelemetry spans capture cross-layer latency when enabled.
- Guardrails sanitize user-visible payloads.

## Data Contracts

Trace entry:

```json
{
  "agent": "faq",
  "action": "hybrid_route",
  "status": "done",
  "result": {
    "route_source": "regex+dataset",
    "faq_confidence": 0.92,
    "semantic_match_score": 0.0,
    "dataset_match_score": 0.91,
    "fuzzy_match_used": true,
    "ambiguity_detected": false,
    "policy_source": "cag",
    "latency_ms": 42.1
  }
}
```

## Config Contracts

- `OTEL_EXPORTER_OTLP_ENDPOINT` enables OTLP export.
- logging config controls verbosity.
- FAQ and orchestration config control whether explainability metadata is returned to clients.

## Response Semantics

Client responses may include a `routing` object for diagnostics. This object must contain operational explanations, not chain-of-thought or hidden prompts.

## Fallback Behavior

If OpenTelemetry is unavailable, structured logs and `agent_trace` remain available.

## Latency Expectations

Observability overhead should stay below 20 ms per request in default local logging mode.

## CPU Constraints

Tracing must not serialize large provider payloads or full dataframe snapshots. Only counts, keys, scores, and bounded strings are logged.

## Extensibility Expectations

New agents must emit the same trace shape: `agent`, `action`, `status`, `result`, `latency_ms`, and optional `error`.

## Failure Handling

Observability failures must never break user requests. Exporter failures are logged and ignored.

## Observability Requirements

Required events:

- `faq_route`
- `booking_route`
- `hybrid_route`
- `semantic_match`
- `regex_match`
- `fuzzy_match`
- `ambiguity_resolution`
- `CAG_policy_used`
- `DatasetFacts_used`
- `confidence_scores`
- layer latency

