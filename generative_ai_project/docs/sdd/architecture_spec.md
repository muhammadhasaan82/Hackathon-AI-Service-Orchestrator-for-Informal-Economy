# Architecture Specification — SDD Document

## 1. System Overview

The AI Service Orchestrator is a **guardrail-first, config-driven multi-agent system** for Pakistan's informal economy. It processes natural language requests in English, Urdu, and Roman Urdu, discovers relevant local providers, ranks them with deterministic math, and requires explicit user confirmation before finalizing a booking in the conversational flow.

The implementation is intentionally split into:

- **Deterministic layers** for policy checks, retrieval filters, scoring, state transitions, and follow-up scheduling
- **Probabilistic layers** for multilingual intent interpretation, ranking explanations, booking copy, and follow-up message phrasing
- **Soft-coded control planes** in YAML so behavior can be tuned without changing core agent code

## 2. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM Runtime | Gemma 4 via Unsloth or Ollama | Intent extraction, reasoning, message generation |
| Embeddings | BAAI/bge-m3 | Multilingual semantic query + provider representation (EN / Urdu / Roman Urdu) |
| Reranking | BAAI/bge-reranker-base | Precision candidate re-ordering after retrieval |
| Vector Store | Weaviate | Hybrid retrieval over provider metadata + vectors |
| Session State | Redis with in-memory fallback | Conversation history, ranked options, booking workflow state |
| Booking Store | PostgreSQL with SQLite fallback | Booking persistence and status transitions |
| CAG | Rust/PyO3 with Python fallback | Golden knowledge lookup |
| API | FastAPI | Mobile-facing REST interface |
| Observability | OpenTelemetry | Request and agent tracing |
| Deployment | Docker Compose | Local orchestration for core services |

## 3. Soft-Coded Control Layers

### 3.1 Guardrails

- Source of truth: `config/guardrails_config.yaml`
- Runtime component: `src/guardrails/engine.py`
- Policy domains:
  - input validation and risk patterns
  - intent confidence thresholds
  - booking confirmation rules
  - output sanitization and internal detail redaction

### 3.2 Agent Behavior

- Source of truth: `config/agents_config.yaml`
- Defines pipeline roles, search behavior, reminder tiers, lifecycle events, and notification actions

### 3.3 Deterministic Ranking

- Source of truth: `config/scoring_config.yaml`
- Governs weights, availability scoring, distance decay, verification bonuses, and recommendation thresholds

### 3.4 Prompting and Context

- Source of truth: `config/prompts_config.yaml`
- LLM instructions are dynamically combined with runtime guardrail directives before generation

## 4. Hybrid AI Knowledge Engine

### Tier 1: Guardrails

- Blocks prompt injection and internal prompt exposure requests
- Redirects out-of-scope queries
- Escalates emergency-like requests away from the booking workflow
- Enforces explicit confirmation before conversational booking

### Tier 2: Prompt + Context Engineering

- Prompt templates are YAML-driven
- Runtime guardrail directives are appended to model instructions
- Conversation history is injected for better multi-turn intent continuity

### Tier 3: Agentic RAG + Reranking

- Discovery applies metadata filtering and fallback broadening
- Retrieval supports user coordinates for distance-aware search
- Reranking refines candidate ordering before deterministic scoring

### Tier 4: Deterministic Scoring

- Provider ranking uses fixed math from config
- Scores are grounded in distance, rating, availability, experience, response time, price preference, and verification bonus
- Results are reproducible for the same inputs

### Tier 5: CAG

- Golden knowledge supplements prompts with domain-specific context
- Rust acceleration is optional; Python fallback is supported

### Tier 6: Fine-Tuning

- Optional LoRA/QLoRA path remains part of the design for domain adaptation

### Tier 7: Follow-Up Automation

- Schedule structure is deterministic from config
- Notification phrasing is generated probabilistically by the LLM under guardrail constraints

## 5. Agent Pipeline

```
User Message
  → Input Guardrails
  → Intent Agent
  → Intent Guardrails / Clarification Gate
  → Discovery Agent
  → Ranking Agent
  → Confirmation Gate
  → Booking Agent
  → Follow-Up Agent
```

Each major stage emits OpenTelemetry spans and persists workflow state into the session store.

## 6. State Management

### Session Store

Stored per `session_id`:

- conversation history
- current structured intent
- ranked provider shortlist
- selected provider
- booking object
- follow-up plan
- `awaiting_booking_confirmation` flag
- reasoning trace

### Booking Store

Booking persistence includes:

- booking record
- lifecycle status (`PENDING`, `CONFIRMED`, `IN_PROGRESS`, `COMPLETED`, `CANCELLED`)
- event log of booking transitions

## 7. Data Flow

1. Client sends `POST /api/v1/chat` with message and optional coordinates.
2. Orchestrator creates or resumes the session.
3. Input guardrails evaluate the message before any downstream agent work.
4. Intent Agent extracts structured intent using multilingual LLM reasoning.
5. Intent guardrails decide whether to clarify, escalate, or proceed.
6. Discovery Agent retrieves provider candidates with retrieval fallback logic.
7. Ranking Agent computes deterministic scores and generates a grounded explanation.
8. Orchestrator stores the shortlist and returns `awaiting_booking_confirmation` when confirmation is required.
9. User confirms a specific option on a later turn.
10. Booking Agent creates a confirmed booking record.
11. Follow-Up Agent generates a deterministic notification schedule plus guarded LLM-generated copy.
12. Final response returns booking, follow-up plan, trace, and ranked provider payload.
