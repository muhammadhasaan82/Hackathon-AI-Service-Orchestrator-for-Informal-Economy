# Agent Specifications — SDD Document

## Guardrail Engine

- **Role**: Policy enforcement across the entire conversational workflow
- **Input**: Raw user message, extracted intent, ranked providers, outbound payloads
- **Output**: Policy decision, clarification trigger, confirmation resolution, or sanitized payload
- **Source of Truth**: `config/guardrails_config.yaml`
- **Runtime Component**: `src/guardrails/engine.py`

## Agent 1: Intent Agent

- **Role**: Multilingual natural language understanding
- **Input**: User message, recent conversation history, available service categories, available cities
- **Probabilistic Layer**: Gemma 4 extracts structured intent from English, Urdu, or Roman Urdu
- **Deterministic Layer**: Field normalization against known categories and cities
- **Output**: Structured JSON `{service_type, city, area, time_preference, urgency, language_detected, price_preference, confidence, needs_clarification}`
- **Guardrails Applied**:
  - runtime LLM instructions from `GuardrailEngine.compose_system_instruction("intent")`
  - clarification triggered by missing or low-confidence required fields
- **Failure Mode**: Returns empty intent structure with clarification requirements

## Agent 2: Discovery Agent

- **Role**: Provider search via Agentic RAG
- **Input**: Structured intent and optional user coordinates
- **Deterministic Layer**:
  - metadata filtering by category, city, area, and availability
  - fallback broadening sequence when results are sparse
- **Output**: Candidate providers with retrieval and rerank metadata
- **Tools**: Weaviate vector search, embedding generation, reranking pipeline
- **Failure Mode**: Falls back from filtered search to broader search to unfiltered fallback query

## Agent 3: Ranking Agent

- **Role**: Deterministic scoring plus grounded recommendation reasoning
- **Input**: Candidate providers, intent, scoring config, optional user coordinates
- **Deterministic Layer**:
  - weighted composite score from `config/scoring_config.yaml`
  - reproducible ranking order for the same inputs
- **Probabilistic Layer**: Gemma 4 explains why the shortlisted providers fit the request
- **Output**: Top-N ranked providers with `score_result` and natural language reasoning
- **Guardrails Applied**:
  - ranking explanation constrained to already-ranked providers
  - outbound reasoning sanitized before return

## Agent 4: Booking Agent

- **Role**: Booking finalization after confirmation
- **Input**: Selected provider, structured intent, `session_id`, target booking status
- **Deterministic Layer**:
  - time slot generation
  - booking persistence in PostgreSQL or SQLite fallback
- **Probabilistic Layer**: Gemma 4 generates user-facing booking confirmation copy
- **Output**: Booking record with confirmation message
- **Guardrails Applied**:
  - booking is created by the orchestrator only after confirmation gate success in conversational mode
  - booking copy is generated under shared guardrail instructions and sanitized before return

## Agent 5: Follow-Up Agent

- **Role**: Post-booking lifecycle automation
- **Input**: Booking object and current intent
- **Deterministic Layer**:
  - reminder schedule from configured reminder tiers and urgency multipliers
  - status timeline and post-completion actions from config
- **Probabilistic Layer**: Gemma 4 writes concise notification text per event and reminder
- **Output**: Follow-up plan with scheduled reminders, status events, post-completion actions, and immediate notification
- **Guardrails Applied**:
  - shared notification instructions injected into LLM calls
  - generated payloads sanitized before return

## Orchestrator (Root Agent)

- **Role**: Workflow routing, state management, and guardrail orchestration
- **Input**: User message, `session_id`, optional coordinates
- **Deterministic Responsibilities**:
  - evaluate input guardrails before agent execution
  - merge multi-turn intent state
  - persist session state and reasoning trace
  - enforce explicit confirmation before conversational booking
- **Output**: Full response with status, trace, intent, ranked providers, and optional booking/follow-up payloads
- **Key Session State**:
  - `intent`
  - `ranked_providers`
  - `selected_provider`
  - `booking`
  - `followup`
  - `awaiting_booking_confirmation`
- **Observability**: OpenTelemetry spans for each stage
