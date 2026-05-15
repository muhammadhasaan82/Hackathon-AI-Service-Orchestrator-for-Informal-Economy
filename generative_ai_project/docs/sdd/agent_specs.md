# Agent Specifications — SDD Document

## Agent 1: Intent Agent
- **Role**: Natural Language Understanding
- **Input**: User message (English/Urdu/Roman Urdu), conversation history
- **Output**: Structured JSON `{service_type, city, area, time_preference, urgency, language_detected, price_preference, confidence}`
- **Model**: Gemma 4 E4B with structured JSON output
- **Tools**: None (pure LLM reasoning)
- **Failure**: Returns `needs_clarification` list for missing fields

## Agent 2: Discovery Agent
- **Role**: Provider search via Agentic RAG
- **Input**: Structured intent from Intent Agent
- **Output**: List of candidate providers with retrieval + rerank scores
- **Tools**: Weaviate vector search, BGE embedder, BGE reranker
- **Failure**: 3-phase fallback (filtered → relaxed → unfiltered)

## Agent 3: Ranking Agent
- **Role**: Deterministic scoring + probabilistic reasoning
- **Input**: Candidate providers, intent, scoring config
- **Output**: Top-N ranked providers with score breakdowns + natural language reasoning
- **Tools**: `score_provider()` (deterministic), Gemma 4 (reasoning)
- **Failure**: Returns best available even if below threshold

## Agent 4: Booking Agent
- **Role**: Booking simulation and confirmation
- **Input**: Selected provider, intent, session_id
- **Output**: Booking record with confirmation message
- **Tools**: `BookingStore.create_booking()`, `generate_time_slot()`
- **Failure**: Returns error with retry suggestion

## Agent 5: Follow-Up Agent
- **Role**: Post-booking automation
- **Input**: Booking ID
- **Output**: Reminder messages, status updates
- **Tools**: `BookingStore.get_booking()`, Gemma 4 (message generation)

## Orchestrator (Root Agent)
- **Role**: Pipeline routing and session management
- **Input**: User message, session_id
- **Output**: Full response with reasoning trace, booking, provider list
- **Sub-agents**: All 5 agents above
- **State**: Redis session with conversation history and reasoning trace
- **Observability**: OpenTelemetry spans for each step
