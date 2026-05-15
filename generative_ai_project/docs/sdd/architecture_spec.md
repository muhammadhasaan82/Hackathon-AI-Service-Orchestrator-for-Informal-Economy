# Architecture Specification — SDD Document

## 1. System Overview

The AI Service Orchestrator is a **decentralized, multi-agent system** that automates the full lifecycle of service requests for Pakistan's informal economy. It processes natural language input (English, Urdu, Roman Urdu) and routes through a pipeline of specialized agents to deliver provider recommendations and booking confirmations.

## 2. Technology Stack (Fully Open-Source)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Gemma 4 E4B (via Ollama) | Reasoning, NLU, text generation |
| Embeddings | BAAI/bge-large-en-v1.5 | 1024-dim semantic vectors |
| Reranking | BAAI/bge-reranker-base | Cross-encoder precision re-scoring |
| Vector Store | Weaviate | Hybrid semantic + keyword search |
| Session Cache | Redis | Short-term conversation state |
| DBMS | PostgreSQL | Booking records, episodic memory |
| CAG | Rust/PyO3 | KV-cache for golden knowledge |
| Fine-Tuning | LoRA/QLoRA (PEFT) | Domain adaptation for informal economy |
| Orchestration | Google ADK V2 | Agent framework with LiteLLM bridge |
| API | FastAPI | REST endpoints with SSE streaming |
| Observability | OpenTelemetry | Distributed tracing and spans |
| Deployment | Docker Compose | Container orchestration |

## 3. Hybrid AI Knowledge Engine (4-Tier)

### Tier 1: Prompt Engineering + Context Engineering
- System-level context assembly via `ContextEngine`
- Dynamic prompt templates from `prompts_config.yaml`
- Token budget management with priority-based truncation
- Anti-hallucination guardrails

### Tier 2: Agentic RAG + Reranking
- **Phase 1**: Weaviate metadata filtering (city, category, availability)
- **Phase 2**: BGE embedding similarity search (cosine distance)
- **Phase 3**: Cross-encoder reranking (BAAI/bge-reranker-base)
- 50K provider dataset indexed with 1024-dim vectors

### Tier 3: Cache-Augmented Generation (CAG)
- Static "Golden Knowledge" preloaded into context window
- Rust PyO3 KV-cache for thread-safe, low-latency access
- Domain: service categories, city areas, Roman Urdu mappings, business rules
- Python fallback when Rust module not compiled

### Tier 4: Fine-Tuning (LoRA)
- QLoRA 4-bit quantization on Gemma 4
- Target modules: q_proj, k_proj, v_proj, o_proj
- Domain-specific training data: intent parsing in Roman Urdu
- Hot-swappable adapter loading at inference time

## 4. Agent Pipeline

```
User Message → [Intent Agent] → [Discovery Agent] → [Ranking Agent] → [Booking Agent] → [Follow-Up Agent]
                    ↓                   ↓                  ↓                ↓                  ↓
              NLU/JSON          RAG+Rerank          Score+Reason      Create Booking       Reminder
```

Each agent step produces an OpenTelemetry span for full observability.

## 5. State Management

- **Redis**: Conversation history, agent state, reasoning traces (TTL-based expiry)
- **PostgreSQL**: Booking records with ACID guarantees, event sourcing
- **Graceful Fallback**: In-memory dict (sessions) and SQLite (bookings) when services unavailable

## 6. Data Flow

1. User sends message via `POST /api/v1/chat`
2. Orchestrator creates/resumes session in Redis
3. Intent Agent extracts structured intent via Gemma 4
4. Discovery Agent queries Weaviate with BGE embeddings + metadata filters
5. Reranker rescores top-K results with cross-encoder
6. Ranking Agent computes deterministic scores from `scoring_config.yaml`
7. Gemma 4 generates natural language reasoning for rankings
8. Booking Agent creates record in PostgreSQL
9. Full response returned with reasoning trace
