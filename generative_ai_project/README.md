# 🤖 AI Service Orchestrator for Informal Economy

**Google Antigravity Hackathon 2026** | Fully Open-Source Stack | No API Keys Required

An Agentic AI system powered by **Gemma 4** that automates the end-to-end lifecycle of service requests — from multilingual intent understanding through provider discovery, probabilistic ranking, booking simulation, and follow-up automation.

## 🏗️ Architecture

```
User Message (English/Urdu/Roman Urdu)
    │
    ▼
┌──────────────────────────────────────┐
│   Google ADK V2 Orchestrator         │
│   (Triage Router + Context Engine)   │
└──────┬───────────────────────────────┘
       │
  ┌────▼────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Intent  │→ │Discovery │→ │ Ranking  │→ │ Booking  │→ │Follow-Up │
  │  Agent  │  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │
  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘
       │            │             │              │
  Gemma 4      RAG+Rerank    Score+LLM      PostgreSQL
  (NLU)        (Weaviate)    (Reasoning)    (Bookings)
```

## 🧠 Hybrid AI Knowledge Engine (6-Tier)

| Tier | Technology | Purpose |
|------|-----------|---------|
| **Prompt Engineering** | YAML-driven templates | Context-engineered system prompts |
| **Context Engineering** | `ContextEngine` class | System-level information flow design |
| **Agentic RAG** | BGE + Weaviate | Semantic search over 50K providers |
| **Reranking** | BGE-Reranker (Cross-Encoder) | Precision re-scoring of candidates |
| **CAG** | Rust/PyO3 KV-Cache | Golden knowledge in context window |
| **Fine-Tuning** | LoRA/QLoRA on Gemma 4 | Domain adaptation for informal economy |

## 🚀 Quick Start

### Prerequisites
- **Docker** (for Weaviate, Redis, PostgreSQL)
- **Ollama** (for Gemma 4 local inference)
- **Python 3.12+**
- **Rust** (optional, for CAG performance)

### 1. Start Infrastructure
```bash
cd generative_ai_project
docker-compose up -d
```

### 2. Pull Gemma 4 Model
```bash
ollama pull gemma4:e4b
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Embeddings (first time only)
```bash
python scripts/build_embeddings.py
```

### 5. (Optional) Build Rust CAG Module
```bash
cd rust_cag
pip install maturin
maturin develop --release
cd ..
```

### 6. Start Server
```bash
uvicorn src.api.app:app --reload --port 8000
```

### 7. Test
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mujhe kal subah G-13 mein AC technician chahiye"}'
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Main conversational endpoint |
| `/api/v1/health` | GET | System health check |
| `/api/v1/agents/trace/{session_id}` | GET | Full reasoning trace |
| `/api/v1/bookings/{booking_id}` | GET | Booking status |
| `/api/v1/bookings/{booking_id}/cancel` | POST | Cancel booking |
| `/api/v1/providers/categories` | GET | Service categories |
| `/api/v1/providers/cities` | GET | Available cities |

## ⚙️ Configuration (Fully Soft-Coded)

| File | Purpose |
|------|---------|
| `config/model_config.yaml` | Gemma 4 + Ollama settings, reranking config |
| `config/agents_config.yaml` | Agent definitions, routing rules |
| `config/scoring_config.yaml` | Provider ranking weights |
| `config/prompts_config.yaml` | Context-engineered prompt templates |
| `config/fine_tuning_config.yaml` | LoRA training hyperparameters |
| `config/logging_config.yaml` | OpenTelemetry tracing config |
| `data/cache/golden_knowledge.yaml` | CAG static domain knowledge |

## 🔧 Tech Stack (100% Open-Source)

| Component | Technology |
|-----------|-----------|
| LLM | Gemma 4 E4B via Ollama (Apache 2.0) |
| Embeddings | BAAI/bge-large-en-v1.5 (MIT) |
| Reranker | BAAI/bge-reranker-base (MIT) |
| Vector Store | Weaviate (BSD-3) |
| Session Cache | Redis (BSD-3) |
| DBMS | PostgreSQL (PostgreSQL License) |
| CAG Cache | Rust + PyO3 (Apache 2.0) |
| Agent Framework | Google ADK V2 (Apache 2.0) |
| API | FastAPI (MIT) |
| Observability | OpenTelemetry (Apache 2.0) |

## 📊 Dataset

50,000 service providers across 6 Pakistani cities:
- **14 Categories**: Plumber, Electrician, AC Technician, Beautician, Tutor, Carpenter, etc.
- **6 Cities**: Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Peshawar
- **Rich Metadata**: Geo-coordinates, ratings, availability, experience, pricing, verification

## 📐 SDD Documentation

- `docs/sdd/architecture_spec.md` — System architecture
- `docs/sdd/agent_specs.md` — Agent interface contracts
- `docs/sdd/api_spec.md` — API specification

## License
Built for Google Antigravity Hackathon 2026
