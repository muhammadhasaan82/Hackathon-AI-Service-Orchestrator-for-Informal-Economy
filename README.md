# 🤖 AI Service Orchestrator for Informal Economy
 
**Google Antigravity Hackathon 2026** | Fully Open-Source Stack | No API Keys Required
 
An Agentic AI system powered by **Gemma 4** that handles multilingual service requests for Pakistan's informal economy through a **guardrail-first**, **config-driven** workflow: input safety checks, multilingual intent extraction, provider discovery, deterministic ranking, explicit booking confirmation, and follow-up automation.
 
## 🏗️ Architecture
 
``` 
User Message (English/Urdu/Roman Urdu)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Config-Driven Orchestrator                   │
│ Guardrails + Session State + Prompt Utilities│
└───────────────┬──────────────────────────────┘
                │
        ┌────────▼────────┐
        │ Guardrail Layer │
        │ input / intent  │
        │ output / policy │
        └────────┬────────┘
                │
  ┌────▼────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  ┌──────────┐
  │ Intent  │→ │Discovery │→ │ Ranking  │→ │ Confirmation Gate  │→ │Follow-Up │
  │  Agent  │  │  Agent   │  │  Agent   │  │ + Booking Agent    │  │  Agent   │
  └────┬────┘  └────┬─────┘  └────┬─────┘  └─────────┬──────────┘  └──────────┘
       │            │             │                  │
  Gemma 4      RAG+Rerank   Deterministic      PostgreSQL / SQLite
  (JSON NLU)   (Weaviate)   score + LLM        confirmed booking
```
 
## 🧠 Hybrid AI Knowledge Engine (7-Tier)
 
| Tier | Technology | Purpose |
|------|-----------|---------|
| **Guardrails** | `guardrails_config.yaml` + `GuardrailEngine` | Soft-coded policy enforcement for input, intent, booking, and output |
| **Prompt Engineering** | YAML-driven templates | Context-engineered system prompts |
| **Context Engineering** | `ContextEngine` class | System-level information flow design |
| **Agentic RAG** | BGE + Weaviate | Semantic search over 50K providers |
| **Reranking** | BGE-Reranker (Cross-Encoder) | Precision re-scoring of candidates |
| **CAG** | Rust/PyO3 KV-Cache | Golden knowledge in context window |
| **Fine-Tuning** | LoRA/QLoRA on Gemma 4 | Domain adaptation for informal economy |
 
## 🚀 Quick Start
 
### Prerequisites
- **Docker** (for Weaviate, Redis, PostgreSQL)
- **Python 3.12+**
- **Unsloth-compatible Gemma weights** for the default runtime, or **Ollama** if you switch providers in config/environment
- **Rust** (optional, for CAG performance)
 
### 1. Configure Model Runtime
Default runtime is **Unsloth**. Set your environment before starting the API:
```bash
set MODEL_PROVIDER=unsloth
set UNSLOTH_MODEL_ID=<your-local-or-hf-model-id>
set HF_TOKEN=<your-huggingface-token-if-required>
```
 
If you prefer Ollama instead, switch the provider and pull the model:
```bash
set MODEL_PROVIDER=ollama
ollama pull gemma4:31b
```
 
### 2. Start Infrastructure
```bash
cd generative_ai_project
docker-compose up -d
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
 
### 7. Test the Guarded Booking Flow
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mujhe kal subah G-13 mein AC technician chahiye", "user_lat": 33.6844, "user_lon": 73.0479}'
```
 
The first response should return ranked options with status like `awaiting_booking_confirmation`.
Then confirm in the same session:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<same-session-id>", "message": "book option 1"}'
```
 
## 📡 API Endpoints
 
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Main conversational endpoint |
| `/api/v1/health` | GET | System health check |
| `/api/v1/agents/trace/{session_id}` | GET | Full reasoning trace |
| `/api/v1/bookings/{booking_id}` | GET | Booking status |
| `/api/v1/bookings/{booking_id}/cancel` | POST | Cancel booking |
| `/api/v1/discovery/categories` | GET | Service categories |
| `/api/v1/discovery/cities` | GET | Available cities |
 
## ⚙️ Configuration (Fully Soft-Coded)
 
| File | Purpose |
|------|---------|
| `generative_ai_project/config/model_config.yaml` | Gemma runtime selection, generation settings, reranking config |
| `generative_ai_project/config/agents_config.yaml` | Agent definitions, routing rules |
| `generative_ai_project/config/scoring_config.yaml` | Provider ranking weights |
| `generative_ai_project/config/prompts_config.yaml` | Context-engineered prompt templates |
| `generative_ai_project/config/guardrails_config.yaml` | Input, intent, booking, and output guardrail policies |
| `generative_ai_project/config/fine_tuning_config.yaml` | LoRA training hyperparameters |
| `generative_ai_project/config/logging_config.yaml` | OpenTelemetry tracing config |
| `generative_ai_project/data/cache/golden_knowledge.yaml` | CAG static domain knowledge |
 
## 🔧 Tech Stack (100% Open-Source)
 
| Component | Technology |
|-----------|-----------|
| LLM | Gemma 4 via Unsloth or Ollama (Apache 2.0) |
| Embeddings | BAAI/bge-m3 — multilingual EN/UR/Roman-UR (MIT) |
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

- `generative_ai_project/docs/sdd/architecture_spec.md` — System architecture
- `generative_ai_project/docs/sdd/agent_specs.md` — Agent interface contracts
- `generative_ai_project/docs/sdd/api_spec.md` — API specification

## License
Built for Google Antigravity Hackathon 2026
