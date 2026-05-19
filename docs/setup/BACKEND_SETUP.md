# Backend Setup

Backend source lives in:

```text
backend/generative_ai_project
```

The backend is the Python/FastAPI agentic AI service orchestrator. It handles natural-language intent extraction, provider discovery, ranking, booking, and follow-up workflows.

## Placeholder Local Commands

Run these later on your local machine or GCP VM after choosing the correct Python/runtime environment:

```bash
cd backend/generative_ai_project
python -m venv .venv
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Environment

Use `.env.example` as the template for required variables. Do not commit `.env`.

For GCP, store secrets in environment variables, Secret Manager, or your deployment platform configuration.
