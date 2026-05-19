# AI Service Orchestrator for the Informal Economy

This repository is a hackathon monorepo for an agentic AI service booking system. The backend is a Python/FastAPI AI orchestrator for intent extraction, provider discovery, ranking, booking, and follow-up workflows. The mobile app is the Flutter client used by end users to chat with the AI assistant and browse providers.

## Folder Structure

```text
.
├── backend/
│   └── generative_ai_project/       # Python/FastAPI agentic backend
├── mobile/
│   └── ProFixer/                    # Flutter mobile application
├── frontend/
│   └── README.md                    # Optional future web frontend notes
├── docs/
│   ├── architecture/                # Architecture PDFs and notes
│   ├── challenge/                   # Hackathon challenge PDF and generated pages
│   └── setup/                       # Setup and structure guides
├── scripts/                         # Shared repo-level helper scripts
├── README.md
└── .gitignore
```

## Backend Location

Backend code lives in:

```text
backend/generative_ai_project
```

This folder contains the FastAPI app, agent code, model/config files, RAG components, tests, Dockerfile, docker-compose file, and environment example.

Key paths:

- `backend/generative_ai_project/src/api/app.py`
- `backend/generative_ai_project/src/agents/`
- `backend/generative_ai_project/config/`
- `backend/generative_ai_project/requirements.txt`
- `backend/generative_ai_project/.env.example`

## Mobile App Location

The Flutter app lives in:

```text
mobile/ProFixer
```

This is the mobile client for the backend. It includes Android, iOS, web, desktop platform folders, `lib/`, `test/`, `pubspec.yaml`, and `pubspec.lock`.

## Frontend Status

A separate web frontend is optional for the hackathon and is not currently implemented. Flutter web support exists under:

```text
mobile/ProFixer/web
```

A dedicated web frontend can be added later under `frontend/` if needed.

## Local Setup Instructions

Setup commands are documented as placeholders only:

- Backend: `docs/setup/BACKEND_SETUP.md`
- Mobile: `docs/setup/MOBILE_SETUP.md`
- Structure: `docs/setup/PROJECT_STRUCTURE.md`
- Smoke tests: `docs/setup/SMOKE_TESTS.md`

Do not commit local virtual environments, dependency folders, build outputs, cache folders, or secrets.

## GCP Deployment Note

For GCP, deploy the backend from `backend/generative_ai_project` and configure the Flutter app to point at the deployed API/WebSocket URL. Keep runtime secrets in environment variables or GCP Secret Manager, not in git.

## Important Git Hygiene

Do not commit:

- `.env`
- cache folders
- `__pycache__/`
- `.pytest_cache/`
- Flutter/Dart build folders
- Android/iOS generated build artifacts
- local editor folders such as `.vscode/` or `.idea/`
